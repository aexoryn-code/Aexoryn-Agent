# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
core/planner.py
───────────────
Task Planner & Executor — translates a user task into an ordered execution
plan, dispatches the jury, applies the consensus solution to the workspace,
and runs any follow-up terminal commands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from core.consensus import ConsensusEngine
from core.memory import MemoryStore
from core.registry import ModelRegistry
from core.tools import CommandResult, DiffTool, FileSystemTool, TerminalTool


# ── System Prompts ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are the Aexoryn Agent Task Planner.
Analyze the user task and produce a precise, numbered execution plan.

Rules:
- Each step must be one of: READ_FILE, WRITE_FILE, PATCH_FILE, CREATE_DIR,
  DELETE_FILE, RUN_COMMAND, or THINK.
- Output ONLY valid JSON. No markdown fences, no prose.
- Format: {"plan": [{"step": 1, "action": "...", "target": "...", "detail": "..."}]}
- For THINK steps, describe what reasoning is needed.
- Keep steps minimal and non-redundant.
"""

SOLVER_SYSTEM = """\
You are an expert software engineer working inside the Aexoryn Agent.
You will be given a task and relevant file context.
Produce complete, production-ready code or changes.

If the task requires MULTIPLE files, use this exact format:
=== FILE: path/to/file.ext ===
<complete file content here>
=== FILE: another/file.ext ===
<complete file content here>

If the task requires only ONE file, output the file content directly.
Output ONLY file content — no explanations, no markdown code fences.
"""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step: int
    action: str      # READ_FILE | WRITE_FILE | PATCH_FILE | CREATE_DIR | DELETE_FILE | RUN_COMMAND | THINK
    target: str      # file path or command string
    detail: str      # extra context for the action


@dataclass
class ExecutionResult:
    success: bool
    steps_completed: int
    files_modified: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    diffs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    master_solution: str = ""
    total_tokens: int = 0
    contributing_models: List[str] = field(default_factory=list)
    jury_total: int = 0
    finalists: int = 0
    consensus_mode: str = ""
    workspace_files_read: List[str] = field(default_factory=list)
    iterations: int = 0
    test_passed: Optional[bool] = None


# ── Planner ───────────────────────────────────────────────────────────────────

class TaskPlanner:
    """
    End-to-end pipeline:
      1. Call a planning model to decompose the task into steps.
      2. Execute READ steps to gather context.
      3. Dispatch the full jury for the solution phase.
      4. Run Super-Judge consensus.
      5. Apply the solution (WRITE / PATCH / RUN_COMMAND steps).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        fst: FileSystemTool,
        terminal: TerminalTool,
        ui_callback: Optional[Callable[[str, str], None]] = None,
        confirm_callback: Optional[Callable[[str, str, str], bool]] = None,
    ) -> None:
        self.registry = registry
        self.fst = fst
        self.terminal = terminal
        self.consensus = ConsensusEngine(registry)
        self.memory = MemoryStore()
        self._log = ui_callback or (lambda status, msg: logger.info(f"[{status}] {msg}"))
        self._confirm = confirm_callback  # fn(filepath, old_content, new_content) -> bool

    def _emit(self, status: str, message: str) -> None:
        self._log(status, message)

    # ── Plan Generation ───────────────────────────────────────────────────────

    async def generate_plan(self, task: str) -> List[PlanStep]:
        self._emit("THINKING", "Decomposing task into execution plan…")
        # Try judge model first; if it fails, fall back to best available model
        model = self.registry.get_judge_model()
        result = await self.registry.call_model(
            model,
            messages=[{"role": "user", "content": f"Task:\n{task}"}],
            system_prompt=PLANNER_SYSTEM,
        )
        if result.get("error") or not result.get("content"):
            enabled = self.registry.get_enabled_models()
            fallback_model = next(
                (m for m in enabled if m.id != model.id), None
            )
            if fallback_model:
                logger.info(f"[Planner] Judge unavailable, using {fallback_model.id} for planning.")
                result = await self.registry.call_model(
                    fallback_model,
                    messages=[{"role": "user", "content": f"Task:\n{task}"}],
                    system_prompt=PLANNER_SYSTEM,
                )
        raw = result["content"].strip() if result.get("content") else ""
        try:
            parsed = json.loads(raw)
            steps = [PlanStep(**s) for s in parsed.get("plan", [])]
            self._emit("PLAN", f"Generated {len(steps)} step(s).")
            return steps
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"[Planner] Could not parse plan JSON: {exc}. Using smart fallback.")
            return self._smart_fallback_plan(task)

    def _smart_fallback_plan(self, task: str) -> List[PlanStep]:
        """Infer a best-effort plan from the task string when the planner model fails."""
        import re as _re
        lower = task.lower()

        # Try to detect a filename mentioned in the task
        file_match = _re.search(
            r'[\w\-\.]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|txt|sh|sql)',
            task, _re.IGNORECASE
        )
        filename = file_match.group(0) if file_match else None

        # If no filename, infer one from task keywords
        if not filename:
            if any(k in lower for k in ["html", "webpage", "web page", "website"]):
                filename = "index.html"
            elif any(k in lower for k in ["react", "component", "tsx"]):
                filename = "Component.tsx"
            elif any(k in lower for k in ["python", ".py", "script", "flask", "fastapi"]):
                filename = "main.py"
            elif any(k in lower for k in ["css", "style", "stylesheet"]):
                filename = "styles.css"
            elif any(k in lower for k in ["javascript", "js", "node"]):
                filename = "index.js"

        if filename:
            self._emit("PLAN", f"Smart fallback: WRITE_FILE → {filename}")
            return [PlanStep(step=1, action="WRITE_FILE", target=filename, detail=task)]

        # Generic fallback — just think
        return [PlanStep(step=1, action="THINK", target="", detail=task)]

    # ── Context Gathering ─────────────────────────────────────────────────────

    def _gather_context(self, steps: List[PlanStep]) -> str:
        context_parts: List[str] = []
        for s in steps:
            if s.action == "READ_FILE" and s.target:
                try:
                    content = self.fst.read_file(s.target)
                    context_parts.append(f"### {s.target}\n```\n{content}\n```")
                    self._emit("READING", f"Read file: {s.target}")
                except Exception as exc:
                    logger.warning(f"[Planner] Could not read {s.target}: {exc}")
        return "\n\n".join(context_parts)

    # ── Auto-Scan Workspace ───────────────────────────────────────────────────

    _CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
        ".json", ".yaml", ".yml", ".md", ".sql", ".sh", ".txt",
        ".vue", ".svelte", ".php", ".go", ".rs", ".java", ".cs",
    }
    _MAX_FILE_CHARS = 3000   # truncate each file to avoid context overflow
    _MAX_TOTAL_CHARS = 12000 # total workspace context cap
    _SKIP_DIRS = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "dist", "build", ".next", ".nuxt", ".cache",
    }

    def _auto_scan_workspace(self, task: str) -> tuple[str, List[str]]:
        """
        Scan workspace directory and read files relevant to the task.
        Returns (context_string, list_of_files_read).
        """
        workspace = self.fst.root
        if not workspace.exists():
            return "", []

        task_lower = task.lower()
        task_keywords = set(re.findall(r'\w+', task_lower))

        # Collect candidate files
        candidates: List[tuple[int, Path]] = []
        for path in workspace.rglob("*"):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in self._SKIP_DIRS):
                continue
            if path.suffix.lower() not in self._CODE_EXTENSIONS:
                continue

            # Score relevance: filename keywords matching task
            name_keywords = set(re.findall(r'\w+', path.stem.lower()))
            score = len(name_keywords & task_keywords)
            candidates.append((score, path))

        # Sort by relevance, then recency (mtime)
        candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)

        parts: List[str] = []
        files_read: List[str] = []
        total_chars = 0

        for score, path in candidates[:20]:  # max 20 files considered
            if total_chars >= self._MAX_TOTAL_CHARS:
                break
            try:
                rel = path.relative_to(workspace)
                content = path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue
                snippet = content[:self._MAX_FILE_CHARS]
                if len(content) > self._MAX_FILE_CHARS:
                    snippet += f"\n… [{len(content) - self._MAX_FILE_CHARS} chars truncated]"
                chunk = f"### {rel}\n```{path.suffix.lstrip('.')}\n{snippet}\n```"
                parts.append(chunk)
                files_read.append(str(rel))
                total_chars += len(chunk)
            except Exception:
                continue

        if files_read:
            self._emit("READING", f"Auto-scanned {len(files_read)} workspace file(s): {', '.join(files_read[:3])}{'…' if len(files_read) > 3 else ''}")

        return "\n\n".join(parts), files_read

    # ── Solution Execution ────────────────────────────────────────────────────

    async def execute(
        self,
        task: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(success=False, steps_completed=0)
        total_tokens = 0
        history = history or []

        # 1. Plan
        plan = await self.generate_plan(task)

        # 2. Gather explicit read context from plan steps
        plan_context = self._gather_context(plan)

        # 3. Auto-scan workspace for relevant files
        ws_context, ws_files = self._auto_scan_workspace(task)
        result.workspace_files_read = ws_files

        # Recall relevant past solutions from long-term memory
        memory_context = self.memory.recall(task, top_k=3)
        if memory_context:
            self._emit("INFO", "Long-term memory: injecting relevant past solutions as context.")

        # Merge contexts
        context_parts = [c for c in [plan_context, ws_context, memory_context] if c]
        context = "\n\n".join(context_parts)

        # 4. Determine routing tier
        estimated_tokens = len(task.split()) * 2 + len(context.split()) * 2
        tier = self.registry.route(estimated_tokens, tags=["code"])
        self._emit("ROUTING", f"Dispatching jury (tier={tier}, ~{estimated_tokens} tokens)…")

        # 5. Build messages — prepend conversation history for chat continuity
        user_content = f"Task:\n{task}"
        if context:
            user_content += f"\n\nWorkspace Context:\n{context}"

        messages = history + [{"role": "user", "content": user_content}]

        # 6. Parallel jury call
        self._emit("JURY", f"Calling {len(self.registry.get_enabled_models(tier=tier))} model(s) in parallel…")
        jury_results = await self.registry.call_jury(
            messages=messages,
            system_prompt=SOLVER_SYSTEM,
            tier=tier or None,
        )
        total_tokens += sum(r.get("tokens_used", 0) for r in jury_results)

        # 5. Consensus
        self._emit("CONSENSUS", "Super-Judge synthesising Master Solution…")
        consensus_result = await self.consensus.resolve(task, jury_results, context=context)
        master_solution = consensus_result["solution"]
        total_tokens += consensus_result.get("tokens_used", 0)
        result.master_solution = master_solution
        result.contributing_models = consensus_result.get("contributing_models", [])
        result.jury_total = consensus_result.get("jury_total", 0)
        result.finalists = consensus_result.get("finalists", 0)
        result.consensus_mode = consensus_result.get("mode", "")

        # Log jury stats
        if result.jury_total:
            self._emit(
                "JURY",
                f"{result.jury_total} responded → "
                f"{result.finalists} finalists → Super-Judge  "
                f"[mode: {result.consensus_mode}]",
            )

        if consensus_result.get("conflicts_found"):
            for conflict in consensus_result["conflicts_found"]:
                self._emit("CONFLICT", conflict)

        # 6. Execute write/command steps
        for s in plan:
            if s.action in ("READ_FILE", "THINK"):
                result.steps_completed += 1
                continue

            elif s.action == "WRITE_FILE":
                self._emit("WRITING", f"Writing: {s.target}")
                if not self._confirmed_write(s.target, master_solution):
                    self._emit("INFO", f"Skipped (user declined): {s.target}")
                    result.steps_completed += 1
                    continue
                op = self.fst.write_file(s.target, master_solution)
                if op.diff:
                    result.diffs.append(op.diff)
                result.files_modified.append(s.target)
                result.steps_completed += 1

            elif s.action == "PATCH_FILE":
                self._emit("PATCHING", f"Patching: {s.target}")
                old_snip, new_snip = self._extract_patch(s.detail, master_solution)
                op = self.fst.patch_file(s.target, old_snip, new_snip)
                if op.diff:
                    result.diffs.append(op.diff)
                if op.success:
                    result.files_modified.append(s.target)
                else:
                    result.errors.append(op.message)
                result.steps_completed += 1

            elif s.action == "CREATE_DIR":
                self._emit("MKDIR", f"Creating directory: {s.target}")
                self.fst.create_dir(s.target)
                result.steps_completed += 1

            elif s.action == "DELETE_FILE":
                self._emit("DELETE", f"Deleting: {s.target}")
                self.fst.delete_file(s.target)
                result.steps_completed += 1

            elif s.action == "RUN_COMMAND":
                self._emit("EXEC", f"$ {s.target}")
                cmd_result = await self.terminal.run_with_retry(
                    s.target,
                    on_error_callback=self._make_error_fixer(s.target),
                )
                result.commands_run.append(s.target)
                if not cmd_result.success:
                    result.errors.append(f"Command failed: {s.target}\n{cmd_result.stderr}")
                result.steps_completed += 1

        # 7. Handle multi-file output (=== FILE: path === format)
        multi_files = self._parse_multi_file_output(master_solution)
        if multi_files:
            self._emit("WRITING", f"Multi-file output detected: {len(multi_files)} file(s)")
            for filepath, content in multi_files.items():
                self._emit("WRITING", f"Writing: {filepath}")
                if not self._confirmed_write(filepath, content):
                    self._emit("INFO", f"Skipped (user declined): {filepath}")
                    continue
                op = self.fst.write_file(filepath, content)
                if op.diff:
                    result.diffs.append(op.diff)
                result.files_modified.append(filepath)

        # 8. Self-Improving Loop — run tests, fix on failure, retry N times
        si_cfg = self.registry.config.self_improve
        if si_cfg.enabled and result.files_modified:
            total_tokens += await self._self_improve_loop(
                task, result, si_cfg.test_commands, si_cfg.max_iterations
            )

        result.total_tokens = total_tokens
        result.success = len(result.errors) == 0

        # 9. Save to long-term memory & history log
        self.memory.save(
            task=task,
            solution=result.master_solution,
            files_modified=result.files_modified,
            workspace=str(self.fst.root),
        )
        self._save_history(task, result)

        return result

    # ── Diff Confirm Helper ───────────────────────────────────────────────────

    def _confirmed_write(self, filepath: str, new_content: str) -> bool:
        """
        If a confirm_callback is registered, read the existing file (if any),
        call the callback with (filepath, old_content, new_content).
        Returns True if the write should proceed.
        """
        if not self._confirm:
            return True
        existing_path = self.fst.root / filepath
        old_content = existing_path.read_text(encoding="utf-8") if existing_path.exists() else ""
        return self._confirm(filepath, old_content, new_content)

    # ── Self-Improving Loop ───────────────────────────────────────────────────

    async def _self_improve_loop(
        self,
        task: str,
        result: "ExecutionResult",
        test_commands: List[str],
        max_iterations: int,
    ) -> int:
        """
        Autonomously run tests, detect failures, and re-call the jury to fix
        them. Repeats up to max_iterations times.
        Returns total extra tokens consumed during fix iterations.
        """
        workspace = self.fst.root
        extra_tokens = 0

        # Detect which test command applies to this workspace
        test_cmd = self._detect_test_command(workspace, test_commands)
        if not test_cmd:
            return 0

        for iteration in range(1, max_iterations + 1):
            self._emit("EXEC", f"[Self-Improve {iteration}/{max_iterations}] Running: {test_cmd}")
            cmd_result = await self.terminal.run(test_cmd)

            if cmd_result.success and cmd_result.returncode == 0:
                self._emit("DONE", f"All tests passed on iteration {iteration}! ✅")
                result.iterations = iteration
                result.test_passed = True
                break

            # Tests failed — ask jury to fix
            error_summary = (cmd_result.stderr or cmd_result.stdout)[:1500]
            self._emit(
                "SELF-CORRECT",
                f"Tests failed (iteration {iteration}). Asking jury to fix…\n{error_summary[:200]}",
            )

            fix_messages = [
                {
                    "role": "user",
                    "content": (
                        f"Original task: {task}\n\n"
                        f"The following files were written:\n"
                        + "\n".join(f"- {f}" for f in result.files_modified)
                        + f"\n\nTest command `{test_cmd}` failed with:\n```\n{error_summary}\n```\n\n"
                        "Fix the code so all tests pass. Output corrected file(s) using "
                        "=== FILE: path === format if multiple files, or raw content for a single file."
                    ),
                }
            ]

            jury_results = await self.registry.call_jury(
                messages=fix_messages,
                system_prompt=SOLVER_SYSTEM,
            )
            extra_tokens += sum(r.get("tokens_used", 0) for r in jury_results)

            fix_consensus = await self.consensus.resolve(task, jury_results)
            fixed_solution = fix_consensus.get("solution", "")
            extra_tokens += fix_consensus.get("tokens_used", 0)

            if not fixed_solution:
                self._emit("SELF-CORRECT", "No fix produced. Stopping self-improve loop.")
                break

            # Apply the fix
            result.master_solution = fixed_solution
            multi = self._parse_multi_file_output(fixed_solution)
            if multi:
                for filepath, content in multi.items():
                    self._emit("WRITING", f"[Fix] Writing: {filepath}")
                    op = self.fst.write_file(filepath, content)
                    if op.diff:
                        result.diffs.append(op.diff)
                    if filepath not in result.files_modified:
                        result.files_modified.append(filepath)
            elif result.files_modified:
                # Single-file fix: rewrite the first modified file
                primary = result.files_modified[0]
                self._emit("WRITING", f"[Fix] Rewriting: {primary}")
                op = self.fst.write_file(primary, fixed_solution)
                if op.diff:
                    result.diffs.append(op.diff)
        else:
            self._emit("SELF-CORRECT", f"Max iterations ({max_iterations}) reached. Manual review needed.")

        return extra_tokens

    @staticmethod
    def _detect_test_command(workspace: Path, candidates: List[str]) -> Optional[str]:
        """Return the first applicable test command based on workspace contents."""
        markers = {
            "pytest":           ["pytest.ini", "setup.cfg", "pyproject.toml", "conftest.py", "tests/", "test_*.py"],
            "python -m pytest": ["pytest.ini", "tests/", "conftest.py"],
            "npm test":         ["package.json"],
            "npm run test":     ["package.json"],
            "go test ./...":    ["go.mod"],
            "cargo test":       ["Cargo.toml"],
        }
        for cmd in candidates:
            for marker in markers.get(cmd, []):
                if marker.endswith("/"):
                    if (workspace / marker.rstrip("/")).is_dir():
                        return cmd
                elif marker.startswith("*.") or marker.startswith("test_"):
                    if list(workspace.rglob(marker)):
                        return cmd
                elif (workspace / marker).exists():
                    return cmd
        return None

    # ── Multi-File Parser ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_multi_file_output(content: str) -> Dict[str, str]:
        """
        Parse the === FILE: path === delimiter format.
        Returns {filepath: content} dict. Empty if single-file output.
        """
        pattern = re.compile(
            r'={3}\s*FILE:\s*(.+?)\s*={3}\n(.*?)(?=\n={3}\s*FILE:|\Z)',
            re.DOTALL,
        )
        matches = pattern.findall(content)
        if not matches:
            return {}
        return {path.strip(): body.strip() for path, body in matches}

    # ── History Logger ────────────────────────────────────────────────────────

    def _save_history(self, task: str, result: "ExecutionResult") -> None:
        """Append task result to logs/history.json for future reference."""
        history_path = Path(__file__).parent.parent / "logs" / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "workspace":      str(self.fst.root),
            "task":           task,
            "solution":       result.master_solution[:800] + ("…" if len(result.master_solution) > 800 else ""),
            "files_modified": result.files_modified,
            "tokens_used":    result.total_tokens,
            "jury_total":     result.jury_total,
            "consensus_mode": result.consensus_mode,
            "success":        result.success,
        }

        entries: List[Dict] = []
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                entries = []

        entries.append(entry)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_patch(detail: str, solution: str) -> tuple[str, str]:
        """
        Attempt to extract old/new snippets from detail JSON,
        falling back to entire solution as new content.
        """
        try:
            data = json.loads(detail)
            return data["old"], data["new"]
        except Exception:
            return "", solution

    def _make_error_fixer(self, original_cmd: str):
        """Return an async callback that asks the judge model to fix a failed command."""
        async def fixer(cmd_result: CommandResult) -> Optional[str]:
            self._emit("SELF-CORRECT", f"Attempting to fix: {cmd_result.stderr[:100]}…")
            model = self.registry.get_judge_model()
            res = await self.registry.call_model(
                model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"This command failed:\n```\n{original_cmd}\n```\n"
                        f"Error:\n```\n{cmd_result.stderr}\n```\n"
                        "Output ONLY the corrected command as a single line."
                    ),
                }],
            )
            fixed = res["content"].strip().splitlines()[0] if res["content"] else None
            if fixed:
                self._emit("SELF-CORRECT", f"Retrying with: {fixed}")
            return fixed
        return fixer
