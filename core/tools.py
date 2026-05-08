# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
core/tools.py
─────────────
MCP-Style File System & Terminal Tools — the "Hands" of Aexoryn Agent.

Provides:
  FileSystemTool  — create / read / edit / delete files & directories
  TerminalTool    — run shell commands, capture output, self-correct on error
  DiffTool        — produce rich visual diffs of file changes
"""

from __future__ import annotations

import asyncio
import difflib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from rich.syntax import Syntax
from rich.text import Text


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class FileOpResult:
    success: bool
    path: str
    message: str
    diff: Optional[str] = None


@dataclass
class CommandResult:
    success: bool
    command: str
    stdout: str
    stderr: str
    returncode: int


# ── File System Tool ──────────────────────────────────────────────────────────

class FileSystemTool:
    """
    Provides atomic file-system operations within a sandboxed workspace root.
    All paths are resolved relative to workspace_root to prevent escape.
    """

    def __init__(self, workspace_root: str) -> None:
        self.root = Path(workspace_root).resolve()

    def _safe_path(self, relative: str) -> Path:
        resolved = (self.root / relative).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise PermissionError(f"Path escape attempt blocked: {relative}")
        return resolved

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        logger.debug(f"[FST] Read: {p}")
        return p.read_text(encoding="utf-8")

    def list_dir(self, path: str = ".") -> List[str]:
        p = self._safe_path(path)
        return [str(child.relative_to(self.root)) for child in sorted(p.iterdir())]

    def exists(self, path: str) -> bool:
        return self._safe_path(path).exists()

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_file(self, path: str, content: str, create_parents: bool = True) -> FileOpResult:
        p = self._safe_path(path)
        old_content: Optional[str] = None
        if p.exists():
            old_content = p.read_text(encoding="utf-8")
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        diff = DiffTool.compute(old_content or "", content, filename=path) if old_content else None
        action = "Updated" if old_content else "Created"
        logger.info(f"[FST] {action}: {p}")
        return FileOpResult(success=True, path=str(p), message=f"{action}: {path}", diff=diff)

    def create_dir(self, path: str) -> FileOpResult:
        p = self._safe_path(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"[FST] Created dir: {p}")
        return FileOpResult(success=True, path=str(p), message=f"Directory created: {path}")

    def delete_file(self, path: str) -> FileOpResult:
        p = self._safe_path(path)
        if p.is_dir():
            shutil.rmtree(p)
            logger.info(f"[FST] Deleted dir: {p}")
            return FileOpResult(success=True, path=str(p), message=f"Directory deleted: {path}")
        elif p.is_file():
            p.unlink()
            logger.info(f"[FST] Deleted file: {p}")
            return FileOpResult(success=True, path=str(p), message=f"File deleted: {path}")
        else:
            return FileOpResult(success=False, path=str(p), message=f"Path not found: {path}")

    def patch_file(self, path: str, old_snippet: str, new_snippet: str) -> FileOpResult:
        """Precision patch: replace exact old_snippet with new_snippet."""
        content = self.read_file(path)
        if old_snippet not in content:
            return FileOpResult(
                success=False, path=path,
                message=f"Snippet not found in {path} — patch aborted."
            )
        updated = content.replace(old_snippet, new_snippet, 1)
        return self.write_file(path, updated)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_files(self, pattern: str, extensions: List[str] = []) -> List[str]:
        """Return all file paths under workspace matching a glob or extension filter."""
        results = []
        for p in self.root.rglob(pattern):
            if extensions and p.suffix not in extensions:
                continue
            results.append(str(p.relative_to(self.root)))
        return results


# ── Terminal Tool ─────────────────────────────────────────────────────────────

class TerminalTool:
    """
    Executes shell commands inside the workspace with self-correction support.
    """

    def __init__(self, workspace_root: str, timeout: int = 120) -> None:
        self.cwd = str(Path(workspace_root).resolve())
        self.timeout = timeout

    async def run(self, command: str, cwd: Optional[str] = None) -> CommandResult:
        """
        Async non-blocking command execution.
        Returns stdout, stderr, and returncode.
        """
        work_dir = cwd or self.cwd
        logger.info(f"[Terminal] $ {command}  (cwd={work_dir})")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            rc = proc.returncode or 0
            if rc != 0:
                logger.warning(f"[Terminal] Non-zero exit ({rc}): {stderr[:200]}")
            return CommandResult(
                success=(rc == 0),
                command=command,
                stdout=stdout,
                stderr=stderr,
                returncode=rc,
            )
        except asyncio.TimeoutError:
            return CommandResult(
                success=False, command=command,
                stdout="", stderr="Command timed out.", returncode=-1
            )
        except Exception as exc:
            return CommandResult(
                success=False, command=command,
                stdout="", stderr=str(exc), returncode=-1
            )

    async def run_with_retry(
        self,
        command: str,
        on_error_callback=None,
        max_retries: int = 2,
    ) -> CommandResult:
        """
        Run command; if it fails, call on_error_callback(result) to get a
        corrected command, then retry. Enables self-correction loops.
        """
        result = await self.run(command)
        for attempt in range(max_retries):
            if result.success:
                break
            logger.info(f"[Terminal] Retry {attempt + 1}/{max_retries}…")
            if on_error_callback:
                corrected_cmd = await on_error_callback(result)
                if corrected_cmd:
                    result = await self.run(corrected_cmd)
                else:
                    break
        return result


# ── Diff Tool ─────────────────────────────────────────────────────────────────

class DiffTool:
    """Produces unified diffs for visual display in the terminal UI."""

    @staticmethod
    def compute(old: str, new: str, filename: str = "file") -> str:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
        return "".join(diff)

    @staticmethod
    def rich_diff(diff_text: str) -> Syntax:
        """Wrap diff text in a Rich Syntax object for coloured rendering."""
        return Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
