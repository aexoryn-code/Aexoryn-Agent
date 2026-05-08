#!/usr/bin/env python3
# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
main.py
───────
Aexoryn Agent — Entry Point

Usage:
  python main.py run "Refactor auth.py to use JWT"
  python main.py run --workspace ./my-project "Add dark mode toggle to App.tsx"
  python main.py run --reload-config "..."
  python main.py list-models
  python main.py run --mode majority_vote "Fix the failing tests"

Environment:
  Copy .env.example → .env and fill in your API keys.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import typer
from dotenv import load_dotenv
from loguru import logger

from core.planner import TaskPlanner
from core.registry import ModelRegistry
from core.tools import FileSystemTool, TerminalTool
from ui.terminal import AexorynUI

load_dotenv()

app = typer.Typer(
    name="aexoryn",
    help="""
    Aexoryn Agent — Autonomous Multi-Model Coding Agent

    Orchestrates hundreds of AI models in parallel to solve coding tasks,
    write files, run commands, and maintain conversation context.

    \b
    QUICK START:
      python main.py run "Create a login page in HTML"
      python main.py run --workspace E:\\MyProject "Fix the bug in auth.py"
      python main.py chat --workspace E:\\MyProject

    \b
    PROFILES (control speed vs quality):
      --profile fast      1 model  — instant, lowest cost
      --profile turbo     5 models — fast multi-model vote
      --profile balanced  15 free models — best free quality
      --profile premium   50 top models — highest quality (uses credits)

    \b
    MODEL SETUP:
      python main.py setup-ollama          Register local Ollama models (free)
      python scripts/fetch_models.py --free-only   Enable only free cloud models
      python scripts/fetch_models.py --enable-all  Enable all 366 cloud models
      python scripts/fetch_models.py --top 20      Enable top 20 models

    \b
    USEFUL COMMANDS:
      python main.py run        Execute a one-shot coding task
      python main.py chat       Interactive mode with memory
      python main.py history    View past tasks
      python main.py list-models  Show all registered models
    """,
    add_completion=False,
)


# ── CLI Commands ──────────────────────────────────────────────────────────────

@app.command()
def run(
    task: str = typer.Argument(..., help="Coding task description, e.g. 'Create auth.py with JWT login'"),
    workspace: str = typer.Option(
        None, "--workspace", "-w",
        help="Path to the VS Code workspace. Defaults to AEXORYN_WORKSPACE env var or '.'",
    ),
    reload_config: bool = typer.Option(
        False, "--reload-config", help="Hot-reload model_config.yaml before running."
    ),
    mode: str = typer.Option(
        None, "--mode", "-m",
        help="Override consensus mode: super_judge | majority_vote | weighted_rank",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p",
        help="Model profile preset: fast | turbo | balanced | premium",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip diff preview and apply all file changes automatically.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """
    Execute an autonomous coding task inside your workspace.

    \b
    The agent will:
      1. Decompose the task into a plan
      2. Scan your workspace files for context
      3. Call multiple AI models in parallel
      4. Synthesise the best solution via Super-Judge
      5. Write the output file(s) directly to your workspace

    \b
    EXAMPLES:
      python main.py run "Create index.html responsive landing page"
      python main.py run --workspace E:\\MyProject "Add dark mode to App.tsx"
      python main.py run --profile fast "Fix the syntax error in utils.py"
      python main.py run --profile balanced "Create a REST API in Flask"
      python main.py run --mode majority_vote "Refactor auth.py"
    """
    _configure_logging(verbose)
    workspace_path = workspace or os.getenv("AEXORYN_WORKSPACE", ".")

    ui = AexorynUI(workspace=workspace_path, verbose=verbose)
    ui.print_banner()
    ui.log("INFO", f"Workspace: {Path(workspace_path).resolve()}")

    registry = ModelRegistry()
    if reload_config:
        registry.reload()
        ui.log("INFO", "Config hot-reloaded.")

    if profile:
        registry.apply_profile(profile)
        ui.log("INFO", f"Profile applied: {profile}")
    elif mode:
        registry.config.consensus.mode = mode
        ui.log("INFO", f"Consensus mode overridden to: {mode}")

    ui.set_mcp_status("active")
    ui.rule("Starting Task")
    ui.log("THINKING", f"Task received: {task[:100]}{'…' if len(task) > 100 else ''}")

    fst = FileSystemTool(workspace_path)
    terminal = TerminalTool(workspace_path)

    def ui_callback(status: str, message: str) -> None:
        ui.log(status, message)

    def confirm_callback(filepath: str, old: str, new: str) -> bool:
        ui.show_diff_preview(filepath, old, new)
        return ui.prompt_confirm(filepath, is_new=(old == ""))

    planner = TaskPlanner(
        registry, fst, terminal,
        ui_callback=ui_callback,
        confirm_callback=None if yes else confirm_callback,
    )

    async def _run() -> None:
        result = await planner.execute(task)

        # Show budget warning if approaching threshold
        cfg = registry.config.token_tracking
        if cfg.enabled and registry.session_tokens >= cfg.warn_threshold:
            ui.print_budget_warning(registry.session_tokens, cfg.session_cap, cfg.warn_threshold)

        ui.rule("Results")

        if result.diffs:
            for i, diff in enumerate(result.diffs, 1):
                ui.print_diff(diff, title=f"Diff {i}")

        if result.master_solution:
            lang = _detect_language(result.master_solution)
            ui.print_solution(result.master_solution, language=lang)

        ui.print_summary(
            task=task,
            files_modified=result.files_modified,
            commands_run=result.commands_run,
            tokens_used=result.total_tokens,
            errors=result.errors,
            mode=result.consensus_mode or "super_judge",
            contributing_models=result.contributing_models,
            jury_total=result.jury_total,
            finalists=result.finalists,
            workspace_files_read=result.workspace_files_read,
            iterations=result.iterations,
            test_passed=result.test_passed,
        )

        if not result.success:
            ui.print_error("Task ended with errors", "\n".join(result.errors))
            raise typer.Exit(code=1)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        ui.log("INFO", "Interrupted by user.")
    except Exception as exc:
        ui.print_error("Fatal Error", str(exc))
        logger.exception(exc)
        raise typer.Exit(code=1)


@app.command(name="list-models")
def list_models() -> None:
    """
    List all registered models with their status, weight, and tier.

    \b
    Shows: Model ID, Name, Provider, Tier (cloud/local), Enabled, Weight, Strengths
    ✅ = active in jury    ⬜ = disabled

    \b
    To change which models are active:
      python scripts/fetch_models.py --free-only     Enable only free models
      python scripts/fetch_models.py --enable-all    Enable all 366 models
      python scripts/fetch_models.py --top 20        Enable top 20 models
      python main.py setup-ollama                    Add local Ollama models
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box

    registry = ModelRegistry()
    console = Console()

    table = Table(title="Aexoryn Model Registry", box=box.ROUNDED, border_style="cyan")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Tier")
    table.add_column("Enabled", justify="center")
    table.add_column("Weight", justify="right")
    table.add_column("Strengths")

    for m in registry.config.models:
        enabled_str = "✅" if m.enabled else "⬜"
        table.add_row(
            m.id, m.name, m.provider, m.tier,
            enabled_str,
            f"{m.weight:.1f}",
            ", ".join(m.strengths),
        )
    console.print(table)


@app.command(name="reload-config")
def reload_config_cmd() -> None:
    """
    Hot-reload model_config.yaml without restarting the agent.

    Useful after manually editing model_config.yaml to enable/disable models
    or change consensus settings.
    """
    registry = ModelRegistry()
    registry.reload()
    enabled = registry.get_enabled_models()
    typer.echo(f"Config reloaded. {len(enabled)} model(s) active.")


@app.command(name="setup-ollama")
def setup_ollama(
    base_url: str = typer.Option(
        None, "--base-url", help="Ollama base URL (default: http://localhost:11434 or OLLAMA_BASE_URL)"
    ),
    judge: Optional[str] = typer.Option(
        None, "--judge", help="Set specific Ollama model as Super-Judge (e.g. llama3.3:70b)"
    ),
) -> None:
    """
    Discover locally installed Ollama models and register them as jury members.
    No API key or credits required — runs entirely on your local machine.

    Examples:
      python main.py setup-ollama
      python main.py setup-ollama --judge llama3.3:70b
      python main.py setup-ollama --base-url http://192.168.1.10:11434
    """
    import subprocess
    cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "setup_ollama.py")]
    if base_url:
        cmd += ["--base-url", base_url]
    if judge:
        cmd += ["--judge", judge]
    result = subprocess.run(cmd)
    raise typer.Exit(code=result.returncode)


@app.command(name="chat")
def chat_mode(
    workspace: str = typer.Option(".", "--workspace", "-w", help="Project folder to operate on"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Consensus mode override"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Model profile preset: fast | turbo | balanced | premium"),
) -> None:
    """
    Interactive chat mode — the agent remembers the full conversation.
    Describe your task naturally and follow up with refinements.
    Type 'exit' or 'quit' to end the session. Type 'reset' to clear history.

    Examples:
      python main.py chat
      python main.py chat --workspace E:\\MyProject
    """
    from rich.console import Console
    from rich.rule import Rule
    from rich.panel import Panel

    load_dotenv()
    _configure_logging(verbose=False)

    console = Console()
    workspace_path = Path(workspace).resolve()
    registry = ModelRegistry()
    if profile:
        registry.apply_profile(profile)
    elif mode:
        registry.config.consensus.mode = mode

    fst = FileSystemTool(workspace_path)
    terminal = TerminalTool(workspace_path)
    ui = AexorynUI()
    session_tokens = 0
    session_cost_tasks = 0

    history: List[Dict[str, str]] = []

    console.print(Panel(
        "[bold cyan]Aexoryn Chat Mode[/bold cyan]\n"
        "[dim]Agent remembers the full conversation. Type [bold]exit[/bold] to quit.[/dim]\n"
        f"[dim]Workspace: {workspace_path}[/dim]",
        border_style="cyan",
    ))

    def ui_callback(status: str, message: str) -> None:
        ui.log(status, message)

    planner = TaskPlanner(registry, fst, terminal, ui_callback=ui_callback)

    async def _run_turn(task: str) -> None:
        nonlocal session_tokens, session_cost_tasks
        result = await planner.execute(task, history=history)

        if result.master_solution:
            lang = _detect_language(result.master_solution)
            ui.print_solution(result.master_solution, language=lang)

        ui.print_summary(
            task=task,
            files_modified=result.files_modified,
            commands_run=result.commands_run,
            tokens_used=result.total_tokens,
            errors=result.errors,
            mode=result.consensus_mode or "super_judge",
            contributing_models=result.contributing_models,
            jury_total=result.jury_total,
            finalists=result.finalists,
            workspace_files_read=result.workspace_files_read,
            iterations=result.iterations,
            test_passed=result.test_passed,
        )

        # Update history for next turn
        history.append({"role": "user", "content": task})
        if result.master_solution:
            history.append({"role": "assistant", "content": result.master_solution})

        # Keep history to last 10 turns to avoid context overflow
        if len(history) > 20:
            history[:] = history[-20:]

        session_tokens += result.total_tokens
        session_cost_tasks += 1
        console.print(
            f"[dim]Session: {session_cost_tasks} task(s) | "
            f"{session_tokens:,} tokens used | "
            f"History: {len(history)//2} turn(s)[/dim]"
        )

    while True:
        try:
            console.print()
            task = console.input("[bold cyan]You ›[/bold cyan] ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit", "bye", "q"):
                console.print("[dim]Aexoryn: Goodbye! 👋[/dim]")
                break
            if task.lower() in ("clear history", "reset"):
                history.clear()
                console.print("[dim]Conversation history cleared.[/dim]")
                continue
            ui.rule("Thinking")
            asyncio.run(_run_turn(task))
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except Exception as exc:
            ui.print_error("Error", str(exc))
            logger.exception(exc)

    raise typer.Exit(code=0)


@app.command(name="history")
def show_history(
    last: int = typer.Option(10, "--last", "-n", help="Number of recent entries to show (default: 10)"),
) -> None:
    """
    Show recent task history saved in logs/history.json.

    Every task run or chat turn is automatically saved with:
    timestamp, workspace, task description, files modified, tokens used.

    \b
    EXAMPLES:
      python main.py history            Show last 10 tasks
      python main.py history --last 25  Show last 25 tasks
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import json

    history_path = Path(__file__).parent / "logs" / "history.json"
    console = Console()

    if not history_path.exists():
        console.print("[yellow]No history yet. Run a task first.[/yellow]")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    recent = entries[-last:]
    t = Table(
        title=f"Task History — last {len(recent)} of {len(entries)} total",
        box=box.ROUNDED, border_style="cyan",
    )
    t.add_column("#",         style="dim",   width=4,  no_wrap=True)
    t.add_column("Timestamp", style="dim",   width=19, no_wrap=True)
    t.add_column("Task",      style="bold",  width=40)
    t.add_column("Files",                    width=20)
    t.add_column("Tokens",    justify="right", width=8)
    t.add_column("OK",        justify="center", width=4)

    for i, e in enumerate(recent, start=len(entries) - len(recent) + 1):
        t.add_row(
            str(i),
            e.get("timestamp", "")[:19],
            e.get("task", "")[:40],
            ", ".join(e.get("files_modified", [])[:2]) or "—",
            str(e.get("tokens_used", 0)),
            "✅" if e.get("success") else "❌",
        )
    console.print(t)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    level = "DEBUG" if verbose else os.getenv("AEXORYN_LOG_LEVEL", "INFO")
    log_path = os.getenv("AEXORYN_SESSION_LOG", "./logs/session.log")
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True, format="<dim>{time:HH:mm:ss}</dim> {message}")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="10 MB", retention="7 days", level="DEBUG")


def _detect_language(content: str) -> str:
    """Heuristic language detection for Rich syntax highlighting."""
    first = content.strip()[:200]
    if first.startswith("import ") or "def " in first or "class " in first:
        return "python"
    if first.startswith("<?php"):
        return "php"
    if "const " in first or "function " in first or "=>" in first:
        return "javascript"
    if first.startswith("<") and ">" in first:
        return "html"
    if "{" in first and ":" in first:
        return "json"
    return "text"


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
