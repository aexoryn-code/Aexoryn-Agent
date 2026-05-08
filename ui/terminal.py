# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
ui/terminal.py
──────────────
Rich-powered Terminal UI for Aexoryn Agent.

Renders:
  - AEXORYN banner
  - Live "Thinking" spinner with status log
  - Jury results table
  - Visual diff panels
  - Task Completed / Error summary
  - MCP status bar & Token tracker
"""

from __future__ import annotations

import difflib
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# ── Theme ─────────────────────────────────────────────────────────────────────

AEXORYN_THEME = Theme({
    "banner":        "bold cyan",
    "status.think":  "bold yellow",
    "status.read":   "bold blue",
    "status.write":  "bold green",
    "status.exec":   "bold magenta",
    "status.jury":   "bold cyan",
    "status.judge":  "bold white on dark_blue",
    "status.done":   "bold bright_green",
    "status.error":  "bold red",
    "status.warn":   "bold orange3",
    "status.route":  "dim cyan",
    "dim_text":      "dim white",
    "token_ok":      "bright_green",
    "token_warn":    "yellow",
    "token_cap":     "red",
    "diff.add":      "bright_green",
    "diff.remove":   "red",
})

BANNER = r"""
 █████╗ ███████╗██╗  ██╗ ██████╗ ██████╗ ██╗   ██╗███╗   ██╗
██╔══██╗██╔════╝╚██╗██╔╝██╔═══██╗██╔══██╗╚██╗ ██╔╝████╗  ██║
███████║█████╗   ╚███╔╝ ██║   ██║██████╔╝ ╚████╔╝ ██╔██╗ ██║
██╔══██║██╔══╝   ██╔██╗ ██║   ██║██╔══██╗  ╚██╔╝  ██║╚██╗██║
██║  ██║███████╗██╔╝ ██╗╚██████╔╝██║  ██║   ██║   ██║ ╚████║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝
"""

STATUS_ICONS: Dict[str, str] = {
    "THINKING":    "🤔",
    "PLAN":        "📋",
    "READING":     "📖",
    "ROUTING":     "🔀",
    "JURY":        "⚖️",
    "CONSENSUS":   "🧠",
    "CONFLICT":    "⚠️",
    "WRITING":     "✍️",
    "PATCHING":    "🔧",
    "MKDIR":       "📁",
    "DELETE":      "🗑️",
    "EXEC":        "⚡",
    "SELF-CORRECT":"🔄",
    "DONE":        "✅",
    "ERROR":       "❌",
    "INFO":        "ℹ️",
    "TOKEN":       "🔢",
    "MCP":         "🔌",
}

STATUS_STYLE: Dict[str, str] = {
    "THINKING":    "status.think",
    "PLAN":        "status.think",
    "READING":     "status.read",
    "ROUTING":     "status.route",
    "JURY":        "status.jury",
    "CONSENSUS":   "status.judge",
    "CONFLICT":    "status.warn",
    "WRITING":     "status.write",
    "PATCHING":    "status.write",
    "MKDIR":       "status.write",
    "DELETE":      "status.error",
    "EXEC":        "status.exec",
    "SELF-CORRECT":"status.warn",
    "DONE":        "status.done",
    "ERROR":       "status.error",
    "INFO":        "dim_text",
    "TOKEN":       "dim_text",
    "MCP":         "dim_text",
}


# ── Terminal UI ───────────────────────────────────────────────────────────────

class AexorynUI:
    """
    Central Rich console for Aexoryn Agent.
    Thread-safe via Rich's internal locking.
    """

    def __init__(self, workspace: str = ".", verbose: bool = False) -> None:
        self.console = Console(theme=AEXORYN_THEME, highlight=False)
        self.workspace = workspace
        self.verbose = verbose
        self._token_used: int = 0
        self._token_cap: int = 500_000
        self._mcp_status: str = "idle"
        self._log_lines: List[str] = []

    # ── Banner ────────────────────────────────────────────────────────────────

    def print_banner(self) -> None:
        self.console.print(Text(BANNER, style="banner"))
        self.console.print(
            Panel.fit(
                "[dim_text]Autonomous Multi-Model Coding Agent  •  MIT 2026  •  "
                "Irfan Syazwan / Aexoryn[/dim_text]",
                border_style="cyan",
            )
        )
        self.console.print()

    # ── Status Log ────────────────────────────────────────────────────────────

    def log(self, status: str, message: str) -> None:
        icon = STATUS_ICONS.get(status, "·")
        style = STATUS_STYLE.get(status, "dim_text")
        label = Text(f" {status:<13}", style=style)
        msg = Text(f"{icon}  {message}")
        self.console.print(label, msg)
        self._log_lines.append(f"[{status}] {message}")

    # ── Spinner Context ───────────────────────────────────────────────────────

    @contextmanager
    def thinking(self, description: str = "Aexoryn is thinking…"):
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[status.think]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(description)
            yield progress, task_id

    # ── Jury Table ────────────────────────────────────────────────────────────

    def print_jury_table(self, jury_results: List[Dict[str, Any]]) -> None:
        table = Table(
            title="⚖️  Jury Responses",
            box=box.ROUNDED,
            border_style="cyan",
            show_lines=True,
        )
        table.add_column("Model", style="bold", no_wrap=True)
        table.add_column("Tokens", justify="right", style="dim_text")
        table.add_column("Status", justify="center")
        table.add_column("Preview", style="dim_text", max_width=60)

        for r in jury_results:
            status = "[status.error]FAILED[/]" if r.get("error") else "[status.done]OK[/]"
            preview = (r.get("content") or r.get("error") or "")[:80].replace("\n", " ")
            table.add_row(
                r.get("name", r.get("model_id", "?")),
                str(r.get("tokens_used", 0)),
                status,
                preview,
            )
        self.console.print(table)

    # ── Diff Panel ────────────────────────────────────────────────────────────

    def print_diff(self, diff_text: str, title: str = "Changes") -> None:
        if not diff_text.strip():
            return
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
        self.console.print(Panel(syntax, title=f"  {title}", border_style="green"))

    # ── Solution Panel ────────────────────────────────────────────────────────

    def print_solution(self, solution: str, language: str = "python") -> None:
        syntax = Syntax(solution, language, theme="monokai", line_numbers=True)
        self.console.print(
            Panel(syntax, title="  🧠 Master Solution", border_style="cyan")
        )

    # ── Task Completed Summary ────────────────────────────────────────────────

    def print_summary(
        self,
        task: str,
        files_modified: List[str],
        commands_run: List[str],
        tokens_used: int,
        errors: List[str],
        mode: str = "super_judge",
        contributing_models: List[str] = [],
        jury_total: int = 0,
        finalists: int = 0,
        workspace_files_read: List[str] = [],
        iterations: int = 0,
        test_passed: Optional[bool] = None,
    ) -> None:
        self.console.print()
        self.console.print(Rule("[status.done]  TASK COMPLETED  [/]", style="bright_green"))
        self.console.print()

        info_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        info_table.add_column(style="dim_text", no_wrap=True)
        info_table.add_column(style="bold")

        info_table.add_row("Task",            task[:80] + ("…" if len(task) > 80 else ""))
        info_table.add_row("Consensus Mode",  mode)
        if jury_total:
            info_table.add_row("Jury Pool",  f"{jury_total} models responded")
            info_table.add_row("Finalists",  f"{finalists} → Super-Judge")
        info_table.add_row("Top Contributors", ", ".join(contributing_models[:4]) + (f"  +{len(contributing_models)-4} more" if len(contributing_models) > 4 else "") or "—")
        if workspace_files_read:
            info_table.add_row("Context Read",    ", ".join(workspace_files_read[:3]) + (f"  +{len(workspace_files_read)-3} more" if len(workspace_files_read) > 3 else ""))
        if iterations > 0:
            test_status = "✅ Passed" if test_passed else ("❌ Failed" if test_passed is False else "—")
            info_table.add_row("Self-Improve",   f"{iterations} iteration(s)  |  Tests: {test_status}")
        info_table.add_row("Files Modified",  str(len(files_modified)))
        info_table.add_row("Commands Run",    str(len(commands_run)))
        info_table.add_row("Errors",          str(len(errors)))
        self.console.print(info_table)

        if files_modified:
            self.console.print("\n[bold]Modified Files:[/bold]")
            for f in files_modified:
                self.console.print(f"  [status.write]✔[/]  {f}")

        if commands_run:
            self.console.print("\n[bold]Commands Executed:[/bold]")
            for c in commands_run:
                self.console.print(f"  [status.exec]⚡[/]  {c}")

        if errors:
            self.console.print("\n[bold]Errors:[/bold]")
            for e in errors:
                self.console.print(f"  [status.error]✘[/]  {e}")

        self._print_token_bar(tokens_used)
        self.console.print()

    # ── Token Status Bar ──────────────────────────────────────────────────────

    def update_tokens(self, used: int, cap: int = 500_000) -> None:
        self._token_used += used
        self._token_cap = cap

    def _print_token_bar(self, session_tokens: int) -> None:
        pct = min(session_tokens / self._token_cap * 100, 100)
        if pct < 60:
            style = "token_ok"
        elif pct < 85:
            style = "token_warn"
        else:
            style = "token_cap"
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        self.console.print(
            f"\n[dim_text]Token Usage:[/]  [{style}]{bar}[/]  "
            f"[{style}]{session_tokens:,} / {self._token_cap:,}[/]  "
            f"([{style}]{pct:.1f}%[/])"
        )

    # ── MCP Status Bar ────────────────────────────────────────────────────────

    def set_mcp_status(self, status: str) -> None:
        self._mcp_status = status
        icon = "🟢" if status == "active" else "🔴" if status == "error" else "🟡"
        self.console.print(f"[dim_text]MCP:[/]  {icon}  {status.upper()}")

    # ── Error Panel ───────────────────────────────────────────────────────────

    def print_error(self, title: str, message: str) -> None:
        self.console.print(
            Panel(
                Text(message, style="status.error"),
                title=f"  ❌ {title}",
                border_style="red",
            )
        )

    # ── Diff Preview ──────────────────────────────────────────────────────────

    def show_diff_preview(self, filepath: str, old: str, new: str) -> None:
        """Render a coloured unified diff of old vs new content."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm="",
        ))

        if not diff:
            self.console.print(f"[dim]  (no changes to {filepath})[/dim]")
            return

        lines: List[str] = []
        for line in diff:
            if line.startswith("+++") or line.startswith("---"):
                lines.append(f"[bold]{line}[/bold]")
            elif line.startswith("+"):
                lines.append(f"[green]{line}[/green]")
            elif line.startswith("-"):
                lines.append(f"[red]{line}[/red]")
            elif line.startswith("@@"):
                lines.append(f"[cyan]{line}[/cyan]")
            else:
                lines.append(f"[dim]{line}[/dim]")

        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"  📄 Diff Preview — {filepath}",
                border_style="yellow",
                expand=False,
            )
        )

    def prompt_confirm(self, filepath: str, is_new: bool = False) -> bool:
        """Ask the user to confirm writing a file. Returns True to apply."""
        label = "[bold yellow]NEW[/bold yellow]" if is_new else "[bold cyan]MODIFY[/bold cyan]"
        self.console.print(f"\n  {label}  {filepath}")
        try:
            answer = self.console.input("  Apply changes? [Y/n] ").strip().lower()
            return answer in ("", "y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False

    def print_budget_warning(self, used: int, cap: int, threshold: int) -> None:
        """Display a token budget warning panel."""
        pct = used / cap * 100
        remaining = cap - used
        self.console.print(
            Panel(
                f"  [bold yellow]⚠️  Token budget warning[/bold yellow]\n\n"
                f"  Used:      [yellow]{used:,}[/yellow] / {cap:,}  ({pct:.1f}%)\n"
                f"  Remaining: [yellow]{remaining:,}[/yellow] tokens\n\n"
                f"  Tip: Use [bold]--profile fast[/bold] to reduce token usage.",
                border_style="yellow",
                title="  💰 Budget Alert",
            )
        )

    # ── Separator ─────────────────────────────────────────────────────────────

    def rule(self, title: str = "") -> None:
        self.console.print(Rule(title, style="cyan"))
