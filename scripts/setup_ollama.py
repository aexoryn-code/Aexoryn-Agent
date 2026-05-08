#!/usr/bin/env python3
# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
scripts/setup_ollama.py
───────────────────────
Auto-discovers all locally installed Ollama models and registers them
into model_config.yaml as tier:local jury members.

Usage:
  python scripts/setup_ollama.py
  python scripts/setup_ollama.py --base-url http://localhost:11434
  python scripts/setup_ollama.py --judge llama3.3:70b

After running, Ollama models will be added to the jury automatically.
No API key or credits needed — runs 100% locally.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv(Path(__file__).parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent.parent / "model_config.yaml"
console = Console()


# ── Ollama API ────────────────────────────────────────────────────────────────

async def check_ollama(base_url: str) -> bool:
    """Ping Ollama to check if it's running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def fetch_ollama_models(base_url: str) -> List[Dict[str, Any]]:
    """Fetch list of locally installed Ollama models."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{base_url.rstrip('/')}/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])


# ── Weight Heuristics ─────────────────────────────────────────────────────────

def _assign_weight(model_name: str) -> float:
    n = model_name.lower()
    if any(x in n for x in ["70b", "72b", "65b", "405b", "236b"]):
        return 1.4
    if any(x in n for x in ["32b", "34b", "30b"]):
        return 1.1
    if any(x in n for x in ["13b", "14b", "20b", "22b"]):
        return 0.9
    if any(x in n for x in ["7b", "8b", "9b"]):
        return 0.7
    if any(x in n for x in ["3b", "3.8b", "4b"]):
        return 0.5
    return 0.8


def _assign_strengths(model_name: str) -> List[str]:
    n = model_name.lower()
    strengths = ["general"]
    if any(x in n for x in ["coder", "code", "deepseek-coder", "starcoder", "qwen-coder"]):
        strengths += ["code"]
    if any(x in n for x in ["think", "r1", "reasoning"]):
        strengths += ["reasoning"]
    if any(x in n for x in ["vision", "llava", "moondream", "bakllava"]):
        strengths += ["vision"]
    return list(dict.fromkeys(strengths))


def _safe_id(name: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9\-_.]", "-", name).strip("-")


def build_ollama_entry(raw: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    model_path = raw.get("name", "")
    size_bytes = raw.get("size", 0)
    size_gb = round(size_bytes / 1e9, 1) if size_bytes else 0

    return {
        "id":         f"ollama-{_safe_id(model_path)}",
        "name":       f"{model_path} (Local)",
        "provider":   "ollama",
        "model_path": model_path,
        "base_url":   base_url,
        "tier":       "local",
        "enabled":    True,
        "weight":     _assign_weight(model_path),
        "strengths":  _assign_strengths(model_path),
        "max_tokens": 4096,
        "temperature": 0.2,
        "size_gb":    size_gb,
    }


# ── Config Update ─────────────────────────────────────────────────────────────

def update_config(new_ollama_entries: List[Dict], judge_id: Optional[str] = None) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models: List[Dict] = cfg.get("models", [])

    # Remove old ollama entries, keep cloud ones
    models = [m for m in models if m.get("provider") != "ollama"]

    # Prepend new local models so they appear first
    cfg["models"] = new_ollama_entries + models

    # Optionally update judge to a local model
    if judge_id:
        cfg.setdefault("consensus", {})["judge_model_id"] = judge_id
        console.print(f"[cyan]Super-Judge set to:[/cyan] [bold]{judge_id}[/bold]")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(base_url: str, judge: Optional[str]) -> None:
    console.print(f"\n[cyan]Checking Ollama at[/cyan] [bold]{base_url}[/bold]…")

    if not await check_ollama(base_url):
        console.print(
            "[red bold]✗ Ollama is not running![/red bold]\n"
            f"  Start it with: [bold]ollama serve[/bold]\n"
            f"  Then re-run this script."
        )
        sys.exit(1)

    console.print("[green]✓ Ollama is running![/green]")
    raw_models = await fetch_ollama_models(base_url)

    if not raw_models:
        console.print(
            "[yellow]No models found.[/yellow] "
            "Pull a model first:\n"
            "  [bold]ollama pull llama3.2[/bold]\n"
            "  [bold]ollama pull qwen2.5-coder:7b[/bold]\n"
            "  [bold]ollama pull phi4[/bold]"
        )
        sys.exit(0)

    entries = [build_ollama_entry(m, base_url) for m in raw_models]

    # Determine judge model id
    judge_id = None
    if judge:
        judge_id = f"ollama-{_safe_id(judge)}"
    elif entries:
        # Auto-pick highest weight model as judge
        best = max(entries, key=lambda e: e["weight"])
        judge_id = best["id"]
        console.print(f"[dim]Auto-selected judge: {best['name']}[/dim]")

    update_config(entries, judge_id=judge_id)

    # Print summary table
    t = Table(
        title=f"Ollama Local Models — {len(entries)} installed",
        box=box.ROUNDED, border_style="green",
    )
    t.add_column("Model", style="bold")
    t.add_column("Size", justify="right")
    t.add_column("Weight", justify="right")
    t.add_column("Strengths")
    t.add_column("ID (config)")

    for e in sorted(entries, key=lambda x: x["weight"], reverse=True):
        t.add_row(
            e["name"],
            f"{e['size_gb']} GB" if e.get("size_gb") else "—",
            str(e["weight"]),
            ", ".join(e["strengths"]),
            e["id"],
        )
    console.print(t)

    console.print(
        f"\n[bold green]✅ Done![/bold green] "
        f"{len(entries)} local Ollama model(s) added to model_config.yaml\n"
        f"[dim]These models run FREE — no API key or credits needed.[/dim]\n"
        f"[dim]Run [bold]python main.py list-models[/bold] to verify.[/dim]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register Ollama local models into Aexoryn Agent")
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--judge", default=None,
        help="Set a specific Ollama model as Super-Judge (e.g. llama3.3:70b)",
    )
    args = parser.parse_args()
    asyncio.run(main(base_url=args.base_url, judge=args.judge))
