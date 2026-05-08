#!/usr/bin/env python3
# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
scripts/fetch_models.py
───────────────────────
Fetches ALL available models from the OpenRouter API and writes them into
model_config.yaml — preserving gateway / routing / consensus settings.

Usage:
  python scripts/fetch_models.py
  python scripts/fetch_models.py --enable-all
  python scripts/fetch_models.py --top 50        (enable only top 50 by weight)

After running, restart the agent or use --reload-config to pick up the changes.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

load_dotenv(Path(__file__).parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent.parent / "model_config.yaml"
OR_MODELS_URL = "https://openrouter.ai/api/v1/models"

console = Console()


# ── Weight Heuristics ─────────────────────────────────────────────────────────

def _assign_weight(model_id: str, name: str) -> float:
    """
    Assign a consensus weight (0.5 – 2.0) based on known model quality tiers.
    Higher = more influence in the tournament ranking.
    """
    mid = model_id.lower()
    nm  = name.lower()

    # Tier 1 — Frontier reasoning models
    if any(x in mid for x in ["o3", "o4", "gpt-5", "claude-4", "claude-opus-4",
                                "gemini-2.5-pro", "deepseek-r2", "grok-3"]):
        return 1.8

    # Tier 2 — Strong general-purpose
    if any(x in mid for x in ["gpt-4o", "gpt-4.1", "claude-3.7", "claude-3.5",
                                "claude-sonnet", "gemini-2.0-pro", "deepseek-r1",
                                "llama-4-maverick", "llama-4-scout", "mistral-large",
                                "command-r-plus", "qwen-2.5-72b", "qwen3-235b"]):
        return 1.4

    # Tier 3 — Capable mid-range
    if any(x in mid for x in ["gpt-4", "claude-3", "gemini-1.5-pro", "gemini-flash",
                                "llama-3.3", "llama-3.1-70b", "mixtral-8x22b",
                                "deepseek-chat", "qwen-2.5-32b", "qwen3-32b",
                                "phi-4", "nova-pro"]):
        return 1.1

    # Tier 4 — Fast / lightweight
    if any(x in mid for x in ["gpt-3.5", "gpt-4o-mini", "claude-haiku",
                                "gemini-flash", "llama-3.1-8b", "llama-3.2",
                                "mistral-7b", "mistral-nemo", "phi-3",
                                "qwen-2.5-7b", "qwen3-8b", "gemma"]):
        return 0.7

    return 0.9  # default


def _assign_strengths(model_id: str) -> List[str]:
    mid = model_id.lower()
    strengths = ["general"]
    if any(x in mid for x in ["coder", "code", "qwen-coder", "deepseek-coder", "starcoder"]):
        strengths += ["code"]
    if any(x in mid for x in ["o1", "o3", "o4", "r1", "r2", "reasoning", "think"]):
        strengths += ["reasoning"]
    if any(x in mid for x in ["vision", "vl", "llava", "pixtral"]):
        strengths += ["vision"]
    if any(x in mid for x in ["instruct", "chat"]):
        strengths += ["chat"]
    return list(dict.fromkeys(strengths))  # deduplicate, preserve order


def _safe_id(model_path: str) -> str:
    """Convert 'openai/gpt-4o' → 'openai-gpt-4o' as a YAML-safe id.
    Strips ~ prefix used by OpenRouter dynamic aliases."""
    clean = model_path.lstrip("~")
    safe = re.sub(r"[^a-zA-Z0-9\-_.]", "-", clean)
    return safe.strip("-")


def _context_to_max_tokens(context_length: Optional[int]) -> int:
    # Keep jury max_tokens small — reduces cost per call significantly.
    # Free models ignore this; paid models bill per output token.
    return 2048


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def fetch_openrouter_models(api_key: str) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/aexoryn/aexoryn-agent",
        "X-Title": "Aexoryn Agent",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(OR_MODELS_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data.get("data", [])


# ── Build Model Entry ─────────────────────────────────────────────────────────

def build_model_entry(
    raw: Dict[str, Any],
    enable_all: bool = False,
    top_ids: Optional[set] = None,
    free_only: bool = False,
) -> Dict[str, Any]:
    model_path = raw.get("id", "")
    name       = raw.get("name", model_path)
    ctx_len    = raw.get("context_length")
    is_free    = ":free" in model_path.lower()

    if free_only:
        enabled = is_free
    elif enable_all:
        enabled = True
    elif top_ids and model_path in top_ids:
        enabled = True
    else:
        enabled = False

    return {
        "id":          _safe_id(model_path),
        "name":        name,
        "provider":    "openrouter",
        "model_path":  model_path,
        "tier":        "cloud",
        "enabled":     enabled,
        "weight":      _assign_weight(model_path, name),
        "strengths":   _assign_strengths(model_path),
        "max_tokens":  _context_to_max_tokens(ctx_len),
        "temperature": 0.2,
    }


# ── Write Config ──────────────────────────────────────────────────────────────

def update_config(new_models: List[Dict], top: Optional[int] = None) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # If --top N: enable only the top N models by weight
    if top:
        sorted_models = sorted(new_models, key=lambda m: m["weight"], reverse=True)
        for i, m in enumerate(sorted_models):
            m["enabled"] = i < top
        new_models = sorted_models

    cfg["models"] = new_models
    cfg.setdefault("consensus", {})["judge_model_id"] = "anthropic-claude-3.7-sonnet"
    cfg["consensus"].setdefault("min_models_required", 1)
    cfg["consensus"].setdefault("tournament_top_k", 8)
    cfg["consensus"].setdefault("max_concurrent", 30)
    cfg["consensus"].setdefault("jury_timeout", 30)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Print Summary Table ───────────────────────────────────────────────────────

def print_summary(models: List[Dict]) -> None:
    enabled = [m for m in models if m["enabled"]]
    t = Table(
        title=f"OpenRouter Model Registry — {len(models)} total / {len(enabled)} enabled",
        box=box.ROUNDED, border_style="cyan", show_lines=False,
    )
    t.add_column("Model Path", style="bold", max_width=45)
    t.add_column("Weight", justify="right")
    t.add_column("Max Tokens", justify="right")
    t.add_column("Enabled", justify="center")
    t.add_column("Strengths")

    for m in sorted(models, key=lambda x: x["weight"], reverse=True)[:30]:
        t.add_row(
            m["model_path"],
            f"{m['weight']:.1f}",
            str(m["max_tokens"]),
            "✅" if m["enabled"] else "⬜",
            ", ".join(m["strengths"]),
        )
    if len(models) > 30:
        t.add_row(f"… {len(models) - 30} more models …", "", "", "", "")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(enable_all: bool, top: Optional[int], free_only: bool = False) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or "xxxx" in api_key:
        console.print("[red]ERROR:[/] OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), transient=True) as p:
        t = p.add_task("Fetching models from OpenRouter API…")
        raw_models = await fetch_openrouter_models(api_key)
        p.update(t, description=f"Fetched {len(raw_models)} models. Building entries…")

        # Build top_ids set for selective enabling
        top_ids: Optional[set] = None
        if not enable_all and not top:
            # Default: enable only models with weight >= 1.1 (quality threshold)
            top_ids = {
                r["id"] for r in raw_models
                if _assign_weight(r.get("id", ""), r.get("name", "")) >= 1.1
            }

        if free_only:
            entries = [build_model_entry(r, enable_all=False, top_ids=None, free_only=True) for r in raw_models]
        else:
            entries = [build_model_entry(r, enable_all=enable_all, top_ids=top_ids) for r in raw_models]
        p.update(t, description="Writing model_config.yaml…")
        update_config(entries, top=top)

    console.print(f"\n[bold green]✅ Done![/] {len(entries)} models written to model_config.yaml")
    print_summary(entries)

    enabled_count = len([m for m in entries if m["enabled"]])
    console.print(
        f"\n[dim]Enabled: [bold]{enabled_count}[/bold] models  |  "
        f"Disabled: [bold]{len(entries) - enabled_count}[/bold] models[/dim]"
    )
    console.print(
        "[dim]Tip: Set enabled: true for any model in model_config.yaml, "
        "or re-run with --enable-all to activate all models.[/dim]"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch all OpenRouter models into model_config.yaml")
    parser.add_argument("--enable-all", action="store_true",
                        help="Enable ALL models in the jury (expensive — uses credits)")
    parser.add_argument("--top", type=int, default=None,
                        help="Enable only top N models by quality weight")
    parser.add_argument("--free-only", action="store_true",
                        help="Enable ONLY free models (:free) — zero credit cost")
    args = parser.parse_args()

    asyncio.run(main(enable_all=args.enable_all, top=args.top, free_only=args.free_only))
