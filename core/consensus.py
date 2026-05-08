# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
core/consensus.py
─────────────────
Super-Judge Consensus Engine — analyses all jury outputs, resolves conflicts,
and synthesises a single authoritative "Master Solution".

Modes:
  super_judge    — The judge model reads all outputs and writes the final answer.
  majority_vote  — The most common / highest-scored answer wins.
  weighted_rank  — Score outputs by model weight; return the top-ranked one.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from core.registry import ModelRegistry


# ── Super-Judge System Prompt ─────────────────────────────────────────────────

SUPER_JUDGE_SYSTEM = """\
You are the Super-Judge of the Aexoryn Agent consensus engine.
You will receive responses from multiple AI models (the Jury) for the same task.

Your job:
1. Read every jury response carefully.
2. Identify conflicts, hallucinations, and incorrect reasoning.
3. Extract the best elements from each response.
4. Synthesise ONE definitive, complete, and production-ready Master Solution.
5. Output ONLY the Master Solution — no meta-commentary, no preamble.

If the task involves code:
- The solution must be complete, runnable, and follow best practices.
- Use inline comments to annotate non-obvious logic.
- Do NOT omit any part with placeholders like "..." or "TODO".
"""


# ── Consensus Engine ──────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Orchestrates the jury, resolves conflicts, and returns the Master Solution.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.cfg = registry.config.consensus

    # ── Public Entry Point ────────────────────────────────────────────────────

    async def resolve(
        self,
        task: str,
        jury_results: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Given the raw jury results, run the configured consensus mode and
        return the Master Solution dict:
          {solution, mode, contributing_models, tokens_used, conflicts_found}
        """
        valid = [r for r in jury_results if not r.get("error") and r.get("content")]
        failed = [r for r in jury_results if r.get("error")]

        if failed:
            for f in failed:
                logger.warning(f"[Consensus] Jury member {f['model_id']} failed: {f['error']}")

        min_required = self.registry.config.consensus.min_models_required
        if len(valid) < min_required:
            self._raise_rate_limit_error(failed, len(valid))

        mode = self.cfg.mode
        logger.info(f"[Consensus] Running mode={mode} with {len(valid)} jury member(s).")

        if mode == "super_judge":
            return await self._super_judge(task, valid, context)
        elif mode == "majority_vote":
            return self._majority_vote(valid)
        elif mode == "weighted_rank":
            return self._weighted_rank(valid)
        else:
            raise ValueError(f"Unknown consensus mode: {mode}")

    # ── Super-Judge Mode (with Tournament Filter) ─────────────────────────────

    async def _super_judge(
        self,
        task: str,
        valid: List[Dict[str, Any]],
        context: Optional[str],
    ) -> Dict[str, Any]:
        judge_model = self.registry.get_judge_model()
        top_k = self.cfg.tournament_top_k
        all_count = len(valid)

        # Tournament: when jury > top_k, keep only the highest-weight finalists.
        # This prevents context overflow when 100s of models respond.
        if all_count > top_k:
            weight_map = {m.id: m.weight for m in self.registry.get_enabled_models()}
            finalists = sorted(
                valid,
                key=lambda r: weight_map.get(r["model_id"], 1.0),
                reverse=True,
            )[:top_k]
            logger.info(
                f"[Consensus] Tournament: {all_count} responses → "
                f"top {top_k} finalists selected for Super-Judge."
            )
        else:
            finalists = valid

        jury_block = self._format_jury_block(finalists)
        user_content = (
            f"## Original Task\n{task}\n\n"
            + (f"## Workspace Context\n{context}\n\n" if context else "")
            + f"## Jury Responses ({len(finalists)} finalists of {all_count} total)\n"
            + jury_block + "\n\n"
            + "## Instruction\nSynthesise the single best Master Solution."
        )

        result = await self.registry.call_model(
            judge_model,
            messages=[{"role": "user", "content": user_content}],
            system_prompt=SUPER_JUDGE_SYSTEM,
        )

        conflicts = self._detect_conflicts(finalists)
        total_tokens = sum(r.get("tokens_used", 0) for r in valid) + result.get("tokens_used", 0)

        if result.get("error") or not result.get("content"):
            # Super-Judge failed (e.g. 402 out of credits) — fall back to the
            # top-weighted finalist's response so we always return something useful.
            logger.warning(
                f"[Consensus] Super-Judge failed ({result.get('error', 'empty response')}). "
                f"Falling back to top-weighted finalist: {finalists[0]['name']}"
            )
            return {
                "solution": finalists[0]["content"],
                "mode": "super_judge_fallback",
                "judge": f"{judge_model.name} (failed — used fallback)",
                "jury_total": all_count,
                "finalists": len(finalists),
                "contributing_models": [r["name"] for r in finalists],
                "all_models": [r["name"] for r in valid],
                "tokens_used": total_tokens,
                "conflicts_found": conflicts,
                "raw_jury": valid,
            }

        return {
            "solution": result["content"],
            "mode": "super_judge",
            "judge": judge_model.name,
            "jury_total": all_count,
            "finalists": len(finalists),
            "contributing_models": [r["name"] for r in finalists],
            "all_models": [r["name"] for r in valid],
            "tokens_used": total_tokens,
            "conflicts_found": conflicts,
            "raw_jury": valid,
        }

    # ── Majority Vote Mode ────────────────────────────────────────────────────

    def _majority_vote(self, valid: List[Dict[str, Any]]) -> Dict[str, Any]:
        from collections import Counter
        votes: Counter = Counter()
        for r in valid:
            snippet = r["content"][:200].strip()
            votes[snippet] += 1
        best_snippet = votes.most_common(1)[0][0]
        winner = next(r for r in valid if r["content"].startswith(best_snippet))
        return {
            "solution": winner["content"],
            "mode": "majority_vote",
            "contributing_models": [r["name"] for r in valid],
            "tokens_used": sum(r.get("tokens_used", 0) for r in valid),
            "conflicts_found": [],
            "raw_jury": valid,
        }

    # ── Weighted Rank Mode ────────────────────────────────────────────────────

    def _weighted_rank(self, valid: List[Dict[str, Any]]) -> Dict[str, Any]:
        weight_map = {m.id: m.weight for m in self.registry.get_enabled_models()}
        ranked = sorted(
            valid,
            key=lambda r: weight_map.get(r["model_id"], 1.0),
            reverse=True,
        )
        winner = ranked[0]
        return {
            "solution": winner["content"],
            "mode": "weighted_rank",
            "winner": winner["name"],
            "contributing_models": [r["name"] for r in valid],
            "tokens_used": sum(r.get("tokens_used", 0) for r in valid),
            "conflicts_found": [],
            "raw_jury": valid,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_jury_block(results: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(
                f"### Response {i} — {r['name']} (weight={r.get('weight', '?')})\n"
                f"{r['content']}\n"
                f"{'─' * 60}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _raise_rate_limit_error(failed: List[Dict], valid_count: int) -> None:
        """
        Inspect failed jury results and raise a clear, actionable error.
        Detects 429 rate limits vs 402 no-credits and shows reset time.
        """
        codes = []
        reset_ts: Optional[int] = None

        for f in failed:
            err_str = str(f.get("error", ""))
            # Extract HTTP error code
            code_match = re.search(r"Error code: (\d+)", err_str)
            if code_match:
                codes.append(int(code_match.group(1)))
            # Extract reset timestamp from metadata headers
            ts_match = re.search(r"'X-RateLimit-Reset':\s*'(\d+)'", err_str)
            if ts_match:
                ts = int(ts_match.group(1))
                if reset_ts is None or ts < reset_ts:
                    reset_ts = ts  # take earliest reset

        has_402  = 402 in codes
        has_429  = 429 in codes
        has_429d = any("free-models-per-day" in str(f.get("error", "")) for f in failed)
        has_429m = any("free-models-per-min" in str(f.get("error", "")) for f in failed)

        # Build reset time string
        reset_str = ""
        if reset_ts:
            try:
                dt = datetime.fromtimestamp(reset_ts / 1000, tz=timezone.utc).astimezone()
                reset_str = f"\n  Reset at: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            except Exception:
                pass

        if has_402:
            raise RuntimeError(
                "\n\n"
                "  ❌  No OpenRouter credits remaining.\n"
                "  All paid models returned 402 Insufficient Credits.\n\n"
                "  Fix options:\n"
                "    1. Add credits at https://openrouter.ai/settings/credits\n"
                "    2. Use only free models: python scripts/fetch_models.py --free-only\n"
                "    3. Run a local Ollama model: python main.py setup-ollama\n"
            )

        if has_429d:
            raise RuntimeError(
                "\n\n"
                f"  ⏳  Daily free model limit reached (50 req/day).{reset_str}\n\n"
                "  Fix options:\n"
                "    1. Wait until tomorrow (limit resets daily)\n"
                "    2. Add 10 credits to unlock 1000 free req/day:\n"
                "         https://openrouter.ai/settings/credits\n"
                "    3. Run a local Ollama model (no limit):\n"
                "         python main.py setup-ollama\n"
            )

        if has_429m:
            raise RuntimeError(
                "\n\n"
                f"  ⏳  Per-minute rate limit hit.{reset_str}\n\n"
                "  Fix options:\n"
                "    1. Wait ~1 minute and retry\n"
                "    2. Use --profile fast (fewer parallel calls):\n"
                "         python main.py run --profile fast \"your task\"\n"
                "    3. Run a local Ollama model: python main.py setup-ollama\n"
            )

        # Generic fallback
        raise RuntimeError(
            f"Consensus requires {valid_count >= 1 and valid_count or 1} valid "
            f"response(s); got {valid_count}. "
            f"All {len(failed)} jury model(s) failed. Check your API key and model config."
        )

    @staticmethod
    def _detect_conflicts(results: List[Dict[str, Any]]) -> List[str]:
        """
        Heuristic conflict detection: flag if responses diverge significantly
        in their opening lines (rough proxy for directional disagreement).
        """
        if len(results) < 2:
            return []
        snippets = [r["content"][:150].strip().lower() for r in results]
        unique = set(snippets)
        if len(unique) > 1:
            return [f"Divergence detected among {len(unique)} jury members."]
        return []
