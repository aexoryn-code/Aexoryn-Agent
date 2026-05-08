# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
core/memory.py
──────────────
Long-Term Memory Store for Aexoryn Agent.

Persists task→solution pairs in logs/memory.json.
Uses keyword-overlap scoring (no heavy ML deps) to retrieve
the most relevant past solutions as context for new tasks.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

MEMORY_PATH = Path(__file__).parent.parent / "logs" / "memory.json"
MAX_MEMORY_ENTRIES = 500          # cap to avoid unbounded growth
MAX_CONTEXT_CHARS   = 4000        # max chars injected as memory context
SNIPPET_CHARS       = 600         # chars shown per recalled memory


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lowercase word tokens, stop-words removed."""
    STOP = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
        "or", "is", "it", "with", "that", "this", "be", "as", "by",
        "do", "we", "i", "you", "my", "your", "from", "into", "create",
        "make", "add", "write", "build", "fix", "update", "use", "using",
    }
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOP]


def _tf_idf_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Simple TF-IDF-inspired overlap score between query and document."""
    if not query_tokens or not doc_tokens:
        return 0.0
    q_set  = set(query_tokens)
    d_freq = Counter(doc_tokens)
    d_len  = len(doc_tokens)
    score  = 0.0
    for tok in q_set:
        tf = d_freq.get(tok, 0) / d_len if d_len else 0
        # IDF approximation: boost rare terms (inverse of query freq)
        score += tf
    return score


# ── Memory Store ──────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Persistent task-memory that helps the agent recall relevant past solutions.

    Usage:
        mem = MemoryStore()
        mem.save(task, solution, files_modified, workspace)
        ctx = mem.recall(new_task, top_k=3)   # returns injected context string
    """

    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _dump(self, entries: List[Dict]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        task: str,
        solution: str,
        files_modified: List[str],
        workspace: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Persist a task→solution pair to long-term memory."""
        entries = self._load()

        entry = {
            "id":             len(entries) + 1,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "workspace":      workspace,
            "task":           task,
            "solution":       solution[:1500],
            "files_modified": files_modified,
            "tags":           tags or [],
            "tokens":         _tokenise(task + " " + solution[:500]),
        }
        entries.append(entry)

        # Cap memory size
        if len(entries) > MAX_MEMORY_ENTRIES:
            entries = entries[-MAX_MEMORY_ENTRIES:]

        self._dump(entries)
        logger.debug(f"[Memory] Saved entry #{entry['id']}: {task[:60]}")

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(self, task: str, top_k: int = 3) -> str:
        """
        Search memory for the most relevant past solutions.
        Returns a formatted context string to inject into the jury prompt.
        """
        entries = self._load()
        if not entries:
            return ""

        query_tokens = _tokenise(task)
        if not query_tokens:
            return ""

        scored: List[Tuple[float, Dict]] = []
        for e in entries:
            doc_tokens = e.get("tokens") or _tokenise(e.get("task", "") + " " + e.get("solution", ""))
            score = _tf_idf_score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((score, e))

        if not scored:
            return ""

        # Sort by score desc, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        parts: List[str] = ["### Relevant Past Solutions (Long-Term Memory)"]
        total_chars = 0

        for score, e in top:
            if total_chars >= MAX_CONTEXT_CHARS:
                break
            snippet = e.get("solution", "")[:SNIPPET_CHARS]
            files   = ", ".join(e.get("files_modified", [])) or "—"
            chunk   = (
                f"\n**Past Task [{e['timestamp'][:10]}]:** {e['task']}\n"
                f"**Files:** {files}\n"
                f"**Solution snippet:**\n```\n{snippet}\n```"
            )
            parts.append(chunk)
            total_chars += len(chunk)

        if len(parts) == 1:
            return ""

        logger.debug(f"[Memory] Recalled {len(parts)-1} relevant past solution(s) for: {task[:60]}")
        return "\n".join(parts)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        entries = self._load()
        return {
            "total_entries": len(entries),
            "path": str(self._path),
            "size_kb": round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
        }
