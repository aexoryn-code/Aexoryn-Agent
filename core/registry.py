# MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn
"""
core/registry.py
────────────────
Model Registry — loads model_config.yaml and provides Plug-and-Play access
to all registered LLMs. Supports hot-reload, tier routing, and weight lookup.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel


CONFIG_PATH = Path(__file__).parent.parent / "model_config.yaml"


# ── Data Models ──────────────────────────────────────────────────────────────

class ModelEntry(BaseModel):
    id: str
    name: str
    provider: str                  # openrouter | litellm | ollama
    model_path: str
    tier: str                      # cloud | local
    enabled: bool = True
    weight: float = 1.0
    strengths: List[str] = []
    max_tokens: int = 4096
    temperature: float = 0.2
    base_url: Optional[str] = None


class GatewayConfig(BaseModel):
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 120
    max_retries: int = 3


class HybridRoutingConfig(BaseModel):
    local_threshold_tokens: int = 2000
    cloud_force_tags: List[str] = []


class ConsensusConfig(BaseModel):
    mode: str = "super_judge"
    min_models_required: int = 1
    judge_model_id: str = "anthropic-claude-3.7-sonnet"
    conflict_resolution: str = "weighted"
    tournament_top_k: int = 8
    max_concurrent: int = 30
    jury_timeout: int = 30


class TokenTrackingConfig(BaseModel):
    enabled: bool = True
    warn_threshold: int = 50000
    session_cap: int = 500000


class SelfImproveConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 3
    test_commands: List[str] = ["pytest", "npm test"]


class AgentConfig(BaseModel):
    gateway: GatewayConfig = GatewayConfig()
    hybrid_routing: HybridRoutingConfig = HybridRoutingConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    token_tracking: TokenTrackingConfig = TokenTrackingConfig()
    self_improve: SelfImproveConfig = SelfImproveConfig()
    models: List[ModelEntry] = []


# ── Registry ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads model_config.yaml and provides:
      - get_enabled_models() — all active jury members
      - get_model(id)        — single model by id
      - route(task)          — hybrid routing decision (cloud vs local)
      - build_client(model)  — returns a configured AsyncOpenAI client
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self._config_path = config_path
        self.config: AgentConfig = self._load()
        self._clients: Dict[str, AsyncOpenAI] = {}
        self._session_tokens: int = 0

    # ── Load / Reload ─────────────────────────────────────────────────────────

    def _load(self) -> AgentConfig:
        with open(self._config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        config = AgentConfig(**raw)
        logger.debug(f"[Registry] Loaded {len(config.models)} model(s) from config.")
        return config

    def reload(self) -> None:
        """Hot-reload config without restarting the agent."""
        self.config = self._load()
        self._clients.clear()
        logger.info("[Registry] Configuration hot-reloaded.")

    # ── Token Budget ──────────────────────────────────────────────────────────────

    @property
    def session_tokens(self) -> int:
        return self._session_tokens

    def add_tokens(self, count: int) -> None:
        self._session_tokens += count

    def check_budget(self) -> None:
        """
        Enforce the token session_cap hard limit.
        Emits a warning at warn_threshold.
        Raises BudgetExceededError at session_cap.
        """
        cfg = self.config.token_tracking
        if not cfg.enabled:
            return
        used = self._session_tokens
        if used >= cfg.session_cap:
            raise RuntimeError(
                f"\n\n"
                f"  🛑  Token session cap reached ({used:,} / {cfg.session_cap:,}).\n"
                f"  No more requests will be sent this session.\n\n"
                f"  Tip: Start a new session or increase session_cap in model_config.yaml."
            )
        if used >= cfg.warn_threshold:
            logger.warning(
                f"[Registry] Token budget warning: {used:,} / {cfg.session_cap:,} used "
                f"({used/cfg.session_cap*100:.1f}%). Approaching session cap."
            )

    # ── Model Access ───────────────────────────────────────────────────────────────

    def get_enabled_models(self, tier: Optional[str] = None) -> List[ModelEntry]:
        """Return all enabled models, optionally filtered by tier."""
        models = [m for m in self.config.models if m.enabled]
        if tier:
            models = [m for m in models if m.tier == tier]
        return models

    def get_model(self, model_id: str) -> ModelEntry:
        for m in self.config.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Model '{model_id}' not found in registry.")

    def get_judge_model(self) -> ModelEntry:
        return self.get_model(self.config.consensus.judge_model_id)

    # ── Model Profiles ────────────────────────────────────────────────────────

    PROFILES: Dict[str, Dict] = {
        "fast":     {"mode": "weighted_rank", "top_n": 1,  "free_only": False},
        "turbo":    {"mode": "majority_vote", "top_n": 5,  "free_only": True},
        "balanced": {"mode": "super_judge",   "top_n": 15, "free_only": True},
        "premium":  {"mode": "super_judge",   "top_n": 50, "free_only": False},
    }

    def apply_profile(self, profile: str) -> None:
        """
        Apply a named profile preset, adjusting consensus mode and active model pool.
          fast     — 1 model (weighted_rank), lowest cost, fastest
          turbo    — 5 free models (majority_vote), quick consensus
          balanced — 15 free models (super_judge), best free quality
          premium  — top 50 models including paid (super_judge), highest quality
        """
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(self.PROFILES)}")

        cfg = self.PROFILES[profile]
        self.config.consensus.mode = cfg["mode"]

        enabled_models = sorted(
            [m for m in self.config.models if not cfg["free_only"] or ":free" in (m.model_path or "").lower()],
            key=lambda m: m.weight,
            reverse=True,
        )
        top_ids = {m.id for m in enabled_models[: cfg["top_n"]]}

        for m in self.config.models:
            m.enabled = m.id in top_ids

        logger.info(
            f"[Registry] Profile '{profile}' applied — "
            f"mode={cfg['mode']}, {len(top_ids)} model(s) active."
        )

    # ── Hybrid Routing ────────────────────────────────────────────────────────

    def route(self, estimated_tokens: int, tags: List[str] = []) -> str:
        """
        Returns 'local' or 'cloud' based on hybrid routing rules.
        Force-cloud if any tag matches cloud_force_tags.
        """
        force_tags = self.config.hybrid_routing.cloud_force_tags
        if any(t in force_tags for t in tags):
            return "cloud"
        if estimated_tokens <= self.config.hybrid_routing.local_threshold_tokens:
            local_models = self.get_enabled_models(tier="local")
            if local_models:
                return "local"
        return "cloud"

    # ── Client Builder ────────────────────────────────────────────────────────

    def build_client(self, model: ModelEntry) -> AsyncOpenAI:
        """Build (or reuse) a provider-appropriate AsyncOpenAI-compatible client."""
        if model.id in self._clients:
            return self._clients[model.id]

        import os
        if model.provider == "ollama":
            base = model.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            client = AsyncOpenAI(
                api_key="ollama",
                base_url=f"{base.rstrip('/')}/v1",
            )
        elif model.provider in ("openrouter", "litellm"):
            api_key = (
                os.getenv("OPENROUTER_API_KEY", "")
                if model.provider == "openrouter"
                else os.getenv("LITELLM_API_KEY", "")
            )
            base = (
                os.getenv("OPENROUTER_BASE_URL", self.config.gateway.base_url)
                if model.provider == "openrouter"
                else os.getenv("LITELLM_BASE_URL", self.config.gateway.base_url)
            )
            client = AsyncOpenAI(api_key=api_key, base_url=base)
        else:
            raise ValueError(f"Unknown provider: {model.provider}")

        self._clients[model.id] = client
        return client

    # ── Single Inference ──────────────────────────────────────────────────────

    async def call_model(
        self,
        model: ModelEntry,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Call a single model and return a structured result dict.
        Returns: {model_id, name, content, tokens_used, error}
        """
        client = self.build_client(model)
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = await client.chat.completions.create(
                model=model.model_path,
                messages=full_messages,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            self.add_tokens(tokens)
            return {
                "model_id": model.id,
                "name": model.name,
                "content": content,
                "tokens_used": tokens,
                "error": None,
            }
        except Exception as exc:
            logger.warning(f"[Registry] Model {model.id} failed: {exc}")
            return {
                "model_id": model.id,
                "name": model.name,
                "content": "",
                "tokens_used": 0,
                "error": str(exc),
            }

    # ── Parallel Jury Call (Semaphore-Limited) ───────────────────────────────

    async def call_jury(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        tier: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fire parallel inference across all enabled models (the Jury).
        Uses asyncio.Semaphore to cap concurrent API calls at max_concurrent.
        Returns list of result dicts sorted by weight descending.
        """
        self.check_budget()
        jury = self.get_enabled_models(tier=tier)
        if not jury:
            raise RuntimeError("No enabled models available for jury call.")

        max_concurrent = self.config.consensus.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            f"[Registry] Jury pool: {len(jury)} model(s) | "
            f"concurrency cap: {max_concurrent}"
        )

        jury_timeout = self.config.consensus.jury_timeout

        async def _guarded_call(model: ModelEntry) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.call_model(model, messages, system_prompt),
                        timeout=jury_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[Registry] Model {model.id} timed out after {jury_timeout}s")
                    return {
                        "model_id": model.id,
                        "name": model.name,
                        "content": "",
                        "tokens_used": 0,
                        "error": f"Timed out after {jury_timeout}s",
                    }

        tasks = [_guarded_call(m) for m in jury]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        weight_map = {m.id: m.weight for m in jury}
        results.sort(key=lambda r: weight_map.get(r["model_id"], 1.0), reverse=True)
        return list(results)
