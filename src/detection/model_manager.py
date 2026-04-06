"""
Model Manager — vLLM lifecycle management for two-tier VLM classification.

Handles model swapping on a single GPU: stops one vLLM model and starts
another via a configurable swap mechanism (shell script or HTTP API).

Usage::

    manager = ModelManager(
        vllm_url="http://10.10.2.222:8000",
        tier1=ModelConfig(name="soccer-8b-lora", ...),
        tier2=ModelConfig(name="qwen3-vl-32b-fp8", ...),
    )

    with manager.tier("tier1"):
        # vLLM is serving the 8B model
        ...

    with manager.tier("tier2"):
        # vLLM swapped to 32B model
        ...
"""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one VLM tier."""
    name: str
    """Model name as registered in vLLM (--served-model-name)."""

    tier: str
    """Tier identifier: 'tier1' or 'tier2'."""

    max_tokens: int = 500
    """Default max_tokens for requests to this model."""

    temperature: float = 0.0
    """Default temperature for requests to this model."""

    model_path: str = ""
    """HuggingFace model path or local path (used by swap script)."""

    lora_path: str = ""
    """LoRA adapter path if applicable."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manage vLLM model lifecycle for two-tier classification.

    Supports two swap strategies:

    1. **Script-based** (recommended): Calls an external shell script that
       stops the current vLLM process and starts a new one.  The script
       receives the target tier's ModelConfig fields as environment variables.

    2. **Single-model**: When only one tier is configured, no swapping is
       needed.  The manager validates that the requested model is loaded and
       returns immediately.

    The manager is intentionally simple — it does not manage the vLLM process
    directly; it delegates to the swap script.  This keeps operational
    concerns (systemd, Docker, SSH) outside the Python pipeline.
    """

    def __init__(
        self,
        vllm_url: str,
        *,
        tier1: Optional[ModelConfig] = None,
        tier2: Optional[ModelConfig] = None,
        swap_script: str = "",
        swap_timeout_sec: int = 120,
        health_poll_interval_sec: float = 2.0,
    ):
        self._vllm_url = vllm_url.rstrip("/")
        self._configs: dict[str, ModelConfig] = {}
        if tier1:
            self._configs["tier1"] = tier1
        if tier2:
            self._configs["tier2"] = tier2

        self._swap_script = swap_script
        self._swap_timeout = swap_timeout_sec
        self._poll_interval = health_poll_interval_sec

        # Track which tier is believed to be loaded (None = unknown)
        self._current_tier: Optional[str] = None

        # Detect initial state
        self._current_tier = self._detect_current_tier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_tier(self) -> Optional[str]:
        """Currently loaded tier ('tier1', 'tier2', or None if unknown)."""
        return self._current_tier

    @property
    def current_config(self) -> Optional[ModelConfig]:
        """ModelConfig for the currently loaded tier."""
        if self._current_tier:
            return self._configs.get(self._current_tier)
        return None

    @property
    def current_model_name(self) -> Optional[str]:
        """Model name for the currently loaded tier."""
        cfg = self.current_config
        return cfg.name if cfg else None

    def config_for(self, tier: str) -> Optional[ModelConfig]:
        """Get ModelConfig for a given tier."""
        return self._configs.get(tier)

    def is_healthy(self) -> bool:
        """Check if vLLM is responding."""
        try:
            import httpx
            r = httpx.get(f"{self._vllm_url}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_loaded_models(self) -> list[str]:
        """Query vLLM for currently loaded model names."""
        try:
            import httpx
            r = httpx.get(f"{self._vllm_url}/v1/models", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    def ensure_tier(self, tier: str) -> bool:
        """Ensure the given tier's model is loaded.  Swaps if needed.

        Returns True if the tier is ready, False if swap failed.
        """
        if tier not in self._configs:
            log.error("model_manager.unknown_tier", tier=tier,
                      available=list(self._configs.keys()))
            return False

        # Already loaded?
        if self._current_tier == tier:
            if self.is_healthy():
                log.debug("model_manager.already_loaded", tier=tier)
                return True
            # Tier was set but server is unhealthy — try to detect again
            self._current_tier = self._detect_current_tier()
            if self._current_tier == tier:
                return True

        # Need to swap
        target = self._configs[tier]
        log.info("model_manager.swapping",
                 from_tier=self._current_tier,
                 to_tier=tier,
                 target_model=target.name)

        success = self._swap_to(tier)
        if success:
            self._current_tier = tier
            log.info("model_manager.swap_complete", tier=tier,
                     model=target.name)
        else:
            log.error("model_manager.swap_failed", tier=tier,
                      model=target.name)
        return success

    @contextmanager
    def tier(self, tier_name: str):
        """Context manager that ensures the requested tier is loaded.

        Usage::

            with manager.tier("tier1"):
                # vLLM is serving tier1's model
                verdicts = verifier.verify(candidates)

        Raises RuntimeError if the swap fails.
        """
        if not self.ensure_tier(tier_name):
            raise RuntimeError(
                f"Failed to load tier '{tier_name}' model "
                f"({self._configs.get(tier_name, '?')})"
            )
        yield self._configs[tier_name]

    # ------------------------------------------------------------------
    # Internal — model detection
    # ------------------------------------------------------------------

    def _detect_current_tier(self) -> Optional[str]:
        """Query vLLM and match loaded model(s) to a tier config."""
        loaded = self.get_loaded_models()
        if not loaded:
            return None

        for tier_name, cfg in self._configs.items():
            if cfg.name in loaded:
                log.debug("model_manager.detected_tier",
                          tier=tier_name, model=cfg.name)
                return tier_name

        log.warning("model_manager.unknown_loaded_model",
                    loaded=loaded,
                    known={k: v.name for k, v in self._configs.items()})
        return None

    # ------------------------------------------------------------------
    # Internal — model swapping
    # ------------------------------------------------------------------

    def _swap_to(self, tier: str) -> bool:
        """Execute the model swap.  Tries script-based swap first,
        falls back to waiting for the model to appear (manual swap)."""
        target = self._configs[tier]

        if self._swap_script:
            return self._swap_via_script(target)

        # No swap script — check if model is already loaded (maybe the
        # user manually switched models or only one tier is configured)
        log.warning("model_manager.no_swap_script",
                    hint="Set MODEL_SWAP_SCRIPT to enable automatic model swapping")
        return self._wait_for_model(target.name, timeout_sec=10)

    def _swap_via_script(self, target: ModelConfig) -> bool:
        """Call the swap script with target model info as env vars."""
        import os

        script = self._swap_script
        if not Path(script).exists():
            log.error("model_manager.swap_script_not_found", path=script)
            return False

        env = {
            **os.environ,
            "SWAP_TARGET_NAME": target.name,
            "SWAP_TARGET_PATH": target.model_path,
            "SWAP_TARGET_LORA": target.lora_path,
            "SWAP_TARGET_TIER": target.tier,
            "SWAP_VLLM_URL": self._vllm_url,
        }

        log.info("model_manager.running_swap_script",
                 script=script, target=target.name)

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                ["bash", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=self._swap_timeout,
            )
            elapsed = time.monotonic() - t0

            if result.returncode != 0:
                log.error("model_manager.swap_script_failed",
                          returncode=result.returncode,
                          stderr=result.stderr[-500:] if result.stderr else "",
                          elapsed_sec=round(elapsed, 1))
                return False

            log.info("model_manager.swap_script_ok",
                     elapsed_sec=round(elapsed, 1))

        except subprocess.TimeoutExpired:
            log.error("model_manager.swap_script_timeout",
                      timeout_sec=self._swap_timeout)
            return False
        except Exception as exc:
            log.error("model_manager.swap_script_error", error=str(exc))
            return False

        # Wait for the new model to appear on the health endpoint
        return self._wait_for_model(target.name, timeout_sec=self._swap_timeout)

    def _wait_for_model(self, model_name: str, timeout_sec: float) -> bool:
        """Poll vLLM until the target model is loaded or timeout."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            loaded = self.get_loaded_models()
            if model_name in loaded:
                return True
            time.sleep(self._poll_interval)

        log.warning("model_manager.wait_timeout",
                    model=model_name, timeout_sec=timeout_sec,
                    loaded=self.get_loaded_models())
        return False
