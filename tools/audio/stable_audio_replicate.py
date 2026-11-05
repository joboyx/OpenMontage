"""Stability AI Stable Audio 2.5 via Replicate.

Replicate model: stability-ai/stable-audio-2.5 (official model, flat price per run)

Generates one continuous instrumental music or sound track from a text prompt,
up to 190 seconds long, so a whole short-form bed comes from one generation
with no joins.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

MODEL_SLUG = "stability-ai/stable-audio-2.5"
MAX_DURATION = 190
MAX_STEPS = 8
# Flat USD price per prediction, from the Replicate model page.
_PRICE_PER_RUN = 0.20


class StableAudioReplicate(BaseTool):
    name = "stable_audio_replicate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "stability"
    hosting_provider = "replicate"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set REPLICATE_API_TOKEN to your Replicate API token.\n"
        "  Get one at https://replicate.com/account/api-tokens"
    )
    agent_skills = ["music"]

    capabilities = ["text_to_music", "text_to_audio"]
    supports = {
        "instrumental": True,
        "vocals": False,
        "duration_seconds_max": MAX_DURATION,
        "seed": True,
    }
    best_for = [
        "one continuous instrumental bed up to 190 s (no joins between chunks)",
        "ambient, cinematic, lo-fi and calm background music for short-form video",
        "cheap flat-price music generation through an existing Replicate token",
    ]
    not_good_for = [
        "songs with lyrics or vocals",
        "exact tempo or key control (describe them in the prompt instead)",
        "offline generation",
    ]
    fallback_tools = ["music_gen", "google_music", "elevenlabs_music", "pixabay_music"]
    quality_score = 0.8

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Genre, instruments, mood, tempo and key, e.g. 'calm ambient piano, 60 BPM, F major'.",
            },
            "duration": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_DURATION,
                "default": 60,
                "description": "Track length in seconds.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_STEPS,
                "default": MAX_STEPS,
                "description": "Diffusion steps; more is better quality and slower.",
            },
            "cfg_scale": {
                "type": "number",
                "minimum": 1,
                "maximum": 25,
                "default": 1,
                "description": "Prompt adherence.",
            },
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "duration", "steps", "cfg_scale", "seed"]
    side_effects = ["writes audio file to output_path", "calls Replicate API"]
    user_visible_verification = ["Listen for mood match, no abrupt endings and no artifacts"]

    def _get_api_token(self) -> str | None:
        return os.environ.get("REPLICATE_API_TOKEN")

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._get_api_token() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return _PRICE_PER_RUN

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 30.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        duration = int(inputs.get("duration") or 60)
        steps = int(inputs.get("steps") or MAX_STEPS)
        if not 1 <= duration <= MAX_DURATION:
            return ToolResult(success=False, error=f"duration must be 1-{MAX_DURATION}, got {duration}")
        if not 1 <= steps <= MAX_STEPS:
            return ToolResult(success=False, error=f"steps must be 1-{MAX_STEPS}, got {steps}")

        token = self._get_api_token()
        if not token:
            return ToolResult(
                success=False, error="REPLICATE_API_TOKEN not set. " + self.install_instructions
            )

        payload_input: dict[str, Any] = {
            "prompt": inputs["prompt"],
            "duration": duration,
            "steps": steps,
            "cfg_scale": float(inputs.get("cfg_scale") or 1),
        }
        if inputs.get("seed") is not None:
            payload_input["seed"] = inputs["seed"]

        start = time.time()
        try:
            from tools.audio._replicate import download, run_prediction

            pred = run_prediction(
                token, {"input": payload_input},
                url=f"https://api.replicate.com/v1/models/{MODEL_SLUG}/predictions",
            )
            output_path = download(pred, inputs.get("output_path"), "stable_audio_output")
        except Exception as e:
            return ToolResult(success=False, error=f"Replicate Stable Audio generation failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "stability",
                "gateway": "replicate",
                "model": MODEL_SLUG,
                "prompt": inputs["prompt"],
                "duration_seconds": duration,
                "steps": steps,
                "seed": pred.get("input", {}).get("seed"),
                "output": str(output_path),
                "output_path": str(output_path),
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL_SLUG,
        )
