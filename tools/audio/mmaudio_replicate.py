"""MMAudio V2 (video-to-audio) via Replicate.

Replicate model: zsxkib/mmaudio (community model, pinned version, billed by GPU time)

Generates sound effects and ambience that follow what happens in a video clip
(wind when grass bends, waves when water gets choppy, a rumble on a lightning
flash). Also works text-only. Music is suppressed by default through the
negative prompt.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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

MODEL_SLUG = "zsxkib/mmaudio"
MODEL_VERSION = "62871fb59889b2d7c13777f08deb3b36bdff88f7e1d53a50ad7694548a41b484"
# Replicate model page: about $0.0058 per run for the default 8 s; scale by length.
_PRICE_PER_8S = 0.006
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac"}


class MMAudioReplicate(BaseTool):
    name = "mmaudio_replicate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "mmaudio"
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
    agent_skills = ["sound-effects"]

    capabilities = ["video_to_audio", "synchronized_sound_effects", "text_to_audio"]
    supports = {
        "video_input": True,
        "image_input": True,
        "music": False,
        "seed": True,
    }
    best_for = [
        "ambience and sound effects synced to a silent generated clip (wind, water, weather, footsteps)",
        "replacing patchy native clip audio with one consistent sound pass per clip",
        "very cheap sound design through an existing Replicate token",
    ]
    not_good_for = [
        "music or speech",
        "clips much longer than about 10 s in one run (split them)",
        "offline generation",
    ]
    fallback_tools = ["elevenlabs_soundtrack", "freesound_music"]
    quality_score = 0.75

    input_schema = {
        "type": "object",
        "properties": {
            "video": {
                "type": "string",
                "description": "Local path or URL of the clip to add sound to.",
            },
            "image": {"type": "string", "description": "Optional still image instead of a video."},
            "prompt": {
                "type": "string",
                "default": "",
                "description": "Sounds to produce, e.g. 'gusty wind over a lake, reeds rustling'.",
            },
            "negative_prompt": {"type": "string", "default": "music"},
            "duration": {
                "type": "number",
                "default": 8,
                "description": "Audio length in seconds; match the clip length.",
            },
            "num_steps": {"type": "integer", "default": 25},
            "cfg_strength": {"type": "number", "default": 4.5},
            "seed": {"type": "integer"},
            "output_path": {
                "type": "string",
                "description": "Audio suffix (.wav/.m4a/...) extracts the track; otherwise the provider file is saved as is.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["video", "image", "prompt", "negative_prompt", "duration", "seed"]
    side_effects = ["writes audio or video file to output_path", "calls Replicate API"]
    user_visible_verification = ["Listen with the clip: sounds match the action and timing"]

    def _get_api_token(self) -> str | None:
        return os.environ.get("REPLICATE_API_TOKEN")

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._get_api_token() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        seconds = float(inputs.get("duration") or 8)
        return round(_PRICE_PER_8S * max(seconds, 8) / 8, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 30.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        token = self._get_api_token()
        if not token:
            return ToolResult(
                success=False, error="REPLICATE_API_TOKEN not set. " + self.install_instructions
            )
        if not (inputs.get("video") or inputs.get("image") or inputs.get("prompt")):
            return ToolResult(success=False, error="Give a video, an image or a prompt")

        start = time.time()
        payload_input: dict[str, Any] = {
            "prompt": inputs.get("prompt") or "",
            "negative_prompt": inputs.get("negative_prompt", "music"),
            "duration": float(inputs.get("duration") or 8),
            "num_steps": int(inputs.get("num_steps") or 25),
            "cfg_strength": float(inputs.get("cfg_strength") or 4.5),
        }
        if inputs.get("seed") is not None:
            payload_input["seed"] = inputs["seed"]

        requested = Path(inputs["output_path"]) if inputs.get("output_path") else None
        try:
            from tools.audio._replicate import download, run_prediction, to_input_uri

            if inputs.get("video"):
                payload_input["video"] = to_input_uri(token, inputs["video"])
            if inputs.get("image"):
                payload_input["image"] = to_input_uri(token, inputs["image"])
            pred = run_prediction(token, {"version": MODEL_VERSION, "input": payload_input})

            want_audio_only = requested is not None and requested.suffix.lower() in _AUDIO_SUFFIXES
            raw_target = None if want_audio_only else (str(requested) if requested else None)
            raw = download(pred, raw_target, str(requested.with_suffix("")) + ".raw" if requested else "mmaudio_output")
            output_path = raw
            if want_audio_only and raw.suffix.lower() not in _AUDIO_SUFFIXES:
                if not shutil.which("ffmpeg"):
                    raise RuntimeError("ffmpeg is needed to extract the audio track")
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vn", str(requested)],
                    check=True,
                )
                raw.unlink()
                output_path = requested
            elif want_audio_only:
                raw.rename(requested)
                output_path = requested
        except Exception as e:
            return ToolResult(success=False, error=f"Replicate MMAudio generation failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "mmaudio",
                "gateway": "replicate",
                "model": f"{MODEL_SLUG}:{MODEL_VERSION[:12]}",
                "prompt": payload_input["prompt"],
                "duration_seconds": payload_input["duration"],
                "seed": pred.get("input", {}).get("seed"),
                "output": str(output_path),
                "output_path": str(output_path),
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL_SLUG,
        )
