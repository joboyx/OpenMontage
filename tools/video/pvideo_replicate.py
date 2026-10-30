"""Pruna P-Video 2 Pro (reference-to-video) via Replicate.

Replicate model: prunaai/p-video-2-pro-reference

Generates an MP4 (24 fps) with generated sound from a prompt plus optional
reference images (up to 9) and reference audio. Cheap compared with the
premium providers; billed per second of output video plus per reference input.
"""

from __future__ import annotations

import base64
import mimetypes
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

MODEL_SLUG = "prunaai/p-video-2-pro-reference"
MAX_REFERENCE_IMAGES = 9
DURATIONS = [str(s) for s in range(5, 16)]
RESOLUTIONS = ["480p", "768p", "1080p", "2k", "4k"]
ASPECT_RATIOS = ["adaptive", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1"]
MODES = ["cost", "speed"]
PROMPT_UPSAMPLERS = ["off", "turbo", "max"]

# USD per second of output video, keyed by (mode, generation resolution).
# 1080p / 2k / 4k generate at 768p and then upscale. Replicate does not
# publish a separate rate for them, so the estimate uses the 768p rate and
# may under-count the upscale step.
_OUTPUT_RATE = {
    ("cost", "480p"): 0.015,
    ("cost", "768p"): 0.025,
    ("speed", "480p"): 0.02,
    ("speed", "768p"): 0.035,
}
# USD per input megapixel of reference image.
_IMAGE_MP_RATE = {"cost": 0.005, "speed": 0.01}
# USD per second of reference audio.
_AUDIO_SEC_RATE = {"cost": 0.0015, "speed": 0.0025}

# mimetypes on some platforms returns audio/x-wav or nothing for these.
_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class PVideoReplicate(BaseTool):
    name = "pvideo_replicate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "pruna"
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
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "reference_to_video"]
    supports = {
        "text_to_video": True,
        "reference_to_video": True,
        "reference_image": True,
        "native_audio": True,
        "aspect_ratio": True,
        "seed": True,
    }
    best_for = [
        "cheap reference-to-video: up to 9 reference images (characters, objects, style) in one clip",
        "low-cost drafts and social clips (9:16, 1:1) with generated sound",
        "clips that must use a supplied soundtrack (one reference audio with keep_audio)",
        "5-15 second clips at 480p or 768p",
    ]
    not_good_for = [
        "dialogue lip-sync (not documented by the provider)",
        "offline generation",
        "true 1080p+ detail (1080p/2k/4k are 768p upscaled)",
    ]
    fallback_tools = ["seedance_replicate", "seedance_video", "kling_video", "minimax_video"]
    # No measured eval yet; set below the premium providers until clips are reviewed.
    quality_score = 0.8

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_REFERENCE_IMAGES,
                "description": (
                    "Reference images as local paths, http(s) URLs or data URIs. "
                    "Local files are sent inline as base64 data URIs. Sent as `image`."
                ),
            },
            "reference_audio": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Reference audio as local paths, http(s) URLs or data URIs. Sent as `audio`."
                ),
            },
            "keep_audio": {
                "type": "boolean",
                "default": False,
                "description": "Use the single reference_audio file as the clip soundtrack.",
            },
            "duration": {
                "type": "string",
                "enum": DURATIONS,
                "default": "5",
                "description": "Output length in seconds (5-15).",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ASPECT_RATIOS,
                "default": "16:9",
            },
            "resolution": {
                "type": "string",
                "enum": RESOLUTIONS,
                "default": "768p",
                "description": "1080p, 2k and 4k generate at 768p and upscale.",
            },
            "mode": {
                "type": "string",
                "enum": MODES,
                "default": "cost",
                "description": "cost = cheaper and slower, speed = faster and dearer.",
            },
            "prompt_upsampler": {
                "type": "string",
                "enum": PROMPT_UPSAMPLERS,
                "default": "off",
                "description": "off sends the prompt as written; turbo/max let the provider rewrite it.",
            },
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "prompt", "duration", "aspect_ratio", "resolution", "mode", "seed",
        "reference_images", "reference_audio",
    ]
    side_effects = ["writes video file to output_path", "calls Replicate API"]
    user_visible_verification = [
        "Watch generated clip for reference likeness, motion coherence and sound quality"
    ]

    def _get_api_token(self) -> str | None:
        return os.environ.get("REPLICATE_API_TOKEN")

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._get_api_token() else ToolStatus.UNAVAILABLE

    @staticmethod
    def _reference_images(inputs: dict[str, Any]) -> list[str]:
        """Collect reference images, including the video_selector key names."""
        return [
            *(inputs.get("reference_images") or []),
            *(inputs.get("reference_image_paths") or []),
            *(inputs.get("reference_image_urls") or []),
        ]

    @staticmethod
    def _reference_audio(inputs: dict[str, Any]) -> list[str]:
        """Collect reference audio, including the video_selector key names."""
        return [
            *(inputs.get("reference_audio") or []),
            *(inputs.get("reference_audio_paths") or []),
            *(inputs.get("reference_audio_urls") or []),
        ]

    @staticmethod
    def _is_remote(ref: str) -> bool:
        return ref.startswith(("http://", "https://", "data:"))

    @staticmethod
    def _image_megapixels(ref: str) -> float:
        """Megapixels of a local image; 1.0 when it cannot be measured."""
        if PVideoReplicate._is_remote(ref):
            return 1.0
        try:
            from PIL import Image

            with Image.open(ref) as img:
                width, height = img.size
            return width * height / 1_000_000
        except Exception:
            return 1.0

    @staticmethod
    def _audio_seconds(ref: str, fallback: float) -> float:
        """Length of a local audio file via ffprobe; fallback when unknown."""
        if PVideoReplicate._is_remote(ref) or not shutil.which("ffprobe"):
            return fallback
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", ref,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return float(proc.stdout.strip()) if proc.returncode == 0 else fallback
        except (OSError, ValueError, subprocess.SubprocessError):
            return fallback

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Estimate USD cost: output seconds + reference image MP + reference audio seconds."""
        mode = inputs.get("mode") if inputs.get("mode") in MODES else "cost"
        resolution = inputs.get("resolution") or "768p"
        rate_resolution = "480p" if resolution == "480p" else "768p"
        try:
            seconds = int(inputs.get("duration") or 5)
        except (TypeError, ValueError):
            seconds = 5

        cost = seconds * _OUTPUT_RATE[(mode, rate_resolution)]
        megapixels = sum(self._image_megapixels(r) for r in self._reference_images(inputs))
        cost += megapixels * _IMAGE_MP_RATE[mode]
        audio_seconds = sum(
            self._audio_seconds(r, float(seconds)) for r in self._reference_audio(inputs)
        )
        cost += audio_seconds * _AUDIO_SEC_RATE[mode]
        return round(cost, 3)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 60.0 if inputs.get("mode") == "speed" else 120.0

    @staticmethod
    def _to_data_uri(ref: str, mime_overrides: dict[str, str] | None = None) -> str:
        """Pass URLs and data URIs through; inline a local file as a data URI."""
        if PVideoReplicate._is_remote(ref):
            return ref
        path = Path(ref)
        mime_type = (mime_overrides or {}).get(path.suffix.lower())
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _validate(
        self, inputs: dict[str, Any], images: list[str], audio: list[str]
    ) -> str | None:
        """Return an error message for inputs the model would reject, else None."""
        if len(images) > MAX_REFERENCE_IMAGES:
            return (
                f"P-Video 2 Pro accepts at most {MAX_REFERENCE_IMAGES} reference images, "
                f"got {len(images)}"
            )
        if inputs.get("keep_audio") and len(audio) != 1:
            return f"keep_audio needs exactly one reference_audio file, got {len(audio)}"
        if inputs.get("duration") is not None and str(inputs["duration"]) not in DURATIONS:
            return f"duration must be 5-15 seconds, got {inputs['duration']!r}"
        for key, allowed in (
            ("aspect_ratio", ASPECT_RATIOS),
            ("resolution", RESOLUTIONS),
            ("mode", MODES),
            ("prompt_upsampler", PROMPT_UPSAMPLERS),
        ):
            value = inputs.get(key)
            if value is not None and value not in allowed:
                return f"{key} must be one of {allowed}, got {value!r}"
        return None

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        reference_images = self._reference_images(inputs)
        reference_audio = self._reference_audio(inputs)
        error = self._validate(inputs, reference_images, reference_audio)
        if error:
            return ToolResult(success=False, error=error)

        token = self._get_api_token()
        if not token:
            return ToolResult(
                success=False,
                error="REPLICATE_API_TOKEN not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        mode = inputs.get("mode") or "cost"
        resolution = inputs.get("resolution") or "768p"
        aspect_ratio = inputs.get("aspect_ratio") or "16:9"
        payload_input: dict[str, Any] = {
            "prompt": inputs["prompt"],
            "duration": int(inputs.get("duration") or 5),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "mode": mode,
            "prompt_upsampler": inputs.get("prompt_upsampler") or "off",
        }
        if inputs.get("seed") is not None:
            payload_input["seed"] = inputs["seed"]
        if inputs.get("keep_audio"):
            payload_input["keep_audio"] = True

        try:
            if reference_images:
                payload_input["image"] = [self._to_data_uri(r) for r in reference_images]
            if reference_audio:
                payload_input["audio"] = [
                    self._to_data_uri(r, _AUDIO_MIME) for r in reference_audio
                ]
        except OSError as e:
            return ToolResult(success=False, error=f"Cannot read reference file: {e}")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
        }

        try:
            submit = requests.post(
                f"https://api.replicate.com/v1/models/{MODEL_SLUG}/predictions",
                headers=headers,
                json={"input": payload_input},
                timeout=90,
            )
            submit.raise_for_status()
            pred = submit.json()

            # Replicate may return the result synchronously when Prefer: wait
            # is honored; otherwise poll until the prediction settles.
            while pred.get("status") in ("starting", "processing"):
                time.sleep(3)
                get_url = pred.get("urls", {}).get("get")
                if not get_url:
                    return ToolResult(success=False, error="Replicate response missing poll URL")
                poll = requests.get(get_url, headers=headers, timeout=30)
                poll.raise_for_status()
                pred = poll.json()

            status = pred.get("status")
            if status != "succeeded":
                return ToolResult(
                    success=False,
                    error=f"Replicate P-Video generation {status}: {pred.get('error')}",
                )

            output = pred.get("output")
            # Replicate returns either a string URL or a list.
            video_url = output[0] if isinstance(output, list) else output
            if not isinstance(video_url, str):
                return ToolResult(
                    success=False, error=f"Unexpected output shape from Replicate: {output!r}"
                )

            video_response = requests.get(video_url, timeout=180)
            video_response.raise_for_status()

            output_path = Path(inputs.get("output_path", "pvideo_replicate_output.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video_response.content)

        except Exception as e:
            return ToolResult(success=False, error=f"Replicate P-Video generation failed: {e}")

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "pruna",
                "gateway": "replicate",
                "model": MODEL_SLUG,
                "prompt": inputs["prompt"],
                "mode": mode,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "prompt_upsampler": payload_input["prompt_upsampler"],
                "keep_audio": bool(inputs.get("keep_audio")),
                "seed": pred.get("input", {}).get("seed"),
                "reference_image_count": len(reference_images),
                "reference_audio_count": len(reference_audio),
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=MODEL_SLUG,
        )
