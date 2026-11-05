"""Clip join tool: chain image-to-video clips without the start-of-clip flicker.

Image-to-video models (measured on Seedance 2.0) start each clip with about 3
soft frames (sharpness down about 50%) and a brightness or colour slide of 3-7
levels. A hard cut onto those frames reads as a flicker at every join.

Operations:

* ``start_frame`` exports the exact frame the next chained clip starts from,
  optionally ``before_end_seconds`` before the last frame (overlap chaining).
* ``join`` joins the clips and hides the soft start. With ``overlap_frames`` 0
  the next clip was generated from the previous clip's last frame: its frame 0
  (a repeat) is dropped and frames 1..blend-1 blend in from that last frame, so
  every join costs exactly one frame. With ``overlap_frames`` K > 0 the next
  clip was generated from the previous clip's frame (last - K): the previous
  clip plays through the overlap and the next clip fades in after ``settle``
  frames, so every join costs K + 1 frames. Both modes then match the next
  clip's colour to the previous clip at the join, decaying over ``bridge``.
* ``check`` measures flicker: per-pair diagnostics for source clips, or a
  pass/fail verdict per join time for a joined video.

numpy is imported inside the operations, not at module import time.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ResumeSupport,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

# Analysis width for the frame-to-frame change; the max_change_ratio default is
# calibrated at this width. Sharpness is measured at the source resolution,
# because a downscale hides part of the soft-start dip.
_DIFF_WIDTH = 248
# Half width of the window around each join time, in seconds.
_JOIN_WINDOW_SECONDS = 0.2


class ClipJoin(BaseTool):
    """Join, check and chain image-to-video clips without the start-of-clip flicker.

    See the module docstring for the join modes. The ``check`` operation fails a
    join when the largest frame-to-frame change within 0.2 s of the join is more
    than ``max_change_ratio`` times the median change of the video, or when any
    frame in that window has less than ``min_sharpness_step`` of the previous
    frame's sharpness (Laplacian variance at source resolution).
    """

    name = "clip_join"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["cmd:ffmpeg", "cmd:ffprobe", "python:numpy"]
    install_instructions = (
        "Install FFmpeg: https://ffmpeg.org/download.html\n"
        "Windows: winget install FFmpeg\n"
        "macOS: brew install ffmpeg\n"
        "Linux: sudo apt install ffmpeg\n"
        "Install numpy: pip install -r requirements.txt"
    )
    agent_skills = ["ffmpeg", "video-toolkit"]

    capabilities = ["start_frame", "join", "check"]
    supports = {
        "overlap_chaining": True,
        "colour_matching": True,
        "flicker_check": True,
        "audio": False,
    }
    best_for = [
        "joining chained image-to-video clips without the start-of-clip flicker",
        "measuring flicker at clip joins",
        "exporting the exact start frame for the next chained clip",
    ]
    not_good_for = [
        "clips with audio (the joined output has no audio track; mux audio afterwards)",
        "unrelated clips that need cuts or crossfades (use video_stitch)",
        "clips with different frame rates (the first clip's size and frame rate are used)",
    ]
    fallback_tools = ["video_stitch"]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {"type": "string", "enum": ["start_frame", "join", "check"]},
            "clip": {
                "type": "string",
                "description": "start_frame: the clip to take the frame from",
            },
            "clips": {
                "type": "array",
                "items": {"type": "string"},
                "description": "join: clips in order; check: source clips to diagnose in pairs",
            },
            "video": {
                "type": "string",
                "description": "check: a joined video to check at join_times",
            },
            "join_times": {
                "type": "array",
                "items": {"type": "number"},
                "description": "check: join times in seconds in the joined video",
            },
            "output_path": {
                "type": "string",
                "description": "start_frame: .png path; join: video path",
            },
            "before_end_seconds": {
                "type": "number",
                "minimum": 0,
                "default": 0,
                "description": "start_frame: seconds before the last frame (0 = the last frame)",
            },
            "overlap_frames": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "join: K from start_frame; 0 = chained from the last frame",
            },
            "blend_frames": {
                "type": "integer",
                "minimum": 1,
                "default": 8,
                "description": "join: blend length in frames (overlap_frames 0)",
            },
            "settle_frames": {
                "type": "integer",
                "minimum": 0,
                "default": 6,
                "description": "join: next-clip frames to skip before fading it in (overlap_frames > 0)",
            },
            "bridge_seconds": {
                "type": "number",
                "minimum": 0,
                "default": 2.0,
                "description": "join: colour bridge decay in seconds",
            },
            "crf": {"type": "integer", "default": 12},
            "max_change_ratio": {
                "type": "number",
                "default": 2.0,
                "description": "check: max frame change near a join / median frame change",
            },
            "min_sharpness_step": {
                "type": "number",
                "default": 0.7,
                "description": "check: min sharpness of a frame / sharpness of the previous frame near a join",
            },
        },
    }
    output_schema = {
        "type": "object",
        "description": "Fields depend on the operation.",
        "properties": {
            "operation": {"type": "string"},
            "output_path": {"type": "string", "description": "start_frame and join"},
            "frame_index": {"type": "integer", "description": "start_frame"},
            "overlap_frames": {"type": "integer", "description": "start_frame: K for join"},
            "fps": {"type": "number"},
            "frames": {"type": "integer", "description": "join: frames written"},
            "seconds": {"type": "number", "description": "join: output duration"},
            "mode": {"type": "string", "enum": ["pairs", "video"], "description": "check"},
            "pairs": {
                "type": "array",
                "description": "check pairs: per-pair rows and the start-sharpness dip",
                "items": {"type": "object"},
            },
            "joins": {
                "type": "array",
                "description": "check video: one verdict per join time",
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "number"},
                        "peak_change": {"type": "number"},
                        "normal_change": {"type": "number"},
                        "sharpness_dip": {"type": "number"},
                        "ok": {"type": "boolean"},
                    },
                },
            },
            "passed": {"type": "boolean", "description": "check video: every join is ok"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=2048, vram_mb=0, disk_mb=1000, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=0)
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = [
        "operation", "clip", "clips", "before_end_seconds", "overlap_frames",
        "blend_frames", "settle_frames", "bridge_seconds", "crf",
    ]
    side_effects = ["writes a PNG (start_frame) or a video (join) to output_path"]
    user_visible_verification = [
        "Play each join at full speed and look for a soft or bright flash",
        "Run the check operation on the joined video at the join times",
    ]

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Rough runtime in seconds: decoding and x264 encoding dominate."""
        operation = inputs.get("operation")
        if operation == "join":
            return 15.0 * len(inputs.get("clips") or [])
        if operation == "check":
            return 20.0
        return 2.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Run one operation. A failed check is success=True with passed=False."""
        start = time.time()
        operations = {"start_frame": self._start_frame, "join": self._join, "check": self._check}
        operation = inputs.get("operation")
        if operation not in operations:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")
        try:
            result = operations[operation](inputs)
        except Exception as exc:  # the ToolResult is the response boundary
            detail = getattr(exc, "stderr", None)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", "replace")
            return ToolResult(success=False, error=f"{exc}\n{detail.strip()}" if detail else str(exc))
        result.duration_seconds = round(time.time() - start, 2)
        return result

    # ---- operations ----

    def _start_frame(self, inputs: dict[str, Any]) -> ToolResult:
        clip = _existing(inputs.get("clip"), "clip")
        output_path = _required(inputs.get("output_path"), "output_path")
        before_end = float(inputs.get("before_end_seconds") or 0.0)
        count, fps = _frame_count(clip)
        overlap = round(before_end * fps)
        index = count - 1 - overlap
        if index < 0:
            raise ValueError(f"before_end_seconds {before_end}s is longer than the clip ({count} frames)")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vf", f"select='eq(n\\,{index})'",
             "-fps_mode", "passthrough", "-frames:v", "1", output_path],
            capture_output=True, check=True,
        )
        return ToolResult(
            success=True,
            data={"operation": "start_frame", "frame_index": index, "overlap_frames": overlap,
                  "fps": float(fps), "output_path": output_path},
            artifacts=[output_path],
        )

    def _join(self, inputs: dict[str, Any]) -> ToolResult:
        import numpy as np

        clips = [_existing(c, "clips[]") for c in inputs.get("clips") or []]
        if len(clips) < 2:
            raise ValueError("join needs at least 2 clips")
        output_path = _required(inputs.get("output_path"), "output_path")
        k = int(inputs.get("overlap_frames") or 0)
        blend = int(inputs.get("blend_frames") or 8)
        settle = int(inputs.get("settle_frames", 6))
        crf = int(inputs.get("crf", 12))
        if k and k <= settle:
            raise ValueError(f"overlap_frames ({k}) must be larger than settle_frames ({settle})")
        width, height, fps = _probe(clips[0])
        bridge = max(1, round(float(inputs.get("bridge_seconds", 2.0)) * fps))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", output_path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        def write(frames: Any) -> int:
            enc.stdin.write(frames.tobytes())
            return len(frames)

        def to_u8(frames: Any) -> Any:
            return np.clip(frames, 0, 255).astype(np.uint8)

        def mean_rgb(frame: Any) -> Any:
            return frame.astype(np.float32).reshape(-1, 3).mean(0)

        total = 0
        held = None  # previous clip's frames not yet written (the overlap tail)
        try:
            for idx, path in enumerate(clips):
                # Frames stay uint8; only the frames a join changes become float.
                cur = _read_rgb(path, width, height)
                if held is None:
                    body = cur
                elif k == 0:
                    last = held[-1].astype(np.float32)
                    total += write(held)
                    rest = cur[1:]
                    head = rest[:max(bridge, blend - 1)].astype(np.float32)
                    _colour_bridge(head, mean_rgb(last) / mean_rgb(cur[blend]), 0, bridge)
                    for j in range(min(blend - 1, len(head))):
                        a = (j + 1) / blend
                        head[j] = (1 - a) * last + a * head[j]
                    body = np.concatenate([to_u8(head), rest[len(head):]])
                else:
                    # held = previous clip's last K frames; they line up with next-clip frames 1..K.
                    nxt = cur[1:]
                    head = nxt[:k - 1 + bridge].astype(np.float32)
                    _colour_bridge(head, mean_rgb(held[-1]) / mean_rgb(nxt[k - 1]), k - 1, bridge)
                    held_f = held.astype(np.float32)
                    mixed = held_f.copy()
                    for j in range(k):
                        a = 0.0 if j < settle else (j - settle + 1) / (k - settle)
                        a = 0.5 - 0.5 * np.cos(np.pi * min(a, 1.0))  # ease in/out
                        mixed[j] = (1 - a) * held_f[j] + a * head[j]
                    total += write(to_u8(mixed))
                    body = np.concatenate([to_u8(head), nxt[len(head):]])[k:]
                if idx == len(clips) - 1:
                    total += write(body)
                elif k == 0:
                    held = body
                else:
                    total += write(body[:-k])
                    held = body[-k:]
        finally:
            _, err = enc.communicate()
        if enc.returncode != 0:
            raise subprocess.CalledProcessError(enc.returncode, "ffmpeg", stderr=err)
        return ToolResult(
            success=True,
            data={"operation": "join", "frames": total, "seconds": round(total / float(fps), 3),
                  "fps": float(fps), "output_path": output_path},
            artifacts=[output_path],
        )

    def _check(self, inputs: dict[str, Any]) -> ToolResult:
        if inputs.get("video"):
            return self._check_video(inputs)
        clips = [_existing(c, "clips[]") for c in inputs.get("clips") or []]
        if len(clips) < 2:
            raise ValueError("check needs a video with join_times, or at least 2 clips")
        return self._check_pairs(clips)

    def _check_pairs(self, clips: list[str]) -> ToolResult:
        """Last 4 frames of each clip vs the first 6 of the next, at source resolution."""
        pairs = []
        for a, b in zip(clips, clips[1:]):
            tail: deque = deque(maxlen=4)
            count_a = 0
            for frame in _gray_frames(a):
                tail.append(frame)
                count_a += 1
            head = list(_gray_frames(b, max_frames=6))
            labels = [f"A{count_a - len(tail) + i}" for i in range(len(tail))] + [f"B{i}" for i in range(len(head))]
            rows = _rows(list(tail) + head, labels)
            last_a = rows[len(tail) - 1]["sharp"]
            soft_start = rows[len(tail):len(tail) + 3]
            dip = min(r["sharp"] for r in soft_start) / last_a if last_a else 1.0
            pairs.append({"a": a, "b": b, "sharpness_dip": round(dip, 3), "rows": rows})
        return ToolResult(success=True, data={"operation": "check", "mode": "pairs", "pairs": pairs})

    def _check_video(self, inputs: dict[str, Any]) -> ToolResult:
        import numpy as np

        video = _existing(inputs.get("video"), "video")
        join_times = [float(t) for t in inputs.get("join_times") or []]
        if not join_times:
            raise ValueError("check with a video needs join_times")
        max_ratio = float(inputs.get("max_change_ratio", 2.0))
        min_step = float(inputs.get("min_sharpness_step", 0.7))
        _, _, fps_fraction = _probe(video)
        fps = float(fps_fraction)

        diffs = []
        prev = None
        for frame in _gray_frames(video, width=_DIFF_WIDTH):
            if prev is not None:
                diffs.append(float(np.abs(frame - prev).mean()))
            prev = frame
        diffs_arr = np.array(diffs)
        sharp = np.array([_sharpness(f) for f in _gray_frames(video)])
        # Sharpness of each frame relative to the previous frame; frame 0 has no previous.
        steps = np.ones_like(sharp)
        steps[1:] = sharp[1:] / np.maximum(sharp[:-1], 1e-6)
        moving = diffs_arr[diffs_arr > 0.1]  # ignore repeated frames from fps conversion
        normal = float(np.median(moving)) if len(moving) else 0.0

        half = round(_JOIN_WINDOW_SECONDS * fps)
        joins = []
        for t in join_times:
            c = round(t * fps)
            if c <= 0 or c >= len(sharp):
                raise ValueError(f"join time {t}s is outside the video ({len(sharp)} frames at {fps} fps)")
            lo, hi = max(0, c - half), min(len(diffs_arr), c + half)
            peak = float(diffs_arr[lo:hi].max())
            dip = float(steps[max(1, c - half):min(len(steps), c + half + 1)].min())
            ok = peak <= max_ratio * normal and dip >= min_step
            joins.append({"time": t, "peak_change": round(peak, 2), "normal_change": round(normal, 2),
                          "sharpness_dip": round(dip, 3), "ok": bool(ok)})
        return ToolResult(
            success=True,
            data={"operation": "check", "mode": "video", "fps": fps, "joins": joins,
                  "passed": all(j["ok"] for j in joins)},
        )


# ---- helpers ----


def _required(value: Any, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return str(value)


def _existing(value: Any, name: str) -> str:
    path = _required(value, name)
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _probe(path: str) -> tuple[int, int, Fraction]:
    """Return (width, height, fps) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"]), Fraction(stream["r_frame_rate"])


def _frame_count(path: str) -> tuple[int, Fraction]:
    """Return (number of frames, fps) by decoding the stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["nb_read_frames"]), Fraction(stream["r_frame_rate"])


def _read_rgb(path: str, width: int, height: int) -> Any:
    """Decode a clip to uint8 RGB frames of shape (n, h, w, 3)."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def _colour_bridge(frames: Any, ratio: Any, start: int, length: int) -> None:
    """Multiply frames by ratio**w in place, w falling from 1 at `start` to 0 over `length` frames."""
    for j in range(len(frames)):
        w = max(0.0, 1.0 - (j - start) / length) if j >= start else 1.0
        frames[j] *= ratio ** w


def _gray_frames(path: str, width: int | None = None, max_frames: int | None = None) -> Iterator[Any]:
    """Stream float32 gray frames; ``width`` None keeps the source resolution."""
    import numpy as np

    src_w, src_h, _ = _probe(path)
    w = width or src_w
    h = src_h if width is None else round(src_h * width / src_w / 2) * 2
    cmd = ["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=area,format=gray",
           "-fps_mode", "passthrough"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    proc = subprocess.Popen(cmd + ["-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = w * h
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
    finally:
        proc.stdout.close()
        err = proc.stderr.read()
        proc.stderr.close()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, "ffmpeg", stderr=err)


def _sharpness(frame: Any) -> float:
    """Variance of the 4-neighbour Laplacian (same as scipy.ndimage.laplace, mode reflect)."""
    import numpy as np

    p = np.pad(frame, 1, mode="symmetric")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * frame
    return float(lap.var())


def _rows(frames: list[Any], labels: list[str]) -> list[dict[str, Any]]:
    """Per-frame sharpness, mean brightness and change from the previous frame."""
    import numpy as np

    out = []
    for i, f in enumerate(frames):
        d = 0.0 if i == 0 else float(np.abs(f - frames[i - 1]).mean())
        out.append({"frame": labels[i], "sharp": round(_sharpness(f), 1),
                    "y": round(float(f.mean()), 1), "diff": round(d, 2)})
    return out
