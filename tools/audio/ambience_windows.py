"""One continuous ambience bed for a timeline joined from separately generated clips.

Sound generated per clip changes character at every clip join. ``generate``
(billed through ``mmaudio_replicate``) runs MMAudio over overlapping windows of
the joined timeline (default: 10 s windows every 5 s), so each join falls in the
middle of a window, where the model hears one continuous shot.

``shape`` (local, free) turns the windows into one bed:

1. high-pass each window (cuts wind-on-microphone rumble; a lower 50 Hz cut
   inside ``low_keep_spans``, e.g. a storm with thunder), bring each window to
   a common level, and overlap-add with equal-power cross-fades;
2. measure the bed every 100 ms, smooth over about 4 s, and apply the gain that
   moves it onto ``curve`` (short hits such as thunder survive; slow
   window-to-window level swings do not);
3. fade the bed out by ``fade_end_seconds``. With ``music_path``, mix the bed
   under the music and normalise the mix with two-pass loudnorm; without it,
   write the shaped bed only (for a later audio_mixer pass).

numpy is imported inside the functions that need it, not at module import time.
"""

from __future__ import annotations

import json
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from tools.analysis.audio_probe import probe_duration
from tools.base_tool import (
    _EXECUTE_DEPTH,
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)
from tools.provider_pricing import PriceQuoteRequired

SAMPLE_RATE = 48000
FRAME_SECONDS = 0.1
WINDOW_RMS = 0.03
LOW_KEEP_HIGHPASS_HZ = 50
DEFAULT_NEGATIVE_PROMPT = (
    "music, speech, voices, microphone wind noise, low hum, drone, birds, insects, engine, cars"
)
MMAUDIO_TOOL = "mmaudio_replicate"


# --- Pure logic (no ffmpeg, no API) -------------------------------------------


def window_starts(total_seconds: float, window_seconds: float, hop_seconds: float) -> list[float]:
    """Return the start time of every full window that fits in ``total_seconds``.

    A window starts every ``hop_seconds``; a window that would run past the end
    is dropped, so a timeline shorter than one window gives no windows.
    """
    starts: list[float] = []
    s = 0.0
    while s + window_seconds <= total_seconds + 1e-6:
        starts.append(round(s, 3))
        s += hop_seconds
    return starts


def window_prompt(base_prompt: str, extra_prompts: list[dict[str, Any]], start: float) -> str:
    """Return the prompt for the window that starts at ``start``.

    Each ``extra_prompts`` item ``{start, end, text}`` appends ``text`` (as
    given, so include any leading ", ") when ``start`` is in ``[start, end]``.
    """
    return base_prompt + "".join(
        str(e["text"]) for e in extra_prompts if float(e["start"]) <= start <= float(e["end"])
    )


def window_entry(start: float, ok: bool, error: Optional[str]) -> dict[str, Any]:
    """Return the windows.json entry for one window; ``file`` is relative to the folder."""
    return {"start": start, "file": f"w{start:05.1f}".replace(".", "_") + ".wav", "ok": ok, "error": error}


def windows_record(window_seconds: float, hop_seconds: float, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the windows.json content that ``generate`` writes and ``shape`` reads."""
    return {"window": window_seconds, "hop": hop_seconds, "windows": entries}


def parse_curve(curve: list[Any]) -> list[tuple[float, float]]:
    """Validate a level curve given as ``[[t, dB], ...]`` and return float pairs.

    Raises ValueError when the curve is empty, a point is not a pair, or the
    times do not strictly increase.
    """
    if not curve:
        raise ValueError("curve needs at least one [t, dB] point")
    points: list[tuple[float, float]] = []
    for point in curve:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"curve point {point!r} is not a [t, dB] pair")
        points.append((float(point[0]), float(point[1])))
    times = [p[0] for p in points]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("curve times must strictly increase")
    return points


def _gaussian(x: Any, sigma: float) -> Any:
    """1-D Gaussian smoothing; same result as scipy.ndimage.gaussian_filter1d defaults."""
    import numpy as np

    radius = int(4 * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(x, radius, mode="symmetric"), kernel, mode="valid")


def overlap_add(
    windows: list[tuple[float, Any]],
    window_seconds: float,
    hop_seconds: float,
    total_seconds: float,
    sr: int = SAMPLE_RATE,
) -> Any:
    """Overlap-add stereo windows into one bus with equal-power cross-fades.

    ``windows`` is a start-ordered list of ``(start_seconds, samples)`` where
    ``samples`` has shape ``(n, 2)``. Each window is first scaled to a common
    RMS so the cross-fades do not swing; a near-silent window is left as is.
    The first window has no fade-in and the last has no fade-out.
    """
    import numpy as np

    xfade = window_seconds - hop_seconds
    n = int(total_seconds * sr)
    bus = np.zeros((n, 2))
    for i, (start, samples) in enumerate(windows):
        x = np.asarray(samples, dtype=np.float64)[: int(window_seconds * sr)]
        t = np.arange(len(x)) / sr
        g = np.ones(len(x))
        if i > 0:
            g *= np.sin(np.clip(t / xfade, 0, 1) * np.pi / 2)
        if i < len(windows) - 1:
            g *= np.cos(np.clip((t - (window_seconds - xfade)) / xfade, 0, 1) * np.pi / 2)
        rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
        if rms > 1e-6:
            x = x / rms * WINDOW_RMS
        a = int(start * sr)
        b = min(n, a + len(x))
        if b > a:
            bus[a:b] += (x * g[:, None])[: b - a]
    return bus


def gain_curve_db(
    bus: Any, curve: list[tuple[float, float]], max_boost_db: float, sr: int = SAMPLE_RATE
) -> tuple[Any, Any]:
    """Return ``(frame_times, gain_db)`` that move the bus level onto ``curve``.

    The level is measured every 100 ms and smoothed over about 4 s, so the gain
    follows slow swings only; a short hit keeps its level above the bed. The
    gain is clipped to ``[-24, max_boost_db]`` and smoothed over about 1 s.
    Curve levels are linearly interpolated; the ends are held flat.
    """
    import numpy as np

    hop_n = int(sr * FRAME_SECONDS)
    frames = len(bus) // hop_n
    level = 10 * np.log10(np.mean(bus[: frames * hop_n].reshape(frames, hop_n, -1) ** 2, axis=(1, 2)) + 1e-12)
    smooth = _gaussian(np.maximum(level, -70), 4.0 / FRAME_SECONDS)
    t = (np.arange(frames) + 0.5) * FRAME_SECONDS
    target = np.interp(t, [c[0] for c in curve], [c[1] for c in curve])
    return t, _gaussian(np.clip(target - smooth, -24, max_boost_db), 1.0 / FRAME_SECONDS)


def shape_bus(
    bus: Any,
    curve: list[tuple[float, float]],
    max_boost_db: float,
    fade_end_seconds: float,
    fade_seconds: float,
    coverage_end_seconds: Optional[float] = None,
    sr: int = SAMPLE_RATE,
) -> Any:
    """Apply the curve gain and the end fade to the bus; keep peaks under 0.98.

    The level is measured only up to ``coverage_end_seconds`` (the end of the
    last window; default: the whole bus), so the silent tail after it does not
    pull the smoothed level down and swell the gain; the last gain is held.
    """
    import numpy as np

    covered = bus if coverage_end_seconds is None else bus[: int(coverage_end_seconds * sr)]
    t, gain_db = gain_curve_db(covered, curve, max_boost_db, sr)
    st = np.arange(len(bus)) / sr
    g = 10 ** (np.interp(st, t, gain_db) / 20)
    g *= np.sin(np.clip((fade_end_seconds - st) / fade_seconds, 0, 1) * np.pi / 2)
    amb = bus * g[:, None]
    peak = float(np.max(np.abs(amb))) if amb.size else 0.0
    if peak > 0.98:
        amb *= 0.98 / peak
    return amb


def per_second_db(samples: Any, total_seconds: float, sr: int = SAMPLE_RATE) -> list[float]:
    """Return the RMS level (dB) of each whole second of ``samples``."""
    import numpy as np

    secs = min(int(total_seconds), len(samples) // sr)
    block = samples[: secs * sr].reshape(secs, sr, -1)
    return [round(float(v), 2) for v in 10 * np.log10(np.mean(block**2, axis=(1, 2)) + 1e-12)]


def write_wav16(path: str | Path, samples: Any, sr: int = SAMPLE_RATE) -> None:
    """Write float samples (``(n,)`` or ``(n, channels)``, clipped to [-1, 1]) as 16-bit PCM WAV."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1.0, 1.0) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# --- Tool -----------------------------------------------------------------------


class AmbienceWindows(BaseTool):
    """Generate and shape one continuous ambience bed over a joined timeline.

    ``generate`` is billed: it calls ``mmaudio_replicate`` once per overlapping
    window. ``shape`` is local and free: it cross-fades the windows, levels them
    onto a curve and, with ``music_path``, mixes and normalises the soundtrack.
    """

    name = "ambience_windows"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "audio_processing"
    provider = "mmaudio"
    hosting_provider = "replicate"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.HYBRID  # shape is local; generate calls Replicate

    dependencies = ["cmd:ffmpeg", "cmd:ffprobe", "python:numpy"]
    install_instructions = (
        "shape (local, free) needs FFmpeg (ffmpeg and ffprobe) and numpy.\n"
        "generate is billed per window through mmaudio_replicate and also needs "
        "REPLICATE_API_TOKEN.\n  Get one at https://replicate.com/account/api-tokens"
    )
    agent_skills = ["sound-effects", "ffmpeg"]

    capabilities = ["windowed_video_to_audio", "ambience_crossfade", "level_curve_shaping", "music_mix"]
    supports = {
        "video_input": True,
        "per_span_prompts": True,
        "level_curve": True,
        "music_mix": True,
        "two_pass_loudnorm": True,
        "music": False,
    }
    best_for = [
        "one continuous ambience bed for a timeline joined from separately generated clips",
        "removing level jumps between generated sound segments",
        "generate: billed per window through mmaudio_replicate; shape: local and free",
    ]
    not_good_for = [
        "speech, dialogue or music generation",
        "one short clip (call mmaudio_replicate directly)",
        "precisely timed hard sound effects (window timing is approximate)",
        "offline use of operation=generate",
    ]
    fallback_tools = ["mmaudio_replicate", "audio_mixer"]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["generate", "shape"],
                "description": (
                    "generate (billed via mmaudio_replicate): MMAudio over overlapping windows; "
                    "writes w*.mp4/w*.wav and windows.json to output_dir. shape (local, free): "
                    "cross-fade and level the windows; with music_path also mix and normalise."
                ),
            },
            "timeline": {"type": "string", "description": "generate: joined timeline video."},
            "output_dir": {"type": "string", "description": "generate: folder for windows and windows.json."},
            "prompt": {"type": "string", "description": "generate: base prompt for every window."},
            "extra_prompts": {
                "type": "array",
                "description": "generate: {start, end, text} appended to windows whose start is in [start, end].",
                "items": {
                    "type": "object",
                    "required": ["start", "end", "text"],
                    "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}},
                },
                "default": [],
            },
            "negative_prompt": {"type": "string", "default": DEFAULT_NEGATIVE_PROMPT},
            "max_seconds": {"type": "number", "description": "generate: cover at most this much of the timeline."},
            "window_seconds": {"type": "number", "default": 10, "exclusiveMinimum": 0},
            "hop_seconds": {"type": "number", "default": 5, "description": "0 < hop_seconds < window_seconds."},
            "workers": {"type": "integer", "default": 6, "minimum": 1},
            "windows_dir": {"type": "string", "description": "shape: folder that holds windows.json."},
            "music_path": {"type": "string", "description": "shape: optional music bed to mix under."},
            "music_gain_db": {"type": "number", "default": -7.6},
            "total_seconds": {"type": "number", "description": "shape: output length."},
            "curve": {
                "type": "array",
                "description": "shape: ambience level keypoints [[t_seconds, dB_RMS], ...].",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "fade_end_seconds": {
                "type": "number",
                "description": "shape: ambience is silent from here; default and maximum is the window coverage end.",
            },
            "fade_seconds": {"type": "number", "default": 3, "exclusiveMinimum": 0},
            "highpass_hz": {"type": "number", "default": 120},
            "low_keep_spans": {
                "type": "array",
                "description": "shape: [[start, end], ...] window starts that use a 50 Hz cut.",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "default": [],
            },
            "max_boost_db": {"type": "number", "default": 12},
            "target_lufs": {"type": "number", "default": -14, "minimum": -40, "maximum": 0},
            "output_path": {"type": "string", "description": "shape: output .wav."},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "windows_json": {"type": "string", "description": "generate"},
            "windows": {"type": "array", "items": {"type": "object"}, "description": "generate"},
            "failed_windows": {"type": "integer", "description": "generate"},
            "output_path": {"type": "string", "description": "shape"},
            "ambience_path": {"type": "string", "description": "shape"},
            "premix_path": {"type": ["string", "null"], "description": "shape, with music_path"},
            "integrated_lufs": {"type": ["number", "null"], "description": "shape, with music_path"},
            "ambience_db_per_second": {"type": "array", "items": {"type": "number"}, "description": "shape"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=1024, vram_mb=0, disk_mb=500, network_required=True)
    retry_policy = RetryPolicy(max_retries=0)
    side_effects = [
        "generate: billed Replicate call per window through mmaudio_replicate; "
        "writes w*.mp4, w*.wav and windows.json to output_dir",
        "shape (local, free): writes output_path; with music_path also <stem>-ambience.wav and <stem>-premix.wav",
    ]
    user_visible_verification = [
        "Listen across every clip join: no change of sound character or level step",
        "Check that short hits (thunder) survive and the bed follows the planned curve",
        "With music_path: check integrated loudness of output_path against target_lufs",
    ]

    # ---- Status ----

    @staticmethod
    def _mmaudio() -> Optional[BaseTool]:
        from tools.tool_registry import registry

        registry.ensure_discovered()
        return registry.get(MMAUDIO_TOOL)

    def operation_statuses(self) -> dict[str, str]:
        """Return per-operation readiness: shape needs local deps; generate also mmaudio_replicate."""
        local = self.get_status().value
        generate = local
        if local == "available":
            mm = self._mmaudio()
            generate = mm.get_status().value if mm is not None else "unavailable"
        return {"generate": generate, "shape": local}

    def get_info(self) -> dict[str, Any]:
        """Base contract info plus ``operation_statuses``."""
        info = super().get_info()
        info["operation_statuses"] = self.operation_statuses()
        return info

    # ---- Inputs ----

    @staticmethod
    def _window_params(inputs: dict[str, Any]) -> tuple[float, float]:
        return float(inputs.get("window_seconds", 10)), float(inputs.get("hop_seconds", 5))

    @staticmethod
    def _generate_length(inputs: dict[str, Any]) -> Optional[float]:
        """The shorter of the probed timeline length and ``max_seconds``; None if neither is known."""
        timeline = inputs.get("timeline")
        probed = probe_duration(timeline) if timeline and Path(timeline).is_file() else None
        limit = inputs.get("max_seconds")
        known = [v for v in (probed, None if limit is None else float(limit)) if v is not None]
        return min(known) if known else None

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """Return a list of input problems (empty when the inputs are valid)."""
        operation = inputs.get("operation")
        if operation not in ("generate", "shape"):
            return [f"operation must be 'generate' or 'shape', got {operation!r}"]
        errors: list[str] = []
        window, hop = self._window_params(inputs)
        if not 0 < hop < window:
            errors.append("need 0 < hop_seconds < window_seconds")
        if operation == "generate":
            errors += [f"{k} is required for generate" for k in ("timeline", "output_dir", "prompt") if not inputs.get(k)]
            if inputs.get("timeline") and not Path(inputs["timeline"]).is_file():
                errors.append(f"timeline not found: {inputs['timeline']}")
            if int(inputs.get("workers", 6)) < 1:
                errors.append("workers must be at least 1")
            for item in inputs.get("extra_prompts") or []:
                if not isinstance(item, dict) or not {"start", "end", "text"} <= item.keys():
                    errors.append(f"extra_prompts item {item!r} needs start, end and text")
            return errors

        errors += [
            f"{k} is required for shape"
            for k in ("windows_dir", "total_seconds", "curve", "output_path")
            if inputs.get(k) in (None, "", [])
        ]
        if inputs.get("windows_dir") and not (Path(inputs["windows_dir"]) / "windows.json").is_file():
            errors.append(f"windows.json not found in {inputs['windows_dir']}")
        if inputs.get("music_path") and not Path(inputs["music_path"]).is_file():
            errors.append(f"music_path not found: {inputs['music_path']}")
        if inputs.get("curve"):
            try:
                parse_curve(inputs["curve"])
            except (TypeError, ValueError) as e:
                errors.append(str(e))
        if float(inputs.get("fade_seconds", 3)) <= 0:
            errors.append("fade_seconds must be positive")
        for span in inputs.get("low_keep_spans") or []:
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                errors.append(f"low_keep_spans item {span!r} is not a [start, end] pair")
        return errors

    # ---- Cost / runtime / preflight ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Total MMAudio cost for generate (windows x per-window price); 0 for shape.

        Raises PriceQuoteRequired for generate when the timeline length is unknown.
        """
        if inputs.get("operation") != "generate":
            return 0.0
        length = self._generate_length(inputs)
        mm = self._mmaudio()
        if length is None or mm is None:
            raise PriceQuoteRequired("generate needs a readable timeline or max_seconds, and mmaudio_replicate")
        window, hop = self._window_params(inputs)
        return round(len(window_starts(length, window, hop)) * float(mm.estimate_cost({"duration": window})), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Rough wall time: about 30 s per batch of ``workers`` windows for generate; 15 s for shape."""
        length = self._generate_length(inputs) if inputs.get("operation") == "generate" else None
        if length is None:
            return 15.0
        window, hop = self._window_params(inputs)
        batches = -(-len(window_starts(length, window, hop)) // max(1, int(inputs.get("workers", 6))))
        return 30.0 * batches

    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Preflight: cost, status of the requested operation, input problems and window plan."""
        out = super().dry_run(inputs)
        operation = inputs.get("operation")
        errors = self.validate_inputs(inputs)
        status = self.operation_statuses().get(operation, "unavailable")
        out.update(operation=operation, status=status, errors=errors,
                   would_execute=not errors and status == "available")
        length = self._generate_length(inputs) if operation == "generate" and not errors else None
        if length is not None:
            out["window_starts"] = window_starts(length, *self._window_params(inputs))
        return out

    # ---- Execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Run ``generate`` or ``shape``; see the module docstring."""
        start = time.time()
        errors = self.validate_inputs(inputs)
        if errors:
            return ToolResult(success=False, error="; ".join(errors))
        try:
            self.check_dependencies()
            if inputs["operation"] == "generate":
                result = self._generate(inputs)
            else:
                result = self._shape(inputs)
        except Exception as e:  # DependencyError text names the missing dependency
            return ToolResult(success=False, error=str(e))
        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        mm = self._mmaudio()
        if mm is None or mm.get_status().value != "available":
            return ToolResult(success=False, error="generate needs mmaudio_replicate with REPLICATE_API_TOKEN set. "
                              + self.install_instructions)
        window, hop = self._window_params(inputs)
        length = self._generate_length(inputs)
        if length is None:
            return ToolResult(success=False, error=f"could not read timeline length: {inputs['timeline']}")
        starts = window_starts(length, window, hop)
        if not starts:
            return ToolResult(success=False, error=f"timeline ({length:.1f} s) is shorter than one window")

        out_dir = Path(inputs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        extras = list(inputs.get("extra_prompts") or [])
        # Worker threads start at depth 0; carry this call's depth so the
        # nested MMAudio costs are not counted again next to our total.
        parent_depth = getattr(_EXECUTE_DEPTH, "value", 0)

        def run(win_start: float) -> tuple[dict[str, Any], float]:
            _EXECUTE_DEPTH.value = parent_depth
            wav = out_dir / window_entry(win_start, True, None)["file"]
            seg = wav.with_suffix(".mp4")
            try:
                self.run_command(
                    ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(win_start), "-t", str(window),
                     "-i", str(inputs["timeline"]), "-an", "-c:v", "libx264", "-crf", "20",
                     "-preset", "veryfast", str(seg)]
                )
                res = mm.execute({
                    "video": str(seg),
                    "prompt": window_prompt(inputs["prompt"], extras, win_start),
                    "negative_prompt": inputs.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                    "duration": window,
                    "output_path": str(wav),
                })
            except Exception as e:  # one failed window must not lose the others
                return window_entry(win_start, False, str(e)), 0.0
            return window_entry(win_start, res.success, res.error), float(res.cost_usd or 0.0)

        with ThreadPoolExecutor(max_workers=int(inputs.get("workers", 6))) as pool:
            results = list(pool.map(run, starts))

        entries = [entry for entry, _ in results]
        windows_json = out_dir / "windows.json"
        windows_json.write_text(json.dumps(windows_record(window, hop, entries), indent=2))
        failed = [e for e in entries if not e["ok"]]
        return ToolResult(
            success=not failed,
            error="; ".join(f"window {e['start']}s failed: {e['error']}" for e in failed) or None,
            data={"operation": "generate", "windows_json": str(windows_json), "windows": entries,
                  "failed_windows": len(failed)},
            artifacts=[str(out_dir / e["file"]) for e in entries if e["ok"]] + [str(windows_json)],
            cost_usd=round(sum(cost for _, cost in results), 4),
            model=MMAUDIO_TOOL,
        )

    def _shape(self, inputs: dict[str, Any]) -> ToolResult:
        windows_dir = Path(inputs["windows_dir"])
        meta = json.loads((windows_dir / "windows.json").read_text())
        window, hop = float(meta["window"]), float(meta["hop"])
        if not 0 < hop < window:
            return ToolResult(success=False, error="windows.json needs 0 < hop < window")
        windows = sorted(meta["windows"], key=lambda w: float(w["start"]))
        if not windows:
            return ToolResult(success=False, error="windows.json lists no windows")
        bad = [w for w in windows if not w.get("ok", True) or not (windows_dir / w["file"]).is_file()]
        if bad:
            return ToolResult(success=False, error="windows missing or failed: "
                              + ", ".join(f"{w['start']}s" for w in bad))

        total = float(inputs["total_seconds"])
        coverage_end = min(total, float(windows[-1]["start"]) + window)
        fade_end = inputs.get("fade_end_seconds")
        fade_end = coverage_end if fade_end is None else float(fade_end)
        if fade_end > coverage_end + 1e-6:
            return ToolResult(success=False, error=(
                f"fade_end_seconds {fade_end} is past the window coverage end {coverage_end}; "
                "the ambience would cut hard there"))

        highpass = float(inputs.get("highpass_hz", 120))
        keep = [(float(a), float(b)) for a, b in inputs.get("low_keep_spans") or []]
        decoded = []
        for w in windows:
            s = float(w["start"])
            hp = LOW_KEEP_HIGHPASS_HZ if any(a <= s <= b for a, b in keep) else highpass
            decoded.append((s, self._decode_highpassed(windows_dir / w["file"], hp)))
        amb = shape_bus(overlap_add(decoded, window, hop, total), parse_curve(inputs["curve"]),
                        float(inputs.get("max_boost_db", 12)), fade_end, float(inputs.get("fade_seconds", 3)),
                        coverage_end)

        out = Path(inputs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"operation": "shape", "output_path": str(out),
                                "ambience_db_per_second": per_second_db(amb, total)}
        if not inputs.get("music_path"):
            write_wav16(out, amb)
            data.update(ambience_path=str(out), premix_path=None, integrated_lufs=None)
            return ToolResult(success=True, data=data, artifacts=[str(out)], cost_usd=0.0)

        amb_path = out.with_name(out.stem + "-ambience.wav")
        premix = out.with_name(out.stem + "-premix.wav")
        write_wav16(amb_path, amb)
        fc = (
            f"[0:a]atrim=0:{total},asetpts=PTS-STARTPTS[a];"
            f"[1:a]atrim=0:{total},asetpts=PTS-STARTPTS,aresample={SAMPLE_RATE},"
            f"volume={float(inputs.get('music_gain_db', -7.6))}dB,"
            f"afade=t=in:d=0.3,afade=t=out:st={max(0.0, total - 3)}:d=3[m];"
            f"[a][m]amix=inputs=2:normalize=0,atrim=0:{total}[x]"
        )
        self.run_command(["ffmpeg", "-y", "-loglevel", "error", "-i", str(amb_path), "-i", str(inputs["music_path"]),
                          "-filter_complex", fc, "-map", "[x]", "-c:a", "pcm_s24le", str(premix)])
        target = max(-40.0, min(0.0, float(inputs.get("target_lufs", -14))))
        data.update(ambience_path=str(amb_path), premix_path=str(premix),
                    integrated_lufs=self._two_pass_loudnorm(premix, out, target))
        return ToolResult(success=True, data=data, artifacts=[str(out), str(amb_path), str(premix)], cost_usd=0.0)

    @staticmethod
    def _decode_highpassed(path: Path, highpass_hz: float) -> Any:
        """Decode ``path`` to float64 stereo at 48 kHz after a high-pass filter."""
        import numpy as np

        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path), "-af", f"highpass=f={highpass_hz}",
             "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)

    def _two_pass_loudnorm(self, src: Path, dst: Path, target_lufs: float) -> float:
        """Measure ``src`` with loudnorm, apply a linear loudnorm to ``dst``; return its integrated LUFS.

        audio_mixer's loudnorm helper is a single-pass filter edge; a linear
        second pass needs the first pass's measured values, so it is built here.
        """
        base = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

        def stats(stderr: str) -> dict[str, Any]:
            return json.loads(stderr[stderr.rindex("{"): stderr.rindex("}") + 1])

        j = stats(self.run_command(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-af",
                                    f"{base}:print_format=json", "-f", "null", "-"]).stderr)
        ln = (f"{base}:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
              f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true:print_format=json")
        second = self.run_command(["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(src), "-af",
                                   f"{ln},aresample={SAMPLE_RATE}", "-c:a", "pcm_s16le", str(dst)])
        return float(stats(second.stderr)["output_i"])
