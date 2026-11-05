"""ambience_windows: overlapping MMAudio windows shaped into one ambience bed.

Per-clip sound changes character at every clip join. The tool generates over
overlapping windows (each join falls mid-window), then cross-fades and levels
the windows so slow window-to-window level swings go away while short hits
survive. No test here calls the paid API: generate is covered through cost,
status, preflight, input validation and the windows.json record only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.audio.ambience_windows import (
    AmbienceWindows,
    gain_curve_db,
    overlap_add,
    parse_curve,
    per_second_db,
    shape_bus,
    window_entry,
    window_prompt,
    window_starts,
    windows_record,
    write_wav16,
)
from tools.base_tool import ToolStatus
from tools.provider_pricing import PriceQuoteRequired
from tools.tool_registry import registry

registry.discover()

REPO_ROOT = Path(__file__).resolve().parents[2]
SR = 48000
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe required",
)


def _noise(seconds: float, amplitude: float, seed: int) -> np.ndarray:
    """Stereo pink noise (1/f power, shaped in the frequency domain) at RMS ``amplitude``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal((n, 2)), axis=0)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))[:, None]
    x = np.fft.irfft(spectrum, n=n, axis=0)
    return amplitude * x / np.sqrt(np.mean(x**2))


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x**2) + 1e-12))


# --- Window plan and windows.json record -----------------------------------------


@pytest.mark.parametrize(
    "total, window, hop, expected",
    [
        (60, 10, 5, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        (63, 10, 5, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        (12, 10, 5, [0]),
        (10, 10, 5, [0]),
        (9, 10, 5, []),
        (20, 8, 4, [0, 4, 8, 12]),
    ],
    ids=["60s", "63s-tail-dropped", "12s", "exactly-one", "too-short", "8-every-4"],
)
def test_window_starts(total, window, hop, expected):
    assert window_starts(total, window, hop) == expected


EXTRAS = [
    {"start": 35, "end": 45, "text": ", gusty wind"},
    {"start": 40, "end": 50, "text": ", distant thunder"},
]


@pytest.mark.parametrize(
    "start, expected",
    [
        (30, "rain"),
        (35, "rain, gusty wind"),
        (40, "rain, gusty wind, distant thunder"),
        (45, "rain, gusty wind, distant thunder"),
        (50, "rain, distant thunder"),
        (55, "rain"),
    ],
)
def test_window_prompt_appends_extras_whose_span_holds_the_start(start, expected):
    assert window_prompt("rain", EXTRAS, start) == expected


def test_windows_record_written_by_generate():
    record = windows_record(10.0, 5.0, [window_entry(0.0, True, None), window_entry(35.0, False, "timeout")])
    assert record == {
        "window": 10.0,
        "hop": 5.0,
        "windows": [
            {"start": 0.0, "file": "w000_0.wav", "ok": True, "error": None},
            {"start": 35.0, "file": "w035_0.wav", "ok": False, "error": "timeout"},
        ],
    }


# --- Curve ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "curve, message",
    [
        ([], "at least one"),
        ([[0, -40], [0, -30]], "strictly increase"),
        ([[10, -40], [5, -30]], "strictly increase"),
        ([[0, -40, 1]], "pair"),
    ],
)
def test_parse_curve_rejects_bad_curves(curve, message):
    with pytest.raises(ValueError, match=message):
        parse_curve(curve)


@pytest.mark.parametrize(
    "t, target_db",
    [(2, -40), (9, -35), (19, -30), (29, -37.5), (38, -45)],
    ids=["flat-start", "ramp-up", "flat", "ramp-down", "held-end"],
)
def test_gain_moves_a_steady_bed_onto_the_interpolated_curve(t, target_db):
    bus = _noise(40, 0.03, 6)
    bed_db = _db(bus)
    curve = parse_curve([[0, -40], [4, -40], [14, -30], [24, -30], [34, -45]])
    times, gain_db = gain_curve_db(bus, curve, 30, SR)
    assert float(np.interp(t, times, gain_db)) == pytest.approx(target_db - bed_db, abs=0.5)


def test_gain_is_clipped_to_max_boost():
    bus = _noise(20, 0.001, 5)  # about -60 dB, far under a -20 dB curve
    _, gain_db = gain_curve_db(bus, parse_curve([[0, -20]]), max_boost_db=12, sr=SR)
    assert gain_db.max() == pytest.approx(12, abs=0.01)


# --- Numpy shaping ------------------------------------------------------------------


def test_overlap_add_levels_windows_and_cross_fades_without_dips():
    # Three windows 30 dB apart: after the common-level step and equal-power
    # cross-fades the bus is flat at the window RMS (0.03 -> -30.5 dB).
    windows = [(0.0, _noise(10, 0.3, 1)), (5.0, _noise(10, 0.01, 2)), (10.0, _noise(10, 0.1, 3))]
    levels = per_second_db(overlap_add(windows, 10, 5, 20, SR), 20, SR)
    assert len(levels) == 20
    for second, level in enumerate(levels):
        assert level == pytest.approx(-30.5, abs=1.0), (second, levels)


def test_shape_bus_removes_slow_swing_and_keeps_short_hit():
    # A +-6 dB swing over 60 s: slow compared with the ~4 s level smoothing.
    seconds = 60
    t = np.arange(seconds * SR) / SR
    bus = _noise(seconds, 0.03, 4) * (10 ** (6 * np.sin(2 * np.pi * t / 60) / 20))[:, None]
    hit = slice(int(30.0 * SR), int(30.3 * SR))
    bus[hit] *= 10 ** (15 / 20)  # 0.3 s thunder-like hit, +15 dB

    amb = shape_bus(bus, parse_curve([[0, -30]]), 12, fade_end_seconds=99, fade_seconds=3, sr=SR)

    levels = per_second_db(amb, seconds, SR)
    for second in range(2, 58):
        if second != 30:
            assert levels[second] == pytest.approx(-30, abs=1.5), (second, levels)
    around = np.concatenate([amb[int(28 * SR): int(29.5 * SR)], amb[int(31 * SR): int(32.5 * SR)]])
    assert _db(amb[hit]) - _db(around) > 12


# --- Contract, registry, status -------------------------------------------------------


def test_registry_finds_ambience_windows():
    tool = registry.get("ambience_windows")
    assert isinstance(tool, AmbienceWindows)
    assert tool in registry.get_by_capability("audio_processing")


def test_contract_metadata():
    tool = AmbienceWindows()
    info = tool.get_info()
    assert info["input_schema"]["properties"]["operation"]["enum"] == ["generate", "shape"]
    assert info["output_schema"]["properties"]
    assert info["runtime"] == "hybrid"
    assert info["provider"] == "mmaudio"
    assert tool.hosting_provider == "replicate"
    assert info["resource_profile"]["network_required"] is True
    assert tool.dependencies == ["cmd:ffmpeg", "cmd:ffprobe", "python:numpy"]
    assert "REPLICATE_API_TOKEN" in tool.install_instructions
    assert "one continuous ambience bed for a timeline joined from separately generated clips" in tool.best_for
    assert "removing level jumps between generated sound segments" in tool.best_for
    assert tool.not_good_for and tool.fallback_tools and tool.user_visible_verification
    assert set(info["operation_statuses"]) == {"generate", "shape"}
    for skill in tool.agent_skills:
        assert (REPO_ROOT / ".agents" / "skills" / skill / "SKILL.md").exists(), skill


def test_module_import_does_not_load_numpy():
    code = "import sys, tools.audio.ambience_windows; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
                         env={"PYTHONPATH": str(REPO_ROOT)}, check=True)
    assert out.stdout.strip() == "False"


@pytest.fixture
def timeline_file(tmp_path):
    # Validation checks only that the file exists; ffprobe cannot read it, so
    # max_seconds gives the length.
    path = tmp_path / "timeline.mp4"
    path.write_bytes(b"not a real video")
    return path


def _generate_inputs(timeline, tmp_path, **extra):
    return {
        "operation": "generate",
        "timeline": str(timeline),
        "output_dir": str(tmp_path / "windows"),
        "prompt": "light rain on a lake",
        **extra,
    }


@needs_ffmpeg
def test_status_without_token(monkeypatch, timeline_file, tmp_path):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    tool = AmbienceWindows()
    assert tool.get_status() == ToolStatus.AVAILABLE  # shape is usable
    assert tool.operation_statuses() == {"generate": "unavailable", "shape": "available"}
    assert tool.get_info()["operation_statuses"]["generate"] == "unavailable"

    inputs = _generate_inputs(timeline_file, tmp_path, max_seconds=60)
    plan = tool.dry_run(inputs)
    assert plan["status"] == "unavailable"
    assert plan["would_execute"] is False
    assert plan["estimated_cost_usd"] == pytest.approx(0.0825)

    result = tool.execute(inputs)
    assert not result.success
    assert "REPLICATE_API_TOKEN" in result.error
    assert not (tmp_path / "windows").exists()


@needs_ffmpeg
def test_generate_preflight_with_token(monkeypatch, timeline_file, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token-not-used")
    tool = AmbienceWindows()
    assert tool.operation_statuses() == {"generate": "available", "shape": "available"}
    plan = tool.dry_run(_generate_inputs(timeline_file, tmp_path, max_seconds=63))
    assert plan["would_execute"] is True
    assert plan["errors"] == []
    assert plan["window_starts"] == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert plan["estimated_cost_usd"] == pytest.approx(0.0825)


# --- Cost (no API call) -----------------------------------------------------------------


def test_estimate_cost_is_total_of_all_windows(timeline_file, tmp_path):
    mmaudio_10s = registry.get("mmaudio_replicate").estimate_cost({"duration": 10})
    assert mmaudio_10s == pytest.approx(0.0075)
    cost = AmbienceWindows().estimate_cost(_generate_inputs(timeline_file, tmp_path, max_seconds=60))
    assert cost == pytest.approx(0.0825)
    assert cost == pytest.approx(11 * mmaudio_10s)


def test_estimate_cost_needs_a_length(tmp_path):
    tool = AmbienceWindows()
    inputs = {"operation": "generate", "timeline": str(tmp_path / "missing.mp4")}
    with pytest.raises(PriceQuoteRequired):
        tool.estimate_cost(inputs)
    assert tool.dry_run(inputs)["cost_status"] == "quote_required"


def test_shape_is_free():
    assert AmbienceWindows().estimate_cost({"operation": "shape"}) == 0.0


@pytest.fixture
def video_63s(tmp_path):
    video = tmp_path / "timeline.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=63",
         "-c:v", "libx264", "-preset", "ultrafast", str(video)],
        check=True, timeout=60,
    )
    return video


@needs_ffmpeg
@pytest.mark.parametrize(
    "max_seconds, expected_windows",
    [(None, 11), (30, 5), (200, 11)],
    ids=["probed", "limited", "limit-past-timeline-end"],
)
def test_estimate_cost_uses_shorter_of_timeline_and_max_seconds(video_63s, tmp_path, max_seconds, expected_windows):
    extra = {} if max_seconds is None else {"max_seconds": max_seconds}
    cost = AmbienceWindows().estimate_cost(_generate_inputs(video_63s, tmp_path, **extra))
    assert cost == pytest.approx(expected_windows * 0.0075)


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"operation": "mix"}, "operation must be"),
        ({"operation": "generate"}, "timeline is required"),
        ({"operation": "generate", "timeline": "/no/such.mp4", "output_dir": "x", "prompt": "p"},
         "timeline not found"),
        ({"operation": "generate", "hop_seconds": 10}, "hop_seconds < window_seconds"),
        ({"operation": "generate", "extra_prompts": [{"start": 1}]}, "needs start, end and text"),
        ({"operation": "shape"}, "windows_dir is required"),
        ({"operation": "shape", "music_path": "/no/such.mp3"}, "music_path not found"),
        ({"operation": "shape", "curve": [[5, -30], [1, -20]]}, "strictly increase"),
    ],
)
def test_validation_errors(inputs, message):
    tool = AmbienceWindows()
    assert any(message in e for e in tool.validate_inputs(inputs))
    assert tool.dry_run(inputs)["would_execute"] is False
    result = tool.execute(inputs)
    assert not result.success and message in result.error


# --- shape end to end (ffmpeg) ------------------------------------------------------------


def _write_windows(directory: Path, amplitudes: list[float], window=10.0, hop=5.0) -> None:
    """Write windows like generate does: relative file names in windows.json."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, amplitude in enumerate(amplitudes):
        entry = window_entry(i * hop, True, None)
        write_wav16(directory / entry["file"], _noise(window, amplitude, 10 + i), SR)
        entries.append(entry)
    (directory / "windows.json").write_text(json.dumps(windows_record(window, hop, entries)))


def _write_music(path: Path, seconds: float) -> None:
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * SR)) / SR)
    write_wav16(path, np.stack([tone, tone], axis=1), SR)


def _shape(windows_dir: Path, out: Path, **extra):
    return AmbienceWindows().execute({
        "operation": "shape",
        "windows_dir": str(windows_dir),
        "total_seconds": 20,
        "curve": [[0, -36], [20, -28]],
        "output_path": str(out),
        **extra,
    })


@needs_ffmpeg
def test_shape_with_music_mixes_and_normalises(tmp_path):
    _write_windows(tmp_path / "windows", [0.3, 0.01, 0.1])
    music = tmp_path / "music.wav"
    _write_music(music, 20)
    out = tmp_path / "soundtrack.wav"

    result = _shape(tmp_path / "windows", out, music_path=str(music))

    assert result.success, result.error
    assert result.cost_usd == 0.0
    assert out.exists()
    assert Path(result.data["ambience_path"]) == tmp_path / "soundtrack-ambience.wav"
    assert Path(result.data["premix_path"]).exists()
    assert result.data["integrated_lufs"] == pytest.approx(-14, abs=1.0)
    levels = result.data["ambience_db_per_second"]
    assert len(levels) == 20
    for second in range(2, 16):
        expected = -36 + 8 * (second + 0.5) / 20
        assert levels[second] == pytest.approx(expected, abs=3.0), levels
    # No step at the window joins (5 s, 10 s, 15 s): consecutive seconds stay close.
    assert max(abs(b - a) for a, b in zip(levels[:16], levels[1:17])) < 2.0, levels


@needs_ffmpeg
def test_shape_without_music_writes_the_bed_only_from_a_moved_folder(tmp_path):
    _write_windows(tmp_path / "first", [0.1, 0.02, 0.05])
    moved = tmp_path / "elsewhere" / "windows"
    shutil.copytree(tmp_path / "first", moved)
    shutil.rmtree(tmp_path / "first")
    out = tmp_path / "bed.wav"

    result = _shape(moved, out)

    assert result.success, result.error
    assert result.artifacts == [str(out)]
    assert result.data["premix_path"] is None and result.data["integrated_lufs"] is None
    assert not (tmp_path / "bed-premix.wav").exists()
    levels = result.data["ambience_db_per_second"]
    assert levels[8] == pytest.approx(-36 + 8 * 8.5 / 20, abs=3.0), levels


@needs_ffmpeg
def test_shape_fades_out_by_the_window_coverage_end(tmp_path):
    # Two windows cover 0-15 s of a 20 s output: the bed must be silent from 15 s.
    _write_windows(tmp_path / "windows", [0.1, 0.1])
    result = _shape(tmp_path / "windows", tmp_path / "bed.wav")
    assert result.success, result.error
    levels = result.data["ambience_db_per_second"]
    # (coverage_end - 9) .. (coverage_end - 4): on the curve, no swell before the fade.
    for second, expected in zip(range(6, 12), [-33.4, -33.0, -32.6, -32.2, -31.8, -31.4]):
        assert levels[second] == pytest.approx(expected, abs=1.5), (second, levels)
    assert levels[13] > levels[14] > -80
    assert all(level < -100 for level in levels[15:]), levels


def test_shape_bus_ignores_silent_tail_after_coverage_end():
    # Windows cover 0-20 s of a 30 s bus; the tail is silent. Measured over the
    # tail, the smoothed level would sag near 20 s and swell the gain.
    bus = np.zeros((30 * SR, 2))
    bus[: 20 * SR] = _noise(20, 0.03, 7)
    amb = shape_bus(bus, parse_curve([[0, -30]]), 12, fade_end_seconds=20, fade_seconds=3,
                    coverage_end_seconds=20, sr=SR)
    levels = per_second_db(amb, 30, SR)
    for second in range(11, 16):  # (coverage_end - 9) .. (coverage_end - 4)
        assert levels[second] == pytest.approx(-30, abs=1.5), (second, levels)
    assert levels[16] < -29  # the fade starts at 17 s; no swell before it


@needs_ffmpeg
def test_shape_rejects_fade_end_past_coverage(tmp_path):
    _write_windows(tmp_path / "windows", [0.1, 0.1])
    result = _shape(tmp_path / "windows", tmp_path / "bed.wav", fade_end_seconds=18)
    assert not result.success
    assert "coverage end 15.0" in result.error


@needs_ffmpeg
def test_shape_refuses_failed_windows(tmp_path):
    windows_dir = tmp_path / "windows"
    _write_windows(windows_dir, [0.1, 0.1])
    meta = json.loads((windows_dir / "windows.json").read_text())
    meta["windows"][1]["ok"] = False
    (windows_dir / "windows.json").write_text(json.dumps(meta))

    result = _shape(windows_dir, tmp_path / "out.wav")
    assert not result.success
    assert "5.0s" in result.error
