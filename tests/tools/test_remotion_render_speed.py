"""Tests for the Remotion render-speed flags in video_compose.

Checks render only the frames they inspect, review passes use a draft
(half scale, fast x264 preset), and the final render passes no speed flag.
A final render adds the hardware encode flags only when the host has a
hardware video encoder; every test pins that choice, so results do not
depend on the machine.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.video.remotion_caption_burn import RemotionCaptionBurn  # noqa: E402
from tools.video.video_compose import VideoCompose  # noqa: E402

HW_FLAGS = [
    "--hardware-acceleration=if-possible",
    "--video-bitrate=16M",
    "--offthreadvideo-threads=8",
]


def _set_hardware(monkeypatch, available: bool) -> None:
    monkeypatch.setattr("lib.render_profile.hardware_video_available", lambda: available)
    monkeypatch.setattr("os.cpu_count", lambda: 16)


@pytest.fixture(autouse=True)
def _no_hardware_by_default(monkeypatch):
    _set_hardware(monkeypatch, False)


@pytest.mark.parametrize(
    ("opts", "expected"),
    [
        ({}, []),
        ({"frames": 120}, ["--frames=120"]),
        ({"frames": "100-220"}, ["--frames=100-220"]),
        ({"frames": [100, 220]}, ["--frames=100-220"]),
        ({"frames": 0}, ["--frames=0"]),
        ({"draft": True}, ["--scale=0.5", "--x264-preset=veryfast"]),
        (
            {"draft": True, "scale": 0.25, "x264_preset": "ultrafast"},
            ["--scale=0.25", "--x264-preset=ultrafast"],
        ),
        ({"concurrency": 12}, ["--concurrency=12"]),
        ({"concurrency": "75%"}, ["--concurrency=75%"]),
        ({"gl": "angle"}, ["--gl=angle"]),
        (
            {"frames": "0-59", "draft": True, "concurrency": 10},
            ["--frames=0-59", "--scale=0.5", "--x264-preset=veryfast", "--concurrency=10"],
        ),
    ],
)
def test_speed_flags(opts, expected):
    assert VideoCompose._remotion_speed_flags(opts) == expected


def test_frames_range_needs_two_values():
    with pytest.raises(ValueError, match="start, end"):
        VideoCompose._remotion_speed_flags({"frames": [1, 2, 3]})


@pytest.mark.parametrize(
    ("opts", "expected"),
    [
        ({}, HW_FLAGS),
        ({"encode": "auto"}, HW_FLAGS),
        ({"concurrency": 10, "gl": "angle"}, ["--concurrency=10", "--gl=angle", *HW_FLAGS]),
        ({"draft": True}, ["--scale=0.5", "--x264-preset=veryfast"]),
        ({"frames": 120}, ["--frames=120"]),
        ({"frames": 0}, ["--frames=0"]),
        ({"x264_preset": "slow"}, ["--x264-preset=slow"]),
        ({"scale": 2}, ["--scale=2"]),
        ({"crf": 16}, []),
        ({"encode": "crf"}, []),
    ],
)
def test_final_render_gets_hardware_flags_only_when_final_quality(monkeypatch, opts, expected):
    _set_hardware(monkeypatch, True)
    assert VideoCompose._remotion_speed_flags(opts) == expected


def test_unknown_encode_value_is_rejected():
    with pytest.raises(ValueError, match="encode"):
        VideoCompose._remotion_speed_flags({"encode": "prores"})


def test_remotion_render_passes_render_speed(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/npx")
    tool = VideoCompose()
    seen = {}

    def fake_run_command(cmd, *a, **k):
        seen["cmd"] = cmd
        return None  # output file intentionally absent

    monkeypatch.setattr(tool, "run_command", fake_run_command)
    tool._remotion_render(
        {
            "composition_data": {"cuts": []},
            "output_path": str(tmp_path / "out.mp4"),
            "render_speed": {"frames": "540-660", "draft": True},
        }
    )

    assert "--frames=540-660" in seen["cmd"]
    assert "--scale=0.5" in seen["cmd"]
    assert "--x264-preset=veryfast" in seen["cmd"]


def _caption_burn_render_cmd(tmp_path, monkeypatch) -> list[str]:
    composer_root = tmp_path / "remotion-composer"
    composer_root.mkdir()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake video")
    output = tmp_path / "rendered.mp4"
    tool = RemotionCaptionBurn()
    monkeypatch.setattr(tool, "_find_remotion_root", lambda: composer_root)
    render_cmds: list[list[str]] = []

    def fake_run_command(cmd, *args, **kwargs):
        if "format=duration" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="1.0\n")
        if "stream=width,height" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="1080x1920\n")
        render_cmds.append(list(cmd))
        output.write_bytes(b"rendered video")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(tool, "run_command", fake_run_command)
    result = tool._render_remotion(
        input_path=str(source),
        output_path=str(output),
        captions=[{"word": "hello", "startMs": 0, "endMs": 500}],
        words_per_page=4,
        font_size=52,
        highlight_color="#22D3EE",
    )
    assert result.success, result.error
    return render_cmds[-1]


def test_caption_burn_uses_hardware_encode_instead_of_crf(tmp_path, monkeypatch):
    _set_hardware(monkeypatch, True)
    cmd = _caption_burn_render_cmd(tmp_path, monkeypatch)
    assert "--crf=18" not in cmd
    assert all(flag in cmd for flag in HW_FLAGS)


def test_caption_burn_keeps_crf_without_hardware(tmp_path, monkeypatch):
    cmd = _caption_burn_render_cmd(tmp_path, monkeypatch)
    assert "--crf=18" in cmd
    assert not any(flag.startswith("--hardware-acceleration") for flag in cmd)
