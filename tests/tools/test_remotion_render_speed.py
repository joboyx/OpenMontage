"""Tests for the Remotion render-speed flags in video_compose.

Checks render only the frames they inspect, review passes use a draft
(half scale, fast x264 preset), and the final render passes no speed flag.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.video.video_compose import VideoCompose  # noqa: E402


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
