"""clip_join: start frame export, join frame accounting and the flicker check.

The clips are tiny synthetic testsrc2 renders made with real ffmpeg. A chained
clip starts on the previous clip's last frame, like an image-to-video clip
generated from that frame.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.base_tool import ToolStatus
from tools.tool_registry import registry
from tools.video.clip_join import ClipJoin

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg required",
)

CLIP_FRAMES = 24  # 1 s at 24 fps
# The image-to-video soft start: frames 0-2 blurred and brighter.
SOFT_START = "gblur=sigma=2:enable='lt(n,3)',eq=brightness=0.04:enable='lt(n,3)'"


def _make_clip(path: Path, start_frame: int = 0, vf: str = "") -> Path:
    """Render CLIP_FRAMES frames of testsrc2 (64x112, 24 fps) from `start_frame`."""
    end = start_frame + CLIP_FRAMES
    chain = f"trim=start_frame={start_frame}:end_frame={end},setpts=PTS-STARTPTS"
    if vf:
        chain += f",{vf}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=64x112:r=24:d=10",
         "-vf", chain, "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _chained_clips(tmp_path: Path, count: int, vf: str = "", overlap: int = 0) -> list[str]:
    """`count` clips, each starting on frame (last - overlap) of the previous clip.

    `vf` is applied to every clip except the first.
    """
    step = CLIP_FRAMES - 1 - overlap
    return [
        str(_make_clip(tmp_path / f"c{i}.mp4", start_frame=i * step, vf=vf if i else ""))
        for i in range(count)
    ]


def _hard_cut(clips: list[str], out: Path) -> Path:
    """Concatenate clips with plain cuts."""
    inputs = [arg for clip in clips for arg in ("-i", clip)]
    streams = "".join(f"[{i}:v]" for i in range(len(clips)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
         f"{streams}concat=n={len(clips)}:v=1[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "4",
         "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def _count_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


# ---- metadata, registry, status ----


def test_contract_metadata():
    tool = ClipJoin()
    info = tool.get_info()
    assert info["name"] == "clip_join"
    assert info["tier"] == "core"
    assert info["capability"] == "video_post"
    assert info["provider"] == "ffmpeg"
    assert info["runtime"] == "local"
    assert info["determinism"] == "deterministic"
    assert info["agent_skills"] == ["ffmpeg", "video-toolkit"]
    assert set(info["dependencies"]) == {"cmd:ffmpeg", "cmd:ffprobe", "python:numpy"}
    assert "joining chained image-to-video clips without the start-of-clip flicker" in info["best_for"]
    assert tool.input_schema["properties"]["operation"]["enum"] == ["start_frame", "join", "check"]
    assert "joins" in tool.output_schema["properties"]
    assert tool.estimate_cost({"operation": "join"}) == 0.0
    assert tool.estimate_runtime({"operation": "join", "clips": ["a", "b"]}) > 0


def test_module_import_does_not_load_numpy():
    code = "import sys, tools.video.clip_join; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[2])
    assert out.stdout.strip() == "False"


def test_registry_discovers_clip_join():
    registry.discover()
    assert registry.get("clip_join") is not None


@needs_ffmpeg
def test_status_available_with_ffmpeg():
    assert ClipJoin().get_status() == ToolStatus.AVAILABLE


def test_status_unavailable_without_ffmpeg_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ClipJoin().get_status() == ToolStatus.UNAVAILABLE


# ---- operations ----


@needs_ffmpeg
@pytest.mark.parametrize(
    ("before_end", "frame_index", "overlap_frames"),
    [(0, 23, 0), (0.5, 11, 12)],
)
def test_start_frame_index(tmp_path, before_end, frame_index, overlap_frames):
    clip = str(_make_clip(tmp_path / "clip.mp4"))
    out = tmp_path / "start.png"
    result = ClipJoin().execute({"operation": "start_frame", "clip": clip, "output_path": str(out),
                                 "before_end_seconds": before_end})
    assert result.success, result.error
    assert result.data["frame_index"] == frame_index
    assert result.data["overlap_frames"] == overlap_frames
    assert result.data["fps"] == 24.0
    assert out.is_file()


@needs_ffmpeg
@pytest.mark.parametrize(
    ("overlap_frames", "expected_frames"),
    [(0, 70), (8, 54)],  # 3 x 24 frames: K=0 costs 1 frame per join, K=8 costs 9
)
def test_join_frame_accounting(tmp_path, overlap_frames, expected_frames):
    clips = _chained_clips(tmp_path, 3, overlap=overlap_frames)
    out = tmp_path / "joined.mp4"
    result = ClipJoin().execute({"operation": "join", "clips": clips, "output_path": str(out),
                                 "overlap_frames": overlap_frames})
    assert result.success, result.error
    assert result.data["frames"] == expected_frames
    assert _count_frames(str(out)) == expected_frames
    assert result.artifacts == [str(out)]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("chained_overlap", "joined_by", "passed"),
    [
        # Chained from the last frame.
        (0, "hard_cut", False),
        (0, "clip_join K=0", True),
        # Chained from frame (last - 8): the join must use overlap mode.
        (8, "hard_cut", False),
        (8, "clip_join K=8", True),
    ],
)
def test_check_video_verdict(tmp_path, chained_overlap, joined_by, passed):
    clips = _chained_clips(tmp_path, 2, vf=SOFT_START, overlap=chained_overlap)
    out = tmp_path / "joined.mp4"
    if joined_by == "hard_cut":
        _hard_cut(clips, out)
    else:
        overlap_frames = int(joined_by.rsplit("=", 1)[1])
        joined = ClipJoin().execute({"operation": "join", "clips": clips, "output_path": str(out),
                                     "overlap_frames": overlap_frames, "settle_frames": 6})
        assert joined.success, joined.error

    # Every variant changes clip at frame 24: the cut, the K=0 blend start, the end of the K=8 overlap.
    result = ClipJoin().execute({"operation": "check", "video": str(out), "join_times": [1.0]})

    assert result.success, result.error
    assert result.data["passed"] is passed
    [join] = result.data["joins"]
    assert join["ok"] is passed
    assert set(join) == {"time", "peak_change", "normal_change", "sharpness_dip", "ok"}


@needs_ffmpeg
def test_check_pairs_reports_the_soft_start(tmp_path):
    clips = _chained_clips(tmp_path, 2, vf=SOFT_START)
    result = ClipJoin().execute({"operation": "check", "clips": clips})
    assert result.success, result.error
    [pair] = result.data["pairs"]
    assert [r["frame"] for r in pair["rows"]] == ["A20", "A21", "A22", "A23", "B0", "B1", "B2", "B3", "B4", "B5"]
    assert pair["sharpness_dip"] < 0.7


@needs_ffmpeg
@pytest.mark.parametrize(
    "inputs",
    [
        {"operation": "unknown"},
        {"operation": "join", "clips": []},
        {"operation": "join", "clips": ["missing-a.mp4", "missing-b.mp4"], "output_path": "x.mp4"},
        {"operation": "start_frame", "clip": "missing.mp4", "output_path": "x.png"},
        {"operation": "check"},
    ],
)
def test_bad_inputs_fail_without_raising(inputs):
    result = ClipJoin().execute(inputs)
    assert result.success is False
    assert result.error


@needs_ffmpeg
def test_bad_inputs_on_real_clips_fail_without_raising(tmp_path):
    clips = _chained_clips(tmp_path, 2)
    tool = ClipJoin()
    cases = [
        {"operation": "join", "clips": clips, "output_path": str(tmp_path / "j.mp4"),
         "overlap_frames": 4, "settle_frames": 6},
        {"operation": "start_frame", "clip": clips[0], "output_path": str(tmp_path / "s.png"),
         "before_end_seconds": 5},
        {"operation": "check", "video": clips[0]},
        {"operation": "check", "video": clips[0], "join_times": [30.0]},
    ]
    for inputs in cases:
        result = tool.execute(inputs)
        assert result.success is False, inputs
        assert result.error
