"""ffmpeg H.264 call sites take their video codec arguments from the render profile."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from lib.media_profiles import ffmpeg_output_args, get_profile
from tools.character import character_animation
from tools.video.video_compose import VideoCompose
from tools.video.video_stitch import VideoStitch
from tools.video.video_trimmer import VideoTrimmer

VT_23 = ["-c:v", "h264_videotoolbox", "-q:v", "70"]
X264_23 = ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]


def _capture(monkeypatch, tool_cls) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(self, cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(tool_cls, "run_command", run)
    return commands


def _contains(cmd: list[str], part: list[str]) -> bool:
    return any(cmd[i:i + len(part)] == part for i in range(len(cmd) - len(part) + 1))


@pytest.mark.parametrize(("available", "video"), [(True, VT_23), (False, X264_23)])
def test_compose_encode_uses_profile(monkeypatch, video_hardware, tmp_path, available, video):
    video_hardware(available)
    commands = _capture(monkeypatch, VideoCompose)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"")
    result = VideoCompose().execute(
        {"operation": "encode", "input_path": str(source), "output_path": str(tmp_path / "out.mp4")}
    )
    assert result.success, result.error
    assert _contains(commands[0], video)
    assert _contains(commands[0], ["-c:a", "aac", "-b:a", "192k"])


def test_compose_encode_explicit_codec_wins(monkeypatch, video_hardware, tmp_path):
    video_hardware(True)
    commands = _capture(monkeypatch, VideoCompose)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"")
    result = VideoCompose().execute({
        "operation": "encode", "input_path": str(source),
        "output_path": str(tmp_path / "out.mp4"), "codec": "libx265", "crf": 28,
    })
    assert result.success, result.error
    assert _contains(commands[0], ["-c:v", "libx265", "-crf", "28", "-preset", "medium"])


@pytest.mark.parametrize(
    ("available", "video"),
    [(True, VT_23), (False, ["-c:v", "libx264", "-crf", "23", "-preset", "fast"])],
)
def test_trimmer_speed_uses_profile(monkeypatch, video_hardware, tmp_path, available, video):
    video_hardware(available)
    commands = _capture(monkeypatch, VideoTrimmer)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"")
    result = VideoTrimmer().execute(
        {"operation": "speed", "input_path": str(source), "speed_factor": 2.0,
         "output_path": str(tmp_path / "out.mp4")}
    )
    assert result.success, result.error
    assert _contains(commands[0], video)


@pytest.mark.parametrize(
    ("available", "video"),
    [(True, ["-c:v", "h264_videotoolbox", "-q:v", "75"]), (False, ["-c:v", "libx264", "-crf", "20"])],
)
def test_media_profile_output_args_use_profile(video_hardware, available, video):
    video_hardware(available)
    args = ffmpeg_output_args(get_profile("tiktok"))
    assert _contains(args, video)
    assert _contains(args, ["-c:a", "aac"])


PROBES = [{"duration": 2.0}, {"duration": 2.0}, {"duration": 2.0}]


def _xfade_commands(monkeypatch, tmp_path, suffix: str, clip_count: int, method: str):
    commands = _capture(monkeypatch, VideoStitch)
    clips = [str(tmp_path / f"c{i}.mp4") for i in range(clip_count)]
    getattr(VideoStitch(), method)(clips, tmp_path / f"out{suffix}", 0.5, PROBES[:clip_count])
    return commands


@pytest.mark.parametrize("method", ["_stitch_crossfade", "_stitch_fade_through_black"])
@pytest.mark.parametrize("clip_count", [2, 3])
@pytest.mark.parametrize(("available", "video"), [(True, VT_23), (False, X264_23)])
def test_xfade_mp4_output_uses_profile(
    monkeypatch, video_hardware, tmp_path, method, clip_count, available, video
):
    video_hardware(available)
    commands = _xfade_commands(monkeypatch, tmp_path, ".mp4", clip_count, method)
    assert _contains(commands[0], [*video, str(tmp_path / "out.mp4")])


@pytest.mark.parametrize("method", ["_stitch_crossfade", "_stitch_fade_through_black"])
@pytest.mark.parametrize("clip_count", [2, 3])
def test_xfade_other_container_keeps_ffmpeg_default(
    monkeypatch, video_hardware, tmp_path, method, clip_count
):
    video_hardware(True)
    commands = _xfade_commands(monkeypatch, tmp_path, ".webm", clip_count, method)
    assert "-c:v" not in commands[0]


class _FakePage:
    def goto(self, *args, **kwargs):
        pass

    def wait_for_timeout(self, *args):
        pass

    def screenshot(self, path):
        Path(path).write_bytes(b"")


class _FakePlaywright:
    def __enter__(self):
        browser = types.SimpleNamespace(new_page=lambda **kw: _FakePage(), close=lambda: None)
        return types.SimpleNamespace(chromium=types.SimpleNamespace(launch=lambda: browser))

    def __exit__(self, *exc):
        return False


def _preview_command(monkeypatch, tmp_path, name: str) -> list[str]:
    fake = types.ModuleType("playwright.sync_api")
    fake.sync_playwright = _FakePlaywright
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake)
    monkeypatch.setattr(character_animation.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    commands: list[list[str]] = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(character_animation.subprocess, "run", run)
    preview = tmp_path / "preview.html"
    preview.write_text("<html></html>", encoding="utf-8")
    character_animation._render_preview_mp4(preview, tmp_path / name, 0.2, 10)
    return commands[0]


@pytest.mark.parametrize(("available", "video"), [(True, VT_23), (False, X264_23)])
def test_character_preview_mp4_uses_profile(monkeypatch, video_hardware, tmp_path, available, video):
    video_hardware(available)
    cmd = _preview_command(monkeypatch, tmp_path, "preview.mp4")
    assert _contains(cmd, [*video, str(tmp_path / "preview.mp4")])


def test_character_preview_other_container_keeps_ffmpeg_default(monkeypatch, video_hardware, tmp_path):
    video_hardware(True)
    cmd = _preview_command(monkeypatch, tmp_path, "preview.webm")
    assert "-c:v" not in cmd
