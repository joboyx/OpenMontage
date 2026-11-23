"""Tests for the shared render profile (hardware video on macOS, CPU elsewhere)."""

from __future__ import annotations

import subprocess

import pytest

from lib import render_profile
from lib.render_profile import (
    ffmpeg_intermediate_args,
    hardware_video_available,
    offthread_threads,
)

ENCODERS_WITH_VT = (
    " V....D libx264              libx264 H.264 / AVC\n"
    " V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)\n"
)
ENCODERS_WITHOUT_VT = " V....D libx264              libx264 H.264 / AVC\n"


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    monkeypatch.delenv("OPENMONTAGE_HW_VIDEO", raising=False)
    render_profile._videotoolbox_encoder_listed.cache_clear()
    yield
    render_profile._videotoolbox_encoder_listed.cache_clear()


def _host(monkeypatch, *, system="Darwin", ffmpeg="/opt/bin/ffmpeg", encoders=ENCODERS_WITH_VT):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("shutil.which", lambda name: ffmpeg if name == "ffmpeg" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=encoders),
    )


def test_darwin_with_videotoolbox_encoder_is_available(monkeypatch):
    _host(monkeypatch)
    assert hardware_video_available() is True


def test_darwin_without_videotoolbox_encoder_is_not_available(monkeypatch):
    _host(monkeypatch, encoders=ENCODERS_WITHOUT_VT)
    assert hardware_video_available() is False


def test_env_override_zero_forces_cpu(monkeypatch):
    _host(monkeypatch)
    monkeypatch.setenv("OPENMONTAGE_HW_VIDEO", "0")
    assert hardware_video_available() is False


def test_env_override_other_value_means_auto(monkeypatch):
    _host(monkeypatch)
    monkeypatch.setenv("OPENMONTAGE_HW_VIDEO", "1")
    assert hardware_video_available() is True


def test_non_darwin_is_not_available(monkeypatch):
    _host(monkeypatch, system="Linux")
    assert hardware_video_available() is False


def test_missing_ffmpeg_is_not_available(monkeypatch):
    _host(monkeypatch, ffmpeg=None)
    assert hardware_video_available() is False


def test_ffmpeg_probe_runs_once(monkeypatch):
    _host(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout=ENCODERS_WITH_VT),
    )
    hardware_video_available()
    hardware_video_available()
    assert calls == [["/opt/bin/ffmpeg", "-hide_banner", "-encoders"]]


def test_intermediate_args_hardware(monkeypatch):
    _host(monkeypatch)
    assert ffmpeg_intermediate_args() == (
        ["-hwaccel", "videotoolbox"],
        ["-c:v", "h264_videotoolbox", "-b:v", "80M", "-pix_fmt", "yuv420p"],
    )


def test_intermediate_args_cpu(monkeypatch):
    _host(monkeypatch, system="Linux")
    assert ffmpeg_intermediate_args() == (
        [],
        ["-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [(16, 8), (15, 7), (3, 2), (1, 2), (None, 2)],
)
def test_offthread_threads(monkeypatch, cpu_count, expected):
    monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
    assert offthread_threads() == expected
