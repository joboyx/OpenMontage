"""Tests for the shared render profile (hardware video on macOS, CPU elsewhere)."""

from __future__ import annotations

import subprocess

import pytest

from lib import render_profile
from lib.render_profile import (
    ffmpeg_default_h264_args,
    ffmpeg_h264_encode_args,
    ffmpeg_intermediate_args,
    ffmpeg_video_codec_args,
    hardware_video_available,
    offthread_threads,
    videotoolbox_quality,
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


def _host(
    monkeypatch, *, system="Darwin", machine="arm64", ffmpeg="/opt/bin/ffmpeg",
    encoders=ENCODERS_WITH_VT,
):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    monkeypatch.setattr("shutil.which", lambda name: ffmpeg if name == "ffmpeg" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=encoders),
    )


def test_darwin_with_videotoolbox_encoder_is_available(monkeypatch):
    _host(monkeypatch)
    assert hardware_video_available() is True


def test_intel_mac_is_not_available(monkeypatch):
    _host(monkeypatch, machine="x86_64")
    assert hardware_video_available() is False


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


def _hardware(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(render_profile, "hardware_video_available", lambda: available)


@pytest.mark.parametrize(
    ("crf", "expected"),
    [
        # Measured points.
        (12, 90), (16, 85), (18, 80), (20, 75), (23, 70), (28, 55),
        # Interpolated between points.
        (14, 88), (22, 72), (25, 64),
        # Clamped outside the measured range.
        (1, 90), (10, 90), (30, 55), (51, 55),
    ],
)
def test_videotoolbox_quality(crf, expected):
    assert videotoolbox_quality(crf) == expected


def test_h264_args_hardware(monkeypatch):
    _hardware(monkeypatch, True)
    assert ffmpeg_h264_encode_args(18, "fast") == ["-c:v", "h264_videotoolbox", "-q:v", "80"]


def test_h264_args_hardware_defaults(monkeypatch):
    _hardware(monkeypatch, True)
    assert ffmpeg_h264_encode_args() == ["-c:v", "h264_videotoolbox", "-q:v", "70"]


def test_h264_args_cpu(monkeypatch):
    _hardware(monkeypatch, False)
    assert ffmpeg_h264_encode_args(18, "fast") == [
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    ]
    assert ffmpeg_h264_encode_args() == ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]


def test_h264_lossless_stays_on_libx264_with_hardware(monkeypatch):
    _hardware(monkeypatch, True)
    assert ffmpeg_h264_encode_args(0, "medium") == [
        "-c:v", "libx264", "-crf", "0", "-preset", "medium",
    ]


def test_video_codec_args_libx264_uses_hardware(monkeypatch):
    _hardware(monkeypatch, True)
    assert ffmpeg_video_codec_args("libx264", 20, "medium") == [
        "-c:v", "h264_videotoolbox", "-q:v", "75",
    ]


def test_video_codec_args_libx264_cpu_fills_defaults(monkeypatch):
    _hardware(monkeypatch, False)
    assert ffmpeg_video_codec_args("libx264") == [
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
    ]


def test_video_codec_args_explicit_codec_passes_through(monkeypatch):
    _hardware(monkeypatch, True)
    assert ffmpeg_video_codec_args("libx265", 28, "slow") == [
        "-c:v", "libx265", "-crf", "28", "-preset", "slow",
    ]
    assert ffmpeg_video_codec_args("libvpx-vp9", 30) == ["-c:v", "libvpx-vp9", "-crf", "30"]
    assert ffmpeg_video_codec_args("prores_ks") == ["-c:v", "prores_ks"]


@pytest.mark.parametrize("name", ["out.mp4", "out.MOV", "dir.x/out.m4v"])
def test_default_h264_args_for_h264_containers(monkeypatch, name):
    _hardware(monkeypatch, True)
    assert ffmpeg_default_h264_args(name) == ["-c:v", "h264_videotoolbox", "-q:v", "70"]
    _hardware(monkeypatch, False)
    assert ffmpeg_default_h264_args(name) == ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]


@pytest.mark.parametrize("name", ["out.webm", "out.mkv", "out.gif", "out"])
def test_default_h264_args_leave_other_containers_alone(monkeypatch, name):
    _hardware(monkeypatch, True)
    assert ffmpeg_default_h264_args(name) == []
