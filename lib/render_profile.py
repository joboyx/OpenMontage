"""Shared render profile: hardware video encode/decode on macOS, CPU elsewhere.

Remotion final renders and ffmpeg intermediate files use the flags from this
module, so every tool picks the same encoder for the same host.

Measured on one Remotion clip (Apple M5 Pro, 15 cores; 1080x1920, 60 fps,
1,124 frames): libx264 CRF 16 took 52-69 s; VideoToolbox H.264 at 16 Mb/s plus
8 OffthreadVideo decode threads took 40 s, at VMAF 96.2 against the CRF 16
file. The hardware encoder frees CPU cores, and the extra decode threads use
them; either change alone gave no gain beyond run-to-run noise.

Set ``OPENMONTAGE_HW_VIDEO=0`` to force the CPU path.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess

_HW_ENV_VAR = "OPENMONTAGE_HW_VIDEO"


@functools.lru_cache(maxsize=1)
def _videotoolbox_encoder_listed() -> bool:
    """Probe the ffmpeg on PATH once for the VideoToolbox H.264 encoder.

    Returns:
        True when ``ffmpeg -hide_banner -encoders`` lists ``h264_videotoolbox``.
        False when ffmpeg is not on PATH, because then no hardware encode
        is possible.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return "h264_videotoolbox" in result.stdout


def hardware_video_available() -> bool:
    """Tell whether this host can encode and decode video in hardware.

    Hardware video means VideoToolbox on macOS. The ffmpeg probe runs once per
    process and its result is cached; the environment override is read on
    every call.

    Returns:
        True when the host is macOS and the ffmpeg on PATH lists
        ``h264_videotoolbox``. False when ``OPENMONTAGE_HW_VIDEO=0`` is set,
        on other operating systems, or when ffmpeg is missing. Any other
        value of the environment variable means auto-detect.
    """
    if os.environ.get(_HW_ENV_VAR) == "0":
        return False
    if platform.system() != "Darwin":
        return False
    return _videotoolbox_encoder_listed()


def offthread_threads() -> int:
    """Get the thread count for Remotion OffthreadVideo frame decode.

    Half of the CPU cores, because Chrome tabs and the encoder use the rest.

    Returns:
        ``max(2, cpu_count // 2)``, with a cpu_count of 4 when the OS does
        not report one.
    """
    return max(2, (os.cpu_count() or 4) // 2)


def remotion_encode_flags(video_bitrate: str = "16M") -> list[str]:
    """Build the Remotion CLI encode flags for a final-quality render.

    With hardware video, Remotion encodes with VideoToolbox H.264 at a fixed
    bitrate and decodes OffthreadVideo sources with more threads. The two
    flags need each other: hardware encode frees CPU cores, and the extra
    decode threads use them. Remotion does not accept ``--crf`` together
    with ``--video-bitrate``, so callers must not add ``--crf`` when this
    list is not empty.

    Args:
        video_bitrate: Target bitrate for the hardware encoder, in Remotion's
            ``--video-bitrate`` format (for example ``"16M"``).

    Returns:
        The three hardware flags when hardware video is available, else an
        empty list, so Remotion's own CPU defaults (libx264, CRF) apply.
    """
    if not hardware_video_available():
        return []
    return [
        "--hardware-acceleration=if-possible",
        f"--video-bitrate={video_bitrate}",
        f"--offthreadvideo-threads={offthread_threads()}",
    ]


def ffmpeg_intermediate_args(bitrate: str = "80M") -> tuple[list[str], list[str]]:
    """Build ffmpeg arguments for a near-lossless intermediate video file.

    Use this for prep steps (crop, scale, trim) whose output is a source for
    a later render. With hardware video, a crop/scale of a 4K VP9 source took
    32 s against 69 s for libx264 CRF 10 preset medium.

    Do not use ProRes intermediates for Remotion OffthreadVideo sources: the
    10-bit ProRes software decode made the Remotion render slower (103 s
    against 52-69 s for H.264 sources).

    Args:
        bitrate: Target bitrate for the hardware encoder, in ffmpeg ``-b:v``
            format (for example ``"80M"``).

    Returns:
        A tuple ``(decode_args, encode_args)``. Put ``decode_args`` before
        ``-i``; put ``encode_args`` with the output options. ``decode_args``
        is empty on the CPU path.
    """
    if hardware_video_available():
        return (
            ["-hwaccel", "videotoolbox"],
            ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p"],
        )
    return (
        [],
        ["-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
