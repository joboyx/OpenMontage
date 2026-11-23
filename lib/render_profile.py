"""Shared render profile: hardware video encode/decode on macOS, CPU elsewhere.

Remotion final renders, ffmpeg intermediate files and ffmpeg H.264 encodes
use the flags from this module, so every tool picks the same encoder
for the same host. No helper adds ``-hwaccel`` for a plain encode: a
decode-only pass of a 4K VP9 source with a CPU crop/scale was slower with
``-hwaccel videotoolbox`` (9.5-12.9 s against 5.8-9.7 s).

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

    Hardware video means VideoToolbox on an Apple Silicon Mac. Intel Macs do
    not count: the VideoToolbox H.264 constant-quality mode (``-q:v``) that
    ``ffmpeg_h264_encode_args`` uses works on Apple Silicon only. The ffmpeg
    probe runs once per process and its result is cached; the environment
    override is read on every call.

    Returns:
        True when the host is macOS on ``arm64`` and the ffmpeg on PATH lists
        ``h264_videotoolbox``. False when ``OPENMONTAGE_HW_VIDEO=0`` is set,
        on other operating systems or CPU types, or when ffmpeg is missing.
        Any other value of the environment variable means auto-detect.
    """
    if os.environ.get(_HW_ENV_VAR) == "0":
        return False
    if platform.system() != "Darwin" or platform.machine() != "arm64":
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


# libx264 CRF -> h264_videotoolbox -q:v, as (crf, q) points sorted by CRF.
# Each q is the smallest tested q (45-95, step 5) whose VMAF is at least the
# libx264 VMAF at that CRF (preset medium, yuv420p) on both calibration
# samples: (a) 1080x1920 60 fps gameplay with a face camera, (b) 1920x1080
# 30 fps motion graphics (assets/signal-from-tomorrow-demo.mp4); 8 s each,
# against a lossless yuv420p reference; Apple M5 Pro, ffmpeg 9.0.2.
#
#   CRF  x264 VMAF a / b   q   VT VMAF a / b   VT/x264 bitrate a / b
#   12   97.72 / 97.87     90  97.92 / 98.01   2.86 / 2.88
#   16   97.22 / 97.31     85  97.72 / 97.68   2.46 / 2.44
#   18   96.68 / 96.93     80  97.31 / 97.15   2.01 / 1.94
#   20   95.98 / 96.16     75  96.73 / 96.51   1.84 / 1.74
#   23   94.58 / 94.63     70  96.04 / 95.78   2.10 / 2.03
#   28   89.82 / 90.54     55  91.99 / 91.49   1.80 / 1.75
_CRF_TO_VT_Q: tuple[tuple[int, int], ...] = (
    (12, 90),
    (16, 85),
    (18, 80),
    (20, 75),
    (23, 70),
    (28, 55),
)


def videotoolbox_quality(crf: int) -> int:
    """Map a libx264 CRF to the VideoToolbox H.264 ``-q:v`` of equal quality.

    The map comes from the measured points in ``_CRF_TO_VT_Q``. A CRF between
    two points gets the linear interpolation, rounded to an integer. A CRF
    outside the measured range gets the q of the nearest end point.

    Args:
        crf: The libx264 constant rate factor (lower is better).

    Returns:
        The ``-q:v`` value for ``h264_videotoolbox`` (1-100, higher is better).
    """
    first_crf, first_q = _CRF_TO_VT_Q[0]
    if crf <= first_crf:
        return first_q
    for (crf_lo, q_lo), (crf_hi, q_hi) in zip(_CRF_TO_VT_Q, _CRF_TO_VT_Q[1:]):
        if crf <= crf_hi:
            return round(q_lo + (q_hi - q_lo) * (crf - crf_lo) / (crf_hi - crf_lo))
    return _CRF_TO_VT_Q[-1][1]


def ffmpeg_h264_encode_args(crf: int = 23, preset: str = "medium") -> list[str]:
    """Build ffmpeg arguments for an H.264 video encode at a libx264 quality.

    With hardware video, the encode uses ``h264_videotoolbox`` in constant
    quality mode at the ``-q:v`` that ``videotoolbox_quality`` maps from
    ``crf``. VideoToolbox does not accept ``-crf``, ``-preset``, ``-tune`` or
    ``-x264-params``, so ``preset`` has no effect there. At equal VMAF the
    VideoToolbox file is about 1.7-2.9 times larger than the libx264 file
    (see ``_CRF_TO_VT_Q``).

    CRF 0 (lossless) always uses libx264, because VideoToolbox has no
    lossless mode.

    The caller adds the pixel format (for example ``-pix_fmt yuv420p``).

    Args:
        crf: The libx264 constant rate factor (lower is better).
        preset: The libx264 preset for the CPU path.

    Returns:
        ``["-c:v", "h264_videotoolbox", "-q:v", q]`` with hardware video,
        else ``["-c:v", "libx264", "-crf", crf, "-preset", preset]``.
    """
    if crf > 0 and hardware_video_available():
        return ["-c:v", "h264_videotoolbox", "-q:v", str(videotoolbox_quality(crf))]
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]


def ffmpeg_video_codec_args(
    codec: str, crf: int | None = None, preset: str | None = None
) -> list[str]:
    """Build ffmpeg video codec arguments for a codec that a caller names.

    Use this where a tool input names the codec and ``libx264`` is the
    default. ``libx264`` means "H.264 at this quality" and goes through
    ``ffmpeg_h264_encode_args``, so it uses the hardware encoder when the host
    has one. Any other codec is the caller's explicit choice and is passed
    through as given.

    Args:
        codec: The ffmpeg video encoder name (for example ``"libx264"``).
        crf: The constant rate factor, or None to leave it out. For
            ``libx264``, None means 23 (the libx264 default).
        preset: The encoder preset, or None to leave it out. For
            ``libx264``, None means ``"medium"`` (the libx264 default).

    Returns:
        The ``-c:v`` arguments with quality and preset.
    """
    if codec == "libx264":
        return ffmpeg_h264_encode_args(
            23 if crf is None else crf, "medium" if preset is None else preset
        )
    args = ["-c:v", codec]
    if crf is not None:
        args += ["-crf", str(crf)]
    if preset is not None:
        args += ["-preset", preset]
    return args


# Containers whose ffmpeg default video encoder is libx264.
_H264_DEFAULT_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})


def ffmpeg_default_h264_args(output_path: str | os.PathLike[str]) -> list[str]:
    """Build video codec arguments for a command that names no video codec.

    Without ``-c:v``, ffmpeg picks the encoder from the output extension: for
    MP4, MOV and M4V that is libx264 at its defaults (CRF 23, preset medium).
    For those extensions this returns ``ffmpeg_h264_encode_args()`` at the
    same quality, so the hardware encoder applies. For any other extension
    it returns an empty list and ffmpeg keeps its own choice.

    Args:
        output_path: The output file of the ffmpeg command.

    Returns:
        The H.264 encode arguments, or an empty list.
    """
    if os.path.splitext(os.fspath(output_path))[1].lower() in _H264_DEFAULT_SUFFIXES:
        return ffmpeg_h264_encode_args()
    return []
