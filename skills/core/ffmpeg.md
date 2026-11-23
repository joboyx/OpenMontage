# FFmpeg Skill

## When to Use

Use FFmpeg-backed tools for any video/audio processing that does not require AI inference:
cutting, trimming, speed adjustment, concatenation, audio extraction, mixing,
subtitle burn-in, overlay compositing, encoding, face enhancement, color grading,
and audio cleanup.

## Tools That Use FFmpeg

### Core Pipeline

| Tool | Capability |
|------|-----------|
| `video_trimmer` | Cut, trim, speed adjust, concat video segments |
| `video_compose` | Full composition: cuts + subtitles + overlays + encode |
| `audio_mixer` | Mix speech/music/SFX, ducking, fades, extract audio |
| `frame_sampler` | Extract representative frames from video |

### Enhancement Layer

| Tool | Capability | Key Presets |
|------|-----------|-------------|
| `face_enhance` | Skin smoothing, sharpening, warm/cool tones | `talking_head_standard`, `soft_skin`, `sharpen` |
| `color_grade` | Cinematic color grading with intensity control | `cinematic_warm`, `cinematic_cool`, `moody_dark` |
| `audio_enhance` | Noise reduction, loudness normalization, EQ | `clean_speech`, `voice_clarity`, `podcast` |

## Key Patterns

### Enhancement Chain Order

Apply enhancements in this order to avoid filter interactions:

1. **Subtitles first** — burn into the base video
2. **Face enhance** — smoothing/sharpening works best on ungraded footage
3. **Color grade** — applies look after face is already enhanced
4. **Audio enhance** — independent of video, apply last

Each step is optional and gracefully skipped if the tool is unavailable.

### Lossless vs Re-encode

- Use `-c copy` (codec copy) when you only need to cut or concat without altering frames. This is instant and lossless.
- Re-encode (`-c:v libx264`) when applying filters (speed change, subtitles, overlays, scaling).
- Default CRF is 23. Use 18-20 for higher quality when the output is the final deliverable.

### Intermediate Files

For a prep step whose output feeds a later render (crop, scale, trim of a source clip), get the codec arguments from `lib.render_profile.ffmpeg_intermediate_args()`:

```python
from lib.render_profile import ffmpeg_intermediate_args

decode_args, encode_args = ffmpeg_intermediate_args()
cmd = ["ffmpeg", "-y", *decode_args, "-i", src, "-vf", vf, *encode_args, out]
```

- On macOS with VideoToolbox: `-hwaccel videotoolbox` decode and `h264_videotoolbox -b:v 80M` encode. Elsewhere: libx264 CRF 10, preset medium. Both use `yuv420p`.
- Measured on a crop/scale of a 4K VP9 source (Apple M5 Pro): 32 s with hardware against 69 s with libx264 CRF 10 preset medium.
- Do not use ProRes intermediates for Remotion `OffthreadVideo` sources. The 10-bit ProRes software decode made the Remotion render slower (103 s against 52–69 s with H.264 sources).
- `OPENMONTAGE_HW_VIDEO=0` forces the CPU path.

### H.264 Encodes

For any other H.264 encode, get the codec arguments from `lib.render_profile.ffmpeg_h264_encode_args(crf, preset)`; the caller still adds `-pix_fmt`. On an Apple Silicon Mac with VideoToolbox it returns `h264_videotoolbox -q:v <q>`; elsewhere `libx264 -crf <crf> -preset <preset>`. Where a tool input names the codec, `ffmpeg_video_codec_args(codec, crf, preset)` uses it for `libx264` and passes any other codec through; a command with no `-c:v` uses `ffmpeg_default_h264_args(output_path)`, which adds the CRF 23 arguments only for `.mp4`, `.mov` and `.m4v` outputs. The CRF to `-q:v` map is the smallest q with a VMAF at least that of libx264 on two test clips: CRF 12 → 90, 16 → 85, 18 → 80, 20 → 75, 23 → 70, 28 → 55 (interpolated between, clamped outside); CRF 0 (lossless) stays on libx264. At equal VMAF the VideoToolbox file is about 1.7–2.9 times larger than the libx264 file. Do not add `-hwaccel` by default: a decode-only 4K VP9 pass with a CPU crop/scale measured slower with `-hwaccel videotoolbox` (9.5–12.9 s against 5.8–9.7 s); the hardware gain is in the encode.

### Subtitle Burn-in

- Prefer SRT format for simple word subtitles.
- Use `force_style` with full ASS color format: `&H00FFFFFF` (not `&HFFFFFF`).
- Always escape Windows paths (`C\:` not `C:`) in the subtitles filter.
- Vertical video: `font_size: 18`, `max 3 words/cue`, `margin_v: 50`.
- Horizontal video: `font_size: 22`, `max 6 words/cue`, `margin_v: 40`.

### Audio Enhancement Targets

| Platform | Target LUFS | Loudness Range |
|----------|-------------|----------------|
| Social media (TikTok, Reels) | -14 LUFS | 5-7 LU |
| YouTube | -14 to -16 LUFS | 7-11 LU |
| Podcast | -16 LUFS | 7-11 LU |
| Broadcast | -24 LUFS | 7 LU |

The `clean_speech` preset targets -16 LUFS with 11 LU range — good for YouTube/social media.

### Color Grade Intensity

- `intensity: 0.85` — recommended default for cinematic_warm on talking heads
- `intensity: 0.5` — subtle, barely noticeable
- `intensity: 1.0` — full effect, may look over-processed on some footage

### Audio Ducking

- Use `sidechaincompress` with speech as the key signal to lower music volume during dialogue.
- Typical settings: threshold=0.02, ratio=9, attack=200ms, release=500ms.

### Concatenation

- Use the concat demuxer (`-f concat -safe 0`) for same-codec segments.
- For mixed codecs or different resolutions, re-encode all segments first.

## Quality Checklist

- [ ] Output plays without artifacts on desktop and mobile
- [ ] Audio and video remain in sync after processing
- [ ] Subtitles are in the bottom 20% of frame, never covering the face
- [ ] Audio loudness is within target range for the platform
- [ ] Enhancement is visible but natural — skin tones look healthy, not orange
- [ ] No audio clipping or silence gaps at cut points
- [ ] File size is reasonable for the target platform
