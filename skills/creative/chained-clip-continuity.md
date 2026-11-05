# Chained Clip Continuity — Image-to-Video Joins Without Flicker or Sound Jumps

## When to Use

Use this skill when one continuous shot (a timelapse, a slow push-in, a locked-off
landscape) is built from several image-to-video clips, each started from a frame of
the previous clip. Without it, the cut at every join shows a short flicker, and
per-clip sound changes character at every join.

## Prerequisites

| Resource | Why |
|----------|-----|
| `clip_join` tool (`start_frame`, `join`, `check`) | Start frames, flicker-free join, join measurement |
| `ambience_windows` tool (`generate`, `shape`) | One continuous ambience bed over the joined timeline |
| `mmaudio_replicate` tool | Video-to-audio model that `ambience_windows.generate` calls |
| `skills/creative/video-stitching.md` § 3 | General AI clip chaining rules (FPS, colour, motion direction) |
| `skills/creative/sound-design.md` | LUFS targets and mixing levels |
| `skills/core/remotion.md` § Render Speed | Fast frame-range and draft renders for checks |
| Layer 3: `ffmpeg`, `video-toolkit`, `sound-effects` | Underlying technology |

## Why Joins Flicker (Measured)

Measured on Seedance 2.0 image-to-video (mini, 480p, 24 fps, 10 s clips), chained from
the previous clip's last frame:

| Clip region | Sharpness (Laplacian variance) | Brightness |
|-------------|-------------------------------|------------|
| Last frame of clip N | highest of the clip (extra sharp) | reference |
| Frames 0-2 of clip N+1 | down to about 50% | slides 3-7 levels over the first 8 frames |
| Frame 8 onward | normal | settled |

So it is a model behaviour, not a tooling error: the model re-renders the start image
softly and then settles. A hard cut from an extra-sharp frame onto soft frames reads as a
flicker every clip length (every 10 s).

## Process

### Step 1: Plan the chain (prevention)

Choose the chaining mode before generation, because the start frame decides it.

| Mode | Start frame for clip N+1 | `join` setting | Timeline cost per join |
|------|--------------------------|----------------|------------------------|
| **Overlap (preferred)** | frame about 1 s before the end of clip N | `overlap_frames` = the value `start_frame` returns (24 at 24 fps) | K + 1 frames |
| Last frame (fallback) | last frame of clip N | `overlap_frames: 0` | 1 frame |

Overlap mode hides the soft start completely: clip N keeps playing through the overlap,
and clip N+1 fades in only after it settles (`settle_frames`, default 6). Generate each
clip about 1 s longer than its screen time to pay for the overlap.

Also:
- Export start frames as **PNG** (lossless). A JPEG start frame adds its own softness.
- Keep the source frame rate (24 fps) through the join. Convert to 30 fps only once, at the
  final render; a 24 to 30 fps pulldown repeats every 4th frame.

🚨 Last-frame mode is verified on a real 60 s, 6-clip project. Overlap mode passes the
join check on synthetic chained clips, but not yet on a real generated chain. Generate
two clips first, join them, check the join with Step 5, then generate the rest.

### Step 2: Export each start frame

```python
clip_join.execute({"operation": "start_frame", "clip": "assets/video/c1.mp4",
                   "output_path": "assets/images/frames/c2-start.png",
                   "before_end_seconds": 1.0})   # 0 for last-frame mode
# -> data.frame_index, data.overlap_frames (pass to join), data.fps
```

Send the PNG to the video tool as the start image (`image_url` as a data URI for
`seedance_replicate`; a local path is not converted).

### Step 3: Join the clips

```python
clip_join.execute({"operation": "join",
                   "clips": ["c1.mp4", "c2.mp4", "c3.mp4", "c4.mp4", "c5.mp4", "c6.mp4"],
                   "output_path": "assets/video/timeline.mp4",
                   "overlap_frames": 0})
```

Last-frame mode drops the repeated frame 0 of each next clip, blends frames 1-7 from the
previous clip's last frame (`blend_frames: 8`), and matches colour with a per-channel
ratio that decays over `bridge_seconds` (2.0). Six 241-frame clips give 1441 frames.

Use the joined timeline as ONE video scene in the composition. Do not let the
composition cut between the clips again.

### Step 4: Build the ambience on the joined timeline

Never generate sound per clip: each clip gets a different sound character, and the
change is heard at every join.

```python
ambience_windows.execute({"operation": "generate", "timeline": "assets/video/timeline.mp4",
                          "output_dir": "assets/audio/windows",
                          "prompt": "gentle wind over a calm lake, soft water lapping",
                          "extra_prompts": [{"start": 35, "end": 45,
                                             "text": ", gusty wind, distant rolling thunder"}]})
ambience_windows.execute({"operation": "shape", "windows_dir": "assets/audio/windows",
                          "music_path": "assets/music/bed.mp3", "total_seconds": 63,
                          "curve": [[0, -40], [20, -37], [38, -29], [42, -26], [50, -25],
                                    [54, -31], [58, -40], [63, -46]],
                          "fade_end_seconds": 59, "low_keep_spans": [[35, 45]],
                          "output_path": "assets/audio/soundtrack.wav"})
```

- `generate` cuts 10 s windows every 5 s (`window_seconds`, `hop_seconds`), so every join
  sits in the middle of a window. Cost: 11 windows for 60 s, about $0.0075 each
  (`estimate_cost` with `max_seconds`).
- `generate` is a paid batch. Per AGENT_GUIDE.md § Decision Communication Contract,
  announce the tool (`ambience_windows` calling `mmaudio_replicate`), provider, model,
  reason and cost first. Run a sample first (`max_seconds: 10` gives one window), listen,
  then run the batch.
- The default negative prompt removes "microphone wind noise, low hum, drone". MMAudio adds
  a rumble below 150 Hz to wind scenes otherwise.
- `shape` high-passes each window (120 Hz, or 50 Hz inside `low_keep_spans` so thunder
  keeps its weight), cross-fades with equal power, then moves the smoothed level onto
  `curve`. Plan the curve from the story: calm, build, peak, release. The curve sets the
  relative shape only: the final loudnorm sets the absolute level, so the measured
  ambience can sit several dB under the curve values.
- Fade the ambience out before the end card (`fade_end_seconds`; default and maximum: the
  end of the last window).
- With `music_path`, `shape` also mixes the music under the bed and normalises to
  `target_lufs` (-14 for social platforms). Without it, `shape` writes the shaped bed only;
  mix it with narration in `audio_mixer`.

### Step 5: Check the joins

```python
clip_join.execute({"operation": "check", "video": "assets/video/timeline.mp4",
                   "join_times": [10, 20, 30, 40, 50]})
# -> data.passed, data.joins[i].peak_change / sharpness_dip / ok
```

A join fails when, within 0.2 s of the join, the biggest frame-to-frame change is more
than `max_change_ratio` (2.0) times the median change of the video, or a frame is less
than `min_sharpness_step` (0.7) as sharp as the frame before it. Measured: raw hard cuts
drop to 0.50-0.62, fixed joins stay at 0.82-0.90, normal motion stays above 0.81.

Run it on the joined timeline, then on a fast render of each join
(`render_speed: {"frames": "<start>-<end>"}`, see `skills/core/remotion.md`). Render the
final without `render_speed`.

Listen to every join on headphones and on a phone speaker: no jump in level, tone or
background.

### Step 6: Self-Evaluate

| Criterion | 1 (fail) | 3 (acceptable) | 5 (excellent) |
|-----------|----------|----------------|---------------|
| Visual join | flicker visible at normal speed | `check` passes, slight softness seen frame by frame | no visible change frame by frame |
| Colour at join | brightness jump | small shift that settles within 2 s | no shift |
| Sound at join | character or level change heard | small level change | one continuous bed |
| Loudness | off target by > 1 LU | within 1 LU | within 0.3 LU, true peak below -1.5 dBTP |
| Cost | regenerated whole chain | regenerated single clips | first pass used |

Score 3 or better on every row before the edit stage is approved.

## Common Pitfalls

- **Treating the flicker as a tooling bug**: frame-exact extraction does not fix it; the
  model start does. Use overlap mode, or the blend in last-frame mode.
- **Hard cuts between chained clips in the composition**: join first with `clip_join`,
  then compose one timeline scene.
- **Per-clip sound passes**: each pass changes character; use `ambience_windows`.
- **Loudness normalising per window** (`dynaudnorm`): it pumps between windows. Use the
  `shape` curve.
- **Checking with full renders**: use `render_speed` frames or draft renders for checks.
- **Wrong `overlap_frames`**: pass the exact value `start_frame` returned. The check finds
  flicker, not a small jump back in time, so a join with the wrong mode can still pass.
- **Wrong frame maps after crop or fps change**: measure on the 24 fps timeline, or pass
  join times in the output's own timeline.
