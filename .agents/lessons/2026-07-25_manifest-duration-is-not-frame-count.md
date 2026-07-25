---
date: 2026-07-25
experiment: VidaForge-3M shard-00000 real-media adapter review
category: data
severity: important
---

# Manifest Duration Is Not an Exact Video Frame Count

## What Happened

A preprocessing adapter derived `num_frames` with
`floor(fps * duration_sec)`. Synthetic metadata tests passed, but official
VidaForge-3M clips disagreed with that estimate in both directions. One
25 FPS clip recorded as 2.04 seconds decoded to 50 frames instead of 51,
while a clip recorded as 1.74 seconds decoded to 44 frames instead of 43.

An overestimate can produce an out-of-range decoder index. An underestimate
can reject a clip that actually satisfies a target frame count.

## Root Cause

Manifest timing describes segmentation boundaries or rounded duration, not
necessarily the final container's exact indexed frame count. Encoding,
timestamp quantization, and boundary handling can each shift the relationship
between duration and frame count.

## Fix / Workaround

Probe the selected media container and use its indexed video stream metadata.
If the container does not expose a frame count, decode and count frames as a
fallback. Treat manifest width, height, FPS, and duration as validation and
provenance fields rather than exact decoder bounds.

## Prevention

- Include at least one valid video container in manifest-adapter tests.
- Test metadata that deliberately disagrees with the encoded media.
- Before trusting a derived media field, compare it against several real
  upstream artifacts, including non-integral durations.
- Never use `fps * duration` as the upper bound for frame indices.
