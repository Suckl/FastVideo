# SPDX-License-Identifier: Apache-2.0
"""Pure VidaForge Stage 5 bucket planning helpers.

The temporal, spatial, and dynamic-forward-batch formulas in this module
match GAIR-NLP/VidaForge commit 4562d3fbcbd4861fc74c2859950c0237363681bb.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

VIDAFORGE_BUCKETING_REFERENCE_REVISION = "4562d3fbcbd4861fc74c2859950c0237363681bb"
VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def valid_frame_count_at_or_below(frame_count: int, *, stride: int) -> int:
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    return ((int(frame_count) - 1) // int(stride)) * int(stride) + 1


def duration_bucket_frame_counts(durations_sec: list[float], *, fps: float, stride: int) -> list[int]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"fps must be finite and > 0, got {fps!r}")
    if not durations_sec:
        raise ValueError("bucket.durations_sec must not be empty")
    frame_counts = []
    for duration_sec in durations_sec:
        duration = float(duration_sec)
        if duration <= 0 or not math.isfinite(duration):
            raise ValueError(f"bucket.durations_sec values must be finite and > 0, got {duration_sec!r}")
        target_frame_count = max(1, int(math.floor(duration * fps + 1e-9)))
        valid_frame_count = valid_frame_count_at_or_below(target_frame_count, stride=stride)
        if valid_frame_count not in frame_counts:
            frame_counts.append(valid_frame_count)
    frame_counts.sort()
    return frame_counts


def select_bucket_frame_count(
    *,
    source_duration_sec: float,
    fps: float,
    durations_sec: list[float],
    stride: int,
) -> int:
    duration = float(source_duration_sec)
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError(f"duration_sec must be finite and > 0, got {source_duration_sec!r}")
    input_frame_count = int(math.floor(duration * fps + 1e-9))
    if input_frame_count < 1:
        raise ValueError("duration_sec and fps produce no input frames: "
                         f"duration_sec={source_duration_sec!r}, fps={fps!r}")
    candidates = duration_bucket_frame_counts(durations_sec, fps=fps, stride=stride)
    eligible = [frame_count for frame_count in candidates if frame_count <= input_frame_count]
    if not eligible:
        raise ValueError("clip is shorter than the smallest temporal bucket: "
                         f"input_frame_count={input_frame_count}, "
                         f"min_bucket_frame_count={candidates[0]}, fps={fps}")
    return eligible[-1]


def resolution_pixel_budget(resolution: str) -> int:
    text = str(resolution).strip().lower()
    if not text.endswith("p"):
        raise ValueError(f"bucket.resolution must look like '480p', got {resolution!r}")
    reference_height = int(text[:-1])
    if reference_height <= 0:
        raise ValueError("bucket.resolution height must be > 0")
    return int(round(reference_height * reference_height * 16 / 9))


def resolve_bucket_resolution(
    *,
    source_width: int,
    source_height: int,
    resolution: str,
    size_multiple: int,
    upscale: bool,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source width and height must be > 0")
    if size_multiple <= 0:
        raise ValueError("size_multiple must be > 0")
    max_pixels = resolution_pixel_budget(resolution)
    source_pixels = source_width * source_height
    scale = math.sqrt(max_pixels / source_pixels) if upscale or source_pixels > max_pixels else 1.0
    width = int(math.floor(source_width * scale / size_multiple)) * size_multiple
    height = int(math.floor(source_height * scale / size_multiple)) * size_multiple
    if width <= 0 or height <= 0:
        raise ValueError("resolved bucket resolution is empty after size_multiple alignment: "
                         f"source={source_width}x{source_height}, resolution={resolution!r}, "
                         f"size_multiple={size_multiple}")
    if width * height > max_pixels:
        raise ValueError("resolved bucket resolution exceeds pixel budget: "
                         f"resolved={width}x{height}, resolution={resolution!r}, "
                         f"max_pixels={max_pixels}")
    return width, height


def scaled_forward_batch_size(
    *,
    dynamic_forward_batch_size: int,
    reference_frame_count: int,
    reference_pixels: int,
    bucket_frame_count: int,
    bucket_resolution: tuple[int, int],
) -> int:
    if dynamic_forward_batch_size <= 0:
        raise ValueError("dynamic_forward_batch_size must be > 0")
    if reference_frame_count <= 0:
        raise ValueError("reference_frame_count must be > 0")
    if reference_pixels <= 0:
        raise ValueError("reference_pixels must be > 0")
    if bucket_frame_count <= 0:
        raise ValueError("bucket_frame_count must be > 0")
    bucket_width, bucket_height = (
        int(bucket_resolution[0]),
        int(bucket_resolution[1]),
    )
    if bucket_width <= 0 or bucket_height <= 0:
        raise ValueError("bucket_resolution must contain positive width and height")

    reference_cost = reference_frame_count * reference_pixels
    bucket_cost = bucket_frame_count * bucket_width * bucket_height
    return max(
        1,
        int(math.floor(dynamic_forward_batch_size * reference_cost / bucket_cost)),
    )


@dataclass(frozen=True, slots=True)
class VidaForgeBucket:
    """One homogeneous Stage 5 temporal/spatial bucket."""

    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")

    @property
    def key(self) -> tuple[int, int, int]:
        return self.frame_count, self.width, self.height

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class VidaForgeBucketPlanner:
    """Compose the pinned VidaForge formulas without depending on FastVideo config types."""

    resolution: str
    upscale: bool
    durations_sec: tuple[float, ...]
    temporal_stride: int
    input_size_multiple: int
    dynamic_forward_batch_size: int
    reference_fps: float | None = None

    def __post_init__(self) -> None:
        resolution_pixel_budget(self.resolution)
        if not self.durations_sec:
            raise ValueError("bucket.durations_sec must not be empty")
        for duration_sec in self.durations_sec:
            duration = float(duration_sec)
            if duration <= 0 or not math.isfinite(duration):
                raise ValueError(f"bucket.durations_sec values must be finite and > 0, got {duration_sec!r}")
        if self.temporal_stride <= 0:
            raise ValueError("temporal_stride must be > 0")
        if self.input_size_multiple <= 0:
            raise ValueError("input_size_multiple must be > 0")
        if self.dynamic_forward_batch_size <= 0:
            raise ValueError("dynamic_forward_batch_size must be > 0")
        if self.reference_fps is not None and (self.reference_fps <= 0 or not math.isfinite(self.reference_fps)):
            raise ValueError(f"reference_fps must be finite and > 0, got {self.reference_fps!r}")

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        temporal_stride: int = 4,
        input_size_multiple: int = 16,
        reference_fps: float | None = None,
    ) -> VidaForgeBucketPlanner:
        """Build a planner from the duck-typed ``PreprocessConfig`` surface."""
        resolution = str(config.vidaforge_bucket_resolution).strip().lower()
        if not resolution:
            raise ValueError("VidaForge multi-bucket mode is not enabled")
        if not isinstance(config.vidaforge_bucket_upscale, bool):
            raise ValueError("vidaforge_bucket_upscale must be a bool")
        return cls(
            resolution=resolution,
            upscale=config.vidaforge_bucket_upscale,
            durations_sec=tuple(float(value) for value in config.vidaforge_bucket_durations_sec),
            temporal_stride=int(temporal_stride),
            input_size_multiple=int(input_size_multiple),
            dynamic_forward_batch_size=int(config.vidaforge_dynamic_forward_batch_size),
            reference_fps=reference_fps,
        )

    def bucket_for_item(self, item: Mapping[str, Any]) -> VidaForgeBucket:
        """Resolve a normalized FastVideo item or an official Stage 4 row."""
        source_width, source_height = _source_resolution(item)
        frame_count = select_bucket_frame_count(
            source_duration_sec=float(item["duration_sec"]),
            fps=float(item["fps"]),
            durations_sec=list(self.durations_sec),
            stride=self.temporal_stride,
        )
        width, height = resolve_bucket_resolution(
            source_width=source_width,
            source_height=source_height,
            resolution=self.resolution,
            size_multiple=self.input_size_multiple,
            upscale=self.upscale,
        )
        return VidaForgeBucket(frame_count=frame_count, width=width, height=height)

    def forward_batch_size(self, bucket: VidaForgeBucket, *, reference_fps: float | None = None) -> int:
        """Scale an encoder batch using the maximum FPS of the production set."""
        effective_reference_fps = self.reference_fps if reference_fps is None else float(reference_fps)
        if effective_reference_fps is None:
            raise ValueError("reference_fps is required to match VidaForge dynamic forward batching")
        reference_frame_count = max(
            duration_bucket_frame_counts(
                list(self.durations_sec),
                fps=effective_reference_fps,
                stride=self.temporal_stride,
            ))
        return scaled_forward_batch_size(
            dynamic_forward_batch_size=self.dynamic_forward_batch_size,
            reference_frame_count=reference_frame_count,
            reference_pixels=resolution_pixel_budget(self.resolution),
            bucket_frame_count=bucket.frame_count,
            bucket_resolution=bucket.resolution,
        )

    def to_producer_config(self) -> dict[str, object]:
        """Return a canonical JSON-serializable description of the policy."""
        return {
            "resolution": self.resolution,
            "upscale": self.upscale,
            "durations_sec": list(self.durations_sec),
            "dynamic_forward_batch_size": self.dynamic_forward_batch_size,
            "temporal_stride": self.temporal_stride,
            "input_size_multiple": self.input_size_multiple,
            "reference_fps": self.reference_fps,
            "reference_revision": VIDAFORGE_BUCKETING_REFERENCE_REVISION,
        }


def _source_resolution(item: Mapping[str, Any]) -> tuple[int, int]:
    if "width" in item and "height" in item:
        return int(item["width"]), int(item["height"])
    resolution = item.get("resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError("item must contain width/height or a resolution mapping")
    try:
        return int(resolution["width"]), int(resolution["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"item resolution is invalid: {resolution!r}") from exc
