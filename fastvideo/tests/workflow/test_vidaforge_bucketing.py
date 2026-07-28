# SPDX-License-Identifier: Apache-2.0
"""CPU golden tests for the pinned VidaForge Stage 5 bucket policy."""

import json
from types import SimpleNamespace

import pytest

from fastvideo.configs.configs import (
    DatasetType,
    PreprocessConfig,
    PreprocessOutputType,
    VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC as CONFIG_DEFAULT_BUCKET_DURATIONS_SEC,
)
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.workflow.preprocess.vidaforge_bucketing import (
    VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC,
    VidaForgeBucket,
    VidaForgeBucketPlanner,
    duration_bucket_frame_counts,
    resolution_pixel_budget,
    resolve_bucket_resolution,
    scaled_forward_batch_size,
    select_bucket_frame_count,
    valid_frame_count_at_or_below,
)

_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _automodel_config(**overrides) -> PreprocessConfig:
    values = {
        "dataset_path": "manifest.parquet",
        "dataset_type": DatasetType.VIDAFORGE,
        "output_type": PreprocessOutputType.VIDAFORGE_AUTOMODEL,
        "vidaforge_model_name": _MODEL_NAME,
        "num_frames": 17,
        "max_height": 144,
        "max_width": 256,
    }
    values.update(overrides)
    return PreprocessConfig(**values)


def test_vidaforge_temporal_bucket_golden_values() -> None:
    durations = list(VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC)

    assert CONFIG_DEFAULT_BUCKET_DURATIONS_SEC == VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC
    assert valid_frame_count_at_or_below(48, stride=4) == 45
    assert valid_frame_count_at_or_below(49, stride=4) == 49
    assert duration_bucket_frame_counts(durations, fps=24, stride=4) == [
        45,
        69,
        93,
        117,
        141,
        189,
        237,
    ]
    assert select_bucket_frame_count(
        source_duration_sec=4.0,
        fps=24,
        durations_sec=durations,
        stride=4,
    ) == 93


def test_vidaforge_temporal_bucket_deduplicates_derived_frame_counts() -> None:
    assert duration_bucket_frame_counts([0.05, 0.06, 0.07], fps=24, stride=4) == [1]


def test_vidaforge_temporal_bucket_rejects_invalid_or_short_input() -> None:
    with pytest.raises(ValueError, match="stride must be > 0"):
        valid_frame_count_at_or_below(1, stride=0)
    with pytest.raises(ValueError, match="frame_count must be >= 1"):
        valid_frame_count_at_or_below(0, stride=4)
    with pytest.raises(ValueError, match="fps must be finite and > 0"):
        duration_bucket_frame_counts([2.0], fps=float("inf"), stride=4)
    with pytest.raises(ValueError, match="shorter than the smallest temporal bucket"):
        select_bucket_frame_count(
            source_duration_sec=1.0,
            fps=24,
            durations_sec=[2.0],
            stride=4,
        )


def test_vidaforge_spatial_bucket_golden_values() -> None:
    assert resolution_pixel_budget("480p") == 409_600
    assert resolve_bucket_resolution(
        source_width=1920,
        source_height=1080,
        resolution="480p",
        size_multiple=16,
        upscale=False,
    ) == (848, 480)
    assert resolve_bucket_resolution(
        source_width=720,
        source_height=1280,
        resolution="480p",
        size_multiple=16,
        upscale=False,
    ) == (480, 848)
    assert resolve_bucket_resolution(
        source_width=320,
        source_height=240,
        resolution="480p",
        size_multiple=16,
        upscale=False,
    ) == (320, 240)
    assert resolve_bucket_resolution(
        source_width=320,
        source_height=240,
        resolution="480p",
        size_multiple=16,
        upscale=True,
    ) == (736, 544)


def test_vidaforge_spatial_bucket_rejects_invalid_alignment() -> None:
    with pytest.raises(ValueError, match="size_multiple must be > 0"):
        resolve_bucket_resolution(
            source_width=1920,
            source_height=1080,
            resolution="480p",
            size_multiple=0,
            upscale=False,
        )


def test_vidaforge_dynamic_forward_batch_golden_values() -> None:
    values = {
        "dynamic_forward_batch_size": 4,
        "reference_frame_count": 237,
        "reference_pixels": 409_600,
        "bucket_resolution": (848, 480),
    }

    assert scaled_forward_batch_size(bucket_frame_count=45, **values) == 21
    assert scaled_forward_batch_size(bucket_frame_count=93, **values) == 10
    assert scaled_forward_batch_size(bucket_frame_count=237, **values) == 4
    assert scaled_forward_batch_size(
        dynamic_forward_batch_size=1,
        reference_frame_count=1,
        reference_pixels=1,
        bucket_frame_count=237,
        bucket_resolution=(848, 480),
    ) == 1


def test_vidaforge_bucket_planner_handles_fastvideo_and_stage4_rows() -> None:
    config = SimpleNamespace(
        vidaforge_bucket_resolution="480p",
        vidaforge_bucket_upscale=False,
        vidaforge_bucket_durations_sec=VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC,
        vidaforge_dynamic_forward_batch_size=4,
    )
    planner = VidaForgeBucketPlanner.from_config(config, reference_fps=24)

    normalized_bucket = planner.bucket_for_item({
        "duration_sec": 4.0,
        "fps": 24.0,
        "resolution": {
            "width": 1920,
            "height": 1080,
        },
    })
    stage4_bucket = planner.bucket_for_item({
        "duration_sec": 4.0,
        "fps": 24.0,
        "width": 1920,
        "height": 1080,
    })

    assert normalized_bucket == VidaForgeBucket(frame_count=93, width=848, height=480)
    assert normalized_bucket.key == (93, 848, 480)
    assert normalized_bucket.to_dict() == {
        "frame_count": 93,
        "width": 848,
        "height": 480,
    }
    assert stage4_bucket == normalized_bucket
    assert planner.forward_batch_size(normalized_bucket) == 10
    assert json.loads(json.dumps(planner.to_producer_config()))["durations_sec"] == [
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
    ]


def test_vidaforge_bucket_planner_requires_reference_fps_for_dynamic_batch() -> None:
    config = SimpleNamespace(
        vidaforge_bucket_resolution="480p",
        vidaforge_bucket_upscale=False,
        vidaforge_bucket_durations_sec=VIDAFORGE_DEFAULT_BUCKET_DURATIONS_SEC,
        vidaforge_dynamic_forward_batch_size=4,
    )
    planner = VidaForgeBucketPlanner.from_config(config)

    with pytest.raises(ValueError, match="reference_fps is required"):
        planner.forward_batch_size(VidaForgeBucket(frame_count=93, width=848, height=480))


def test_vidaforge_bucket_cli_is_canonicalized() -> None:
    parser = FlexibleArgumentParser()
    PreprocessConfig.add_cli_args(parser)
    args = parser.parse_args([
        "--preprocess.dataset-path",
        "manifest",
        "--preprocess.vidaforge-bucket-resolution",
        " 480P ",
        "--preprocess.vidaforge-bucket-upscale",
        "--preprocess.vidaforge-bucket-durations-sec",
        "10",
        "2",
        "2",
        "4",
        "--preprocess.vidaforge-dynamic-forward-batch-size",
        "3",
    ])

    config = PreprocessConfig.from_kwargs(vars(args))

    assert config is not None
    assert config.vidaforge_bucket_resolution == "480p"
    assert config.vidaforge_bucket_upscale is True
    assert config.vidaforge_bucket_durations_sec == (2.0, 4.0, 10.0)
    assert config.vidaforge_dynamic_forward_batch_size == 3


def test_vidaforge_multi_bucket_config_does_not_require_fixed_geometry() -> None:
    config = _automodel_config(
        vidaforge_bucket_resolution="480p",
        num_frames=16,
        max_height=15,
        max_width=15,
    )

    config.check_preprocess_config()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vidaforge_bucket_resolution", "480", "must look like '480p'"),
        ("vidaforge_bucket_durations_sec", (), "must not be empty"),
        ("vidaforge_bucket_durations_sec", (2.0, float("nan")), "finite and greater than 0"),
        ("vidaforge_bucket_upscale", "false", "must be a bool"),
        ("vidaforge_dynamic_forward_batch_size", 0, "must be greater than 0"),
    ],
)
def test_vidaforge_multi_bucket_config_rejects_invalid_policy(field: str, value: object, message: str) -> None:
    config = _automodel_config(vidaforge_bucket_resolution="480p")
    setattr(config, field, value)

    with pytest.raises(ValueError, match=message):
        config.check_preprocess_config()
