import hashlib
import io
import os
import pickle
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from torch.utils.data import DataLoader

from fastvideo.configs.configs import (
    DatasetType,
    PreprocessConfig,
    PreprocessOutputType,
    VideoLoaderType,
)
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.preprocess.preprocess_stages import (
    VideoTransformStage,
    resolve_file_backed_video_path,
)
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.workflow.preprocess import vidaforge_manifest
from fastvideo.workflow.preprocess.components import (
    VidaForgeWanDataValidator,
    VideoForwardBatchBuilder,
    build_dataset,
)
from fastvideo.workflow.preprocess.preprocess_workflow_t2v import PreprocessWorkflowT2V

_VIDAFORGE_RELEASE_REVISION = "091bdc02d82b8c89a4e4eff54945d286fb328b47"
_VIDAFORGE_HEVC_SMOKE_CLIP_ID = "video-9a2221ec0d47d85c:clip:00002:02"


def _sample_name(clip_id: str) -> str:
    return "vidaforge-" + hashlib.sha256(clip_id.encode("utf-8")).hexdigest()


def _identity_collate(batch):
    return batch


def _keep_all(_row):
    return True


def _drop_all(_row):
    return False


class _SequentialBroadcastGroup:

    def __init__(self):
        self.value = None

    def broadcast_object(self, value, src=0):
        assert src == 0
        if value is not None:
            self.value = value
        return self.value


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _video_path(
    data_root: Path,
    relative_path: str,
    *,
    frame_count: int = 4,
    fps: int = 24,
    width: int = 32,
    height: int = 24,
) -> Path:
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            pixels = np.full((height, width, 3), frame_index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _row(
    *,
    clip_id: str,
    clip_path: str,
    clip_ok: int = 1,
    caption_ok: int = 1,
    select_ok: int = 1,
    select_pass: int = 1,
    caption_level_0: str = "short caption",
    caption_level_3: str = "dense caption",
    width: int = 32,
    height: int = 24,
    fps: float = 24.0,
    duration_sec: float = 4.0,
) -> dict:
    return {
        "clip_id": clip_id,
        "clip_path": clip_path,
        "clip_ok": clip_ok,
        "caption_ok": caption_ok,
        "select_ok": select_ok,
        "select_pass": select_pass,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "caption_level_0": caption_level_0,
        "caption_level_3": caption_level_3,
    }


def _config(manifest_path: Path, *, data_root: Path | None = None, **kwargs) -> PreprocessConfig:
    video_loader_type = kwargs.pop("video_loader_type", VideoLoaderType.TORCHVISION)
    return PreprocessConfig(
        dataset_path=str(manifest_path),
        dataset_type=DatasetType.VIDAFORGE,
        video_loader_type=video_loader_type,
        vidaforge_data_root="" if data_root is None else str(data_root),
        **kwargs,
    )


def _release_row(
    *,
    clip_id: str,
    clip_path: str,
    tar_path: str,
    tar_offset: int,
    filesize_bytes: int,
    sha256: str,
    caption_level_3: str = "dense caption",
    width: int = 32,
    height: int = 24,
    fps: float = 25.0,
    duration_sec: float = 4.0,
) -> dict:
    row = _row(
        clip_id=clip_id,
        clip_path=clip_path,
        caption_level_3=caption_level_3,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
    )
    for field in ("clip_ok", "caption_ok", "select_ok", "select_pass"):
        del row[field]
    row.update({
        "tar_path": tar_path,
        "tar_offset": tar_offset,
        "filesize_bytes": filesize_bytes,
        "sha256": sha256,
    })
    return row


def _indexed_tar(
    data_root: Path,
    source_video: Path,
    *,
    clip_path: str = "00/00/clip.mp4",
    tar_path: str = "data/shard-00000.tar",
) -> tuple[Path, int, int, str]:
    output_path = data_root / tar_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, mode="w") as archive:
        archive.add(source_video, arcname=clip_path)
    with tarfile.open(output_path, mode="r") as archive:
        member = archive.getmember(clip_path)
    payload = source_video.read_bytes()
    return output_path, member.offset_data, member.size, hashlib.sha256(payload).hexdigest()


@pytest.fixture(autouse=True)
def _single_process_world(monkeypatch):
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 1)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: 0)


def test_dataset_type_accepts_vidaforge():
    assert DatasetType.from_string("VIDAFORGE") == DatasetType.VIDAFORGE
    assert "vidaforge" in DatasetType.choices()


def test_manifest_fingerprint_changes_with_manifest_content(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path="a.mp4")])
    first = vidaforge_manifest.build_vidaforge_manifest_fingerprint(manifest_path)

    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path="a.mp4", caption_level_3="updated")])
    second = vidaforge_manifest.build_vidaforge_manifest_fingerprint(manifest_path)

    assert len(first) == 64
    assert first != second


@pytest.mark.parametrize(
    ("decoded_frames", "expected"),
    [
        (17, True),
        (14, True),
        (13, False),
    ],
)
def test_vidaforge_wan_validator_uses_decoded_frame_contract(decoded_frames: int, expected: bool) -> None:
    validator = VidaForgeWanDataValidator(num_frames=17)
    row = {
        "caption": "caption",
        "fps": 120.0,
        "num_frames": decoded_frames,
        "resolution": {
            "width": 256,
            "height": 144,
        },
    }

    assert validator(row) is expected


def test_vidaforge_multibucket_validator_defers_temporal_choice_to_planner() -> None:
    validator = VidaForgeWanDataValidator(num_frames=237, multi_bucket=True)
    row = {
        "caption": "caption",
        "fps": 24.0,
        "num_frames": 49,
        "resolution": {
            "width": 256,
            "height": 144,
        },
    }

    assert validator(row) is True
    assert "decoded_frame_validator" not in validator.validators


def test_vidaforge_cli_options_populate_preprocess_config():
    parser = FlexibleArgumentParser()
    PreprocessConfig.add_cli_args(parser)
    args = parser.parse_args([
        "--preprocess.dataset-path",
        "manifest",
        "--preprocess.dataset-type",
        "vidaforge",
        "--preprocess.vidaforge-data-root",
        "data-root",
        "--preprocess.vidaforge-materialize-dir",
        "materialized",
        "--preprocess.vidaforge-caption-field",
        "caption_level_2",
        "--preprocess.vidaforge-selection",
        "reject",
        "--preprocess.video-loader-type",
        "torchcodec",
        "--preprocess.output-type",
        "vidaforge_automodel",
        "--preprocess.vidaforge-model-name",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--preprocess.num-frames",
        "17",
        "--preprocess.max-height",
        "144",
        "--preprocess.max-width",
        "256",
    ])

    config = PreprocessConfig.from_kwargs(vars(args))

    assert config is not None
    assert config.dataset_type == DatasetType.VIDAFORGE
    assert isinstance(config.dataset_type, DatasetType)
    assert config.video_loader_type == VideoLoaderType.TORCHCODEC
    assert isinstance(config.video_loader_type, VideoLoaderType)
    assert config.output_type == PreprocessOutputType.VIDAFORGE_AUTOMODEL
    assert isinstance(config.output_type, PreprocessOutputType)
    assert config.dataset_path == "manifest"
    assert config.vidaforge_data_root == "data-root"
    assert config.vidaforge_materialize_dir == "materialized"
    assert config.vidaforge_caption_field == "caption_level_2"
    assert config.vidaforge_selection == "reject"


def test_build_vidaforge_dataset_normalizes_relative_clip_path(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    relative_path = "data/stage2_segmentation/step2_clip/run_id_demo/ab/cd/clip.mp4"
    expected_video_path = _video_path(data_root, relative_path)
    manifest_path = tmp_path / "manifest" / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=relative_path)])

    dataset = build_dataset(_config(manifest_path, data_root=data_root), split="train", validator=lambda row: True)

    assert list(dataset) == [{
        "video": str(expected_video_path.resolve()),
        "name": _sample_name("clip-a"),
        "clip_id": "clip-a",
        "resolution": {
            "width": 32,
            "height": 24,
        },
        "fps": 24.0,
        "num_frames": 4,
        "caption": "dense caption",
    }]


def test_build_vidaforge_dataset_accepts_absolute_clip_path_without_data_root(tmp_path: Path):
    video_path = _video_path(tmp_path, "absolute.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])

    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    assert next(iter(dataset))["video"] == str(video_path)


def test_build_vidaforge_dataset_filters_status_and_empty_caption(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    rows = []
    for clip_id, overrides in [
        ("keep", {}),
        ("clip-failed", {
            "clip_ok": 0
        }),
        ("caption-failed", {
            "caption_ok": 0
        }),
        ("selection-failed", {
            "select_ok": 0,
            "select_pass": 0,
        }),
        ("selection-rejected", {
            "select_pass": 0
        }),
        ("empty-caption", {
            "caption_level_3": " "
        }),
    ]:
        relative_path = f"data/{clip_id}.mp4"
        _video_path(data_root, relative_path)
        rows.append(_row(clip_id=clip_id, clip_path=relative_path, **overrides))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)

    dataset = build_dataset(_config(manifest_path, data_root=data_root), split="train", validator=lambda row: True)

    assert [row["name"] for row in dataset] == [_sample_name("keep")]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip_ok", 1.9),
        ("caption_ok", 1.5),
        ("select_ok", float("nan")),
        ("select_pass", float("inf")),
        ("clip_ok", "1"),
        ("caption_ok", True),
        ("select_ok", 1.0),
    ],
)
def test_build_vidaforge_dataset_rejects_non_integer_status(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = _row(
        clip_id="invalid-status",
        clip_path="data/clip.mp4",
    )
    row[field] = value
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [row])

    with pytest.raises(ValueError, match=rf"requires integer {field}"):
        build_dataset(
            _config(manifest_path, data_root=tmp_path),
            split="train",
            validator=lambda sample: True,
        )


@pytest.mark.parametrize(
    ("overrides", "selection", "invalid_field"),
    [
        ({"clip_ok": 0, "caption_ok": 1.5}, "auto", "caption_ok"),
        ({"select_pass": 1.5}, "all", "select_pass"),
        (
            {
                "caption_level_3": "",
                "select_ok": 1.5,
            },
            "auto",
            "select_ok",
        ),
    ],
)
def test_build_vidaforge_dataset_validates_all_statuses_before_filtering(
    tmp_path: Path,
    overrides: dict[str, object],
    selection: str,
    invalid_field: str,
) -> None:
    row = _row(
        clip_id="invalid-short-circuit-status",
        clip_path="data/clip.mp4",
    )
    row.update(overrides)
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [row])

    with pytest.raises(
        ValueError,
        match=rf"requires integer {invalid_field}",
    ):
        build_dataset(
            _config(
                manifest_path,
                data_root=tmp_path,
                vidaforge_selection=selection,
            ),
            split="train",
            validator=lambda sample: True,
        )


@pytest.mark.parametrize(
    "clip_id",
    [
        "../../target",
        "/absolute/target",
        r"C:\absolute\target",
        _VIDAFORGE_HEVC_SMOKE_CLIP_ID,
        "CON",
        "trailing-dot.",
    ],
)
def test_build_vidaforge_dataset_encodes_unsafe_clip_id(
    tmp_path: Path,
    clip_id: str,
) -> None:
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [_row(clip_id=clip_id, clip_path=str(video_path))],
    )

    dataset = build_dataset(
        _config(manifest_path),
        split="train",
        validator=lambda sample: True,
    )
    sample_name = next(iter(dataset))["name"]

    assert sample_name == _sample_name(clip_id)
    assert "/" not in sample_name
    assert "\\" not in sample_name
    assert ":" not in sample_name


def test_build_vidaforge_dataset_output_names_do_not_fold_extensions_or_case(
    tmp_path: Path,
) -> None:
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    collision_target = "collision-target"
    clip_ids = [
        "foo.mp4",
        "foo.webm",
        "clip",
        "CLIP",
        collision_target,
        _sample_name(collision_target),
    ]
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _row(clip_id=clip_id, clip_path=str(video_path))
            for clip_id in clip_ids
        ],
    )

    dataset = build_dataset(
        _config(manifest_path),
        split="train",
        validator=lambda sample: True,
    )
    sample_names = [row["name"] for row in dataset]

    assert sample_names == [_sample_name(clip_id) for clip_id in clip_ids]
    assert len(set(sample_names)) == len(clip_ids)
    assert len({name.casefold() for name in sample_names}) == len(clip_ids)


@pytest.mark.parametrize("clip_id", [" clip", "clip ", "\tclip", "clip\n"])
def test_build_vidaforge_dataset_rejects_clip_id_whitespace(
    tmp_path: Path,
    clip_id: str,
) -> None:
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [_row(clip_id=clip_id, clip_path="data/clip.mp4")],
    )

    with pytest.raises(
        ValueError,
        match="leading or trailing whitespace",
    ):
        build_dataset(
            _config(manifest_path, data_root=tmp_path),
            split="train",
            validator=lambda sample: True,
        )


def test_build_vidaforge_dataset_rejects_manifest_without_eligible_rows(tmp_path: Path):
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [_row(clip_id="clip-failed", clip_path="data/clip.mp4", clip_ok=0)],
    )

    with pytest.raises(ValueError, match="produced no eligible rows"):
        build_dataset(_config(manifest_path, data_root=tmp_path), split="train", validator=lambda row: True)


@pytest.mark.parametrize(
    ("selection", "expected_names"),
    [
        ("auto", ["passed"]),
        ("pass", ["passed"]),
        ("reject", ["rejected"]),
        ("all", ["passed", "rejected"]),
    ],
)
def test_build_vidaforge_dataset_selection_modes(
    tmp_path: Path,
    selection: str,
    expected_names: list[str],
):
    data_root = tmp_path / "vidaforge"
    rows = []
    for clip_id, select_ok, select_pass in [
        ("passed", 1, 1),
        ("rejected", 1, 0),
        ("selection-error", 0, 0),
    ]:
        relative_path = f"data/{clip_id}.mp4"
        _video_path(data_root, relative_path)
        rows.append(
            _row(
                clip_id=clip_id,
                clip_path=relative_path,
                select_ok=select_ok,
                select_pass=select_pass,
            ))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)

    dataset = build_dataset(
        _config(manifest_path, data_root=data_root, vidaforge_selection=selection),
        split="train",
        validator=lambda row: True,
    )

    assert [row["name"] for row in dataset] == [
        _sample_name(clip_id) for clip_id in expected_names
    ]


def test_build_vidaforge_dataset_uses_configured_caption_field(tmp_path: Path):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])

    dataset = build_dataset(
        _config(manifest_path, vidaforge_caption_field="caption_level_0"),
        split="train",
        validator=lambda row: True,
    )

    assert next(iter(dataset))["caption"] == "short caption"


def test_build_vidaforge_dataset_reads_only_direct_parquet_shards(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    manifest_dir = tmp_path / "manifest"
    rows = []
    for index in range(2):
        relative_path = f"data/clip-{index}.mp4"
        _video_path(data_root, relative_path)
        row = _row(clip_id=f"clip-{index}", clip_path=relative_path)
        rows.append(row)
        _write_parquet(manifest_dir / f"clip-{index:05d}.parquet", [row])
    _write_parquet(manifest_dir / "pass" / "clip-00000.parquet", rows)

    dataset = build_dataset(_config(manifest_dir, data_root=data_root), split="train", validator=lambda row: True)

    assert [row["name"] for row in dataset] == [
        _sample_name("clip-0"),
        _sample_name("clip-1"),
    ]


def test_build_vidaforge_dataset_rejects_missing_required_columns(tmp_path: Path):
    manifest_path = tmp_path / "clip-00000.parquet"
    row = _row(clip_id="clip-a", clip_path="data/clip.mp4")
    del row["width"]
    _write_parquet(manifest_path, [row])

    with pytest.raises(ValueError, match="missing required columns: width"):
        build_dataset(_config(manifest_path, data_root=tmp_path), split="train", validator=lambda row: True)


def test_build_vidaforge_dataset_requires_data_root_for_relative_path(tmp_path: Path):
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path="data/clip.mp4")])

    with pytest.raises(ValueError, match="set preprocess.vidaforge-data-root"):
        list(build_dataset(_config(manifest_path), split="train", validator=lambda row: True))


def test_build_vidaforge_dataset_rejects_missing_clip_file(tmp_path: Path):
    manifest_path = tmp_path / "clip-00000.parquet"
    missing_path = (tmp_path / "missing.mp4").resolve()
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(missing_path))])

    with pytest.raises(FileNotFoundError, match="clip_path does not exist"):
        list(build_dataset(_config(manifest_path), split="train", validator=lambda row: True))


def test_build_vidaforge_dataset_rejects_path_outside_data_root(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    data_root.mkdir()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path="../outside.mp4")])

    with pytest.raises(ValueError, match="has clip_path outside"):
        list(
            build_dataset(
                _config(manifest_path, data_root=data_root),
                split="train",
                validator=lambda row: True,
            ))


def test_build_vidaforge_dataset_uses_decoded_media_metadata(tmp_path: Path):
    video_path = _video_path(
        tmp_path,
        "clip.mp4",
        frame_count=50,
        fps=25,
        width=40,
        height=30,
    ).resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _row(
                clip_id="clip-a",
                clip_path=str(video_path),
                width=40,
                height=30,
                fps=25.0,
                duration_sec=2.04,
            )
        ],
    )

    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    row = next(iter(dataset))
    assert row["resolution"] == {
        "width": 40,
        "height": 30,
    }
    assert row["fps"] == 25.0
    assert row["num_frames"] == 50
    assert row["duration_sec"] == 2.04
    assert row["vidaforge_manifest_fps"] == 25.0
    assert row["vidaforge_manifest_resolution"] == {
        "width": 40,
        "height": 30,
    }


def test_normalize_row_preserves_manifest_duration(tmp_path: Path, monkeypatch) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.touch()
    monkeypatch.setattr(
        vidaforge_manifest,
        "_probe_video",
        lambda *_args, **_kwargs: {
            "width": 40,
            "height": 30,
            "fps": 25.0,
            "num_frames": 50,
        },
    )

    normalized = vidaforge_manifest._normalize_row(
        _row(
            clip_id="clip-a",
            clip_path=str(clip_path),
            width=40,
            height=30,
            duration_sec=2.04,
        ),
        caption_field="caption_level_3",
        manifest_kind=vidaforge_manifest.VidaForgeManifestKind.STAGE4,
        data_root=None,
        materialize_root=None,
        video_loader_type=VideoLoaderType.TORCHVISION,
    )

    assert normalized["duration_sec"] == 2.04
    assert normalized["resolution"] == {
        "width": 40,
        "height": 30,
    }
    assert normalized["vidaforge_manifest_fps"] == 24.0
    assert normalized["vidaforge_manifest_resolution"] == {
        "width": 40,
        "height": 30,
    }


def test_normalize_row_rejects_manifest_resolution_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.touch()
    monkeypatch.setattr(
        vidaforge_manifest,
        "_probe_video",
        lambda *_args, **_kwargs: {
            "width": 40,
            "height": 30,
            "fps": 25.0,
            "num_frames": 50,
        },
    )

    with pytest.raises(ValueError, match="does not match decoded media"):
        vidaforge_manifest._normalize_row(
            _row(
                clip_id="clip-a",
                clip_path=str(clip_path),
                width=1280,
                height=720,
            ),
            caption_field="caption_level_3",
            manifest_kind=vidaforge_manifest.VidaForgeManifestKind.STAGE4,
            data_root=None,
            materialize_root=None,
            video_loader_type=VideoLoaderType.TORCHVISION,
        )


def test_build_vidaforge_stage4_defers_media_probe_until_iteration(tmp_path: Path, monkeypatch):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])
    probe_spy = MagicMock(wraps=vidaforge_manifest._probe_video)
    monkeypatch.setattr(vidaforge_manifest, "_probe_video", probe_spy)

    dataset = build_dataset(_config(manifest_path), split="train", validator=_keep_all)

    probe_spy.assert_not_called()
    assert next(iter(dataset))["name"] == _sample_name("clip-a")
    probe_spy.assert_called_once()


def test_build_vidaforge_release_materializes_indexed_tar_clip(tmp_path: Path):
    source_video = _video_path(
        tmp_path,
        "source.mp4",
        frame_count=5,
        fps=25,
        width=40,
        height=30,
    )
    data_root = tmp_path / "VidaForge-3M"
    clip_path = "00/00/clip.mp4"
    tar_path = "data/shard-00000.tar"
    tar_file, tar_offset, filesize_bytes, sha256 = _indexed_tar(
        data_root,
        source_video,
        clip_path=clip_path,
        tar_path=tar_path,
    )
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path=clip_path,
                tar_path=tar_path,
                tar_offset=tar_offset,
                filesize_bytes=filesize_bytes,
                sha256=sha256,
                width=40,
                height=30,
                duration_sec=2.04,
            )
        ],
    )
    materialize_dir = tmp_path / "materialized"

    dataset = build_dataset(
        _config(
            manifest_path,
            data_root=data_root,
            vidaforge_materialize_dir=str(materialize_dir),
        ),
        split="train",
        validator=lambda row: True,
    )

    assert not materialize_dir.exists()
    rows = list(dataset)
    assert len(rows) == 1
    materialized_path = Path(rows[0]["video"])
    assert materialized_path.read_bytes() == source_video.read_bytes()
    assert materialized_path.parent == materialize_dir.resolve() / sha256[:2]
    assert materialized_path.name == f"{sha256}.mp4"
    assert rows[0]["name"] == _sample_name("release-clip")
    assert rows[0]["resolution"] == {
        "width": 40,
        "height": 30,
    }
    assert rows[0]["fps"] == 25.0
    assert rows[0]["num_frames"] == 5
    tar_file.unlink()
    cached_rows = list(dataset)
    assert cached_rows[0]["video"] == str(materialized_path)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="TorchCodec on Windows requires a separately installed full-shared FFmpeg build",
)
def test_build_vidaforge_release_streams_verified_bytes_for_torchcodec(tmp_path: Path):
    source_video = _video_path(tmp_path, "source.mp4", frame_count=5, fps=25)
    data_root = tmp_path / "VidaForge-3M"
    _, tar_offset, filesize_bytes, sha256 = _indexed_tar(data_root, source_video)
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path="00/00/clip.mp4",
                tar_path="data/shard-00000.tar",
                tar_offset=tar_offset,
                filesize_bytes=filesize_bytes,
                sha256=sha256,
            )
        ],
    )
    output_dir = tmp_path / "output"

    dataset = build_dataset(
        _config(
            manifest_path,
            data_root=data_root,
            dataset_output_dir=str(output_dir),
            video_loader_type=VideoLoaderType.TORCHCODEC,
        ),
        split="train",
        validator=lambda row: True,
    )

    assert not output_dir.exists()
    rows = list(dataset)
    video = rows[0]["video"]
    assert isinstance(video, vidaforge_manifest.VidaForgeTorchCodecVideo)
    assert video.source == source_video.read_bytes()
    decoded = video.get_frames_at(np.asarray([0, 1], dtype=np.int64)).data
    assert tuple(decoded.shape) == (2, 3, 24, 32)
    restored_video = pickle.loads(pickle.dumps(video))
    assert restored_video.source == video.source
    assert not output_dir.exists()


def test_ltx2_audio_resolves_file_backed_vidaforge_torchcodec_source(tmp_path: Path):
    video_path = str(_video_path(tmp_path, "source.mp4").resolve())

    file_backed = vidaforge_manifest.VidaForgeTorchCodecVideo(video_path)
    in_memory = vidaforge_manifest.VidaForgeTorchCodecVideo(b"encoded-video")

    assert resolve_file_backed_video_path(file_backed) == video_path
    assert resolve_file_backed_video_path(in_memory) is None


def test_vidaforge_wan_decode_uses_official_cuda_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    frames = np.arange(5 * 3 * 2 * 2, dtype=np.uint8).reshape(5, 3, 2, 2)

    class _Decoder:

        def __init__(self, source: object, **kwargs: object) -> None:
            captured["source"] = source
            captured["kwargs"] = kwargs

        def __len__(self) -> int:
            return len(frames)

        def get_frames_at(self, indices):
            index_values = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            captured["indices"] = index_values
            return SimpleNamespace(data=torch.from_numpy(frames[index_values]))

    backend = MagicMock(return_value=nullcontext())
    torchcodec_module = ModuleType("torchcodec")
    decoders_module = ModuleType("torchcodec.decoders")
    decoders_module.VideoDecoder = _Decoder  # type: ignore[attr-defined]
    decoders_module.set_cuda_backend = backend  # type: ignore[attr-defined]
    torchcodec_module.decoders = decoders_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders_module)

    decoded = vidaforge_manifest.VidaForgeTorchCodecVideo("clip.mp4").get_vidaforge_wan_frames(3)

    backend.assert_called_once_with("beta")
    assert captured["source"] == "clip.mp4"
    assert captured["kwargs"] == {
        "dimension_order": "NCHW",
        "device": "cuda",
        "seek_mode": "exact",
        "num_ffmpeg_threads": 1,
    }
    assert captured["indices"] == [0, 2, 4]
    assert decoded.is_contiguous()


@pytest.mark.skipif(
    os.environ.get("VIDAFORGE_RUN_OFFICIAL_HEVC_SMOKE") != "1",
    reason="set VIDAFORGE_RUN_OFFICIAL_HEVC_SMOKE=1 to download and decode one official clip",
)
def test_vidaforge_official_hevc_torchcodec_pipeline_smoke(tmp_path: Path):
    metadata_path = hf_hub_download(
        repo_id="VidaForge/VidaForge-3M",
        filename="meta/shard-00000.parquet",
        repo_type="dataset",
        revision=_VIDAFORGE_RELEASE_REVISION,
    )
    official_rows = pq.read_table(
        metadata_path,
        filters=[("clip_id", "=", _VIDAFORGE_HEVC_SMOKE_CLIP_ID)],
    ).to_pylist()
    assert len(official_rows) == 1
    official_row = official_rows[0]

    remote_tar_path = (
        f"datasets/VidaForge/VidaForge-3M@{_VIDAFORGE_RELEASE_REVISION}/{official_row['tar_path']}")
    with HfFileSystem().open(remote_tar_path, "rb") as tar_handle:
        tar_handle.seek(official_row["tar_offset"])
        payload_chunks = []
        remaining = official_row["filesize_bytes"]
        while remaining > 0:
            chunk = tar_handle.read(remaining)
            assert chunk
            payload_chunks.append(chunk)
            remaining -= len(chunk)
    payload = b"".join(payload_chunks)
    assert len(payload) == official_row["filesize_bytes"]
    assert hashlib.sha256(payload).hexdigest() == official_row["sha256"]
    with av.open(io.BytesIO(payload)) as container:
        assert container.streams.video[0].codec_context.name == "hevc"

    data_root = tmp_path / "VidaForge-3M"
    sample_tar_path = data_root / "data" / "official-hevc-sample.tar"
    sample_tar_path.parent.mkdir(parents=True)
    sample_tar_path.write_bytes(b"\0" * 512 + payload)
    smoke_row = dict(official_row)
    smoke_row["tar_path"] = "data/official-hevc-sample.tar"
    smoke_row["tar_offset"] = 512
    manifest_path = data_root / "meta" / "official-hevc-sample.parquet"
    _write_parquet(manifest_path, [smoke_row])
    output_dir = tmp_path / "output"
    config = _config(
        manifest_path,
        data_root=data_root,
        dataset_output_dir=str(output_dir),
        video_loader_type=VideoLoaderType.TORCHCODEC,
        dataloader_num_workers=1,
        preprocess_video_batch_size=1,
    )
    dataset = build_dataset(config, split="train", validator=_keep_all)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=_identity_collate,
    )
    raw_batch = next(iter(dataloader))
    preprocess_batch = VideoForwardBatchBuilder(seed=7)(raw_batch)
    stage = VideoTransformStage(
        train_fps=25,
        num_frames=4,
        max_height=64,
        max_width=64,
        do_temporal_sample=False,
    )
    result = stage.forward(
        preprocess_batch,
        SimpleNamespace(
            preprocess_config=config,
            workload_type=WorkloadType.T2V,
        ),
    )

    assert result.latents is not None
    assert tuple(result.latents.shape) == (1, 3, 4, 64, 64)
    assert not output_dir.exists()


def test_build_vidaforge_release_uses_extracted_clip(tmp_path: Path):
    data_root = tmp_path / "VidaForge-3M"
    clip_path = "00/00/clip.mp4"
    extracted_video = _video_path(
        data_root / "data",
        clip_path,
        frame_count=6,
        fps=25,
    )
    payload = extracted_video.read_bytes()
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path=clip_path,
                tar_path="data/missing.tar",
                tar_offset=512,
                filesize_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        ],
    )

    dataset = build_dataset(
        _config(manifest_path, data_root=data_root),
        split="train",
        validator=lambda row: True,
    )

    rows = list(dataset)
    assert rows[0]["video"] == str(extracted_video.resolve())
    assert rows[0]["num_frames"] == 6


@pytest.mark.parametrize("selection", ["pass", "reject"])
def test_build_vidaforge_release_rejects_unavailable_selection_partition(
    tmp_path: Path,
    selection: str,
):
    manifest_path = tmp_path / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path="00/00/clip.mp4",
                tar_path="data/shard-00000.tar",
                tar_offset=512,
                filesize_bytes=100,
                sha256="0" * 64,
            )
        ],
    )

    with pytest.raises(ValueError, match="select_pass is intentionally absent"):
        build_dataset(
            _config(manifest_path, data_root=tmp_path, vidaforge_selection=selection),
            split="train",
            validator=lambda row: True,
        )


def test_build_vidaforge_release_rejects_checksum_mismatch(tmp_path: Path):
    source_video = _video_path(tmp_path, "source.mp4")
    data_root = tmp_path / "VidaForge-3M"
    _, tar_offset, filesize_bytes, _ = _indexed_tar(data_root, source_video)
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path="00/00/clip.mp4",
                tar_path="data/shard-00000.tar",
                tar_offset=tar_offset,
                filesize_bytes=filesize_bytes,
                sha256="0" * 64,
            )
        ],
    )

    dataset = build_dataset(
        _config(
            manifest_path,
            data_root=data_root,
            vidaforge_materialize_dir=str(tmp_path / "materialized"),
        ),
        split="train",
        validator=lambda row: True,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(dataset)


def test_build_vidaforge_release_requires_data_root(tmp_path: Path):
    manifest_path = tmp_path / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path="00/00/clip.mp4",
                tar_path="data/shard-00000.tar",
                tar_offset=512,
                filesize_bytes=100,
                sha256="0" * 64,
            )
        ],
    )

    with pytest.raises(ValueError, match="requires preprocess.vidaforge-data-root"):
        build_dataset(_config(manifest_path), split="train", validator=lambda row: True)


def test_build_vidaforge_dataset_rejects_more_ranks_than_eligible_rows(tmp_path: Path, monkeypatch):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 2)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: 0)
    monkeypatch.setattr(vidaforge_manifest, "get_world_group", lambda: _SequentialBroadcastGroup())

    with pytest.raises(ValueError, match="requires at least one eligible row per rank"):
        build_dataset(
            _config(manifest_path, dataset_output_dir=str(tmp_path / "output")),
            split="train",
            validator=lambda row: True,
        )


def test_vidaforge_metadata_is_filtered_and_validated_globally_before_rank_sharding(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "vidaforge"
    rows = []
    for index in range(4):
        relative_path = f"data/clip-{index}.mp4"
        _video_path(data_root, relative_path)
        row = _row(clip_id=f"clip-{index}", clip_path=relative_path)
        row["unused_large_column"] = "unused" * 100
        rows.append(row)
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)
    current_rank = {
        "value": 0,
    }
    broadcast_group = _SequentialBroadcastGroup()
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 2)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: current_rank["value"])
    monkeypatch.setattr(vidaforge_manifest, "get_world_group", lambda: broadcast_group)
    checked_lengths = []

    def _record_unique_check(dataset, column):
        checked_lengths.append((column, len(dataset)))

    monkeypatch.setattr(vidaforge_manifest, "_validate_unique_column", _record_unique_check)

    config = _config(
        manifest_path,
        data_root=data_root,
        dataset_output_dir=str(tmp_path / "output"),
    )
    rank_zero_dataset = build_dataset(
        config,
        split="train",
        validator=_keep_all,
    )
    current_rank["value"] = 1
    rank_one_dataset = build_dataset(config, split="train", validator=_keep_all)

    assert checked_lengths == [("clip_id", 4)]
    assert "unused_large_column" not in rank_zero_dataset.metadata.column_names
    assert "unused_large_column" not in rank_one_dataset.metadata.column_names
    assert [row["name"] for row in rank_zero_dataset] == [
        _sample_name("clip-0"),
        _sample_name("clip-1"),
    ]
    assert [row["name"] for row in rank_one_dataset] == [
        _sample_name("clip-2"),
        _sample_name("clip-3"),
    ]


def test_vidaforge_global_filter_balances_eligible_rows_across_ranks(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "vidaforge"
    rows = []
    for index in range(4):
        relative_path = f"data/clip-{index}.mp4"
        _video_path(data_root, relative_path)
        rows.append(
            _row(
                clip_id=f"clip-{index}",
                clip_path=relative_path,
                select_pass=0 if index < 2 else 1,
            ))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)
    current_rank = {
        "value": 0,
    }
    broadcast_group = _SequentialBroadcastGroup()
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 2)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: current_rank["value"])
    monkeypatch.setattr(vidaforge_manifest, "get_world_group", lambda: broadcast_group)
    config = _config(
        manifest_path,
        data_root=data_root,
        dataset_output_dir=str(tmp_path / "output"),
    )

    rank_zero_dataset = build_dataset(config, split="train", validator=_keep_all)
    current_rank["value"] = 1
    rank_one_dataset = build_dataset(config, split="train", validator=_keep_all)

    assert [row["name"] for row in rank_zero_dataset] == [
        _sample_name("clip-2")
    ]
    assert [row["name"] for row in rank_one_dataset] == [
        _sample_name("clip-3")
    ]


def test_vidaforge_global_unique_check_catches_duplicate_across_rank_boundary(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "vidaforge"
    rows = []
    clip_ids = ["duplicate", "clip-1", "duplicate", "clip-3"]
    for index, clip_id in enumerate(clip_ids):
        relative_path = f"data/clip-{index}.mp4"
        _video_path(data_root, relative_path)
        rows.append(_row(clip_id=clip_id, clip_path=relative_path))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 2)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: 0)
    monkeypatch.setattr(vidaforge_manifest, "get_world_group", lambda: _SequentialBroadcastGroup())

    with pytest.raises(ValueError, match="duplicate clip_id"):
        build_dataset(
            _config(
                manifest_path,
                data_root=data_root,
                dataset_output_dir=str(tmp_path / "output"),
            ),
            split="train",
            validator=_keep_all,
        )


def test_build_vidaforge_release_shards_rows_across_dataloader_workers(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "VidaForge-3M"
    rows = []
    for index in range(4):
        clip_path = f"00/00/clip-{index}.mp4"
        video_path = _video_path(data_root / "data", clip_path)
        payload = video_path.read_bytes()
        rows.append(
            _release_row(
                clip_id=f"release-{index}",
                clip_path=clip_path,
                tar_path="data/missing.tar",
                tar_offset=512,
                filesize_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ))
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(manifest_path, rows)
    monkeypatch.setattr(vidaforge_manifest, "get_worker_info", lambda: SimpleNamespace(num_workers=2, id=1))

    dataset = build_dataset(
        _config(manifest_path, data_root=data_root),
        split="train",
        validator=lambda row: True,
    )

    assert [row["name"] for row in dataset] == [
        _sample_name("release-2"),
        _sample_name("release-3"),
    ]


def test_vidaforge_workflow_rejects_all_rows_dropped_by_dataloader_workers(tmp_path: Path):
    data_root = tmp_path / "VidaForge-3M"
    clip_path = "00/00/clip.mp4"
    extracted_video = _video_path(data_root / "data", clip_path)
    payload = extracted_video.read_bytes()
    manifest_path = data_root / "meta" / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id="release-clip",
                clip_path=clip_path,
                tar_path="data/missing.tar",
                tar_offset=512,
                filesize_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        ],
    )
    dataset = build_dataset(
        _config(manifest_path, data_root=data_root),
        split="train",
        validator=_drop_all,
    )
    workflow = object.__new__(PreprocessWorkflowT2V)
    workflow.training_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=_identity_collate,
    )
    workflow.processed_dataset_saver = MagicMock()

    with pytest.raises(ValueError, match="produced no samples on this rank"):
        workflow.run()

    workflow.processed_dataset_saver.flush_tables.assert_called_once()
    workflow.processed_dataset_saver.clean_up.assert_called_once()


def test_build_vidaforge_dataset_rejects_duplicate_clip_ids(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    first_path = "data/first.mp4"
    second_path = "data/second.mp4"
    _video_path(data_root, first_path)
    _video_path(data_root, second_path)
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _row(clip_id="duplicate", clip_path=first_path),
            _row(clip_id="duplicate", clip_path=second_path),
        ],
    )

    with pytest.raises(ValueError, match="duplicate clip_id"):
        build_dataset(_config(manifest_path, data_root=data_root), split="train", validator=lambda row: True)


def test_build_vidaforge_release_rejects_duplicate_materialization_paths(tmp_path: Path):
    manifest_path = tmp_path / "shard-00000.parquet"
    _write_parquet(
        manifest_path,
        [
            _release_row(
                clip_id=f"release-{index}",
                clip_path="00/00/shared.mp4",
                tar_path="data/shard-00000.tar",
                tar_offset=512 + index * 100,
                filesize_bytes=100,
                sha256=str(index) * 64,
            )
            for index in range(2)
        ],
    )

    with pytest.raises(ValueError, match="duplicate clip_path"):
        build_dataset(_config(manifest_path, data_root=tmp_path), split="train", validator=lambda row: True)


def test_build_vidaforge_dataset_has_no_validation_split(tmp_path: Path):
    config = _config(tmp_path / "unused.parquet")

    with pytest.raises(ValueError, match="only the train split"):
        build_dataset(config, split="validation", validator=lambda row: True)


def test_build_vidaforge_dataset_applies_fastvideo_validator(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    rows = []
    for clip_id in ["keep", "drop"]:
        relative_path = f"data/{clip_id}.mp4"
        _video_path(data_root, relative_path)
        rows.append(_row(clip_id=clip_id, clip_path=relative_path))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)

    dataset = build_dataset(
        _config(manifest_path, data_root=data_root),
        split="train",
        validator=lambda row: row["name"] == _sample_name("keep"),
    )

    assert [row["name"] for row in dataset] == [_sample_name("keep")]


def test_normalized_vidaforge_row_builds_preprocess_batch(tmp_path: Path):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])
    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    batch = VideoForwardBatchBuilder(seed=7)([next(iter(dataset))])

    assert batch.video_loader == [str(video_path)]
    assert batch.video_file_name == [_sample_name("clip-a")]
    assert batch.height == [24]
    assert batch.width == [32]
    assert batch.fps == [24.0]
    assert batch.num_frames == [4]
    assert batch.prompt == ["dense caption"]


def test_vidaforge_config_rejects_invalid_selection():
    config = PreprocessConfig(
        dataset_path="manifest",
        dataset_type=DatasetType.VIDAFORGE,
        vidaforge_selection="invalid",
    )

    with pytest.raises(ValueError, match="vidaforge_selection must be one of"):
        config.check_preprocess_config()
