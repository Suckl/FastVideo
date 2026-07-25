import hashlib
import tarfile
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastvideo.configs.configs import DatasetType, PreprocessConfig, VideoLoaderType
from fastvideo.utils import FlexibleArgumentParser
from fastvideo.workflow.preprocess import vidaforge_manifest
from fastvideo.workflow.preprocess.components import VideoForwardBatchBuilder, build_dataset


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
    select_pass: int = 1,
    caption_level_0: str = "short caption",
    caption_level_3: str = "dense caption",
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
    duration_sec: float = 4.0,
) -> dict:
    return {
        "clip_id": clip_id,
        "clip_path": clip_path,
        "clip_ok": clip_ok,
        "caption_ok": caption_ok,
        "select_pass": select_pass,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "caption_level_0": caption_level_0,
        "caption_level_3": caption_level_3,
    }


def _config(manifest_path: Path, *, data_root: Path | None = None, **kwargs) -> PreprocessConfig:
    return PreprocessConfig(
        dataset_path=str(manifest_path),
        dataset_type=DatasetType.VIDAFORGE,
        video_loader_type=VideoLoaderType.TORCHVISION,
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
    width: int = 1280,
    height: int = 720,
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
    for field in ("clip_ok", "caption_ok", "select_pass"):
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
    ])

    config = PreprocessConfig.from_kwargs(vars(args))

    assert config is not None
    assert config.dataset_type == DatasetType.VIDAFORGE
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

    assert len(dataset) == 1
    assert dataset[0] == {
        "video": str(expected_video_path.resolve()),
        "name": "clip-a",
        "resolution": {
            "width": 32,
            "height": 24,
        },
        "fps": 24.0,
        "num_frames": 4,
        "caption": "dense caption",
    }


def test_build_vidaforge_dataset_accepts_absolute_clip_path_without_data_root(tmp_path: Path):
    video_path = _video_path(tmp_path, "absolute.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])

    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    assert dataset[0]["video"] == str(video_path)


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

    assert list(dataset["name"]) == ["keep"]


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
    for clip_id, select_pass in [("passed", 1), ("rejected", 0)]:
        relative_path = f"data/{clip_id}.mp4"
        _video_path(data_root, relative_path)
        rows.append(_row(clip_id=clip_id, clip_path=relative_path, select_pass=select_pass))
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, rows)

    dataset = build_dataset(
        _config(manifest_path, data_root=data_root, vidaforge_selection=selection),
        split="train",
        validator=lambda row: True,
    )

    assert list(dataset["name"]) == expected_names


def test_build_vidaforge_dataset_uses_configured_caption_field(tmp_path: Path):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])

    dataset = build_dataset(
        _config(manifest_path, vidaforge_caption_field="caption_level_0"),
        split="train",
        validator=lambda row: True,
    )

    assert dataset[0]["caption"] == "short caption"


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

    assert list(dataset["name"]) == ["clip-0", "clip-1"]


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
        build_dataset(_config(manifest_path), split="train", validator=lambda row: True)


def test_build_vidaforge_dataset_rejects_missing_clip_file(tmp_path: Path):
    manifest_path = tmp_path / "clip-00000.parquet"
    missing_path = (tmp_path / "missing.mp4").resolve()
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(missing_path))])

    with pytest.raises(FileNotFoundError, match="clip_path does not exist"):
        build_dataset(_config(manifest_path), split="train", validator=lambda row: True)


def test_build_vidaforge_dataset_rejects_path_outside_data_root(tmp_path: Path):
    data_root = tmp_path / "vidaforge"
    data_root.mkdir()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path="../outside.mp4")])

    with pytest.raises(ValueError, match="has clip_path outside"):
        build_dataset(_config(manifest_path, data_root=data_root), split="train", validator=lambda row: True)


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
                width=1280,
                height=720,
                fps=25.0,
                duration_sec=2.04,
            )
        ],
    )

    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    assert dataset[0]["resolution"] == {
        "width": 40,
        "height": 30,
    }
    assert dataset[0]["fps"] == 25.0
    assert dataset[0]["num_frames"] == 50


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
    _, tar_offset, filesize_bytes, sha256 = _indexed_tar(
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

    materialized_path = (materialize_dir / clip_path).resolve()
    assert materialized_path.read_bytes() == source_video.read_bytes()
    assert dataset[0]["video"] == str(materialized_path)
    assert dataset[0]["name"] == "release-clip"
    assert dataset[0]["resolution"] == {
        "width": 40,
        "height": 30,
    }
    assert dataset[0]["fps"] == 25.0
    assert dataset[0]["num_frames"] == 5


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

    assert dataset[0]["video"] == str(extracted_video.resolve())
    assert dataset[0]["num_frames"] == 6


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

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_dataset(
            _config(
                manifest_path,
                data_root=data_root,
                vidaforge_materialize_dir=str(tmp_path / "materialized"),
            ),
            split="train",
            validator=lambda row: True,
        )


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


def test_build_vidaforge_dataset_rejects_empty_distributed_shard(tmp_path: Path, monkeypatch):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])
    monkeypatch.setattr(vidaforge_manifest, "get_world_size", lambda: 2)
    monkeypatch.setattr(vidaforge_manifest, "get_world_rank", lambda: 1)

    with pytest.raises(ValueError, match="requires at least one eligible row per rank"):
        build_dataset(_config(manifest_path), split="train", validator=lambda row: True)


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
        validator=lambda row: row["name"] == "keep",
    )

    assert list(dataset["name"]) == ["keep"]


def test_normalized_vidaforge_row_builds_preprocess_batch(tmp_path: Path):
    video_path = _video_path(tmp_path, "clip.mp4").resolve()
    manifest_path = tmp_path / "clip-00000.parquet"
    _write_parquet(manifest_path, [_row(clip_id="clip-a", clip_path=str(video_path))])
    dataset = build_dataset(_config(manifest_path), split="train", validator=lambda row: True)

    batch = VideoForwardBatchBuilder(seed=7)([dataset[0]])

    assert batch.video_loader == [str(video_path)]
    assert batch.video_file_name == ["clip-a"]
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
