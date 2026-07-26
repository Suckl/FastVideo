"""VidaForge manifest inputs for FastVideo preprocessing.

The Stage 4 contract follows VidaForge revision
``4562d3fbcbd4861fc74c2859950c0237363681bb``. The release contract follows
the paired Parquet/indexed-TAR layout published as ``VidaForge/VidaForge-3M``.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import av
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from fastvideo.configs.configs import PreprocessConfig, VideoLoaderType
from fastvideo.distributed.parallel_state import get_world_rank, get_world_size
from fastvideo.logger import init_logger

logger = init_logger(__name__)

VIDAFORGE_REFERENCE_REVISION = "4562d3fbcbd4861fc74c2859950c0237363681bb"
VIDAFORGE_RELEASE_DATASET = "VidaForge/VidaForge-3M"
VIDAFORGE_SELECTION_VALUES: dict[str, int | None] = {
    "pass": 1,
    "reject": 0,
    "all": None,
}

_COMMON_REQUIRED_COLUMNS = frozenset({
    "clip_id",
    "clip_path",
    "duration_sec",
    "fps",
    "height",
    "width",
})
_STAGE4_REQUIRED_COLUMNS = frozenset({
    "caption_ok",
    "clip_ok",
    "select_ok",
    "select_pass",
})
_RELEASE_REQUIRED_COLUMNS = frozenset({
    "filesize_bytes",
    "sha256",
    "tar_offset",
    "tar_path",
})
_COPY_CHUNK_SIZE = 8 * 1024 * 1024


class VidaForgeManifestKind(str, Enum):
    STAGE4 = "stage4"
    RELEASE = "release"


@dataclass(frozen=True)
class VidaForgeTorchCodecVideo:
    """Pickle-safe lazy TorchCodec source used by the release dataset."""

    source: str | bytes

    @property
    def source_path(self) -> str | None:
        """Return a file-backed source for stages that require a path."""
        return self.source if isinstance(self.source, str) else None

    def get_frames_at(self, indices: Any) -> Any:
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(self.source).get_frames_at(indices)


def resolve_vidaforge_parquet_paths(dataset_path: str | Path) -> list[Path]:
    """Resolve a VidaForge manifest file or its direct Parquet shards."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"VidaForge dataset path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"VidaForge dataset file must be Parquet: {path}")
        return [path]

    parquet_paths = sorted(child for child in path.glob("*.parquet") if child.is_file())
    if not parquet_paths:
        raise FileNotFoundError(f"No Parquet shards found directly under VidaForge dataset path: {path}")
    return parquet_paths


def _detect_manifest_kind(dataset: Dataset, caption_field: str) -> VidaForgeManifestKind:
    columns = set(dataset.column_names)
    common_required = _COMMON_REQUIRED_COLUMNS | {caption_field}
    missing_common = sorted(common_required.difference(columns))
    if missing_common:
        raise ValueError("VidaForge manifest is missing required columns: " + ", ".join(missing_common))

    if _STAGE4_REQUIRED_COLUMNS.issubset(columns):
        return VidaForgeManifestKind.STAGE4
    if _RELEASE_REQUIRED_COLUMNS.issubset(columns):
        return VidaForgeManifestKind.RELEASE

    missing_stage4 = sorted(_STAGE4_REQUIRED_COLUMNS.difference(columns))
    missing_release = sorted(_RELEASE_REQUIRED_COLUMNS.difference(columns))
    raise ValueError("VidaForge manifest does not match a supported schema. "
                     f"Stage 4 is missing: {', '.join(missing_stage4) or '<none>'}; "
                     f"VidaForge-3M release metadata is missing: {', '.join(missing_release) or '<none>'}.")


def _status_value(row: dict[str, Any], field: str) -> int:
    clip_id = str(row.get("clip_id") or "<unknown>")
    try:
        return int(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid {field}: {row.get(field)!r}") from exc


def _positive_number(row: dict[str, Any], field: str) -> float:
    clip_id = str(row.get("clip_id") or "<unknown>")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid {field}: {row.get(field)!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"VidaForge row {clip_id!r} requires finite {field} > 0, got {row.get(field)!r}")
    return value


def _integer_value(row: dict[str, Any], field: str, *, minimum: int) -> int:
    clip_id = str(row.get("clip_id") or "<unknown>")
    raw_value = row.get(field)
    if raw_value is None:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid {field}: {raw_value!r}")
    try:
        value = int(raw_value)
        numeric_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid {field}: {raw_value!r}") from exc
    if not math.isfinite(numeric_value) or numeric_value != value or value < minimum:
        raise ValueError(f"VidaForge row {clip_id!r} requires integer {field} >= {minimum}, got {raw_value!r}")
    return value


def _selection_value(manifest_kind: VidaForgeManifestKind, selection: str) -> int | None:
    if selection == "auto":
        return 1 if manifest_kind == VidaForgeManifestKind.STAGE4 else None
    try:
        selection_value = VIDAFORGE_SELECTION_VALUES[selection]
    except KeyError as exc:
        raise ValueError("vidaforge_selection must be one of: auto, pass, reject, all") from exc
    if manifest_kind == VidaForgeManifestKind.RELEASE and selection_value is not None:
        raise ValueError(
            f"vidaforge_selection={selection!r} cannot be applied to the public VidaForge-3M release because "
            "select_pass is intentionally absent; use 'auto' or 'all', then apply a quality recipe separately.")
    return selection_value


def _row_is_eligible(
    row: dict[str, Any],
    *,
    caption_field: str,
    manifest_kind: VidaForgeManifestKind,
    selection_value: int | None,
) -> bool:
    if not str(row[caption_field] or "").strip():
        return False
    if manifest_kind == VidaForgeManifestKind.RELEASE:
        return True
    if (_status_value(row, "clip_ok") != 1 or _status_value(row, "caption_ok") != 1
            or _status_value(row, "select_ok") != 1):
        return False
    return selection_value is None or _status_value(row, "select_pass") == selection_value


def _safe_relative_path(root: Path, raw_path: Any, *, field: str, clip_id: str) -> Path:
    path_text = str(raw_path or "").strip()
    if not path_text:
        raise ValueError(f"VidaForge row {clip_id!r} has an empty {field}")
    relative_path = Path(path_text)
    if relative_path.is_absolute():
        raise ValueError(f"VidaForge row {clip_id!r} requires relative {field}, got {path_text!r}")
    resolved_path = (root / relative_path).resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"VidaForge row {clip_id!r} has {field} outside {root}: {path_text!r}") from exc
    return resolved_path


def _resolve_stage4_clip_path(row: dict[str, Any], data_root: Path | None) -> Path:
    clip_id = str(row.get("clip_id") or "<unknown>")
    raw_clip_path = str(row["clip_path"] or "").strip()
    if not raw_clip_path:
        raise ValueError(f"VidaForge row {clip_id!r} has an empty clip_path")

    clip_path = Path(raw_clip_path).expanduser()
    if clip_path.is_absolute():
        resolved_path = clip_path.resolve()
    else:
        if data_root is None:
            raise ValueError(f"VidaForge row {clip_id!r} uses relative clip_path {raw_clip_path!r}; "
                             "set preprocess.vidaforge-data-root to VidaForge DATA_DIR.")
        resolved_path = _safe_relative_path(data_root, clip_path, field="clip_path", clip_id=clip_id)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"VidaForge row {clip_id!r} clip_path does not exist: {resolved_path}")
    return resolved_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_release_clip(path: Path, *, expected_size: int, expected_sha256: str, clip_id: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"VidaForge row {clip_id!r} expected {expected_size} video bytes at {path}, found {actual_size}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"VidaForge row {clip_id!r} SHA-256 mismatch at {path}: expected {expected_sha256}, got {actual_sha256}")


def _release_storage_values(row: dict[str, Any]) -> tuple[str, int, int, str]:
    clip_id = str(row.get("clip_id") or "").strip()
    expected_size = _integer_value(row, "filesize_bytes", minimum=1)
    tar_offset = _integer_value(row, "tar_offset", minimum=0)
    expected_sha256 = str(row.get("sha256") or "").strip().lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"VidaForge row {clip_id!r} has invalid sha256: {row.get('sha256')!r}")
    return clip_id, expected_size, tar_offset, expected_sha256


def _resolve_extracted_release_clip(row: dict[str, Any], *, data_root: Path) -> Path | None:
    clip_id, expected_size, _, expected_sha256 = _release_storage_values(row)

    extracted_root = (data_root / "data").resolve()
    extracted_path = _safe_relative_path(extracted_root, row["clip_path"], field="clip_path", clip_id=clip_id)
    if extracted_path.is_file():
        _validate_release_clip(
            extracted_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            clip_id=clip_id,
        )
        return extracted_path
    return None


def _release_materialize_destination(row: dict[str, Any], *, materialize_root: Path) -> Path:
    clip_id, _, _, expected_sha256 = _release_storage_values(row)
    validated_clip_path = _safe_relative_path(
        materialize_root,
        row["clip_path"],
        field="clip_path",
        clip_id=clip_id,
    )
    suffix = validated_clip_path.suffix
    return materialize_root / expected_sha256[:2] / f"{expected_sha256}{suffix}"


def _release_tar_path_and_range(
    row: dict[str, Any],
    *,
    data_root: Path,
) -> tuple[str, Path, int, int, str]:
    clip_id, expected_size, tar_offset, expected_sha256 = _release_storage_values(row)
    tar_path = _safe_relative_path(data_root, row["tar_path"], field="tar_path", clip_id=clip_id)
    if not tar_path.is_file():
        raise FileNotFoundError(f"VidaForge row {clip_id!r} tar_path does not exist: {tar_path}")
    tar_size = tar_path.stat().st_size
    if tar_offset + expected_size > tar_size:
        raise ValueError(f"VidaForge row {clip_id!r} byte range [{tar_offset}, {tar_offset + expected_size}) "
                         f"exceeds tar size {tar_size}: {tar_path}")
    return clip_id, tar_path, tar_offset, expected_size, expected_sha256


def _read_release_clip_bytes(row: dict[str, Any], *, data_root: Path) -> bytes:
    clip_id, tar_path, tar_offset, expected_size, expected_sha256 = _release_tar_path_and_range(
        row,
        data_root=data_root,
    )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    remaining = expected_size
    with tar_path.open("rb") as tar_handle:
        tar_handle.seek(tar_offset)
        while remaining > 0:
            chunk = tar_handle.read(min(remaining, _COPY_CHUNK_SIZE))
            if not chunk:
                raise ValueError(f"VidaForge row {clip_id!r} reached EOF while reading {expected_size} bytes "
                                 f"from offset {tar_offset}: {tar_path}")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"VidaForge row {clip_id!r} SHA-256 mismatch while reading {tar_path}: "
                         f"expected {expected_sha256}, got {actual_sha256}")
    return b"".join(chunks)


def _materialize_release_clip(row: dict[str, Any], *, data_root: Path, materialize_root: Path) -> Path:
    clip_id, expected_size, _, expected_sha256 = _release_storage_values(row)
    extracted_path = _resolve_extracted_release_clip(row, data_root=data_root)
    if extracted_path is not None:
        return extracted_path

    destination = _release_materialize_destination(row, materialize_root=materialize_root)
    if destination.is_file():
        _validate_release_clip(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            clip_id=clip_id,
        )
        return destination

    _, tar_path, tar_offset, _, _ = _release_tar_path_and_range(row, data_root=data_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            digest = hashlib.sha256()
            remaining = expected_size
            with tar_path.open("rb") as tar_handle:
                tar_handle.seek(tar_offset)
                while remaining > 0:
                    chunk = tar_handle.read(min(remaining, _COPY_CHUNK_SIZE))
                    if not chunk:
                        raise ValueError(f"VidaForge row {clip_id!r} reached EOF while reading {expected_size} bytes "
                                         f"from offset {tar_offset}: {tar_path}")
                    output_handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"VidaForge row {clip_id!r} SHA-256 mismatch while reading {tar_path}: "
                             f"expected {expected_sha256}, got {actual_sha256}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _release_media_source(
    row: dict[str, Any],
    *,
    data_root: Path,
    materialize_root: Path | None,
) -> str | bytes:
    extracted_path = _resolve_extracted_release_clip(row, data_root=data_root)
    if extracted_path is not None:
        return str(extracted_path)
    if materialize_root is not None:
        return str(_materialize_release_clip(row, data_root=data_root, materialize_root=materialize_root))
    return _read_release_clip_bytes(row, data_root=data_root)


def _probe_video(source: str | bytes, *, clip_id: str, fallback_fps: float) -> dict[str, int | float]:
    av_source: str | io.BytesIO = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        with av.open(av_source) as container:
            try:
                stream = container.streams.video[0]
            except IndexError as exc:
                raise ValueError("contains no video stream") from exc

            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            average_rate = stream.average_rate
            fps = float(average_rate) if average_rate is not None else fallback_fps
            num_frames = int(stream.frames)
            if num_frames <= 0:
                num_frames = sum(1 for _ in container.decode(stream))
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        raise ValueError(f"VidaForge row {clip_id!r} could not probe video: {exc}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid decoded resolution {width}x{height}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"VidaForge row {clip_id!r} has invalid decoded fps {fps}")
    if num_frames <= 0:
        raise ValueError(f"VidaForge row {clip_id!r} has no decodable video frames")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
    }


def _normalize_row(
    row: dict[str, Any],
    *,
    caption_field: str,
    manifest_kind: VidaForgeManifestKind,
    data_root: Path | None,
    materialize_root: Path | None,
    video_loader_type: VideoLoaderType,
) -> dict[str, Any]:
    clip_id = str(row["clip_id"] or "").strip()
    if not clip_id:
        raise ValueError("VidaForge manifest contains an empty clip_id")

    _positive_number(row, "width")
    _positive_number(row, "height")
    manifest_fps = _positive_number(row, "fps")
    _positive_number(row, "duration_sec")

    if manifest_kind == VidaForgeManifestKind.STAGE4:
        clip_path = _resolve_stage4_clip_path(row, data_root)
        media_source: str | bytes = str(clip_path)
        if video_loader_type == VideoLoaderType.TORCHCODEC:
            video: str | VidaForgeTorchCodecVideo = VidaForgeTorchCodecVideo(media_source)
        else:
            video = str(clip_path)
    else:
        if data_root is None:
            raise ValueError("VidaForge-3M release metadata requires vidaforge_data_root")
        media_source = _release_media_source(
            row,
            data_root=data_root,
            materialize_root=materialize_root,
        )
        if video_loader_type == VideoLoaderType.TORCHCODEC:
            video = VidaForgeTorchCodecVideo(media_source)
        elif isinstance(media_source, str):
            video = media_source
        else:
            raise ValueError("VidaForge-3M Torchvision loading requires a materialization root")

    media = _probe_video(media_source, clip_id=clip_id, fallback_fps=manifest_fps)
    return {
        "video": video,
        "name": clip_id,
        "resolution": {
            "width": media["width"],
            "height": media["height"],
        },
        "fps": media["fps"],
        "num_frames": media["num_frames"],
        "caption": str(row[caption_field]).strip(),
    }


def _validate_unique_column(dataset: Dataset, column: str) -> None:
    unique_values = dataset.unique(column)
    if len(unique_values) != len(dataset):
        raise ValueError(f"VidaForge manifest contains duplicate {column} values: "
                         f"{len(dataset)} eligible rows but {len(unique_values)} unique values")


def _resolve_data_root(preprocess_config: PreprocessConfig) -> Path | None:
    if not preprocess_config.vidaforge_data_root.strip():
        return None
    data_root = Path(preprocess_config.vidaforge_data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"VidaForge data root does not exist or is not a directory: {data_root}")
    return data_root


def _resolve_materialize_root(preprocess_config: PreprocessConfig) -> Path | None:
    configured_path = preprocess_config.vidaforge_materialize_dir.strip()
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    if preprocess_config.video_loader_type == VideoLoaderType.TORCHVISION:
        return (Path(preprocess_config.dataset_output_dir).expanduser().resolve() / ".vidaforge_clips")
    return None


class VidaForgeIterableDataset(IterableDataset[dict[str, Any]]):
    """Lazily normalize and validate Stage 4 or public release clips."""

    def __init__(
        self,
        metadata: Dataset,
        *,
        caption_field: str,
        manifest_kind: VidaForgeManifestKind,
        data_root: Path | None,
        materialize_root: Path | None,
        video_loader_type: VideoLoaderType,
        validator: Callable[[dict[str, Any]], bool],
        world_rank: int,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.caption_field = caption_field
        self.manifest_kind = manifest_kind
        self.data_root = data_root
        self.materialize_root = materialize_root
        self.video_loader_type = video_loader_type
        self.validator = validator
        self.world_rank = world_rank

    def __iter__(self) -> Iterator[dict[str, Any]]:
        metadata = self.metadata
        worker_info = get_worker_info()
        if worker_info is not None:
            metadata = metadata.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        normalized_count = 0
        validated_count = 0
        for row in metadata:
            normalized_count += 1
            normalized_row = _normalize_row(
                row,
                caption_field=self.caption_field,
                manifest_kind=self.manifest_kind,
                data_root=self.data_root,
                materialize_root=self.materialize_root,
                video_loader_type=self.video_loader_type,
            )
            if self.validator(normalized_row):
                validated_count += 1
                yield normalized_row

        worker_id = worker_info.id if worker_info is not None else 0
        logger.info(
            "FastVideo validation kept %d of %d normalized VidaForge %s rows on rank %d worker %d",
            validated_count,
            normalized_count,
            self.manifest_kind.value,
            self.world_rank,
            worker_id,
        )
        if worker_info is None and normalized_count > 0 and validated_count == 0:
            raise ValueError("VidaForge manifest produced no rows that satisfy FastVideo preprocessing constraints.")


def build_vidaforge_dataset(
    preprocess_config: PreprocessConfig,
    split: str,
    validator: Callable[[dict[str, Any]], bool],
) -> VidaForgeIterableDataset:
    """Load and normalize a VidaForge Stage 4 or public release manifest."""
    if split != "train":
        raise ValueError("VidaForge manifests provide only the train split")

    parquet_paths = resolve_vidaforge_parquet_paths(preprocess_config.dataset_path)
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_paths]},
        split="train",
    )
    assert isinstance(dataset, Dataset)

    caption_field = preprocess_config.vidaforge_caption_field.strip()
    manifest_kind = _detect_manifest_kind(dataset, caption_field)
    selection_value = _selection_value(manifest_kind, preprocess_config.vidaforge_selection)

    required_columns = _COMMON_REQUIRED_COLUMNS | {caption_field}
    if manifest_kind == VidaForgeManifestKind.STAGE4:
        required_columns |= _STAGE4_REQUIRED_COLUMNS
    else:
        required_columns |= _RELEASE_REQUIRED_COLUMNS
    dataset = dataset.select_columns(sorted(required_columns))

    source_count = len(dataset)
    world_size = get_world_size()
    world_rank = get_world_rank()
    if source_count < world_size:
        raise ValueError(
            f"VidaForge preprocessing requires at least one source row per rank, but found {source_count} rows "
            f"for world size {world_size}; reduce the preprocessing world size.")
    dataset = dataset.shard(num_shards=world_size, index=world_rank, contiguous=True)
    rank_source_count = len(dataset)
    dataset = dataset.filter(
        _row_is_eligible,
        fn_kwargs={
            "caption_field": caption_field,
            "manifest_kind": manifest_kind,
            "selection_value": selection_value,
        },
        desc="Filtering VidaForge rows",
    )
    eligible_count = len(dataset)
    logger.info(
        "VidaForge %s manifest selected %d of %d rows on rank %d "
        "(global source rows=%d, selection=%s, caption_field=%s)",
        manifest_kind.value,
        eligible_count,
        rank_source_count,
        world_rank,
        source_count,
        preprocess_config.vidaforge_selection,
        caption_field,
    )
    if eligible_count == 0:
        raise ValueError(f"VidaForge {manifest_kind.value} manifest produced no eligible rows on rank {world_rank} "
                         f"for selection={preprocess_config.vidaforge_selection!r} "
                         f"and caption_field={caption_field!r}.")

    _validate_unique_column(dataset, "clip_id")
    if manifest_kind == VidaForgeManifestKind.RELEASE:
        _validate_unique_column(dataset, "clip_path")

    data_root = _resolve_data_root(preprocess_config)
    if manifest_kind == VidaForgeManifestKind.RELEASE and data_root is None:
        raise ValueError(
            "VidaForge-3M release metadata requires preprocess.vidaforge-data-root to locate paired TAR shards.")
    materialize_root = (_resolve_materialize_root(preprocess_config)
                        if manifest_kind == VidaForgeManifestKind.RELEASE else None)

    return VidaForgeIterableDataset(
        dataset,
        caption_field=caption_field,
        manifest_kind=manifest_kind,
        data_root=data_root,
        materialize_root=materialize_root,
        video_loader_type=preprocess_config.video_loader_type,
        validator=validator,
        world_rank=world_rank,
    )
