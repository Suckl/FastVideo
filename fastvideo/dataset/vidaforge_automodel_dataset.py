# SPDX-License-Identifier: Apache-2.0
"""Training dataloader for VidaForge Stage 5 AutoModel caches."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.distributed import (
    get_sp_world_size,
    get_world_rank,
    get_world_size,
)


VidaForgeBucketKey = tuple[int, int, int, tuple[int, ...]]


def _passthrough(batch: dict[str, Any]) -> dict[str, Any]:
    return batch


class VidaForgeAutoModelDataset(Dataset[dict[str, Any]]):
    """Lazy reader for the Stage 5 ``metadata.json`` + ``.meta`` contract.

    VidaForge stores one leading-batch-dimension sample in every ``.meta``
    file. The Wan encoder has already applied the model's latent mean/std
    normalization, so callers must not normalize ``video_latents`` again.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        cfg_rate: float = 0.0,
        seed: int = 0,
        expected_model_name: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= cfg_rate <= 1.0:
            raise ValueError(f"cfg_rate must be in [0, 1], got {cfg_rate}")

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cfg_rate = float(cfg_rate)
        self.seed = int(seed)
        self.expected_model_name = (
            None
            if expected_model_name is None
            else str(expected_model_name).strip()
        )
        self.metadata = self._read_metadata()
        if not self.metadata:
            raise ValueError(
                "No VidaForge AutoModel metadata items found under "
                f"{self.cache_dir}"
            )

        self._bucket_keys: list[VidaForgeBucketKey] = []
        self.bucket_groups: dict[VidaForgeBucketKey, list[int]] = {}
        for index, item in enumerate(self.metadata):
            key = _bucket_key_from_item(item)
            self._bucket_keys.append(key)
            self.bucket_groups.setdefault(key, []).append(index)
        self.sorted_bucket_keys = sorted(self.bucket_groups)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.metadata[index]
        path = self._resolve_cache_path(item)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"VidaForge .meta payload must be a dict: {path}")

        video_latents = payload.get("video_latents")
        text_embeddings = payload.get("text_embeddings")
        metadata = payload.get("metadata")
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(f"video_latents must be a tensor: {path}")
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError(f"text_embeddings must be a tensor: {path}")
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict: {path}")
        if video_latents.ndim != 5 or video_latents.shape[0] != 1:
            raise ValueError(
                "video_latents must have shape [1, C, T, H, W]: "
                f"path={path}, shape={tuple(video_latents.shape)}"
            )
        if text_embeddings.ndim != 3 or text_embeddings.shape[0] != 1:
            raise ValueError(
                "text_embeddings must have shape [1, L, D]: "
                f"path={path}, shape={tuple(text_embeddings.shape)}"
            )
        if not video_latents.is_floating_point():
            raise TypeError(f"video_latents must be floating point: {path}")
        if not text_embeddings.is_floating_point():
            raise TypeError(f"text_embeddings must be floating point: {path}")

        model_type = str(metadata.get("model_type", "")).strip().lower()
        if model_type != "wan":
            raise ValueError(
                "FastVideo currently supports only Wan VidaForge AutoModel "
                f"caches, got model_type={model_type!r}: {path}"
            )
        model_name = str(metadata.get("model_name", "")).strip()
        if not model_name:
            raise ValueError(
                f"VidaForge Wan .meta is missing metadata.model_name: {path}"
            )
        if (
            self.expected_model_name
            and not _model_names_compatible(
                model_name,
                self.expected_model_name,
            )
        ):
            raise ValueError(
                "VidaForge cache model_name does not match the FastVideo "
                "checkpoint: "
                f"cache={model_name!r}, "
                f"checkpoint={self.expected_model_name!r}, path={path}"
            )

        key = self._bucket_keys[index]
        actual_latent_shape = tuple(int(value) for value in video_latents.shape)
        if actual_latent_shape != key[3]:
            raise ValueError(
                "VidaForge latent shape mismatch: "
                f"path={path}, shard={key[3]}, actual={actual_latent_shape}"
            )
        payload_resolution = _read_resolution(metadata, where=f"{path} metadata")
        if payload_resolution != (key[1], key[2]):
            raise ValueError(
                "VidaForge bucket_resolution mismatch: "
                f"path={path}, shard={(key[1], key[2])}, "
                f"payload={payload_resolution}"
            )
        payload_frame_count = int(payload.get("bucket_frame_count", 0))
        metadata_frame_count = int(metadata.get("bucket_frame_count", 0))
        if payload_frame_count != key[0] or metadata_frame_count != key[0]:
            raise ValueError(
                "VidaForge bucket_frame_count mismatch: "
                f"path={path}, shard={key[0]}, payload={payload_frame_count}, "
                f"metadata={metadata_frame_count}"
            )

        text_attention_mask = _text_attention_mask(
            payload,
            metadata,
            item,
            sequence_length=int(text_embeddings.shape[1]),
            path=path,
        )
        prompt = str(metadata.get("caption", ""))
        clip_id = str(metadata.get("clip_id", item.get("clip_id", "")))
        return {
            "vae_latent": video_latents,
            "text_embedding": text_embeddings,
            "text_attention_mask": text_attention_mask,
            "info": {
                "prompt": prompt,
                "clip_id": clip_id,
                "model_name": model_name,
                "bucket_frame_count": key[0],
                "bucket_resolution": (key[1], key[2]),
                "meta_path": str(path),
            },
        }

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        samples = [self[index] for index in indices]
        return self.collate(samples, indices=indices)

    def collate(
        self,
        samples: list[dict[str, Any]],
        *,
        indices: list[int],
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty VidaForge batch")
        if len(samples) != len(indices):
            raise ValueError("VidaForge sample and index counts do not match")

        bucket_keys = {self._bucket_keys[index] for index in indices}
        if len(bucket_keys) != 1:
            raise ValueError(
                f"VidaForge batch mixes incompatible buckets: {bucket_keys}"
            )

        latents = _cat_same_shape(samples, "vae_latent")
        embeddings = _cat_same_shape(samples, "text_embedding")
        attention_masks = _cat_same_shape(samples, "text_attention_mask")
        if self.cfg_rate > 0:
            embeddings = embeddings.clone()
            for batch_index, sample_index in enumerate(indices):
                if random.Random(self.seed ^ sample_index).random() < self.cfg_rate:
                    embeddings[batch_index].zero_()

        return {
            "vae_latent": latents,
            "text_embedding": embeddings,
            "text_attention_mask": attention_masks,
            "info_list": [sample["info"] for sample in samples],
        }

    def _read_metadata(self) -> list[dict[str, Any]]:
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing VidaForge metadata.json: {metadata_path}"
            )
        root = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(root, dict):
            raise ValueError(
                f"VidaForge metadata.json must contain an object: {metadata_path}"
            )
        shard_names = root.get("shards")
        if not isinstance(shard_names, list) or not shard_names:
            raise ValueError(
                "VidaForge metadata.json must contain a non-empty shards list: "
                f"{metadata_path}"
            )

        items: list[dict[str, Any]] = []
        seen_shards: set[Path] = set()
        for shard_name in shard_names:
            shard_path = _resolve_child_path(
                self.cache_dir,
                shard_name,
                kind="metadata shard",
            )
            if shard_path in seen_shards:
                raise ValueError(
                    f"Duplicate VidaForge metadata shard: {shard_path}"
                )
            seen_shards.add(shard_path)
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing VidaForge metadata shard: {shard_path}"
                )
            shard_items = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard_items, list):
                raise ValueError(
                    f"VidaForge metadata shard must contain a list: {shard_path}"
                )
            for item in shard_items:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"VidaForge metadata item must be an object: {shard_path}"
                    )
                _bucket_key_from_item(item)
                cache_file = str(item.get("cache_file", "")).strip()
                if not cache_file:
                    raise ValueError(
                        f"VidaForge metadata item is missing cache_file: {item}"
                    )
                if Path(cache_file).suffix != ".meta":
                    raise ValueError(
                        "VidaForge cache_file must end in .meta, got "
                        f"{cache_file!r}"
                    )
                items.append(item)
        return items

    def _resolve_cache_path(self, item: dict[str, Any]) -> Path:
        cache_file_text = str(item["cache_file"])
        cache_file = Path(cache_file_text).expanduser()
        foreign_absolute = (
            PurePosixPath(cache_file_text).is_absolute()
            or PureWindowsPath(cache_file_text).is_absolute()
        )
        if not cache_file.is_absolute() and not foreign_absolute:
            path = _resolve_child_path(
                self.cache_dir,
                cache_file,
                kind="cache file",
            )
        else:
            path = cache_file.resolve()

        if path.is_file():
            return path

        # Stage 5 writes absolute cache paths. If the complete output tree was
        # moved, reconstruct the suffix rooted at "<frames>f/<width>x<height>".
        key = _bucket_key_from_item(item)
        frame_dir = f"{key[0]}f"
        resolution_dir = f"{key[1]}x{key[2]}"
        if PureWindowsPath(cache_file_text).is_absolute():
            parts = PureWindowsPath(cache_file_text).parts
        elif PurePosixPath(cache_file_text).is_absolute():
            parts = PurePosixPath(cache_file_text).parts
        else:
            parts = path.parts
        for start in range(len(parts) - 1):
            if (
                parts[start] == frame_dir
                and start + 1 < len(parts)
                and parts[start + 1] == resolution_dir
            ):
                rebased = self.cache_dir.joinpath(*parts[start:]).resolve()
                if rebased.is_relative_to(self.cache_dir) and rebased.is_file():
                    return rebased
                break
        raise FileNotFoundError(f"Missing VidaForge .meta cache file: {path}")


class VidaForgeBucketBatchSampler(Sampler[list[int]]):
    """Bucket sampler aligned over data-parallel groups and SP replicas."""

    def __init__(
        self,
        dataset: VidaForgeAutoModelDataset,
        *,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_replicas is None:
            sp_world_size = get_sp_world_size()
            num_replicas = get_world_size() // sp_world_size
        if rank is None:
            rank = get_world_rank() // get_sp_world_size()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got {rank}"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        self._length = sum(
            self._usable_count(len(indices)) // self.global_batch_size
            for indices in self.dataset.bucket_groups.values()
        )
        if self._length == 0:
            bucket_sizes = {
                str(key): len(indices)
                for key, indices in self.dataset.bucket_groups.items()
            }
            raise ValueError(
                "VidaForge dataset has no full distributed bucket batch: "
                f"global_batch_size={self.global_batch_size}, "
                f"drop_last={self.drop_last}, bucket_sizes={bucket_sizes}"
            )

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        bucket_keys = list(self.dataset.sorted_bucket_keys)
        if self.shuffle:
            order = torch.randperm(
                len(bucket_keys),
                generator=generator,
            ).tolist()
            bucket_keys = [bucket_keys[index] for index in order]

        for key in bucket_keys:
            indices = list(self.dataset.bucket_groups[key])
            if self.shuffle:
                order = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()
                indices = [indices[index] for index in order]
            indices = self._trim_or_pad(indices)
            for start in range(0, len(indices), self.global_batch_size):
                rank_start = start + self.rank * self.batch_size
                batch = indices[rank_start:rank_start + self.batch_size]
                if len(batch) != self.batch_size:
                    raise RuntimeError(
                        "Internal VidaForge sampler error: incomplete rank batch"
                    )
                yield batch

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _usable_count(self, count: int) -> int:
        if self.drop_last:
            return count // self.global_batch_size * self.global_batch_size
        return math.ceil(count / self.global_batch_size) * self.global_batch_size

    def _trim_or_pad(self, indices: list[int]) -> list[int]:
        usable_count = self._usable_count(len(indices))
        if usable_count <= len(indices):
            return indices[:usable_count]
        padding_count = usable_count - len(indices)
        return indices + [
            indices[index % len(indices)] for index in range(padding_count)
        ]


def build_vidaforge_automodel_dataloader(
    path: str | Path,
    batch_size: int,
    num_data_workers: int,
    *,
    cfg_rate: float = 0.0,
    seed: int = 0,
    expected_model_name: str | Path | None = None,
) -> tuple[VidaForgeAutoModelDataset, StatefulDataLoader]:
    dataset = VidaForgeAutoModelDataset(
        path,
        cfg_rate=cfg_rate,
        seed=seed,
        expected_model_name=expected_model_name,
    )
    sampler = VidaForgeBucketBatchSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        seed=seed,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader


def _bucket_key_from_item(item: dict[str, Any]) -> VidaForgeBucketKey:
    frame_count = int(item.get("bucket_frame_count", 0))
    if frame_count <= 0:
        raise ValueError(
            f"bucket_frame_count must be positive in VidaForge item: {item}"
        )
    width, height = _read_resolution(item, where="metadata shard")
    latent_shape_value = item.get("latent_shape")
    if (
        not isinstance(latent_shape_value, list | tuple)
        or len(latent_shape_value) != 5
    ):
        raise ValueError(
            "latent_shape must be [1, C, T, H, W] in VidaForge item: "
            f"{item}"
        )
    latent_shape = tuple(int(value) for value in latent_shape_value)
    if latent_shape[0] != 1 or any(value <= 0 for value in latent_shape):
        raise ValueError(
            "latent_shape must contain one sample and positive dimensions: "
            f"{latent_shape}"
        )
    return (frame_count, width, height, latent_shape)


def _read_resolution(
    item: dict[str, Any],
    *,
    where: str,
) -> tuple[int, int]:
    value = item.get("bucket_resolution")
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(
            f"bucket_resolution must be [width, height] in {where}"
        )
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"bucket_resolution must contain positive values in {where}"
        )
    return width, height


def _resolve_child_path(
    root: Path,
    value: object,
    *,
    kind: str,
) -> Path:
    child = Path(str(value))
    if child.is_absolute():
        raise ValueError(f"VidaForge {kind} path must be relative: {value!r}")
    path = (root / child).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"VidaForge {kind} escapes dataset directory: {value!r}"
        )
    return path


def _text_attention_mask(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    item: dict[str, Any],
    *,
    sequence_length: int,
    path: Path,
) -> torch.Tensor:
    stored_mask = payload.get("text_mask")
    if stored_mask is not None:
        if not isinstance(stored_mask, torch.Tensor):
            raise TypeError(f"text_mask must be a tensor when present: {path}")
        if stored_mask.ndim == 3 and stored_mask.shape[1] == 1:
            stored_mask = stored_mask.squeeze(1)
        if tuple(stored_mask.shape) != (1, sequence_length):
            raise ValueError(
                "text_mask must have shape [1, L]: "
                f"path={path}, shape={tuple(stored_mask.shape)}, "
                f"L={sequence_length}"
            )
        return stored_mask

    token_length = int(
        metadata.get("caption_token_length", 0)
        or item.get("caption_token_length", 0)
        or 0
    )
    if token_length <= 0:
        raise ValueError(
            "VidaForge Wan .meta has no text_mask and no positive "
            f"caption_token_length: {path}"
        )
    valid_length = min(token_length, sequence_length)
    mask = torch.zeros((1, sequence_length), dtype=torch.float32)
    mask[:, :valid_length] = 1
    return mask


def _cat_same_shape(
    samples: list[dict[str, Any]],
    field: str,
) -> torch.Tensor:
    tensors = [sample[field] for sample in samples]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(f"{field} must contain only tensors")
    first_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != first_shape for tensor in tensors[1:]):
        raise ValueError(f"{field} shapes differ within a VidaForge bucket")
    return torch.cat(tensors, dim=0)


def _model_names_compatible(
    cache_model_name: str,
    checkpoint: str,
) -> bool:
    cache_identity = _normalize_model_identity(cache_model_name)
    checkpoint_identity = _normalize_model_identity(checkpoint)
    if cache_identity == checkpoint_identity:
        return True

    # Hugging Face snapshots commonly look like
    # ".../models--Wan-AI--Wan2.1-.../snapshots/<revision>".
    for part in checkpoint.replace("\\", "/").split("/"):
        if part.startswith("models--"):
            hub_identity = _normalize_model_identity(
                part[len("models--"):].replace("--", "/"),
            )
            if hub_identity == cache_identity:
                return True

    checkpoint_path = Path(checkpoint).expanduser()
    checkpoint_is_local = (
        checkpoint_path.exists()
        or PurePosixPath(checkpoint).is_absolute()
        or PureWindowsPath(checkpoint).is_absolute()
        or checkpoint.startswith(("./", "../", ".\\", "..\\", "~"))
    )
    if checkpoint_is_local:
        return (
            cache_identity.rsplit("/", 1)[-1]
            == checkpoint_identity.rsplit("/", 1)[-1]
        )
    return False


def _normalize_model_identity(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/").lower()


__all__ = [
    "VidaForgeAutoModelDataset",
    "VidaForgeBucketBatchSampler",
    "build_vidaforge_automodel_dataloader",
]
