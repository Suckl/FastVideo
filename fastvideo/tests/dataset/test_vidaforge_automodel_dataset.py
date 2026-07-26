# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the VidaForge Stage 5 AutoModel dataloader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from fastvideo.dataset.vidaforge_automodel_dataset import (
    build_vidaforge_automodel_dataloader,
    VidaForgeAutoModelDataset,
    VidaForgeBucketBatchSampler,
)

_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_VAE_FINGERPRINT = "a" * 64
_TEXT_ENCODER_FINGERPRINT = "b" * 64


def _write_dataset(
    root: Path,
    *,
    count: int = 4,
    frame_count: int = 17,
    resolution: tuple[int, int] = (64, 48),
    latent_shape: tuple[int, ...] = (1, 16, 5, 6, 8),
    cache_paths: list[str] | None = None,
    text_mask: torch.Tensor | None = None,
    include_fingerprints: bool = True,
) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    shard_dir = root / "shards"
    shard_dir.mkdir()
    meta_dir = root / f"{frame_count}f" / (
        f"{resolution[0]}x{resolution[1]}"
    )
    meta_dir.mkdir(parents=True)

    items: list[dict[str, Any]] = []
    paths: list[Path] = []
    for index in range(count):
        path = meta_dir / f"sample-{index}.meta"
        paths.append(path)
        metadata = {
            "model_type": "wan",
            "model_name": _MODEL_NAME,
            "clip_id": f"clip-{index}",
            "caption": f"caption {index}",
            "caption_token_length": index + 2,
            "bucket_resolution": list(resolution),
            "bucket_frame_count": frame_count,
        }
        if include_fingerprints:
            metadata.update({
                "vae_fingerprint": _VAE_FINGERPRINT,
                "text_encoder_fingerprint": (
                    _TEXT_ENCODER_FINGERPRINT
                ),
            })
        payload: dict[str, Any] = {
            "video_latents": torch.full(
                latent_shape,
                index,
                dtype=torch.float16,
            ),
            "text_embeddings": torch.full(
                (1, 8, 12),
                index + 1,
                dtype=torch.bfloat16,
            ),
            "metadata": metadata,
            "bucket_frame_count": frame_count,
            "num_frames": frame_count,
        }
        if text_mask is not None:
            payload["text_mask"] = text_mask
        torch.save(payload, path)
        items.append({
            "cache_file":
            cache_paths[index] if cache_paths is not None else str(path),
            "bucket_resolution":
            list(resolution),
            "bucket_frame_count":
            frame_count,
            "latent_shape":
            list(latent_shape),
            "clip_id":
            f"clip-{index}",
            "caption_token_length":
            index + 2,
        })

    (shard_dir / "metadata-000000.json").write_text(
        json.dumps(items),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps({"shards": ["shards/metadata-000000.json"]}),
        encoding="utf-8",
    )
    return paths


def _open_dataset(
    root: Path,
    **kwargs: Any,
) -> VidaForgeAutoModelDataset:
    kwargs.setdefault("expected_model_name", _MODEL_NAME)
    kwargs.setdefault("expected_vae_fingerprint", _VAE_FINGERPRINT)
    kwargs.setdefault(
        "expected_text_encoder_fingerprint",
        _TEXT_ENCODER_FINGERPRINT,
    )
    return VidaForgeAutoModelDataset(root, **kwargs)


def test_loads_official_wan_fields_without_dtype_conversion(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path)
    dataset = _open_dataset(tmp_path)

    sample = dataset[1]

    assert sample["vae_latent"].dtype == torch.float16
    assert sample["text_embedding"].dtype == torch.bfloat16
    assert sample["text_attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]
    assert sample["info"]["prompt"] == "caption 1"
    assert sample["info"]["clip_id"] == "clip-1"


def test_collate_preserves_bucket_and_deterministic_cfg_dropout(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path)
    dataset = _open_dataset(
        tmp_path,
        cfg_rate=1.0,
        seed=123,
    )

    batch = dataset.__getitems__([0, 1])

    assert batch["vae_latent"].shape == (2, 16, 5, 6, 8)
    assert batch["text_embedding"].shape == (2, 8, 12)
    assert torch.count_nonzero(batch["text_embedding"]) == 0
    assert batch["text_attention_mask"].shape == (2, 8)
    assert [info["clip_id"] for info in batch["info_list"]] == [
        "clip-0",
        "clip-1",
    ]


def test_uses_stored_text_mask_when_present(tmp_path: Path) -> None:
    stored_mask = torch.tensor([[[1, 0, 1, 0, 0, 0, 0, 0]]])
    _write_dataset(tmp_path, count=1, text_mask=stored_mask)
    dataset = _open_dataset(tmp_path)

    assert dataset[0]["text_attention_mask"].tolist() == [
        [1, 0, 1, 0, 0, 0, 0, 0]
    ]


def test_bucket_sampler_aligns_sp_ranks_and_shards_dp_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(tmp_path, count=8)
    dataset = _open_dataset(tmp_path)
    module = "fastvideo.dataset.vidaforge_automodel_dataset"
    monkeypatch.setattr(f"{module}.get_world_size", lambda: 4)
    monkeypatch.setattr(f"{module}.get_sp_world_size", lambda: 2)

    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 0)
    sp_rank_0 = list(
        VidaForgeBucketBatchSampler(
            dataset,
            batch_size=2,
            seed=9,
        )
    )
    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 1)
    sp_rank_1 = list(
        VidaForgeBucketBatchSampler(
            dataset,
            batch_size=2,
            seed=9,
        )
    )
    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 2)
    dp_rank_1 = list(
        VidaForgeBucketBatchSampler(
            dataset,
            batch_size=2,
            seed=9,
        )
    )

    assert sp_rank_0 == sp_rank_1
    assert all(
        set(left).isdisjoint(right)
        for left, right in zip(sp_rank_0, dp_rank_1)
    )


def test_stateful_dataloader_returns_fastvideo_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(tmp_path, count=2)
    module = "fastvideo.dataset.vidaforge_automodel_dataset"
    monkeypatch.setattr(f"{module}.get_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_sp_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 0)
    _, dataloader = build_vidaforge_automodel_dataloader(
        tmp_path,
        batch_size=2,
        num_data_workers=0,
        seed=7,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=_VAE_FINGERPRINT,
        expected_text_encoder_fingerprint=_TEXT_ENCODER_FINGERPRINT,
    )

    batch = next(iter(dataloader))

    assert set(batch) == {
        "vae_latent",
        "text_embedding",
        "text_attention_mask",
        "info_list",
    }
    assert batch["vae_latent"].shape == (2, 16, 5, 6, 8)


def test_stateful_dataloader_resume_preserves_next_batch_and_cfg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(tmp_path, count=6)
    module = "fastvideo.dataset.vidaforge_automodel_dataset"
    monkeypatch.setattr(f"{module}.get_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_sp_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 0)

    _, original = build_vidaforge_automodel_dataloader(
        tmp_path,
        batch_size=2,
        num_data_workers=0,
        cfg_rate=0.5,
        seed=19,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=_VAE_FINGERPRINT,
        expected_text_encoder_fingerprint=_TEXT_ENCODER_FINGERPRINT,
    )
    first_epoch = list(original)
    original_iterator = iter(original)
    next(original_iterator)
    state = original.state_dict()
    assert state["epoch"] == 1
    expected = next(original_iterator)

    _, resumed = build_vidaforge_automodel_dataloader(
        tmp_path,
        batch_size=2,
        num_data_workers=0,
        cfg_rate=0.5,
        seed=19,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=_VAE_FINGERPRINT,
        expected_text_encoder_fingerprint=_TEXT_ENCODER_FINGERPRINT,
    )
    resumed.load_state_dict(state)
    actual = next(iter(resumed))

    assert torch.equal(actual["vae_latent"], expected["vae_latent"])
    assert torch.equal(actual["text_embedding"], expected["text_embedding"])
    assert torch.equal(
        actual["text_attention_mask"],
        expected["text_attention_mask"],
    )
    assert actual["info_list"] == expected["info_list"]
    assert len(first_epoch) == 3


def test_each_epoch_reshuffles_samples_and_cfg_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(tmp_path, count=12)
    module = "fastvideo.dataset.vidaforge_automodel_dataset"
    monkeypatch.setattr(f"{module}.get_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_sp_world_size", lambda: 1)
    monkeypatch.setattr(f"{module}.get_world_rank", lambda: 0)
    _, loader = build_vidaforge_automodel_dataloader(
        tmp_path,
        batch_size=2,
        num_data_workers=0,
        cfg_rate=0.5,
        seed=23,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=_VAE_FINGERPRINT,
        expected_text_encoder_fingerprint=_TEXT_ENCODER_FINGERPRINT,
    )

    epochs = [list(loader), list(loader)]
    orders: list[list[str]] = []
    dropped: list[set[str]] = []
    for batches in epochs:
        epoch_order: list[str] = []
        epoch_dropped: set[str] = set()
        for batch in batches:
            for index, info in enumerate(batch["info_list"]):
                clip_id = info["clip_id"]
                epoch_order.append(clip_id)
                if torch.count_nonzero(batch["text_embedding"][index]) == 0:
                    epoch_dropped.add(clip_id)
        orders.append(epoch_order)
        dropped.append(epoch_dropped)

    assert orders[0] != orders[1]
    assert dropped[0] != dropped[1]


def test_rebases_moved_absolute_cache_paths(tmp_path: Path) -> None:
    old_paths = [
        f"/old/cache/17f/64x48/sample-{index}.meta"
        for index in range(2)
    ]
    _write_dataset(tmp_path, count=2, cache_paths=old_paths)
    dataset = _open_dataset(tmp_path)

    assert dataset[0]["info"]["clip_id"] == "clip-0"


def test_rejects_metadata_shard_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shards": ["../outside.json"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes dataset directory"):
        VidaForgeAutoModelDataset(tmp_path)


def test_rejects_dataset_without_full_distributed_bucket_batch(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, count=1)
    dataset = _open_dataset(tmp_path)

    with pytest.raises(ValueError, match="no full distributed bucket batch"):
        VidaForgeBucketBatchSampler(
            dataset,
            batch_size=1,
            num_replicas=2,
            rank=0,
        )


def test_rejects_non_wan_payload(tmp_path: Path) -> None:
    paths = _write_dataset(tmp_path, count=1)
    payload = torch.load(paths[0], weights_only=True)
    payload["metadata"]["model_type"] = "other"
    torch.save(payload, paths[0])
    dataset = _open_dataset(tmp_path)

    with pytest.raises(ValueError, match="only Wan"):
        dataset[0]


def test_rejects_cache_from_a_different_wan_checkpoint(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, count=1)
    dataset = _open_dataset(
        tmp_path,
        expected_model_name="Wan-AI/another-model",
    )

    with pytest.raises(ValueError, match="does not match"):
        dataset[0]


def test_rejects_same_model_basename_from_a_different_hf_org(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, count=1)
    dataset = _open_dataset(
        tmp_path,
        expected_model_name=(
            "OtherOrg/Wan2.1-T2V-1.3B-Diffusers"
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        dataset[0]


def test_rejects_unverified_component_identity_by_default(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, count=1)
    dataset = VidaForgeAutoModelDataset(
        tmp_path,
        expected_model_name=_MODEL_NAME,
    )

    with pytest.raises(ValueError, match="no verifiable component identity"):
        dataset[0]


def test_rejects_component_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, count=1)
    dataset = _open_dataset(
        tmp_path,
        expected_vae_fingerprint="c" * 64,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        dataset[0]


def test_legacy_cache_requires_explicit_unverified_opt_in(
    tmp_path: Path,
) -> None:
    _write_dataset(
        tmp_path,
        count=1,
        include_fingerprints=False,
    )
    dataset = VidaForgeAutoModelDataset(
        tmp_path,
        expected_model_name=_MODEL_NAME,
        allow_unverified_model=True,
    )

    assert dataset[0]["info"]["model_name"] == _MODEL_NAME
