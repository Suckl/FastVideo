# SPDX-License-Identifier: Apache-2.0
"""Opt-in VidaForge multi-bucket producer-to-training GPU integration test.

The test intentionally uses two real clips from a pinned VidaForge-3M release:
one short landscape clip and one longer portrait clip. It exercises:

* FastVideo's native 144p multi-bucket producer;
* tensor parity with the pinned VidaForge Wan AutoModel encoder;
* the real bucketed training loader, Wan 1.3B, and ``FineTuneMethod``;
* a forward/loss/backward/optimizer step with only ``proj_out`` trainable; and
* ``CheckpointManager`` model, optimizer, and dataloader state restoration;
* the public YAML training entrypoint with two-rank FSDP sharding; and
* a fresh entrypoint process resuming from ``latest`` into the next bucket.

Set ``VIDAFORGE_RUN_MULTIBUCKET_TRAINING_INTEGRATION=1`` and point
``VIDAFORGE_REFERENCE_DIR`` at VidaForge commit
``4562d3fbcbd4861fc74c2859950c0237363681bb`` to run it. The default test is
sized for one Modal L40S; set ``VIDAFORGE_INTEGRATION_WORLD_SIZE=2`` for the
Section 5 distributed gate on two GPUs. Also set
``VIDAFORGE_RUN_ENTRYPOINT_INTEGRATION=1`` for the Section 6 YAML/FSDP gate.
It is skipped during ordinary CPU test runs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

_RUN_ENV = "VIDAFORGE_RUN_MULTIBUCKET_TRAINING_INTEGRATION"
_ENTRYPOINT_ENV = "VIDAFORGE_RUN_ENTRYPOINT_INTEGRATION"
_WORLD_SIZE_ENV = "VIDAFORGE_INTEGRATION_WORLD_SIZE"
_RECEIPT_DIR_ENV = "VIDAFORGE_ENTRYPOINT_RECEIPT_DIR"
_RECEIPT_PHASE_ENV = "VIDAFORGE_ENTRYPOINT_RECEIPT_PHASE"
_RECEIPT_MODEL_TARGET = (
    "fastvideo.tests.workflow.vidaforge_entrypoint_receipt."
    "VidaForgeEntrypointReceiptWanModel"
)
_RECEIPT_CALLBACK_TARGET = (
    "fastvideo.tests.workflow.vidaforge_entrypoint_receipt."
    "Section6ReceiptCallback"
)
_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
_VIDAFORGE_REVISION = "4562d3fbcbd4861fc74c2859950c0237363681bb"
_VIDAFORGE_RELEASE_REVISION = "091bdc02d82b8c89a4e4eff54945d286fb328b47"
_VIDAFORGE_RELEASE_DATASET = "VidaForge/VidaForge-3M"
_DEFAULT_DURATIONS_SEC = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "train"
    / "fixtures"
    / "wan_t2v_finetune_min.yaml"
)
_ENTRYPOINT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "train"
    / "configs"
    / "fine_tuning"
    / "wan"
    / "vidaforge_automodel_t2v_lora.yaml"
)


@dataclass(frozen=True, slots=True)
class _ClipSpec:
    clip_id: str
    file_name: str
    bucket_frame_count: int
    bucket_resolution: tuple[int, int]


_CLIPS = (
    _ClipSpec(
        clip_id="video-500c89b96e6fcb86:clip:00004:00",
        file_name="short-landscape.mp4",
        bucket_frame_count=49,
        bucket_resolution=(240, 144),
    ),
    _ClipSpec(
        clip_id="video-5dc4a0e50e9d29f8:clip:00001:01",
        file_name="long-portrait.mp4",
        bucket_frame_count=125,
        bucket_resolution=(144, 240),
    ),
)


def _timeout_seconds() -> int:
    return int(os.environ.get("VIDAFORGE_INTEGRATION_TIMEOUT_SEC", "7200"))


def _integration_world_size() -> int:
    world_size = int(os.environ.get(_WORLD_SIZE_ENV, "1"))
    if world_size <= 0:
        raise ValueError(f"{_WORLD_SIZE_ENV} must be greater than zero")
    return world_size


def _validate_reference_checkout() -> Path:
    reference_dir = Path(
        os.environ.get("VIDAFORGE_REFERENCE_DIR", ""),
    ).expanduser().resolve()
    if not (reference_dir / "vidaforge").is_dir():
        pytest.fail(
            "VIDAFORGE_REFERENCE_DIR must point to the pinned VidaForge "
            "checkout"
        )
    checkout_revision = subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_revision == _VIDAFORGE_REVISION
    return reference_dir


def _release_rows() -> dict[str, dict[str, Any]]:
    metadata_path = hf_hub_download(
        repo_id=_VIDAFORGE_RELEASE_DATASET,
        filename="meta/shard-00000.parquet",
        repo_type="dataset",
        revision=_VIDAFORGE_RELEASE_REVISION,
    )
    rows: dict[str, dict[str, Any]] = {}
    for spec in _CLIPS:
        matches = pq.read_table(
            metadata_path,
            filters=[("clip_id", "=", spec.clip_id)],
        ).to_pylist()
        assert len(matches) == 1
        rows[spec.clip_id] = matches[0]
    return rows


def _download_release_clips(
    rows: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    filesystem = HfFileSystem()
    video_paths: dict[str, Path] = {}
    for spec in _CLIPS:
        row = rows[spec.clip_id]
        remote_tar_path = (
            f"datasets/{_VIDAFORGE_RELEASE_DATASET}"
            f"@{_VIDAFORGE_RELEASE_REVISION}/{row['tar_path']}"
        )
        remaining = int(row["filesize_bytes"])
        chunks: list[bytes] = []
        with filesystem.open(remote_tar_path, "rb") as tar_handle:
            tar_handle.seek(int(row["tar_offset"]))
            while remaining:
                chunk = tar_handle.read(remaining)
                assert chunk
                chunks.append(chunk)
                remaining -= len(chunk)
        payload = b"".join(chunks)
        assert len(payload) == int(row["filesize_bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        video_path = output_dir / spec.file_name
        video_path.write_bytes(payload)
        video_paths[spec.clip_id] = video_path
    return video_paths


def _write_stage4_manifest(
    rows: dict[str, dict[str, Any]],
    video_paths: dict[str, Path],
    path: Path,
) -> None:
    records = []
    for spec in _CLIPS:
        row = rows[spec.clip_id]
        records.append({
            "clip_id": spec.clip_id,
            "clip_path": str(video_paths[spec.clip_id]),
            "clip_ok": 1,
            "select_ok": 1,
            "select_pass": 1,
            "caption_ok": 1,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "fps": float(row["fps"]),
            "duration_sec": float(row["duration_sec"]),
            "caption_level_3": str(row["caption_level_3"]),
        })
    pq.write_table(pa.Table.from_pylist(records), path)


def _run_fastvideo_producer(
    *,
    model_root: Path,
    manifest_path: Path,
    output_dir: Path,
    world_size: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "fastvideo.pipelines.preprocess.v1_preprocessing_new",
        "--model-path",
        str(model_root),
        "--revision",
        _MODEL_REVISION,
        "--mode",
        "preprocess",
        "--workload-type",
        "t2v",
        "--vae-precision",
        "fp16",
        "--text-encoder-precisions",
        "bf16",
        "--preprocess.dataset-type",
        "vidaforge",
        "--preprocess.dataset-path",
        str(manifest_path),
        "--preprocess.vidaforge-selection",
        "all",
        "--preprocess.video-loader-type",
        "torchcodec",
        "--preprocess.output-type",
        "vidaforge_automodel",
        "--preprocess.vidaforge-model-name",
        _MODEL_NAME,
        "--preprocess.dataset-output-dir",
        str(output_dir),
        "--preprocess.vidaforge-bucket-resolution",
        "144p",
        "--preprocess.vidaforge-bucket-durations-sec",
        *(str(duration) for duration in _DEFAULT_DURATIONS_SEC),
        "--preprocess.vidaforge-dynamic-forward-batch-size",
        "1",
        # Fixed geometry is ignored in multi-bucket mode, but keeping valid
        # values makes this command explicit and backwards-compatible.
        "--preprocess.max-height",
        "144",
        "--preprocess.max-width",
        "256",
        "--preprocess.num-frames",
        "17",
        "--preprocess.train-fps",
        "16",
        "--preprocess.drop-short-ratio",
        "1",
        "--preprocess.preprocess-video-batch-size",
        "2",
        "--preprocess.dataloader-num-workers",
        "0",
        "--preprocess.samples-per-file",
        "1",
    ]
    subprocess.run(command, check=True, timeout=_timeout_seconds())


def _expand_cache_for_data_parallel(
    output_dir: Path,
    *,
    world_size: int,
) -> None:
    """Give every oracle-verified bucket one synthetic DP global batch.

    The aliases are consumed only by the training loader in this test. They
    intentionally retain the verified tensor/source fingerprint and do not
    rewrite the producer's summary; the expanded directory must never be
    treated as a resumable producer output.
    """
    if world_size == 1:
        return

    metadata_path = output_dir / "metadata.json"
    root = json.loads(metadata_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for shard_name in root["shards"]:
        shard_path = output_dir / str(shard_name)
        items.extend(json.loads(shard_path.read_text(encoding="utf-8")))
    assert len(items) == len(_CLIPS)

    expanded_items: list[dict[str, Any]] = []
    for item in items:
        source_path = output_dir / str(item["cache_file"])
        source_payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        expanded_items.append(item)
        for replica in range(1, world_size):
            replica_item = copy.deepcopy(item)
            replica_payload = copy.deepcopy(source_payload)
            replica_clip_id = (
                f"{item['clip_id']}:fastvideo-dp-replica:{replica:05d}"
            )
            digest = hashlib.sha256(
                replica_clip_id.encode("utf-8")
            ).hexdigest()
            replica_path = (
                source_path.parent
                / "section5-dp"
                / f"{digest}.meta"
            )
            replica_path.parent.mkdir(parents=True, exist_ok=True)
            replica_payload["metadata"]["clip_id"] = replica_clip_id
            torch.save(replica_payload, replica_path)
            replica_item["clip_id"] = replica_clip_id
            replica_item["cache_file"] = (
                replica_path.relative_to(output_dir).as_posix()
            )
            expanded_items.append(replica_item)

    expanded_items.sort(key=lambda item: str(item["clip_id"]))
    shard_path = output_dir / "shards" / "section5-distributed.json"
    shard_path.write_text(
        json.dumps(expanded_items, sort_keys=True),
        encoding="utf-8",
    )
    root["shards"] = [shard_path.relative_to(output_dir).as_posix()]
    metadata_path.write_text(
        json.dumps(root, sort_keys=True),
        encoding="utf-8",
    )


def test_expand_cache_for_data_parallel_preserves_bucket_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "stage5"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True)
    items = []
    for index, spec in enumerate(_CLIPS):
        cache_path = (
            output_dir
            / f"{spec.bucket_frame_count}f"
            / f"{spec.bucket_resolution[0]}x{spec.bucket_resolution[1]}"
            / f"sample-{index}.meta"
        )
        cache_path.parent.mkdir(parents=True)
        latent_shape = (
            1,
            16,
            (spec.bucket_frame_count - 1) // 4 + 1,
            spec.bucket_resolution[1] // 8,
            spec.bucket_resolution[0] // 8,
        )
        torch.save(
            {
                "video_latents": torch.zeros(
                    latent_shape,
                    dtype=torch.float16,
                ),
                "text_embeddings": torch.zeros(
                    (1, 8, 12),
                    dtype=torch.bfloat16,
                ),
                "text_mask": torch.ones((1, 8)),
                "metadata": {
                    "model_type": "wan",
                    "model_name": _MODEL_NAME,
                    "clip_id": spec.clip_id,
                    "caption": "test",
                    "caption_token_length": 8,
                    "bucket_resolution": list(spec.bucket_resolution),
                    "bucket_frame_count": spec.bucket_frame_count,
                    "vae_fingerprint": "a" * 64,
                    "text_encoder_fingerprint": "b" * 64,
                },
                "bucket_frame_count": spec.bucket_frame_count,
            },
            cache_path,
        )
        items.append({
            "cache_file": cache_path.relative_to(output_dir).as_posix(),
            "bucket_resolution": list(spec.bucket_resolution),
            "bucket_frame_count": spec.bucket_frame_count,
            "latent_shape": list(latent_shape),
            "clip_id": spec.clip_id,
            "caption_token_length": 8,
        })
    (shard_dir / "metadata.json").write_text(
        json.dumps(items),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({"shards": ["shards/metadata.json"]}),
        encoding="utf-8",
    )

    _expand_cache_for_data_parallel(output_dir, world_size=2)

    from fastvideo.dataset.vidaforge_automodel_dataset import (
        VidaForgeAutoModelDataset,
    )

    dataset = VidaForgeAutoModelDataset(
        output_dir,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint="a" * 64,
        expected_text_encoder_fingerprint="b" * 64,
    )
    assert len(dataset) == 4
    assert sorted(len(indices) for indices in dataset.bucket_groups.values()) == [
        2,
        2,
    ]
    assert len({
        dataset[index]["info"]["clip_id"]
        for index in range(len(dataset))
    }) == 4


def _load_produced_samples(
    output_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    from fastvideo.dataset.vidaforge_automodel_dataset import (
        VidaForgeAutoModelDataset,
    )

    dataset = VidaForgeAutoModelDataset(
        output_dir,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=provenance["vae_fingerprint"],
        expected_text_encoder_fingerprint=(
            provenance["text_encoder_fingerprint"]
        ),
    )
    assert len(dataset) == len(_CLIPS)
    samples = {
        str(sample["info"]["clip_id"]): sample
        for sample in (dataset[index] for index in range(len(dataset)))
    }
    assert set(samples) == {spec.clip_id for spec in _CLIPS}
    return samples


def _assert_oracle_parity(
    *,
    reference_dir: Path,
    model_root: Path,
    rows: dict[str, dict[str, Any]],
    video_paths: dict[str, Path],
    samples: dict[str, dict[str, Any]],
) -> None:
    sys.path.insert(0, str(reference_dir))
    try:
        from vidaforge.packaging.automodel.wan import WanAutoModelEncoder
    finally:
        sys.path.remove(str(reference_dir))

    encoder = WanAutoModelEncoder(
        model_name=model_root,
        device="cuda",
        max_sequence_length=512,
        deterministic_latents=True,
    )
    try:
        for spec in _CLIPS:
            row = rows[spec.clip_id]
            oracle = encoder.encode_batch(
                bucket_frame_count=spec.bucket_frame_count,
                bucket_resolution=spec.bucket_resolution,
                video_paths=[video_paths[spec.clip_id]],
                source_resolutions=[(
                    int(row["width"]),
                    int(row["height"]),
                )],
                source_fps=[float(row["fps"])],
                captions=[str(row["caption_level_3"])],
            )[0]
            sample = samples[spec.clip_id]
            torch.testing.assert_close(
                sample["vae_latent"].float(),
                oracle.video_latents.float(),
                # The producer persists FP16 after an independent CUDA VAE
                # forward, so allow one FP16-scale rounding step on larger
                # latent values while still catching preprocessing drift.
                atol=1e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                sample["text_embedding"].float(),
                oracle.text_embeddings.float(),
                atol=5e-3,
                rtol=5e-3,
            )
            assert int(sample["text_attention_mask"].sum()) == min(
                int(oracle.metadata["caption_token_length"]),
                512,
            )
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()


def _run_training_worker(
    *,
    model_root: Path,
    cache_dir: Path,
    checkpoint_dir: Path,
    result_path: Path,
    provenance: dict[str, Any],
    world_size: int,
) -> None:
    environment = dict(os.environ)
    environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(Path(__file__).resolve()),
        "--training-worker",
        "--model-root",
        str(model_root),
        "--cache-dir",
        str(cache_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--result-path",
        str(result_path),
        "--vae-fingerprint",
        str(provenance["vae_fingerprint"]),
        "--text-encoder-fingerprint",
        str(provenance["text_encoder_fingerprint"]),
    ]
    subprocess.run(
        command,
        check=True,
        timeout=_timeout_seconds(),
        env=environment,
    )


def _checkpoint_tree_fingerprint(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    )
    assert files
    for path in files:
        digest.update(
            path.relative_to(checkpoint_dir).as_posix().encode("utf-8")
        )
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _run_entrypoint_phase(
    *,
    model_root: Path,
    cache_dir: Path,
    checkpoint_dir: Path,
    provenance: dict[str, Any],
    world_size: int,
    max_steps: int,
    resume_from_checkpoint: str | None,
    receipt_dir: Path,
    receipt_phase: str,
    save_steps: int = 1,
) -> None:
    environment = dict(os.environ)
    environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    environment[_RECEIPT_DIR_ENV] = str(receipt_dir)
    environment[_RECEIPT_PHASE_ENV] = receipt_phase
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("WANDB_MODE", "disabled")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "fastvideo.train.entrypoint.train",
        "--config",
        str(_ENTRYPOINT_CONFIG),
        "--models.student._target_",
        _RECEIPT_MODEL_TARGET,
        "--models.student.init_from",
        str(model_root),
        "--callbacks.section6_receipt._target_",
        _RECEIPT_CALLBACK_TARGET,
        "--training.distributed.num_gpus",
        str(world_size),
        "--training.distributed.sp_size",
        "1",
        "--training.distributed.tp_size",
        "1",
        "--training.distributed.hsdp_replicate_dim",
        "1",
        "--training.distributed.hsdp_shard_dim",
        str(world_size),
        "--training.data.data_path",
        str(cache_dir),
        "--training.data.vidaforge_model_name",
        _MODEL_NAME,
        "--training.data.vidaforge_vae_fingerprint",
        str(provenance["vae_fingerprint"]),
        "--training.data.vidaforge_text_encoder_fingerprint",
        str(provenance["text_encoder_fingerprint"]),
        "--training.data.dataloader_num_workers",
        "0",
        "--training.loop.max_train_steps",
        str(max_steps),
        "--training.checkpoint.output_dir",
        str(checkpoint_dir),
        "--training.checkpoint.training_state_checkpointing_steps",
        str(save_steps),
        "--training.checkpoint.checkpoints_total_limit",
        "3",
        "--training.checkpoint.resume_from_checkpoint",
        (
            resume_from_checkpoint
            if resume_from_checkpoint is not None
            else "null"
        ),
    ]
    subprocess.run(
        command,
        check=True,
        timeout=_timeout_seconds(),
        env=environment,
    )


def _assert_entrypoint_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    world_size: int,
    resume_from_checkpoint: str | None,
) -> None:
    assert (checkpoint_dir / "dcp").is_dir()
    for rank in range(world_size):
        assert (checkpoint_dir / f"rng_state_rank{rank}.pt").is_file()

    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["step"] == step
    raw_config = metadata["config"]
    distributed = raw_config["training"]["distributed"]
    assert distributed["num_gpus"] == world_size
    assert distributed["sp_size"] == 1
    assert distributed["tp_size"] == 1
    assert distributed["hsdp_replicate_dim"] == 1
    assert distributed["hsdp_shard_dim"] == world_size
    checkpoint_config = raw_config["training"]["checkpoint"]
    assert checkpoint_config["resume_from_checkpoint"] == (
        resume_from_checkpoint
    )


def _assert_checkpoint_contains_lora_state(
    checkpoint_dir: Path,
) -> None:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(
        str(checkpoint_dir / "dcp")
    ).read_metadata()
    keys = [
        str(key)
        for key in metadata.state_dict_metadata
    ]
    lora_keys = [
        key
        for key in keys
        if ".lora_A" in key or ".lora_B" in key
    ]
    assert lora_keys, (
        "DCP checkpoint does not contain LoRA parameter state; "
        f"sample keys: {keys[:20]}"
    )


def _run_entrypoint_training_and_resume(
    *,
    model_root: Path,
    cache_dir: Path,
    checkpoint_dir: Path,
    provenance: dict[str, Any],
    world_size: int,
    receipt_dir: Path,
) -> None:
    if world_size != 2:
        pytest.fail(
            f"{_ENTRYPOINT_ENV}=1 requires exactly two GPUs so the gate "
            "exercises two-rank FSDP sharding"
        )

    _run_entrypoint_phase(
        model_root=model_root,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        provenance=provenance,
        world_size=world_size,
        max_steps=1,
        resume_from_checkpoint=None,
        receipt_dir=receipt_dir,
        receipt_phase="initial",
    )
    first_checkpoint = checkpoint_dir / "checkpoint-1"
    _assert_entrypoint_checkpoint(
        first_checkpoint,
        step=1,
        world_size=world_size,
        resume_from_checkpoint=None,
    )
    _assert_checkpoint_contains_lora_state(first_checkpoint)
    first_checkpoint_fingerprint = _checkpoint_tree_fingerprint(
        first_checkpoint
    )

    # Start a new torchrun process. A fresh two-step run would save and
    # overwrite checkpoint-1; preserving it proves that ``latest`` resumed at
    # step 1 and executed only the next bucket before writing checkpoint-2.
    _run_entrypoint_phase(
        model_root=model_root,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        provenance=provenance,
        world_size=world_size,
        max_steps=2,
        resume_from_checkpoint="latest",
        receipt_dir=receipt_dir,
        receipt_phase="resumed",
    )
    assert _checkpoint_tree_fingerprint(
        first_checkpoint
    ) == first_checkpoint_fingerprint
    _assert_entrypoint_checkpoint(
        checkpoint_dir / "checkpoint-2",
        step=2,
        world_size=world_size,
        resume_from_checkpoint="latest",
    )

    # A clean uninterrupted two-step run is the oracle for all state that
    # matters at the entrypoint boundary. The instrumented Wan subclass only
    # writes receipts; model construction and training still flow through the
    # public YAML entrypoint.
    _run_entrypoint_phase(
        model_root=model_root,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir.parent / "continuous-checkpoints",
        provenance=provenance,
        world_size=world_size,
        max_steps=2,
        resume_from_checkpoint=None,
        receipt_dir=receipt_dir,
        receipt_phase="continuous",
    )
    initial_batches: list[dict[str, Any]] = []
    resumed_batches: list[dict[str, Any]] = []
    initial_posts: list[dict[str, Any]] = []
    resumed_posts: list[dict[str, Any]] = []
    for rank in range(world_size):
        initial_batch = _read_entrypoint_receipt(
            receipt_dir,
            phase="initial",
            rank=rank,
            kind="batch",
            index=1,
        )
        resumed_batch = _read_entrypoint_receipt(
            receipt_dir,
            phase="resumed",
            rank=rank,
            kind="batch",
            index=1,
        )
        continuous_first_batch = _read_entrypoint_receipt(
            receipt_dir,
            phase="continuous",
            rank=rank,
            kind="batch",
            index=1,
        )
        continuous_second_batch = _read_entrypoint_receipt(
            receipt_dir,
            phase="continuous",
            rank=rank,
            kind="batch",
            index=2,
        )
        _assert_entrypoint_batch_receipts_match(
            initial_batch,
            continuous_first_batch,
            label=f"rank {rank} initial vs continuous step 1",
        )
        initial_batches.append(initial_batch)
        resumed_batches.append(resumed_batch)
        assert initial_batch["clip_ids"] != resumed_batch["clip_ids"]
        assert initial_batch["bucket_frame_counts"] != (
            resumed_batch["bucket_frame_counts"]
        )
        assert resumed_batch["dtensor_parameter_count"] > 0
        assert resumed_batch["sharded_parameter_count"] > 0
        assert any(
            "Shard(" in placement
            for example in resumed_batch["sharded_parameter_examples"]
            for placement in example["placements"]
        )

        initial_post = _read_entrypoint_receipt(
            receipt_dir,
            phase="initial",
            rank=rank,
            kind="post-step",
            index=1,
        )
        resumed_post = _read_entrypoint_receipt(
            receipt_dir,
            phase="resumed",
            rank=rank,
            kind="post-step",
            index=2,
        )
        continuous_first_post = _read_entrypoint_receipt(
            receipt_dir,
            phase="continuous",
            rank=rank,
            kind="post-step",
            index=1,
        )
        continuous_second_post = _read_entrypoint_receipt(
            receipt_dir,
            phase="continuous",
            rank=rank,
            kind="post-step",
            index=2,
        )
        assert initial_post["trainable_model_sha256"] == (
            resumed_batch["pre_step_trainable_model_sha256"]
        ), f"rank {rank} checkpoint did not restore the saved LoRA state"
        assert continuous_first_post["trainable_model_sha256"] == (
            continuous_second_batch["pre_step_trainable_model_sha256"]
        ), f"rank {rank} checkpoint save mutated the live LoRA state"
        _assert_entrypoint_post_step_receipts_match(
            initial_post,
            continuous_first_post,
            label=f"rank {rank} initial vs continuous step 1",
        )
        _assert_entrypoint_batch_receipts_match(
            resumed_batch,
            continuous_second_batch,
            label=f"rank {rank} resumed vs continuous step 2",
        )
        _assert_entrypoint_post_step_receipts_match(
            resumed_post,
            continuous_second_post,
            label=f"rank {rank} resumed vs continuous step 2",
        )
        initial_posts.append(initial_post)
        resumed_posts.append(resumed_post)

    # LoRA parameters use replicated DTensor placements and must begin and
    # remain identical across data-parallel ranks.
    assert len({
        receipt["pre_step_trainable_model_sha256"]
        for receipt in initial_batches
    }) == 1
    assert len({
        receipt["pre_step_trainable_model_sha256"]
        for receipt in resumed_batches
    }) == 1
    assert len({
        receipt["trainable_model_sha256"]
        for receipt in initial_posts
    }) == 1
    assert len({
        receipt["trainable_model_sha256"]
        for receipt in resumed_posts
    }) == 1
    assert len({
        receipt["optimizer_sha256"]
        for receipt in resumed_posts
    }) == 1


def _read_entrypoint_receipt(
    receipt_dir: Path,
    *,
    phase: str,
    rank: int,
    kind: str,
    index: int,
) -> dict[str, Any]:
    path = (
        receipt_dir
        / f"{phase}-rank-{rank:05d}-{kind}-{index:05d}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_entrypoint_batch_receipts_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    keys = (
        "clip_ids",
        "bucket_frame_counts",
        "bucket_resolutions",
        "noise_sha256",
        "timesteps_sha256",
        "pre_step_trainable_model_sha256",
    )
    mismatches = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in keys
        if actual[key] != expected[key]
    }
    assert not mismatches, (
        f"{label}: {json.dumps(mismatches, sort_keys=True)}"
    )


def _assert_entrypoint_post_step_receipts_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    keys = (
        "iteration",
        "trainable_model_sha256",
        "optimizer_sha256",
        "total_loss",
    )
    mismatches = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in keys
        if actual[key] != expected[key]
    }
    assert not mismatches, (
        f"{label}: {json.dumps(mismatches, sort_keys=True)}"
    )


def test_entrypoint_phase_uses_documented_two_rank_fsdp_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(
        command: list[str],
        **kwargs: Any,
    ) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run_entrypoint_phase(
        model_root=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "checkpoints",
        provenance={
            "vae_fingerprint": "a" * 64,
            "text_encoder_fingerprint": "b" * 64,
        },
        world_size=2,
        max_steps=1,
        resume_from_checkpoint=None,
        receipt_dir=tmp_path / "receipts",
        receipt_phase="initial",
    )

    assert len(calls) == 1
    command, kwargs = calls[0]

    def override_value(name: str) -> str:
        return command[command.index(name) + 1]

    assert override_value("--config") == str(_ENTRYPOINT_CONFIG)
    assert override_value("--models.student._target_") == (
        _RECEIPT_MODEL_TARGET
    )
    assert override_value(
        "--callbacks.section6_receipt._target_"
    ) == _RECEIPT_CALLBACK_TARGET
    assert override_value("--training.distributed.num_gpus") == "2"
    assert override_value(
        "--training.distributed.hsdp_replicate_dim"
    ) == "1"
    assert override_value("--training.distributed.hsdp_shard_dim") == "2"
    assert override_value("--training.loop.max_train_steps") == "1"
    assert override_value(
        "--training.checkpoint.resume_from_checkpoint"
    ) == "null"
    assert kwargs["check"] is True
    assert kwargs["env"]["FASTVIDEO_ATTENTION_BACKEND"] == "TORCH_SDPA"


@pytest.mark.skipif(
    os.environ.get(_RUN_ENV) != "1",
    reason=f"set {_RUN_ENV}=1 to run the producer-to-training GPU test",
)
def test_vidaforge_multibucket_producer_training_and_resume(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.fail("The VidaForge multi-bucket integration test requires CUDA")

    world_size = _integration_world_size()
    if torch.cuda.device_count() < world_size:
        pytest.fail(
            f"The VidaForge integration requested {world_size} GPUs via "
            f"{_WORLD_SIZE_ENV}, but found {torch.cuda.device_count()}"
        )

    reference_dir = _validate_reference_checkout()
    rows = _release_rows()
    video_paths = _download_release_clips(rows, tmp_path)
    manifest_path = tmp_path / "stage4.parquet"
    _write_stage4_manifest(rows, video_paths, manifest_path)

    model_root = Path(
        snapshot_download(
            _MODEL_NAME,
            revision=_MODEL_REVISION,
        )
    ).resolve()
    output_dir = tmp_path / "fastvideo-stage5"
    _run_fastvideo_producer(
        model_root=model_root,
        manifest_path=manifest_path,
        output_dir=output_dir,
        world_size=world_size,
    )

    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["model_revision"] == _MODEL_REVISION
    bucket_policy = provenance["producer_config"]["bucket_policy"]
    assert bucket_policy["mode"] == "multi"
    assert bucket_policy["resolution"] == "144p"
    assert bucket_policy["durations_sec"] == list(_DEFAULT_DURATIONS_SEC)
    assert bucket_policy["reference_revision"] == _VIDAFORGE_REVISION
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["input_count"] == 2
    assert summary["ok_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["bucket_counts"] == {
        "49f/240x144": 1,
        "125f/144x240": 1,
    }

    samples = _load_produced_samples(output_dir, provenance)
    actual_buckets = {
        (
            int(sample["info"]["bucket_frame_count"]),
            tuple(sample["info"]["bucket_resolution"]),
        )
        for sample in samples.values()
    }
    expected_buckets = {
        (spec.bucket_frame_count, spec.bucket_resolution)
        for spec in _CLIPS
    }
    assert actual_buckets == expected_buckets
    assert len({frame_count for frame_count, _resolution in actual_buckets}) >= 2
    assert len({resolution for _frame_count, resolution in actual_buckets}) >= 2

    _assert_oracle_parity(
        reference_dir=reference_dir,
        model_root=model_root,
        rows=rows,
        video_paths=video_paths,
        samples=samples,
    )
    del samples
    gc.collect()
    torch.cuda.empty_cache()

    _expand_cache_for_data_parallel(
        output_dir,
        world_size=world_size,
    )
    result_path = tmp_path / "training-result.json"
    _run_training_worker(
        model_root=model_root,
        cache_dir=output_dir,
        checkpoint_dir=tmp_path / "checkpoints",
        result_path=result_path,
        provenance=provenance,
        world_size=world_size,
    )
    results = [
        json.loads(
            _rank_result_path(result_path, rank).read_text(encoding="utf-8")
        )
        for rank in range(world_size)
    ]
    assert len({result["first_clip_id"] for result in results}) == world_size
    assert len({result["resumed_clip_id"] for result in results}) == world_size
    assert len({result["first_raw_temporal"] for result in results}) == 1
    assert len({result["resumed_raw_temporal"] for result in results}) == 1
    for result in results:
        assert result["first_raw_temporal"] > 1
        assert result["first_prepared_temporal"] == result["first_raw_temporal"]
        assert result["resumed_raw_temporal"] > 1
        assert result["resumed_prepared_temporal"] == (
            result["resumed_raw_temporal"]
        )
        assert result["first_raw_temporal"] != result["resumed_raw_temporal"]
        assert result["num_latent_t"] == 1
        assert result["vae_load_attempts"] == 0
        assert result["checkpoint_step"] == 1
        assert math.isfinite(result["loss"])
        assert result["loss"] >= 0
        assert math.isfinite(result["resumed_loss"])
        assert result["resumed_loss"] >= 0

    if os.environ.get(_ENTRYPOINT_ENV) == "1":
        _run_entrypoint_training_and_resume(
            model_root=model_root,
            cache_dir=output_dir,
            checkpoint_dir=tmp_path / "entrypoint-checkpoints",
            provenance=provenance,
            world_size=world_size,
            receipt_dir=tmp_path / "entrypoint-receipts",
        )


def _rank_result_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}-rank-{rank:05d}{path.suffix}")


def _optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state: dict[str, Any] = {}
            for key, value in optimizer.state[parameter].items():
                state[key] = (
                    value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
            snapshots.append(state)
    return snapshots


def _assert_optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
    expected: list[dict[str, Any]],
) -> None:
    actual = _optimizer_snapshot(optimizer)
    assert len(actual) == len(expected)
    for actual_state, expected_state in zip(actual, expected):
        assert actual_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = actual_state[key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(
                    actual_value,
                    expected_value,
                    atol=0,
                    rtol=0,
                )
            else:
                assert actual_value == expected_value


def _module_state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_module_state(
    module: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(
            actual[name].detach().cpu(),
            expected_tensor,
            atol=0,
            rtol=0,
        )


def _batch_snapshot(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_ids": [
            str(info["clip_id"])
            for info in batch["info_list"]
        ],
        "vae_latent": batch["vae_latent"].detach().cpu().clone(),
        "text_embedding": batch["text_embedding"].detach().cpu().clone(),
        "text_attention_mask": (
            batch["text_attention_mask"].detach().cpu().clone()
        ),
    }


def _assert_batch_snapshot(
    batch: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert [
        str(info["clip_id"])
        for info in batch["info_list"]
    ] == expected["clip_ids"]
    for field in (
        "vae_latent",
        "text_embedding",
        "text_attention_mask",
    ):
        assert torch.equal(batch[field].cpu(), expected[field])


def _training_worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-worker", action="store_true", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--vae-fingerprint", required=True)
    parser.add_argument("--text-encoder-fingerprint", required=True)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The VidaForge training worker requires CUDA")

    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.platforms import AttentionBackendEnum
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    import fastvideo.train.models.wan.wan as wan_implementation
    from fastvideo.train.trainer import Trainer
    from fastvideo.train.utils.checkpoint import (
        CheckpointConfig,
        CheckpointManager,
    )
    from fastvideo.train.utils.config import load_run_config

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    original_loader = wan_implementation.load_module_from_path
    vae_load_attempts: list[str] = []

    def tracked_load_module_from_path(*loader_args: Any, **loader_kwargs: Any) -> Any:
        if loader_kwargs.get("module_type") == "vae":
            vae_load_attempts.append(str(loader_kwargs.get("model_path", "")))
            raise AssertionError(
                "VidaForge pre-encoded training must not load the Wan VAE"
            )
        return original_loader(*loader_args, **loader_kwargs)

    wan_implementation.load_module_from_path = tracked_load_module_from_path

    overrides = [
        "--models.student.init_from",
        str(args.model_root),
        "--training.data.data_path",
        str(args.cache_dir),
        "--training.data.preprocessed_data_type",
        "vidaforge_automodel",
        "--training.data.vidaforge_model_name",
        _MODEL_NAME,
        "--training.data.vidaforge_vae_fingerprint",
        args.vae_fingerprint,
        "--training.data.vidaforge_text_encoder_fingerprint",
        args.text_encoder_fingerprint,
        "--training.data.train_batch_size",
        "1",
        "--training.data.dataloader_num_workers",
        "0",
        "--training.data.training_cfg_rate",
        "0",
        "--training.data.num_latent_t",
        "1",
        "--training.distributed.num_gpus",
        str(torch.distributed.get_world_size()),
        "--training.distributed.sp_size",
        "1",
        "--training.distributed.tp_size",
        "1",
        "--training.distributed.hsdp_replicate_dim",
        str(torch.distributed.get_world_size()),
        "--training.distributed.hsdp_shard_dim",
        "1",
        "--training.optimizer.learning_rate",
        "0.001",
        "--training.optimizer.weight_decay",
        "0",
        "--training.loop.max_train_steps",
        "2",
    ]

    try:
        cfg = load_run_config(str(_FIXTURE), overrides=overrides)
        device = torch.device(
            "cuda",
            args.local_rank,
        )

        def build_student() -> Any:
            model = wan_implementation.WanModel(
                init_from=str(args.model_root),
                training_config=cfg.training,
                trainable=True,
                attention_backend=AttentionBackendEnum.TORCH_SDPA,
            )
            for parameter in model.transformer.parameters():
                parameter.requires_grad_(False)
            proj_out = model.transformer.proj_out
            for parameter in proj_out.parameters():
                parameter.requires_grad_(True)
            model.transformer = model.transformer.to(
                device=device,
                dtype=torch.bfloat16,
            )
            # Keep the tiny trainable surface in fp32 so the optimizer step is
            # numerically observable while the 1.3B frozen backbone stays bf16.
            model.transformer.proj_out.float()
            return model

        model = build_student()
        method = FineTuneMethod(
            cfg=cfg,
            role_models={"student": model},
        )
        assert model.vae is None
        assert model.negative_prompt_embeds is None
        assert vae_load_attempts == []
        method.on_train_start()

        dataloader = model.dataloader
        iterator = iter(dataloader)
        first_batch = next(iterator)
        first_raw_temporal = int(first_batch["vae_latent"].shape[2])
        first_clip_id = str(first_batch["info_list"][0]["clip_id"])
        observed_temporal: list[tuple[int, int]] = []
        original_prepare_batch = model.prepare_batch

        def observed_prepare_batch(
            raw_batch: dict[str, Any],
            **kwargs: Any,
        ) -> Any:
            prepared = original_prepare_batch(raw_batch, **kwargs)
            assert prepared.latents is not None
            observed_temporal.append((
                int(raw_batch["vae_latent"].shape[2]),
                int(prepared.latents.shape[1]),
            ))
            return prepared

        model.prepare_batch = observed_prepare_batch
        proj_out = model.transformer.proj_out
        before_step = _module_state_cpu(proj_out)
        method.optimizers_zero_grad(iteration=0)
        loss_map, outputs, _metrics = method.single_train_step(
            first_batch,
            iteration=0,
        )
        loss = loss_map["total_loss"]
        assert torch.isfinite(loss).item()
        loss_value = float(loss.detach().cpu())
        method.backward(loss_map, outputs, grad_accum_rounds=1)
        trainable_parameters = list(proj_out.parameters())
        assert trainable_parameters
        assert all(parameter.grad is not None for parameter in trainable_parameters)
        assert all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable_parameters
            if parameter.grad is not None
        )
        method.optimizers_schedulers_step(iteration=0)
        trained_state = _module_state_cpu(proj_out)
        assert any(
            not torch.equal(before_step[name], trained_state[name])
            for name in trained_state
        )
        first_prepared_temporal = observed_temporal[-1][1]
        assert observed_temporal[-1] == (
            first_raw_temporal,
            first_raw_temporal,
        )
        assert first_raw_temporal > int(cfg.training.data.num_latent_t)
        assert model.vae is None
        assert vae_load_attempts == []

        optimizer_snapshot = _optimizer_snapshot(
            method._student_optimizer,
        )
        checkpoint_manager = CheckpointManager(
            method=method,
            dataloader=dataloader,
            output_dir=str(args.checkpoint_dir),
            config=CheckpointConfig(save_steps=1, keep_last=1),
        )
        checkpoint_manager.save(step=1)
        checkpoint_path = args.checkpoint_dir / "checkpoint-1"
        assert (checkpoint_path / "dcp").is_dir()
        assert (
            checkpoint_path
            / f"rng_state_rank{torch.distributed.get_rank()}.pt"
        ).is_file()

        expected_next_batch = next(iterator)
        expected_next = _batch_snapshot(expected_next_batch)
        resumed_raw_temporal = int(
            expected_next_batch["vae_latent"].shape[2]
        )
        resumed_clip_id = str(
            expected_next_batch["info_list"][0]["clip_id"]
        )
        assert resumed_raw_temporal != first_raw_temporal
        expected_training_batch = original_prepare_batch(
            expected_next_batch,
            generator=method.cuda_generator,
        )
        assert expected_training_batch.noise is not None
        assert expected_training_batch.timesteps is not None
        expected_noise = expected_training_batch.noise.detach().cpu().clone()
        expected_timesteps = (
            expected_training_batch.timesteps.detach().cpu().clone()
        )

        model.prepare_batch = original_prepare_batch
        del (
            checkpoint_manager,
            dataloader,
            expected_next_batch,
            expected_training_batch,
            first_batch,
            iterator,
            loss_map,
            outputs,
            loss,
            method,
            model,
            proj_out,
            original_prepare_batch,
            observed_prepare_batch,
            trainable_parameters,
            _metrics,
        )
        gc.collect()
        torch.cuda.empty_cache()

        resumed_model = build_student()
        fresh_state = _module_state_cpu(resumed_model.transformer.proj_out)
        assert any(
            not torch.equal(fresh_state[name], trained_state[name])
            for name in trained_state
        )
        resumed_method = FineTuneMethod(
            cfg=cfg,
            role_models={"student": resumed_model},
        )
        assert resumed_model.vae is None
        resumed_method.on_train_start()
        resumed_method.seed_optimizer_state_for_resume()
        resumed_manager = CheckpointManager(
            method=resumed_method,
            dataloader=resumed_model.dataloader,
            output_dir=str(args.checkpoint_dir),
            config=CheckpointConfig(save_steps=1, keep_last=1),
        )
        checkpoint_step = resumed_manager.maybe_resume(
            resume_from_checkpoint=str(checkpoint_path),
        )
        assert checkpoint_step == 1
        _assert_module_state(
            resumed_model.transformer.proj_out,
            trained_state,
        )
        _assert_optimizer_snapshot(
            resumed_method._student_optimizer,
            optimizer_snapshot,
        )

        # Exercise Trainer's real ordering: construct the underlying stateful
        # iterator eagerly, then restore RNG last before requesting a batch.
        resumed_stream = object.__new__(Trainer)._iter_dataloader(
            resumed_model.dataloader
        )
        resumed_manager.load_rng_snapshot(str(checkpoint_path))
        resumed_batch = next(resumed_stream)
        _assert_batch_snapshot(resumed_batch, expected_next)
        resumed_observed_temporal: list[tuple[int, int]] = []
        resumed_original_prepare_batch = resumed_model.prepare_batch

        def resumed_observed_prepare_batch(
            raw_batch: dict[str, Any],
            **kwargs: Any,
        ) -> Any:
            prepared = resumed_original_prepare_batch(raw_batch, **kwargs)
            assert prepared.latents is not None
            assert prepared.noise is not None
            assert prepared.timesteps is not None
            torch.testing.assert_close(
                prepared.noise.detach().cpu(),
                expected_noise,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                prepared.timesteps.detach().cpu(),
                expected_timesteps,
                atol=0,
                rtol=0,
            )
            resumed_observed_temporal.append(
                (
                    int(raw_batch["vae_latent"].shape[2]),
                    int(prepared.latents.shape[1]),
                )
            )
            return prepared

        resumed_model.prepare_batch = resumed_observed_prepare_batch
        resumed_method.optimizers_zero_grad(iteration=1)
        resumed_loss_map, resumed_outputs, _resumed_metrics = (
            resumed_method.single_train_step(
                resumed_batch,
                iteration=1,
            )
        )
        resumed_loss = resumed_loss_map["total_loss"]
        assert torch.isfinite(resumed_loss).item()
        resumed_method.backward(
            resumed_loss_map,
            resumed_outputs,
            grad_accum_rounds=1,
        )
        resumed_trainable_parameters = list(
            resumed_model.transformer.proj_out.parameters()
        )
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            for parameter in resumed_trainable_parameters
        )
        resumed_method.optimizers_schedulers_step(iteration=1)
        resumed_prepared_temporal = resumed_observed_temporal[-1][1]
        assert resumed_prepared_temporal == resumed_raw_temporal
        assert resumed_raw_temporal > int(cfg.training.data.num_latent_t)
        assert resumed_model.vae is None
        assert vae_load_attempts == []

        _rank_result_path(
            args.result_path,
            torch.distributed.get_rank(),
        ).write_text(
            json.dumps(
                {
                    "first_clip_id": first_clip_id,
                    "first_raw_temporal": first_raw_temporal,
                    "first_prepared_temporal": first_prepared_temporal,
                    "resumed_clip_id": resumed_clip_id,
                    "resumed_raw_temporal": resumed_raw_temporal,
                    "resumed_prepared_temporal": (
                        resumed_prepared_temporal
                    ),
                    "num_latent_t": int(cfg.training.data.num_latent_t),
                    "vae_load_attempts": len(vae_load_attempts),
                    "checkpoint_step": checkpoint_step,
                    "loss": loss_value,
                    "resumed_loss": float(resumed_loss.detach().cpu()),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        wan_implementation.load_module_from_path = original_loader
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    _training_worker_main()
