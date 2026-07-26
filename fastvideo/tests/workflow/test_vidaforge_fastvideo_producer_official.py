# SPDX-License-Identifier: Apache-2.0
"""Opt-in GPU smoke test for FastVideo's native Stage 5 producer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

from fastvideo.dataset.vidaforge_automodel_dataset import VidaForgeAutoModelDataset

_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
_VIDAFORGE_REVISION = "4562d3fbcbd4861fc74c2859950c0237363681bb"
_VIDAFORGE_RELEASE_REVISION = "091bdc02d82b8c89a4e4eff54945d286fb328b47"
_CLIP_ID = "video-9a2221ec0d47d85c:clip:00002:02"


@pytest.mark.skipif(
    os.environ.get("VIDAFORGE_RUN_FASTVIDEO_PRODUCER_SMOKE") != "1",
    reason="set VIDAFORGE_RUN_FASTVIDEO_PRODUCER_SMOKE=1 to run the native Wan producer",
)
def test_fastvideo_wan_producer_encodes_official_release_clip(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("The FastVideo VidaForge producer smoke test requires CUDA")

    metadata_path = hf_hub_download(
        repo_id="VidaForge/VidaForge-3M",
        filename="meta/shard-00000.parquet",
        repo_type="dataset",
        revision=_VIDAFORGE_RELEASE_REVISION,
    )
    rows = pq.read_table(metadata_path, filters=[("clip_id", "=", _CLIP_ID)]).to_pylist()
    assert len(rows) == 1
    official_row = rows[0]

    remote_tar_path = f"datasets/VidaForge/VidaForge-3M@{_VIDAFORGE_RELEASE_REVISION}/{official_row['tar_path']}"
    with HfFileSystem().open(remote_tar_path, "rb") as tar_handle:
        tar_handle.seek(int(official_row["tar_offset"]))
        payload = tar_handle.read(int(official_row["filesize_bytes"]))
    assert len(payload) == int(official_row["filesize_bytes"])
    assert hashlib.sha256(payload).hexdigest() == official_row["sha256"]
    video_path = tmp_path / "official-hevc.mp4"
    video_path.write_bytes(payload)

    producer_manifest = tmp_path / "stage4.parquet"
    pq.write_table(
        pa.Table.from_pylist([{
            "clip_id": _CLIP_ID,
            "clip_path": str(video_path),
            "clip_ok": 1,
            "select_ok": 1,
            "select_pass": 1,
            "caption_ok": 1,
            "width": int(official_row["width"]),
            "height": int(official_row["height"]),
            "fps": float(official_row["fps"]),
            "duration_sec": float(official_row["duration_sec"]),
            "caption_level_3": str(official_row["caption_level_3"]),
        }]),
        producer_manifest,
    )
    output_dir = tmp_path / "fastvideo-stage5"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            "fastvideo.pipelines.preprocess.v1_preprocessing_new",
            "--model-path",
            _MODEL_NAME,
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
            str(producer_manifest),
            "--preprocess.video-loader-type",
            "torchcodec",
            "--preprocess.output-type",
            "vidaforge_automodel",
            "--preprocess.vidaforge-model-name",
            _MODEL_NAME,
            "--preprocess.dataset-output-dir",
            str(output_dir),
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
            "1",
            "--preprocess.dataloader-num-workers",
            "0",
            "--preprocess.samples-per-file",
            "1",
        ],
        check=True,
        timeout=3600,
    )

    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["model_revision"] == _MODEL_REVISION
    dataset = VidaForgeAutoModelDataset(
        output_dir,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=provenance["vae_fingerprint"],
        expected_text_encoder_fingerprint=provenance["text_encoder_fingerprint"],
    )
    sample = dataset[0]
    assert sample["info"]["clip_id"] == _CLIP_ID
    assert sample["vae_latent"].dtype == torch.float16
    assert sample["text_embedding"].dtype == torch.bfloat16
    assert torch.isfinite(sample["vae_latent"]).all()
    assert torch.isfinite(sample["text_embedding"]).all()
    assert int(sample["text_attention_mask"].sum()) > 0

    reference_dir = Path(os.environ.get("VIDAFORGE_REFERENCE_DIR", "")).expanduser().resolve()
    if not (reference_dir / "vidaforge").is_dir():
        pytest.fail("VIDAFORGE_REFERENCE_DIR must point to the pinned VidaForge checkout")
    checkout_revision = subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checkout_revision == _VIDAFORGE_REVISION
    sys.path.insert(0, str(reference_dir))
    try:
        from vidaforge.packaging.automodel.wan import WanAutoModelEncoder
    finally:
        sys.path.remove(str(reference_dir))
    resolved_model = snapshot_download(_MODEL_NAME, revision=_MODEL_REVISION)
    oracle = WanAutoModelEncoder(
        model_name=resolved_model,
        device="cuda",
        max_sequence_length=512,
        deterministic_latents=True,
    ).encode_batch(
        bucket_frame_count=17,
        bucket_resolution=(256, 144),
        video_paths=[video_path],
        source_resolutions=[(int(official_row["width"]), int(official_row["height"]))],
        source_fps=[float(official_row["fps"])],
        captions=[str(official_row["caption_level_3"])],
    )[0]

    torch.testing.assert_close(
        sample["vae_latent"].float(),
        oracle.video_latents.float(),
        atol=5e-3,
        rtol=5e-3,
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
