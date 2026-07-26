# SPDX-License-Identifier: Apache-2.0
"""Opt-in real VidaForge Stage 5 compatibility smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from huggingface_hub import HfFileSystem, hf_hub_download

from fastvideo.dataset.vidaforge_automodel_dataset import (
    VidaForgeAutoModelDataset, )


_VIDAFORGE_REVISION = "4562d3fbcbd4861fc74c2859950c0237363681bb"
_VIDAFORGE_RELEASE_REVISION = "091bdc02d82b8c89a4e4eff54945d286fb328b47"
_VIDAFORGE_HEVC_SMOKE_CLIP_ID = "video-9a2221ec0d47d85c:clip:00002:02"


@pytest.mark.skipif(
    os.environ.get("VIDAFORGE_RUN_OFFICIAL_AUTOMODEL_SMOKE") != "1",
    reason=(
        "set VIDAFORGE_RUN_OFFICIAL_AUTOMODEL_SMOKE=1 and "
        "VIDAFORGE_REFERENCE_DIR to run the real Stage 5 encoder"
    ),
)
def test_official_wan_automodel_payload_loads_without_conversion(
    tmp_path: Path,
) -> None:
    reference_dir = Path(
        os.environ.get("VIDAFORGE_REFERENCE_DIR", ""), ).expanduser().resolve()
    if not (reference_dir / "vidaforge").is_dir():
        pytest.fail(
            "VIDAFORGE_REFERENCE_DIR must point to a VidaForge checkout"
        )
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
        from vidaforge.packaging.automodel.worker import (
            write_automodel_metafile, )
    finally:
        sys.path.remove(str(reference_dir))

    metadata_path = hf_hub_download(
        repo_id="VidaForge/VidaForge-3M",
        filename="meta/shard-00000.parquet",
        repo_type="dataset",
        revision=_VIDAFORGE_RELEASE_REVISION,
    )
    rows = pq.read_table(
        metadata_path,
        filters=[("clip_id", "=", _VIDAFORGE_HEVC_SMOKE_CLIP_ID)],
    ).to_pylist()
    assert len(rows) == 1
    row = rows[0]

    remote_tar_path = (
        "datasets/VidaForge/VidaForge-3M@"
        f"{_VIDAFORGE_RELEASE_REVISION}/{row['tar_path']}"
    )
    with HfFileSystem().open(remote_tar_path, "rb") as tar_handle:
        tar_handle.seek(int(row["tar_offset"]))
        chunks: list[bytes] = []
        remaining = int(row["filesize_bytes"])
        while remaining:
            chunk = tar_handle.read(remaining)
            assert chunk
            chunks.append(chunk)
            remaining -= len(chunk)
    payload = b"".join(chunks)
    assert len(payload) == int(row["filesize_bytes"])
    assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    video_path = tmp_path / "official-hevc.mp4"
    video_path.write_bytes(payload)

    encoder = WanAutoModelEncoder(
        model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device="cuda",
        max_sequence_length=64,
        deterministic_latents=True,
    )
    samples = encoder.encode_batch(
        bucket_frame_count=17,
        bucket_resolution=(256, 144),
        video_paths=[video_path],
        source_resolutions=[(int(row["width"]), int(row["height"]))],
        source_fps=[float(row["fps"])],
        captions=[str(row["caption_level_3"])],
    )
    assert len(samples) == 1
    encoded = samples[0]

    cache_dir = tmp_path / "stage5"
    meta_path = cache_dir / "17f" / "256x144" / "official.meta"
    write_automodel_metafile(
        meta_path,
        row=row,
        video_path=video_path,
        caption=str(row["caption_level_3"]),
        caption_field="caption_level_3",
        run_id="fastvideo-smoke",
        input_run_id="official-release",
        bucket_config={
            "resolution": "144p",
            "durations_sec": [0.68],
        },
        sample=encoded,
    )
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir()
    shard_item = {
        "cache_file":
        str(meta_path),
        "bucket_resolution": [256, 144],
        "bucket_frame_count":
        17,
        "latent_shape":
        list(encoded.video_latents.shape),
        "clip_id":
        str(row["clip_id"]),
        "caption_token_length":
        int(encoded.metadata["caption_token_length"]),
    }
    (shard_dir / "metadata-000000.json").write_text(
        json.dumps([shard_item]),
        encoding="utf-8",
    )
    (cache_dir / "metadata.json").write_text(
        json.dumps({"shards": ["shards/metadata-000000.json"]}),
        encoding="utf-8",
    )

    loaded = VidaForgeAutoModelDataset(
        cache_dir,
        expected_model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )[0]

    assert loaded["vae_latent"].dtype == torch.float16
    assert torch.equal(loaded["vae_latent"], encoded.video_latents)
    assert torch.equal(loaded["text_embedding"], encoded.text_embeddings)
    assert int(loaded["text_attention_mask"].sum()) == min(
        int(encoded.metadata["caption_token_length"]),
        64,
    )
    assert loaded["info"]["clip_id"] == _VIDAFORGE_HEVC_SMOKE_CLIP_ID
