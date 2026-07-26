# SPDX-License-Identifier: Apache-2.0
"""CPU tests for FastVideo's VidaForge AutoModel producer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.configs import (
    DatasetType,
    PreprocessConfig,
    PreprocessOutputType,
    VideoLoaderType,
)
from fastvideo.dataset.vidaforge_automodel_dataset import VidaForgeAutoModelDataset
from fastvideo.configs.pipelines.wan import WanT2V480PConfig
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.pipelines.preprocess.wan.wan_preprocess_pipelines import PreprocessPipelineT2V
from fastvideo.workflow.preprocess.vidaforge_automodel_writer import (
    build_model_provenance,
    VidaForgeAutoModelWriter,
)
from fastvideo.pipelines.preprocess.wan.vidaforge_stages import (
    clean_vidaforge_prompt,
    VidaForgeWanVideoTransformStage,
)
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow
from fastvideo.workflow.preprocess.preprocess_workflow_vidaforge_automodel import (
    PreprocessWorkflowVidaForgeAutoModel,
)

_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _producer_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "workload_type": "t2v",
        "pipeline_config": "WanT2V480PConfig",
        "caption_field": "caption_level_3",
        "video_loader_type": "torchcodec",
        "max_height": 16,
        "max_width": 16,
        "num_frames": 1,
        "train_fps": 16,
        "do_temporal_sample": False,
        "seed": 42,
        "vae_precision": "fp32",
        "vae_tiling": False,
        "vae_sp": False,
        "text_encoder_precisions": ["fp32"],
        "text_max_lengths": [512],
        "disable_autocast": False,
    }
    config.update(overrides)
    return config


def _write_model(root: Path) -> None:
    for component in ("vae", "text_encoder", "tokenizer"):
        directory = root / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps({"component": component}),
            encoding="utf-8",
        )
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae-weights")
    (root / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"text-weights")
    (root / "tokenizer" / "spiece.model").write_bytes(b"tokenizer-vocabulary")


def _batch(clip_id: str) -> PreprocessBatch:
    return PreprocessBatch(
        data_type="video",
        latents=torch.tensor([[[[[3.0]]], [[[6.0]]]]]),
        prompt_embeds=[torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)],
        prompt_attention_mask=[torch.tensor([[1, 1, 0]])],
        width=[16],
        height=[16],
        num_frames=[1],
        video_file_name=["safe-name"],
        extra={
            "caption_token_lengths": [2],
            "source_metadata": [{
                "clip_id": clip_id,
                "original_filename": "safe-name",
                "original_video_path": None,
                "source_resolution": [32, 24],
                "source_fps": 24.0,
                "source_frame_count": 12,
                "caption": "a test caption",
            }],
        },
    )


def _writer(
    output_dir: Path,
    model_root: Path,
    *,
    generation: str,
    resume: bool = False,
    producer_config: dict[str, object] | None = None,
) -> VidaForgeAutoModelWriter:
    return VidaForgeAutoModelWriter(
        output_dir,
        model_root=model_root,
        model_name=_MODEL_NAME,
        requested_revision="1" * 40,
        producer_config=_producer_config() if producer_config is None else producer_config,
        samples_per_shard=2,
        resume=resume,
        rank=0,
        world_size=1,
        generation=generation,
    )


def test_provenance_hashes_configs_weights_and_revision(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)

    first = build_model_provenance(
        model_root,
        model_name=_MODEL_NAME,
        requested_revision="1" * 40,
    )
    second = build_model_provenance(
        model_root,
        model_name=_MODEL_NAME,
        requested_revision="1" * 40,
    )
    assert first == second
    assert len(first["vae_fingerprint"]) == 64
    assert len(first["text_encoder_fingerprint"]) == 64
    assert first["model_revision"] == "local-content-addressed"

    (model_root / "vae" / "config.json").write_text('{"changed":true}', encoding="utf-8")
    changed = build_model_provenance(
        model_root,
        model_name=_MODEL_NAME,
        requested_revision="1" * 40,
    )
    assert changed["vae_fingerprint"] != first["vae_fingerprint"]
    assert changed["text_encoder_fingerprint"] == first["text_encoder_fingerprint"]

    (model_root / "tokenizer" / "spiece.model").write_bytes(b"changed-tokenizer")
    tokenizer_changed = build_model_provenance(
        model_root,
        model_name=_MODEL_NAME,
        requested_revision="1" * 40,
    )
    assert tokenizer_changed["text_encoder_fingerprint"] != changed["text_encoder_fingerprint"]


def test_writer_produces_verified_stage5_cache_with_safe_paths(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    writer = _writer(output_dir, model_root, generation="a" * 32)
    writer.save_batch(
        _batch("../../outside.pt"),
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )
    writer.write_rank_progress()
    assert writer.publish() == 1

    root = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    shard = json.loads((output_dir / root["shards"][0]).read_text(encoding="utf-8"))
    meta_path = (output_dir / shard[0]["cache_file"]).resolve()
    assert meta_path.is_relative_to(output_dir.resolve())
    assert meta_path.suffix == ".meta"
    assert not (tmp_path / "outside.pt").exists()

    dataset = VidaForgeAutoModelDataset(
        output_dir,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=writer.provenance["vae_fingerprint"],
        expected_text_encoder_fingerprint=writer.provenance["text_encoder_fingerprint"],
    )
    sample = dataset[0]
    assert sample["vae_latent"].dtype == torch.float16
    assert torch.equal(sample["vae_latent"].float(), torch.tensor([[[[[1.0]]], [[[1.0]]]]]))
    assert sample["text_embedding"].dtype == torch.bfloat16
    assert sample["text_attention_mask"].tolist() == [[1.0, 1.0, 0.0]]
    assert sample["info"]["clip_id"] == "../../outside.pt"


def test_writer_rejects_actual_geometry_outside_producer_contract(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    writer = _writer(tmp_path / "cache", model_root, generation="0" * 32)
    batch = _batch("short-clip")
    batch.num_frames = [2]

    with pytest.raises(ValueError, match="geometry does not match"):
        writer.save_batch(
            batch,
            vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
        )


def test_vidaforge_video_stage_samples_across_complete_clip() -> None:

    class _FrameBatch:

        def __init__(self, data: torch.Tensor) -> None:
            self.data = data

    class _Video:

        def __init__(self) -> None:
            self.frames = torch.stack([
                torch.full((3, 16, 16), index, dtype=torch.uint8)
                for index in range(9)
            ])

        def __len__(self) -> int:
            return len(self.frames)

        def get_frames_at(self, indices: list[int]) -> _FrameBatch:
            return _FrameBatch(self.frames[indices])

    batch = PreprocessBatch(
        data_type="video",
        video_loader=[_Video()],
        fps=[24.0],
        num_frames=[9],
        height=[16],
        width=[16],
    )
    stage = VidaForgeWanVideoTransformStage(
        num_frames=5,
        max_height=16,
        max_width=16,
    )
    result = stage.forward(
        batch,
        SimpleNamespace(
            preprocess_config=SimpleNamespace(
                video_loader_type=VideoLoaderType.TORCHCODEC, ), ),  # type: ignore[arg-type]
    )

    assert isinstance(result.latents, torch.Tensor)
    sampled_values = (result.latents[0, 0, :, 0, 0] * 255).tolist()
    assert sampled_values == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_vidaforge_prompt_cleanup_matches_double_html_unescape() -> None:
    assert clean_vidaforge_prompt("  one &amp;amp; two\n three  ") == "one & two three"


def test_resume_skips_completed_clip_and_rejects_changed_model(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="b" * 32)
    first.save_batch(
        _batch("clip-1"),
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )
    first.write_rank_progress()
    first.publish()

    resumed = _writer(output_dir, model_root, generation="c" * 32, resume=True)
    assert resumed.pending_items([{"clip_id": "clip-1"}, {"clip_id": "clip-2"}]) == [{
        "clip_id": "clip-2"
    }]
    resumed.write_rank_progress()
    assert resumed.publish() == 1

    with pytest.raises(ValueError, match="different producer_config_fingerprint"):
        _writer(
            output_dir,
            model_root,
            generation="d" * 32,
            resume=True,
            producer_config=_producer_config(train_fps=24),
        )

    (model_root / "text_encoder" / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="different text_encoder_fingerprint"):
        _writer(output_dir, model_root, generation="f" * 32, resume=True)


def test_resume_rejects_payload_that_disagrees_with_index(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="1" * 32)
    first.save_batch(
        _batch("clip-1"),
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )
    first.write_rank_progress()
    first.publish()
    root = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    item = json.loads((output_dir / root["shards"][0]).read_text(encoding="utf-8"))[0]
    path = output_dir / item["cache_file"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["clip_id"] = "wrong-clip"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="payload clip_id mismatch"):
        _writer(output_dir, model_root, generation="2" * 32, resume=True)


def test_publish_merges_distributed_rank_progress(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    generation = "e" * 32
    writers = [
        VidaForgeAutoModelWriter(
            output_dir,
            model_root=model_root,
            model_name=_MODEL_NAME,
            requested_revision="1" * 40,
            producer_config=_producer_config(),
            samples_per_shard=1,
            resume=False,
            rank=rank,
            world_size=2,
            generation=generation,
        ) for rank in range(2)
    ]
    vae = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])
    writers[0].save_batch(_batch("clip-rank-0"), vae=vae)
    writers[1].save_batch(_batch("clip-rank-1"), vae=vae)
    for writer in writers:
        writer.write_rank_progress()

    assert writers[0].publish() == 2
    dataset = VidaForgeAutoModelDataset(
        output_dir,
        expected_model_name=_MODEL_NAME,
        expected_vae_fingerprint=writers[0].provenance["vae_fingerprint"],
        expected_text_encoder_fingerprint=writers[0].provenance["text_encoder_fingerprint"],
    )
    assert {dataset[index]["info"]["clip_id"] for index in range(2)} == {
        "clip-rank-0",
        "clip-rank-1",
    }


def test_vidaforge_automodel_config_requires_deterministic_wan_geometry() -> None:
    valid = PreprocessConfig(
        dataset_path="manifest.parquet",
        dataset_type=DatasetType.VIDAFORGE,
        output_type=PreprocessOutputType.VIDAFORGE_AUTOMODEL,
        vidaforge_model_name=_MODEL_NAME,
        num_frames=17,
        max_height=144,
        max_width=256,
    )
    valid.check_preprocess_config()

    invalid = PreprocessConfig(
        dataset_path="manifest.parquet",
        dataset_type=DatasetType.VIDAFORGE,
        output_type=PreprocessOutputType.VIDAFORGE_AUTOMODEL,
        vidaforge_model_name=_MODEL_NAME,
        num_frames=16,
        max_height=144,
        max_width=256,
    )
    with pytest.raises(ValueError, match="num_frames=4n\\+1"):
        invalid.check_preprocess_config()

    valid.drop_short_ratio = 0.0
    with pytest.raises(ValueError, match="drop_short_ratio=1"):
        valid.check_preprocess_config()


def test_vidaforge_workflow_requires_official_component_precisions() -> None:
    preprocess_config = PreprocessConfig(
        dataset_path="manifest.parquet",
        dataset_type=DatasetType.VIDAFORGE,
        output_type=PreprocessOutputType.VIDAFORGE_AUTOMODEL,
        vidaforge_model_name=_MODEL_NAME,
        num_frames=17,
        max_height=144,
        max_width=256,
    )
    pipeline_config = WanT2V480PConfig(
        vae_precision="fp16",
        text_encoder_precisions=("bf16", ),
    )
    args = SimpleNamespace(
        preprocess_config=preprocess_config,
        workload_type=WorkloadType.T2V,
        pipeline_config=pipeline_config,
    )
    assert PreprocessWorkflow.get_workflow_cls(args) is PreprocessWorkflowVidaForgeAutoModel  # type: ignore[arg-type]

    pipeline_config.vae_precision = "fp32"
    with pytest.raises(ValueError, match="requires --vae-precision fp16"):
        PreprocessWorkflow.get_workflow_cls(args)  # type: ignore[arg-type]


def test_wan_t2v_pipeline_activates_vidaforge_producer_stages() -> None:
    preprocess_config = PreprocessConfig(
        output_type=PreprocessOutputType.VIDAFORGE_AUTOMODEL,
        num_frames=17,
        max_height=144,
        max_width=256,
    )
    pipeline = object.__new__(PreprocessPipelineT2V)
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.modules = {
        "text_encoder": object(),
        "tokenizer": object(),
        "vae": object(),
    }

    pipeline.create_pipeline_stages(
        SimpleNamespace(preprocess_config=preprocess_config),  # type: ignore[arg-type]
    )

    assert pipeline.text_transform_stage.__class__.__name__ == "VidaForgeTextTransformStage"
    assert pipeline.prompt_encoding_stage.__class__.__name__ == "VidaForgeTextEncodingStage"
    assert pipeline.video_transform_stage.__class__.__name__ == "VidaForgeWanVideoTransformStage"
