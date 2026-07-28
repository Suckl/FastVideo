# SPDX-License-Identifier: Apache-2.0
"""CPU tests for FastVideo's VidaForge AutoModel producer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    _wan_normalize,
    build_model_provenance,
    build_vidaforge_source_fingerprint,
    VidaForgeAutoModelWriter,
)
from fastvideo.pipelines.preprocess.wan.vidaforge_stages import (
    clean_vidaforge_prompt,
    VidaForgeTextEncodingStage,
    VidaForgeWanEncodingStage,
    VidaForgeWanVideoTransformStage,
)
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow
from fastvideo.workflow.preprocess.preprocess_workflow_vidaforge_automodel import (
    PreprocessWorkflowVidaForgeAutoModel,
    plan_vidaforge_forward_batches,
)
from fastvideo.workflow.preprocess.components import VideoForwardBatchBuilder
from fastvideo.workflow.preprocess.vidaforge_bucketing import VidaForgeBucketPlanner

_MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _producer_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "workload_type": "t2v",
        "pipeline_config": "WanT2V480PConfig",
        "caption_field": "caption_level_3",
        "vidaforge_selection": "auto",
        "manifest_fingerprint": "d" * 64,
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


def _source_item(clip_id: str, *, caption: str = "a test caption", video: bytes = b"video") -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "video": video,
        "caption": caption,
        "resolution": {
            "width": 32,
            "height": 24,
        },
        "fps": 24.0,
        "num_frames": 12,
    }


def _batch(clip_id: str, *, caption: str = "a test caption", video: bytes = b"video") -> PreprocessBatch:
    source_fingerprint = build_vidaforge_source_fingerprint(_source_item(clip_id, caption=caption, video=video))
    return PreprocessBatch(
        data_type="video",
        latents=torch.tensor([[[[[3.0]]], [[[6.0]]]]]),
        prompt_embeds=[torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)],
        # Wan pads embeddings to 512 but returns a mask only as long as the
        # longest prompt in the batch. The writer must pad that mask to match.
        prompt_attention_mask=[torch.tensor([[1, 1]])],
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
                "caption": caption,
                "source_fingerprint": source_fingerprint,
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


def test_writer_validates_each_sample_against_multibucket_policy(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    planner = VidaForgeBucketPlanner(
        resolution="480p",
        upscale=False,
        durations_sec=(2.0, 4.0),
        temporal_stride=4,
        input_size_multiple=16,
        dynamic_forward_batch_size=4,
    )
    source_item = {
        "clip_id": "multi-clip",
        "video": b"video",
        "caption": "a test caption",
        "resolution": {
            "width": 1920,
            "height": 1080,
        },
        "fps": 16.0,
        "num_frames": 34,
        "duration_sec": 2.1,
    }
    batch = _batch("multi-clip")
    batch.num_frames = [29]
    batch.width = [848]
    batch.height = [480]
    batch.extra["source_metadata"] = [{
        "clip_id": "multi-clip",
        "original_filename": "safe-name",
        "original_video_path": None,
        "source_resolution": [1920, 1080],
        "source_fps": 16.0,
        "source_frame_count": 34,
        "source_duration_sec": 2.1,
        "caption": "a test caption",
        "source_fingerprint": build_vidaforge_source_fingerprint(source_item),
    }]
    producer_config = _producer_config(
        schema_version=2,
        bucket_policy={
            "mode": "multi",
            **planner.to_producer_config(),
        },
    )
    writer = _writer(
        tmp_path / "valid-cache",
        model_root,
        generation="1" * 32,
        producer_config=producer_config,
    )

    writer.save_batch(
        batch,
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )

    invalid_writer = _writer(
        tmp_path / "invalid-cache",
        model_root,
        generation="2" * 32,
        producer_config=producer_config,
    )
    batch.width = [832]
    with pytest.raises(ValueError, match="actual=\\(29, 832, 480\\), expected=\\(29, 848, 480\\)"):
        invalid_writer.save_batch(
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


def test_vidaforge_video_stage_repeats_at_most_three_missing_frames() -> None:
    stage = VidaForgeWanVideoTransformStage(
        num_frames=17,
        max_height=16,
        max_width=16,
    )

    indices = stage._frame_indices(14)

    assert len(indices) == 17
    assert indices[0] == 0
    assert indices[-1] == 13
    with pytest.raises(ValueError, match="at least 14 decoded frames"):
        stage._frame_indices(13)


def test_vidaforge_video_stage_prefers_official_cuda_decode() -> None:

    class _Video:

        def __init__(self) -> None:
            self.requested_frame_count: int | None = None

        def get_vidaforge_wan_frames(self, frame_count: int) -> torch.Tensor:
            self.requested_frame_count = frame_count
            return torch.zeros((frame_count, 3, 16, 16), dtype=torch.uint8)

        def __len__(self) -> int:
            raise AssertionError("the generic decoder must not be used")

    video = _Video()
    batch = PreprocessBatch(
        data_type="video",
        video_loader=[video],
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
    stage.forward(
        batch,
        SimpleNamespace(
            preprocess_config=SimpleNamespace(
                video_loader_type=VideoLoaderType.TORCHCODEC, ), ),  # type: ignore[arg-type]
    )

    assert video.requested_frame_count == 5


def test_vidaforge_video_stage_uses_homogeneous_dynamic_bucket() -> None:

    class _Video:

        def __init__(self) -> None:
            self.requested_frame_count: int | None = None

        def get_vidaforge_wan_frames(self, frame_count: int) -> torch.Tensor:
            self.requested_frame_count = frame_count
            return torch.zeros((frame_count, 3, 24, 48), dtype=torch.uint8)

    video = _Video()
    batch = PreprocessBatch(
        data_type="video",
        video_loader=[video],
        fps=[24.0],
        num_frames=[40],
        height=[24],
        width=[48],
        extra={
            "vidaforge_bucket": {
                "frame_count": 9,
                "width": 32,
                "height": 16,
            },
        },
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

    assert video.requested_frame_count == 9
    assert result.latents is not None
    assert tuple(result.latents.shape) == (1, 3, 9, 16, 32)
    assert result.num_frames == [9]
    assert result.width == [32]
    assert result.height == [16]


def test_wan_normalization_uses_official_fp16_arithmetic() -> None:
    latents = torch.tensor([[[[[0.1234]]]]], dtype=torch.float32)
    vae = SimpleNamespace(latents_mean=[0.3333], latents_std=[0.07])

    normalized = _wan_normalize(latents, vae)
    expected = (
        latents.to(torch.float16) - torch.tensor([0.3333], dtype=torch.float16).view(1, 1, 1, 1, 1)
    ) / torch.tensor([0.07], dtype=torch.float16).view(1, 1, 1, 1, 1)

    assert normalized.dtype == torch.float16
    assert torch.equal(normalized, expected)


def test_vidaforge_wan_encoding_uses_explicit_fp16_input(monkeypatch: pytest.MonkeyPatch) -> None:

    class _VAE:

        def __init__(self) -> None:
            self.encoded: torch.Tensor | None = None

        def to(self, _device):
            return self

        def encode(self, value: torch.Tensor):
            self.encoded = value
            return SimpleNamespace(mean=torch.ones((1, 1, 1, 1, 1), dtype=value.dtype))

    vae = _VAE()
    stage = VidaForgeWanEncodingStage(vae=vae)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "fastvideo.pipelines.preprocess.wan.vidaforge_stages.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    batch = PreprocessBatch(
        data_type="video",
        latents=torch.tensor([[[[[0.25]]]]], dtype=torch.float32),
    )
    result = stage.forward(
        batch,
        SimpleNamespace(
            pipeline_config=SimpleNamespace(
                vae_precision="fp16",
                vae_tiling=False,
            ),
            vae_cpu_offload=False,
        ),  # type: ignore[arg-type]
    )

    assert vae.encoded is not None
    assert vae.encoded.dtype == torch.float16
    assert vae.encoded.item() == -0.5
    assert result.latents is not None
    assert result.latents.dtype == torch.float16


def test_vidaforge_prompt_cleanup_matches_double_html_unescape() -> None:
    assert clean_vidaforge_prompt("  one &amp;amp; two\n three  ") == "one & two three"


def test_vidaforge_text_encoding_uses_official_fixed_padding() -> None:

    class _Tokenizer:

        def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            assert prompts == ["test prompt"]
            assert kwargs == {
                "add_special_tokens": True,
                "padding": False,
                "truncation": False,
            }
            return {"input_ids": [[1, 2, 3]]}

    stage = VidaForgeTextEncodingStage(
        text_encoders=[object()],
        tokenizers=[_Tokenizer()],
    )
    encoded = torch.ones((1, 512, 8), dtype=torch.bfloat16)
    mask = torch.cat((torch.ones((1, 3)), torch.zeros((1, 509))), dim=1)
    stage.encode_text = MagicMock(return_value=([encoded], [mask]))  # type: ignore[method-assign]
    batch = PreprocessBatch(
        data_type="video",
        prompt=["test prompt"],
        prompt_embeds=[],
        prompt_attention_mask=[],
    )
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[object()],
        ), )  # type: ignore[arg-type]

    result = stage.forward(batch, args)

    stage.encode_text.assert_called_once_with(  # type: ignore[attr-defined]
        ["test prompt"],
        args,
        encoder_index=[0],
        return_attention_mask=True,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    assert result.prompt_embeds[0] is encoded
    assert result.prompt_attention_mask[0] is mask
    assert result.extra["caption_token_lengths"] == [3]


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
    pending_input = [_source_item("clip-1"), _source_item("clip-2")]
    pending = resumed.pending_items(pending_input)
    assert pending == [pending_input[1]]
    resumed.write_rank_progress()
    assert resumed.publish() == 1

    changed_source = _writer(output_dir, model_root, generation="5" * 32, resume=True)
    with pytest.raises(ValueError, match="caption, media, or source metadata changed"):
        changed_source.pending_items([_source_item("clip-1", caption="updated caption")])

    with pytest.raises(ValueError, match="different producer_config_fingerprint"):
        _writer(
            output_dir,
            model_root,
            generation="d" * 32,
            resume=True,
            producer_config=_producer_config(train_fps=24),
        )
    with pytest.raises(ValueError, match="different producer_config_fingerprint"):
        _writer(
            output_dir,
            model_root,
            generation="6" * 32,
            resume=True,
            producer_config=_producer_config(vidaforge_selection="all"),
        )

    (model_root / "text_encoder" / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="different text_encoder_fingerprint"):
        _writer(output_dir, model_root, generation="f" * 32, resume=True)


def test_resume_publishes_only_items_seen_in_current_manifest(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="7" * 32)
    vae = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])
    first.save_batch(_batch("clip-1"), vae=vae)
    first.save_batch(_batch("clip-2"), vae=vae)
    first.write_rank_progress()
    assert first.publish() == 2

    resumed = _writer(output_dir, model_root, generation="8" * 32, resume=True)
    assert resumed.pending_items([_source_item("clip-1")]) == []
    resumed.write_rank_progress()
    assert resumed.publish() == 1

    root = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    items = json.loads((output_dir / root["shards"][0]).read_text(encoding="utf-8"))
    assert [item["clip_id"] for item in items] == ["clip-1"]


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

    resumed = _writer(output_dir, model_root, generation="2" * 32, resume=True)
    with pytest.raises(ValueError, match="payload clip_id mismatch"):
        resumed.pending_items([_source_item("clip-1")])


@pytest.mark.parametrize(
    ("tampered_mask", "message"),
    [
        (torch.tensor([[1.0, 0.0, 0.0]]), "disagrees with caption_token_length"),
        (torch.tensor([[1.0, float("nan"), 0.0]]), "finite binary values"),
    ],
)
def test_resume_rejects_tampered_text_mask(
    tmp_path: Path,
    tampered_mask: torch.Tensor,
    message: str,
) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="3" * 32)
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
    payload["text_mask"] = tampered_mask
    torch.save(payload, path)

    resumed = _writer(output_dir, model_root, generation="4" * 32, resume=True)
    with pytest.raises(ValueError, match=message):
        resumed.pending_items([_source_item("clip-1")])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("video_latents", torch.ones((1, 2, 1, 1, 1), dtype=torch.float32), "FP16"),
        ("text_embeddings", torch.ones((1, 3, 8), dtype=torch.float32), "BF16"),
    ],
)
def test_resume_rejects_tampered_tensor_contract(
    tmp_path: Path,
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="9" * 32)
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
    payload[field] = value
    torch.save(payload, path)

    resumed = _writer(output_dir, model_root, generation="a" * 32, resume=True)
    with pytest.raises(ValueError, match=message):
        resumed.pending_items([_source_item("clip-1")])


@pytest.mark.parametrize("tensor_kind", ["latents", "text_embeddings"])
def test_writer_rejects_nonfinite_new_tensors(tmp_path: Path, tensor_kind: str) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    writer = _writer(tmp_path / "cache", model_root, generation="0" * 32)
    batch = _batch("clip-1")
    if tensor_kind == "latents":
        batch.latents[0, 0, 0, 0, 0] = float("nan")
        message = "VAE latents must be finite"
    else:
        batch.prompt_embeds[0][0, 0, 0] = float("nan")
        message = "text embeddings must be finite"

    with pytest.raises(ValueError, match=message):
        writer.save_batch(
            batch,
            vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
        )


def test_distributed_resume_validates_only_each_ranks_retained_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    first = _writer(output_dir, model_root, generation="1" * 32)
    vae = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])
    first.save_batch(_batch("clip-rank-0"), vae=vae)
    first.save_batch(_batch("clip-rank-1"), vae=vae)
    first.write_rank_progress()
    assert first.publish() == 2

    original_load = torch.load
    loaded_paths: list[Path] = []

    def _tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _tracked_load)
    generation = "2" * 32
    writers = [
        VidaForgeAutoModelWriter(
            output_dir,
            model_root=model_root,
            model_name=_MODEL_NAME,
            requested_revision="1" * 40,
            producer_config=_producer_config(),
            samples_per_shard=1,
            resume=True,
            rank=rank,
            world_size=2,
            generation=generation,
        ) for rank in range(2)
    ]
    assert loaded_paths == []

    assert writers[0].pending_items([_source_item("clip-rank-0")]) == []
    rank_zero_paths = list(loaded_paths)
    assert len(rank_zero_paths) == 1

    assert writers[1].pending_items([_source_item("clip-rank-1")]) == []
    assert len(loaded_paths) == 2
    assert loaded_paths[1] != rank_zero_paths[0]


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


def test_publish_rejects_failure_followed_by_success_on_another_rank(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    generation = "9" * 32
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
    source = _source_item("conflicting-clip")
    writers[0].record_failure(source, stage="encoding", error="decoder failed")
    writers[1].save_batch(
        _batch("conflicting-clip"),
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )
    for writer in writers:
        writer.write_rank_progress()

    with pytest.raises(ValueError, match="both successful and failed"):
        writers[0].publish()


def test_publish_reports_recoverable_failures_and_bucket_counts(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    writer = _writer(output_dir, model_root, generation="f" * 32)
    failed_item = _source_item("failed-clip")
    assert writer.pending_items([failed_item]) == [failed_item]
    writer.record_failure(failed_item, stage="bucket_planning", error=ValueError("too short"))
    writer.save_batch(
        _batch("encoded-clip"),
        vae=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0]),
    )

    writer.write_rank_progress()
    assert writer.publish() == 1

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert summary == {
        "schema_version": 1,
        "format": "vidaforge_automodel",
        "input_count": 2,
        "ok_count": 1,
        "failed_count": 1,
        "bucket_counts": {
            "1f/16x16": 1,
        },
        "failure_stage_counts": {
            "bucket_planning": 1,
        },
    }
    assert failures[0]["clip_id"] == "failed-clip"
    assert failures[0]["error"] == "too short"
    assert metadata["summary_file"] == "summary.json"
    assert metadata["failures_file"] == "failures.json"


def test_publish_keeps_failure_diagnostics_when_no_cache_item_succeeds(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    output_dir = tmp_path / "cache"
    _write_model(model_root)
    writer = _writer(output_dir, model_root, generation="0" * 32)
    failed_item = _source_item("only-failure")
    assert writer.pending_items([failed_item]) == [failed_item]
    writer.record_failure(failed_item, stage="encoding", error="decoder failed")

    writer.write_rank_progress()
    assert writer.publish() == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert summary["ok_count"] == 0
    assert summary["failed_count"] == 1
    assert metadata["shards"] == []


def test_multibucket_planning_groups_shapes_scales_batches_and_reports_short_clips() -> None:
    planner = VidaForgeBucketPlanner(
        resolution="480p",
        upscale=False,
        durations_sec=(2.0, 4.0),
        temporal_stride=4,
        input_size_multiple=16,
        dynamic_forward_batch_size=4,
    )
    items = [{
        "clip_id": f"landscape-{index}",
        "duration_sec": 4.1,
        "fps": 16.0,
        "resolution": {
            "width": 1920,
            "height": 1080,
        },
    } for index in range(6)]
    items.extend([
        {
            "clip_id": "portrait",
            "duration_sec": 2.1,
            "fps": 16.0,
            "resolution": {
                "width": 1080,
                "height": 1920,
            },
        },
        {
            "clip_id": "too-short",
            "duration_sec": 1.0,
            "fps": 16.0,
            "resolution": {
                "width": 1920,
                "height": 1080,
            },
        },
    ])

    batches, failures = plan_vidaforge_forward_batches(items, planner)

    assert [len(batch) for batch in batches] == [4, 2, 1]
    assert {
        tuple(item["_vidaforge_bucket"].values())
        for batch in batches
        for item in batch
    } == {
        (61, 848, 480),
        (29, 480, 848),
    }
    assert all(
        len({
            tuple(item["_vidaforge_bucket"].values())
            for item in batch
        }) == 1 for batch in batches)
    assert len(failures) == 1
    assert failures[0][0]["clip_id"] == "too-short"
    assert "shorter than the smallest temporal bucket" in str(failures[0][1])


def test_multibucket_planning_prefers_manifest_metadata_over_probe_metadata() -> None:
    planner = VidaForgeBucketPlanner(
        resolution="480p",
        upscale=False,
        durations_sec=(2.0, 4.0),
        temporal_stride=4,
        input_size_multiple=16,
        dynamic_forward_batch_size=4,
    )
    items = [{
        "clip_id": f"clip-{index}",
        "duration_sec": 4.1,
        "fps": 8.0,
        "resolution": {
            "width": 40,
            "height": 30,
        },
        "vidaforge_manifest_fps": 16.0,
        "vidaforge_manifest_resolution": {
            "width": 1920,
            "height": 1080,
        },
    } for index in range(5)]

    batches, failures = plan_vidaforge_forward_batches(items, planner)

    assert failures == []
    assert [len(batch) for batch in batches] == [4, 1]
    assert all(
        item["_vidaforge_bucket"] == {
            "frame_count": 61,
            "width": 848,
            "height": 480,
        } for batch in batches for item in batch)


def test_forward_batch_keeps_probe_and_manifest_metadata_distinct(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    batch = VideoForwardBatchBuilder(seed=7)([{
        "clip_id": "clip-a",
        "video": str(video_path),
        "name": "clip.mp4",
        "caption": "caption",
        "duration_sec": 2.04,
        "fps": 30.0,
        "num_frames": 60,
        "resolution": {
            "width": 40,
            "height": 30,
        },
        "vidaforge_manifest_fps": 25.0,
        "vidaforge_manifest_resolution": {
            "width": 1280,
            "height": 720,
        },
    }])

    assert batch.fps == [30.0]
    assert batch.width == [40]
    assert batch.height == [30]
    assert batch.extra["source_metadata"][0]["source_fps"] == 25.0
    assert batch.extra["source_metadata"][0]["source_resolution"] == [
        1280,
        720,
    ]


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

    valid.drop_short_ratio = 1.0
    valid.video_loader_type = VideoLoaderType.TORCHVISION
    with pytest.raises(ValueError, match="requires video_loader_type=torchcodec"):
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
    assert pipeline.video_encoding_stage.__class__.__name__ == "VidaForgeWanEncodingStage"
