# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for normalized VidaForge latents in the Wan plugin."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.models.wan import WanModel


def _training_config(
    preprocessed_data_type: str,
    *,
    num_latent_t: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        data=SimpleNamespace(
            preprocessed_data_type=preprocessed_data_type,
            vidaforge_model_name=(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                if preprocessed_data_type == "vidaforge_automodel"
                else ""
            ),
            num_latent_t=num_latent_t,
        ),
        pipeline_config=SimpleNamespace(
            flow_shift=3.0,
            text_encoder_configs=[
                SimpleNamespace(
                    arch_config=SimpleNamespace(text_len=512),
                ),
            ],
        ),
    )


def _bare_model(training_config: SimpleNamespace) -> WanModel:
    model = object.__new__(WanModel)
    model.vae = None
    model.training_config = training_config
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    model._requires_negative_conditioning = False
    model._input_latents_are_normalized = False
    return model


def test_vidaforge_latents_are_not_normalized_twice() -> None:
    model = object.__new__(WanModel)
    model._input_latents_are_normalized = True
    latents = torch.randn(1, 16, 5, 4, 4)

    result = model._normalize_training_latents(latents)

    assert result is latents


def test_regular_fastvideo_latents_keep_runtime_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(WanModel)
    model._input_latents_are_normalized = False
    model.vae = object()
    latents = torch.randn(1, 16, 5, 4, 4)
    calls: list[tuple[str, torch.Tensor, object]] = []

    def fake_normalize(
        model_type: str,
        value: torch.Tensor,
        vae: object,
    ) -> torch.Tensor:
        calls.append((model_type, value, vae))
        return value + 1

    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.normalize_dit_input",
        fake_normalize,
    )

    result = model._normalize_training_latents(latents)

    assert torch.equal(result, latents + 1)
    assert calls == [("wan", latents, model.vae)]


def test_vidaforge_preprocessors_skip_vae_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_config = _training_config("vidaforge_automodel")
    model = _bare_model(training_config)
    loader = object()
    component_loads: list[str] = []

    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.get_world_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.get_sp_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.load_module_from_path",
        lambda **kwargs: component_loads.append(str(kwargs["module_type"])),
    )
    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_vidaforge_automodel_train_dataloader",
        lambda *_args, **_kwargs: loader,
    )

    model.init_preprocessors(training_config)

    assert model.dataloader is loader
    assert model.vae is None
    assert component_loads == []


def test_regular_preprocessors_still_load_vae_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_config = _training_config("t2v")
    model = _bare_model(training_config)
    vae = object()
    loader = object()
    component_loads: list[str] = []

    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.get_world_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.get_sp_group",
        lambda: object(),
    )

    def fake_load_module_from_path(**kwargs: Any) -> object:
        component_loads.append(str(kwargs["module_type"]))
        return vae

    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.load_module_from_path",
        fake_load_module_from_path,
    )
    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_t2v_train_dataloader",
        lambda *_args, **_kwargs: loader,
    )

    model.init_preprocessors(training_config)

    assert model.dataloader is loader
    assert model.vae is vae
    assert component_loads == ["vae"]


@pytest.mark.parametrize(
    ("preprocessed_data_type", "expected_temporal_length"),
    [
        ("vidaforge_automodel", 5),
        ("t2v", 2),
    ],
)
def test_prepare_batch_preserves_vidaforge_temporal_bucket(
    monkeypatch: pytest.MonkeyPatch,
    preprocessed_data_type: str,
    expected_temporal_length: int,
) -> None:
    training_config = _training_config(
        preprocessed_data_type,
        num_latent_t=2,
    )
    model = _bare_model(training_config)
    model._input_latents_are_normalized = (
        preprocessed_data_type == "vidaforge_automodel"
    )
    model._normalize_training_latents = lambda latents: latents
    model._prepare_dit_inputs = lambda batch, _generator: batch
    model._build_attention_metadata = lambda batch: batch
    monkeypatch.setattr(
        "fastvideo.train.models.base.get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    result = model.prepare_batch(
        {
            "vae_latent": torch.randn(1, 16, 5, 2, 2),
            "text_embedding": torch.randn(1, 4, 8),
            "text_attention_mask": torch.ones(1, 4),
        },
        generator=torch.Generator(device="cpu"),
    )

    assert result.latents is not None
    assert result.latents.shape[2] == expected_temporal_length


def test_decode_latents_lazy_loads_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVAE(torch.nn.Module):
        handles_latent_denorm = False
        latents_mean = [0.0, 0.0]
        latents_std = [1.0, 1.0]

        def __init__(self) -> None:
            super().__init__()
            self.decoded: torch.Tensor | None = None

        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            self.decoded = latents
            return latents

    training_config = _training_config("vidaforge_automodel")
    model = _bare_model(training_config)
    vae = FakeVAE()
    component_loads: list[str] = []

    def fake_load_module_from_path(**kwargs: Any) -> FakeVAE:
        component_loads.append(str(kwargs["module_type"]))
        return vae

    monkeypatch.setattr(
        "fastvideo.train.models.wan.wan.load_module_from_path",
        fake_load_module_from_path,
    )

    result = model.decode_latents(torch.zeros(1, 3, 2, 2, 2))

    assert component_loads == ["vae"]
    assert model.vae is vae
    assert vae.decoded is not None
    assert tuple(vae.decoded.shape) == (1, 2, 3, 2, 2)
    assert torch.equal(result, torch.full_like(result, 0.5))


def test_finetune_disables_unused_negative_prompt_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    training_config = SimpleNamespace()
    student = SimpleNamespace(
        _trainable=True,
        transformer=torch.nn.Linear(1, 1),
        attention_backend_name=None,
        set_requires_negative_conditioning=lambda value: events.append(
            ("negative", value)
        ),
        init_preprocessors=lambda value: events.append(("preprocessors", value)),
    )
    cfg = SimpleNamespace(
        training=training_config,
        method={},
    )
    monkeypatch.setattr(
        FineTuneMethod,
        "_init_optimizers_and_schedulers",
        lambda _self: None,
    )

    FineTuneMethod(
        cfg=cfg,
        role_models={"student": student},
    )

    assert events == [
        ("negative", False),
        ("preprocessors", training_config),
    ]
