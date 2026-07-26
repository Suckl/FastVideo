# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for normalized VidaForge latents in the Wan plugin."""

from __future__ import annotations

import pytest
import torch

from fastvideo.train.models.wan import WanModel


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
