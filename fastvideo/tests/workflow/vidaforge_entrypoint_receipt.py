# SPDX-License-Identifier: Apache-2.0
"""Test-only observability for the VidaForge YAML entrypoint GPU gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch.distributed.tensor import DTensor

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.models.wan import WanModel

_RECEIPT_DIR_ENV = "VIDAFORGE_ENTRYPOINT_RECEIPT_DIR"
_RECEIPT_PHASE_ENV = "VIDAFORGE_ENTRYPOINT_RECEIPT_PHASE"


def _rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _receipt_path(kind: str, index: int) -> Path:
    receipt_dir = Path(os.environ[_RECEIPT_DIR_ENV])
    phase = os.environ[_RECEIPT_PHASE_ENV]
    receipt_dir.mkdir(parents=True, exist_ok=True)
    return (
        receipt_dir
        / f"{phase}-rank-{_rank():05d}-{kind}-{index:05d}.json"
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().contiguous().cpu()


def _update_tensor_digest(
    digest: Any,
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    local = _local_tensor(tensor)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(local.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(local.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(b"\0")


def _trainable_model_digest(model: WanModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.transformer.named_parameters():
        if parameter.requires_grad:
            _update_tensor_digest(
                digest,
                name=name,
                tensor=parameter,
            )
    return digest.hexdigest()


def _optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key in sorted(state):
                name = (
                    f"{group_index}:{parameter_index}:{key}"
                )
                value = state[key]
                if isinstance(value, torch.Tensor):
                    _update_tensor_digest(
                        digest,
                        name=name,
                        tensor=value,
                    )
                else:
                    digest.update(name.encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(
                        json.dumps(value, sort_keys=True).encode("utf-8")
                    )
                    digest.update(b"\0")
    return digest.hexdigest()


def _parameter_placement_receipt(
    model: WanModel,
) -> dict[str, Any]:
    dtensor_parameters = 0
    sharded_parameters: list[dict[str, Any]] = []
    trainable_parameters: list[dict[str, Any]] = []
    for name, parameter in model.transformer.named_parameters():
        if not isinstance(parameter, DTensor):
            continue
        dtensor_parameters += 1
        placements = [repr(placement) for placement in parameter.placements]
        record = {"name": name, "placements": placements}
        if any(type(placement).__name__ == "Shard"
               for placement in parameter.placements):
            sharded_parameters.append(record)
        if parameter.requires_grad:
            trainable_parameters.append(record)
    return {
        "dtensor_parameter_count": dtensor_parameters,
        "sharded_parameter_count": len(sharded_parameters),
        "sharded_parameter_examples": sharded_parameters[:5],
        "trainable_parameter_count": len(trainable_parameters),
        "trainable_parameter_examples": trainable_parameters[:5],
    }


class VidaForgeEntrypointReceiptWanModel(WanModel):
    """WanModel with receipt writes only for the opt-in Section 6 test."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._section6_batch_index = 0

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        prepared = super().prepare_batch(
            raw_batch,
            generator=generator,
            latents_source=latents_source,
        )
        self._section6_batch_index += 1
        infos = list(raw_batch["info_list"])
        assert prepared.noise is not None
        assert prepared.timesteps is not None

        noise_digest = hashlib.sha256()
        _update_tensor_digest(
            noise_digest,
            name="noise",
            tensor=prepared.noise,
        )
        timestep_digest = hashlib.sha256()
        _update_tensor_digest(
            timestep_digest,
            name="timesteps",
            tensor=prepared.timesteps,
        )
        receipt = {
            "rank": _rank(),
            "clip_ids": [str(info["clip_id"]) for info in infos],
            "bucket_frame_counts": [
                int(info["bucket_frame_count"])
                for info in infos
            ],
            "bucket_resolutions": [
                list(info["bucket_resolution"])
                for info in infos
            ],
            "noise_sha256": noise_digest.hexdigest(),
            "timesteps_sha256": timestep_digest.hexdigest(),
            "pre_step_trainable_model_sha256": (
                _trainable_model_digest(self)
            ),
            **_parameter_placement_receipt(self),
        }
        _receipt_path(
            "batch",
            self._section6_batch_index,
        ).write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="utf-8",
        )
        return prepared


class Section6ReceiptCallback(Callback):
    """Record post-step LoRA and optimizer state for resume comparison."""

    def on_training_step_end(
        self,
        method: Any,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        receipt = {
            "rank": _rank(),
            "iteration": iteration,
            "trainable_model_sha256": _trainable_model_digest(
                method.student
            ),
            "optimizer_sha256": _optimizer_digest(
                method._student_optimizer
            ),
            "total_loss": float(loss_dict["total_loss"]),
        }
        _receipt_path("post-step", iteration).write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="utf-8",
        )
