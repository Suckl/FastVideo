# SPDX-License-Identifier: Apache-2.0
import io
import random
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict)
from torch.distributed.tensor import DTensor, Replicate

_REPLICATED_TENSOR_STATE_PREFIX = "__fastvideo_replicated_tensor__."


def _serialize_replicated_tensor(tensor: DTensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.to_local().detach().cpu(), buffer)
    return buffer.getvalue()


def _deserialize_replicated_tensor(payload: bytes) -> torch.Tensor:
    return torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        state_dict = get_model_state_dict(self.model)

        trainable_parameters = {
            name.replace("._checkpoint_wrapped_module.", "."): parameter
            for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        filtered_state_dict = {name: value for name, value in state_dict.items() if name in trainable_parameters}
        # LoRA adapters added after ``fully_shard`` are replicated DTensors,
        # but they are not managed by FSDP.  ``get_model_state_dict`` still
        # exposes them, so using ``setdefault`` here would retain those
        # unmanaged DTensors and DCP would build its load template from them.
        # The default CUDA DCP tensor planner does not reliably materialize
        # these unmanaged, fully-replicated DTensors on a multi-dimensional
        # HSDP mesh. Store each one as a named DCP byte object instead. This
        # remains part of the DCP checkpoint and preserves the adapter name in
        # metadata, while avoiding tensor-planner assumptions intended for
        # FSDP-managed state. Keep normal Shard parameters untouched.
        for name, parameter in trainable_parameters.items():
            if isinstance(parameter, DTensor) and all(
                    isinstance(placement, Replicate) for placement in parameter.placements):
                filtered_state_dict.pop(name, None)
                filtered_state_dict[f"{_REPLICATED_TENSOR_STATE_PREFIX}{name}"] = (
                    _serialize_replicated_tensor(parameter))
            elif name not in filtered_state_dict:
                filtered_state_dict[name] = parameter.detach().clone()

        return filtered_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # LoRA adapters attached after ``fully_shard`` are replicated
        # DTensors that FSDP does not manage. Keep independent copies of
        # their DCP payload, let the official loader handle only normal FSDP
        # keys, then restore the adapters through their local storage.
        named_trainable_parameters = {
            name.replace("._checkpoint_wrapped_module.", "."): parameter
            for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        replicated_trainable_parameters = {
            name: parameter
            for name, parameter in named_trainable_parameters.items() if isinstance(parameter, DTensor) and all(
                isinstance(placement, Replicate) for placement in parameter.placements)
        }
        # Normal keys support checkpoints written by the earlier DTensor
        # representation. New checkpoints use the byte-object prefix below.
        saved_trainable_parameters: dict[str, torch.Tensor] = {
            name: value.detach().clone()
            for name, value in state_dict.items() if name in replicated_trainable_parameters
        }
        for key, payload in state_dict.items():
            if not key.startswith(_REPLICATED_TENSOR_STATE_PREFIX):
                continue
            name = key[len(_REPLICATED_TENSOR_STATE_PREFIX):]
            if name not in replicated_trainable_parameters:
                continue
            if not isinstance(payload, bytes):
                raise TypeError(f"Expected bytes for replicated tensor checkpoint {name!r}, "
                                f"got {type(payload).__name__}")
            saved_trainable_parameters[name] = (_deserialize_replicated_tensor(payload))
        fsdp_state_dict = {
            name: value
            for name, value in state_dict.items()
            if name not in replicated_trainable_parameters and not name.startswith(_REPLICATED_TENSOR_STATE_PREFIX)
        }
        set_model_state_dict(
            self.model,
            model_state_dict=fsdp_state_dict,
            options=StateDictOptions(strict=False),
        )
        with torch.no_grad():
            for name, value in saved_trainable_parameters.items():
                parameter = replicated_trainable_parameters[name]
                destination = (parameter.to_local() if isinstance(parameter, DTensor) else parameter)
                source = (value.to_local() if isinstance(value, DTensor) else value)
                destination.copy_(source)


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(  # type: ignore[no-any-return]
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])


class RandomStateWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, noise_generator: torch.Generator | None = None) -> None:
        self.noise_generator = noise_generator

    def state_dict(self) -> dict[str, Any]:
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            if torch.cuda.device_count() > 1:
                state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.noise_generator is not None:
            state["noise_generator_state"] = self.noise_generator.get_state()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])

        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])

        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])

        # Restore CUDA random state
        if torch.cuda.is_available():
            if "cuda_rng_state" in state_dict:
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            if "cuda_rng_state_all" in state_dict:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state_all"])

        # Restore noise generator state
        if "noise_generator_state" in state_dict and self.noise_generator is not None:
            self.noise_generator.set_state(state_dict["noise_generator_state"])
