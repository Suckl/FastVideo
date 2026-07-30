import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import pytest
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from fastvideo.training.checkpointing_utils import ModelWrapper


class DummyWrappedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._lora_a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self._lora_b = nn.Parameter(torch.tensor([3.0, 4.0]))
        self._frozen = nn.Parameter(torch.tensor([5.0, 6.0]), requires_grad=False)

    def named_parameters(self, *args, **kwargs):
        # Simulate wrapped names returned by named_parameters()
        yield "layer._checkpoint_wrapped_module.lora_A", self._lora_a
        yield "layer._checkpoint_wrapped_module.lora_B", self._lora_b
        yield "layer._checkpoint_wrapped_module.frozen_weight", self._frozen


def test_model_wrapper_filters_wrapped_trainable_params(monkeypatch):
    """Regression test for wrapped parameter name mismatch during checkpoint filtering."""
    model = DummyWrappedModule()
    wrapper = ModelWrapper(model)

    mocked_state_dict = {
        "layer.lora_A": torch.tensor([10.0, 20.0]),
        "layer.lora_B": torch.tensor([30.0, 40.0]),
        "layer.frozen_weight": torch.tensor([50.0, 60.0]),
    }

    def mock_get_model_state_dict(_model):
        return mocked_state_dict

    monkeypatch.setattr(
        "fastvideo.training.checkpointing_utils.get_model_state_dict",
        mock_get_model_state_dict,
    )

    filtered_state_dict = wrapper.state_dict()

    assert set(filtered_state_dict.keys()) == {"layer.lora_A", "layer.lora_B"}
    assert torch.equal(filtered_state_dict["layer.lora_A"], mocked_state_dict["layer.lora_A"])
    assert torch.equal(filtered_state_dict["layer.lora_B"], mocked_state_dict["layer.lora_B"])
    assert "layer.frozen_weight" not in filtered_state_dict


def test_model_wrapper_adds_trainable_params_missing_from_fsdp_state(
        monkeypatch):
    """Save adapters attached after FSDP2 recorded its parameter set."""
    model = DummyWrappedModule()
    wrapper = ModelWrapper(model)

    monkeypatch.setattr(
        "fastvideo.training.checkpointing_utils.get_model_state_dict",
        lambda _model: {},
    )
    state_dict = wrapper.state_dict()

    assert set(state_dict) == {"layer.lora_A", "layer.lora_B"}
    assert torch.equal(state_dict["layer.lora_A"], model._lora_a)
    assert torch.equal(state_dict["layer.lora_B"], model._lora_b)
    assert state_dict["layer.lora_A"].data_ptr() != model._lora_a.data_ptr()
    assert state_dict["layer.lora_B"].data_ptr() != model._lora_b.data_ptr()


def test_model_wrapper_delegates_plain_trainable_params_to_loader(
        monkeypatch):
    """Ordinary trainable tensors still use the official state-dict loader."""
    model = DummyWrappedModule()
    wrapper = ModelWrapper(model)
    saved_state_dict = {
        "layer.lora_A": torch.tensor([10.0, 20.0]),
        "layer.lora_B": torch.tensor([30.0, 40.0]),
    }
    expected_lora_a = saved_state_dict["layer.lora_A"].clone()
    expected_lora_b = saved_state_dict["layer.lora_B"].clone()

    def consuming_set_model_state_dict(
        loaded_model,
        model_state_dict,
        **_kwargs,
    ):
        with torch.no_grad():
            loaded_model._lora_a.copy_(
                model_state_dict["layer.lora_A"])
            loaded_model._lora_b.copy_(
                model_state_dict["layer.lora_B"])

    monkeypatch.setattr(
        "fastvideo.training.checkpointing_utils.set_model_state_dict",
        consuming_set_model_state_dict,
    )
    wrapper.load_state_dict(saved_state_dict)

    assert torch.equal(model._lora_a, expected_lora_a)
    assert torch.equal(model._lora_b, expected_lora_b)
    assert torch.equal(model._frozen, torch.tensor([5.0, 6.0]))


def test_model_wrapper_dcp_round_trip_overrides_unmanaged_replicated_dtensor(
    monkeypatch,
    tmp_path,
):
    """DCP must materialize and restore a late replicated adapter."""
    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    rendezvous = (tmp_path / "dtensor-rendezvous").as_posix()
    dist.init_process_group(
        "gloo",
        init_method=f"file:///{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = DummyWrappedModule()
        current_a = DTensor.from_local(
            torch.tensor([7.0, 8.0]),
            device_mesh=mesh,
            placements=[Replicate()],
        )
        current_b = DTensor.from_local(
            torch.tensor([9.0, 10.0]),
            device_mesh=mesh,
            placements=[Replicate()],
        )
        model._lora_a = nn.Parameter(current_a)
        model._lora_b = nn.Parameter(current_b)
        stale_a = DTensor.from_local(
            torch.tensor([-1.0, -2.0]),
            device_mesh=mesh,
            placements=[Replicate()],
        )
        stale_b = DTensor.from_local(
            torch.tensor([-3.0, -4.0]),
            device_mesh=mesh,
            placements=[Replicate()],
        )

        monkeypatch.setattr(
            "fastvideo.training.checkpointing_utils.get_model_state_dict",
            lambda _model: {
                "layer.lora_A": stale_a,
                "layer.lora_B": stale_b,
            },
        )

        state_dict = ModelWrapper(model).state_dict()

        assert not isinstance(state_dict["layer.lora_A"], DTensor)
        assert torch.equal(
            state_dict["layer.lora_A"],
            torch.tensor([7.0, 8.0]),
        )
        assert torch.equal(
            state_dict["layer.lora_B"],
            torch.tensor([9.0, 10.0]),
        )
        assert state_dict["layer.lora_A"].data_ptr() != (
            model._lora_a.to_local().data_ptr()
        )

        loader_keys = []

        def capture_set_model_state_dict(
            _model,
            model_state_dict,
            **_kwargs,
        ):
            loader_keys.append(set(model_state_dict))

        monkeypatch.setattr(
            "fastvideo.training.checkpointing_utils.set_model_state_dict",
            capture_set_model_state_dict,
        )
        wrapper = ModelWrapper(model)
        checkpoint_dir = tmp_path / "dcp"
        dcp.save(
            {"model": wrapper},
            checkpoint_id=str(checkpoint_dir),
        )
        with torch.no_grad():
            model._lora_a.to_local().zero_()
            model._lora_b.to_local().zero_()

        dcp.load(
            {"model": wrapper},
            checkpoint_id=str(checkpoint_dir),
        )

        assert torch.equal(
            model._lora_a.to_local(),
            torch.tensor([7.0, 8.0]),
        )
        assert torch.equal(
            model._lora_b.to_local(),
            torch.tensor([9.0, 10.0]),
        )
        assert loader_keys == [set()]
    finally:
        dist.destroy_process_group()
