import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from fastvideo.train.utils.lora import (
    _register_replicated_gradient_sync,
)


def _two_rank_replicated_gradient_worker(
    rank,
    init_method,
):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        mesh = init_device_mesh(
            "cpu",
            (1, 2),
            mesh_dim_names=("replicate", "shard"),
        )
        parameter = nn.Parameter(
            DTensor.from_local(
                torch.tensor([1.0, 2.0]),
                device_mesh=mesh,
                placements=[Replicate(), Replicate()],
            )
        )
        _register_replicated_gradient_sync(parameter)
        optimizer = torch.optim.SGD([parameter], lr=0.1)

        # Rank-local gradients are [1, 1] and [3, 3]. The hook must average
        # them to [2, 2] before either optimizer sees the gradient.
        coefficient = 1.0 if rank == 0 else 3.0
        loss = (parameter.to_local() * coefficient).sum()
        loss.backward()

        assert isinstance(parameter.grad, DTensor)
        assert torch.equal(
            parameter.grad.to_local(),
            torch.tensor([2.0, 2.0]),
        )
        optimizer.step()
        assert torch.equal(
            parameter.to_local(),
            torch.tensor([0.8, 1.8]),
        )

        gathered = [
            torch.empty_like(parameter.to_local())
            for _ in range(2)
        ]
        dist.all_gather(gathered, parameter.to_local())
        assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


def test_replicated_lora_gradient_is_averaged_across_hsdp_mesh(
    tmp_path,
):
    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    rendezvous = (tmp_path / "lora-gradient-rendezvous").resolve().as_uri()
    mp.spawn(
        _two_rank_replicated_gradient_worker,
        args=(rendezvous,),
        nprocs=2,
        join=True,
    )
