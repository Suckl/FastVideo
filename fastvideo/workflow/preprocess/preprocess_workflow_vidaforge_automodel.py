# SPDX-License-Identifier: Apache-2.0
"""Wan preprocessing workflow that publishes VidaForge AutoModel caches."""

from __future__ import annotations

import os
import uuid
from typing import Any, TYPE_CHECKING

import torch
from tqdm import tqdm

from fastvideo.distributed import get_world_group, get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.workflow.preprocess.vidaforge_manifest import build_vidaforge_manifest_fingerprint
from fastvideo.workflow.preprocess.preprocess_workflow_t2v import PreprocessWorkflowT2V
from fastvideo.workflow.preprocess.vidaforge_automodel_writer import VidaForgeAutoModelWriter
from fastvideo.workflow.preprocess.vidaforge_bucketing import (
    VidaForgeBucket,
    VidaForgeBucketPlanner,
    vidaforge_source_fps,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.workflow.preprocess.components import VideoForwardBatchBuilder

logger = init_logger(__name__)


def plan_vidaforge_forward_batches(
    items: list[dict[str, Any]],
    planner: VidaForgeBucketPlanner,
) -> tuple[list[list[dict[str, Any]]], list[tuple[dict[str, Any], Exception]]]:
    """Group planned items and split them with VidaForge's cost scaling."""
    grouped_items: dict[tuple[int, int, int], tuple[VidaForgeBucket, list[dict[str, Any]]]] = {}
    failures: list[tuple[dict[str, Any], Exception]] = []
    for item in items:
        try:
            bucket = planner.bucket_for_item(item)
        except Exception as exc:  # noqa: BLE001
            failures.append((item, exc))
            continue
        planned_item = dict(item)
        planned_item["_vidaforge_bucket"] = bucket.to_dict()
        if bucket.key not in grouped_items:
            grouped_items[bucket.key] = (bucket, [])
        grouped_items[bucket.key][1].append(planned_item)

    if not grouped_items:
        return [], failures
    reference_fps = max(
        vidaforge_source_fps(item) for _, bucket_items in grouped_items.values() for item in bucket_items)
    batches: list[list[dict[str, Any]]] = []
    for bucket, bucket_items in grouped_items.values():
        chunk_size = planner.forward_batch_size(bucket, reference_fps=reference_fps)
        batches.extend(bucket_items[start:start + chunk_size] for start in range(0, len(bucket_items), chunk_size))
    return batches, failures


class PreprocessWorkflowVidaForgeAutoModel(PreprocessWorkflowT2V):
    """Run native FastVideo encoders and emit the Stage 5 tensor contract."""

    training_dataloader: DataLoader
    preprocess_pipeline: ComposedPipelineBase
    video_forward_batch_builder: VideoForwardBatchBuilder

    def prepare_system_environment(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        output_dir = os.path.abspath(os.path.expanduser(self.fastvideo_args.preprocess_config.dataset_output_dir))
        os.makedirs(output_dir, exist_ok=True)
        self.training_dataset_output_dir = output_dir
        self.validation_dataset_output_dir = output_dir

    def run(self) -> None:
        config = self.fastvideo_args.preprocess_config
        assert config is not None
        world_rank = get_world_rank()
        world_size = get_world_size()
        generation = uuid.uuid4().hex if world_rank == 0 else None
        generation = get_world_group().broadcast_object(generation, src=0)
        if not isinstance(generation, str):
            raise RuntimeError("Failed to synchronize VidaForge output generation")

        bucket_planner = (VidaForgeBucketPlanner.from_config(config)
                          if config.vidaforge_bucket_resolution.strip() else None)
        if bucket_planner is None:
            bucket_policy: dict[str, object] = {
                "mode": "fixed",
                "frame_count": config.num_frames,
                "width": config.max_width,
                "height": config.max_height,
            }
        else:
            bucket_policy = {
                "mode": "multi",
                **bucket_planner.to_producer_config(),
            }
        writer = VidaForgeAutoModelWriter(
            self.training_dataset_output_dir,
            model_root=self.preprocess_pipeline.model_path,
            model_name=config.vidaforge_model_name.strip(),
            requested_revision=self.fastvideo_args.revision,
            producer_config={
                "schema_version":
                2,
                "workload_type":
                self.fastvideo_args.workload_type.value,
                "pipeline_config":
                self.fastvideo_args.pipeline_config.__class__.__name__,
                "caption_field":
                config.vidaforge_caption_field.strip(),
                "vidaforge_selection":
                config.vidaforge_selection,
                "manifest_fingerprint":
                build_vidaforge_manifest_fingerprint(config.dataset_path),
                "video_loader_type":
                config.video_loader_type.value,
                "preprocess_video_batch_size":
                config.preprocess_video_batch_size,
                "dataloader_num_workers":
                config.dataloader_num_workers,
                "world_size":
                world_size,
                "max_height":
                config.max_height,
                "max_width":
                config.max_width,
                "num_frames":
                config.num_frames,
                "bucket_policy":
                bucket_policy,
                "train_fps":
                config.train_fps,
                "do_temporal_sample":
                config.do_temporal_sample,
                "seed":
                config.seed,
                "vae_precision":
                self.fastvideo_args.pipeline_config.vae_precision,
                "vae_tiling":
                self.fastvideo_args.pipeline_config.vae_tiling,
                "vae_sp":
                self.fastvideo_args.pipeline_config.vae_sp,
                "text_encoder_precisions":
                self.fastvideo_args.pipeline_config.text_encoder_precisions,
                "text_max_lengths": [
                    getattr(encoder.arch_config, "text_len", None)
                    for encoder in self.fastvideo_args.pipeline_config.text_encoder_configs
                ],
                "disable_autocast":
                self.fastvideo_args.disable_autocast,
            },
            samples_per_shard=config.samples_per_file,
            resume=config.vidaforge_resume,
            rank=world_rank,
            world_size=world_size,
            generation=generation,
        )
        vae = self.preprocess_pipeline.get_module("vae")
        processed_samples = 0
        skipped_samples = 0

        def encode_chunk(items: list[dict]) -> int:
            failure_message: str | None = None
            forward_batch: PreprocessBatch | None = None
            try:
                forward_batch = self.video_forward_batch_builder(items)
                forward_batch = self.preprocess_pipeline.forward(forward_batch, self.fastvideo_args)
                writer.save_batch(forward_batch, vae=vae)
                return len(items)
            except Exception as exc:  # noqa: BLE001
                failure_message = f"{type(exc).__name__}: {exc}"
            finally:
                # A failed homogeneous batch can retain large CUDA tensors
                # through local references. Release it before singleton
                # retries so empty_cache can actually reclaim the allocation.
                del forward_batch
                if failure_message is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            assert failure_message is not None
            if len(items) > 1:
                # Match VidaForge's recoverable row semantics while still
                # getting the throughput benefit of homogeneous batches.
                return sum(encode_chunk([item]) for item in items)
            writer.record_failure(items[0], stage="encoding", error=failure_message)
            logger.warning(
                "VidaForge producer skipped clip_id=%r after an encoding failure: %s",
                items[0].get("clip_id"),
                failure_message,
            )
            return 0

        for source_batch in tqdm(
                self.training_dataloader,
                desc="Producing VidaForge AutoModel cache",
                unit="batch",
        ):
            pending_batch = writer.pending_items(source_batch)
            skipped_samples += len(source_batch) - len(pending_batch)
            if not pending_batch:
                continue
            if bucket_planner is None:
                processed_samples += encode_chunk(pending_batch)
                continue

            forward_batches, planning_failures = plan_vidaforge_forward_batches(pending_batch, bucket_planner)
            for item, exc in planning_failures:
                writer.record_failure(item, stage="bucket_planning", error=exc)
                logger.warning(
                    "VidaForge producer skipped clip_id=%r during bucket planning: %s",
                    item.get("clip_id"),
                    exc,
                )
            for items in forward_batches:
                processed_samples += encode_chunk(items)

        writer.write_rank_progress()
        if world_size > 1:
            get_world_group().barrier()
        published_samples = writer.publish() if world_rank == 0 else None
        published_samples = get_world_group().broadcast_object(published_samples, src=0)
        if not isinstance(published_samples, int) or published_samples <= 0:
            self._require_training_samples(0)
        logger.info(
            "VidaForge producer rank %d encoded %d samples, resumed %d, and failed %d; published total=%d",
            world_rank,
            processed_samples,
            skipped_samples,
            writer.failure_count,
            published_samples,
        )
