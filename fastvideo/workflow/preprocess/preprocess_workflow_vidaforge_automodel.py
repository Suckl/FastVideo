# SPDX-License-Identifier: Apache-2.0
"""Wan preprocessing workflow that publishes VidaForge AutoModel caches."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from tqdm import tqdm

from fastvideo.distributed import get_world_group, get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.workflow.preprocess.vidaforge_manifest import build_vidaforge_manifest_fingerprint
from fastvideo.workflow.preprocess.preprocess_workflow_t2v import PreprocessWorkflowT2V
from fastvideo.workflow.preprocess.vidaforge_automodel_writer import VidaForgeAutoModelWriter

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.workflow.preprocess.components import VideoForwardBatchBuilder

logger = init_logger(__name__)


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

        writer = VidaForgeAutoModelWriter(
            self.training_dataset_output_dir,
            model_root=self.preprocess_pipeline.model_path,
            model_name=config.vidaforge_model_name.strip(),
            requested_revision=self.fastvideo_args.revision,
            producer_config={
                "schema_version":
                1,
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
                "max_height":
                config.max_height,
                "max_width":
                config.max_width,
                "num_frames":
                config.num_frames,
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
        for source_batch in tqdm(
                self.training_dataloader,
                desc="Producing VidaForge AutoModel cache",
                unit="batch",
        ):
            pending_batch = writer.pending_items(source_batch)
            skipped_samples += len(source_batch) - len(pending_batch)
            if not pending_batch:
                continue
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(pending_batch)
            forward_batch = self.preprocess_pipeline.forward(forward_batch, self.fastvideo_args)
            writer.save_batch(forward_batch, vae=vae)
            processed_samples += len(pending_batch)

        writer.write_rank_progress()
        if world_size > 1:
            get_world_group().barrier()
        published_samples = writer.publish() if world_rank == 0 else None
        published_samples = get_world_group().broadcast_object(published_samples, src=0)
        if not isinstance(published_samples, int) or published_samples <= 0:
            self._require_training_samples(0)
        logger.info(
            "VidaForge producer rank %d encoded %d samples and resumed %d; published total=%d",
            world_rank,
            processed_samples,
            skipped_samples,
            published_samples,
        )
