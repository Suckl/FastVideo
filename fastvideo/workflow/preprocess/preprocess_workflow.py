import os
from typing import cast

from torch.utils.data import DataLoader

from fastvideo.configs.configs import PreprocessConfig, PreprocessOutputType
from fastvideo.dataset.dataloader.record_schema import (basic_t2v_record_creator, i2v_record_creator)
from fastvideo.dataset.dataloader.schema import (pyarrow_schema_i2v, pyarrow_schema_t2v)
from fastvideo.distributed.parallel_state import get_world_rank
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.workflow.preprocess.components import (ParquetDatasetSaver, PreprocessingDataValidator,
                                                      VidaForgeWanDataValidator, VideoForwardBatchBuilder,
                                                      build_dataset)
from fastvideo.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


class PreprocessWorkflow(WorkflowBase):

    @staticmethod
    def _require_training_samples(total_samples: int) -> None:
        if total_samples == 0:
            raise ValueError("Training preprocessing produced no samples on this rank. "
                             "All input rows may have been rejected by dataset validation; "
                             "check the preprocessing constraints and input metadata.")

    def register_pipelines(self) -> None:
        self.add_pipeline_config("preprocess_pipeline", (PipelineType.PREPROCESS, self.fastvideo_args))

    def register_components(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        preprocess_config: PreprocessConfig = self.fastvideo_args.preprocess_config

        # raw data validator
        if preprocess_config.output_type == PreprocessOutputType.VIDAFORGE_AUTOMODEL:
            raw_data_validator = VidaForgeWanDataValidator(
                num_frames=preprocess_config.num_frames,
                multi_bucket=bool(preprocess_config.vidaforge_bucket_resolution.strip()),
            )
        else:
            raw_data_validator = PreprocessingDataValidator(
                max_height=preprocess_config.max_height,
                max_width=preprocess_config.max_width,
                num_frames=preprocess_config.num_frames,
                train_fps=preprocess_config.train_fps,
                speed_factor=preprocess_config.speed_factor,
                video_length_tolerance_range=preprocess_config.video_length_tolerance_range,
                drop_short_ratio=preprocess_config.drop_short_ratio,
            )
        self.add_component("raw_data_validator", raw_data_validator)

        # training dataset
        try:
            training_dataset = build_dataset(preprocess_config, split="train", validator=raw_data_validator)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Training dataset not found, please use download_dataset.sh to download the dataset first. Error: {e}"
            ) from e

        # we do not use collate_fn here because we use iterable-style Dataset
        # and want to keep the original type of the dataset
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=preprocess_config.preprocess_video_batch_size,
            num_workers=preprocess_config.dataloader_num_workers,
            collate_fn=lambda x: x,
        )
        self.add_component("training_dataloader", training_dataloader)

        # try to load validation dataset if it exists
        try:
            validation_dataset = build_dataset(preprocess_config, split="validation", validator=raw_data_validator)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=preprocess_config.preprocess_video_batch_size,
                num_workers=preprocess_config.dataloader_num_workers,
                collate_fn=lambda x: x,
            )
        except ValueError:
            logger.warning("Validation dataset not found, skipping validation dataset preprocessing.")
            validation_dataloader = None

        self.add_component("validation_dataloader", validation_dataloader)

        # forward batch builder
        video_forward_batch_builder = VideoForwardBatchBuilder(seed=self.fastvideo_args.preprocess_config.seed)
        self.add_component("video_forward_batch_builder", video_forward_batch_builder)

        # record creator
        if self.fastvideo_args.workload_type == WorkloadType.I2V:
            record_creator = i2v_record_creator
            schema = pyarrow_schema_i2v
        else:
            record_creator = basic_t2v_record_creator
            schema = pyarrow_schema_t2v
        processed_dataset_saver = ParquetDatasetSaver(
            flush_frequency=self.fastvideo_args.preprocess_config.flush_frequency,
            samples_per_file=self.fastvideo_args.preprocess_config.samples_per_file,
            schema=schema,
            record_creator=record_creator,
        )
        self.add_component("processed_dataset_saver", processed_dataset_saver)

    def prepare_system_environment(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        dataset_output_dir = self.fastvideo_args.preprocess_config.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(dataset_output_dir, "validation_dataset",
                                                     f"worker_{get_world_rank()}")
        os.makedirs(validation_dataset_output_dir, exist_ok=True)
        self.validation_dataset_output_dir = validation_dataset_output_dir

        training_dataset_output_dir = os.path.join(dataset_output_dir, "training_dataset", f"worker_{get_world_rank()}")
        os.makedirs(training_dataset_output_dir, exist_ok=True)
        self.training_dataset_output_dir = training_dataset_output_dir

    @classmethod
    def get_workflow_cls(cls, fastvideo_args: FastVideoArgs) -> "PreprocessWorkflow":
        assert fastvideo_args.preprocess_config is not None
        if fastvideo_args.preprocess_config.output_type == PreprocessOutputType.VIDAFORGE_AUTOMODEL:
            from fastvideo.configs.pipelines.wan import WanT2V480PConfig
            if fastvideo_args.workload_type != WorkloadType.T2V or not isinstance(fastvideo_args.pipeline_config,
                                                                                  WanT2V480PConfig):
                raise ValueError("vidaforge_automodel output currently supports only Wan text-to-video pipelines")
            if (fastvideo_args.pipeline_config.vae_precision != "fp16"
                    or tuple(fastvideo_args.pipeline_config.text_encoder_precisions) != ("bf16", )):
                raise ValueError("vidaforge_automodel output requires --vae-precision fp16 "
                                 "and --text-encoder-precisions bf16 to match VidaForge Wan")
            if fastvideo_args.pipeline_config.vae_tiling or fastvideo_args.pipeline_config.vae_sp:
                raise ValueError("vidaforge_automodel output requires vae_tiling=false and vae_sp=false")
            from fastvideo.workflow.preprocess.preprocess_workflow_vidaforge_automodel import (
                PreprocessWorkflowVidaForgeAutoModel, )
            return cast(PreprocessWorkflow, PreprocessWorkflowVidaForgeAutoModel)
        is_ltx2_t2v = (fastvideo_args.workload_type == WorkloadType.T2V
                       and fastvideo_args.pipeline_config.__class__.__name__ == "LTX2T2VConfig")
        if is_ltx2_t2v:
            from fastvideo.workflow.preprocess.preprocess_workflow_ltx2_t2v import (PreprocessWorkflowLTX2T2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowLTX2T2V)
        if fastvideo_args.workload_type == WorkloadType.T2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_t2v import (PreprocessWorkflowT2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowT2V)
        elif fastvideo_args.workload_type == WorkloadType.I2V:
            from fastvideo.workflow.preprocess.preprocess_workflow_i2v import (PreprocessWorkflowI2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowI2V)
        else:
            raise ValueError(
                f"Workload type: {fastvideo_args.workload_type} is not supported in preprocessing workflow.")
