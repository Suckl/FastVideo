import dataclasses
import gc
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset, Video, load_dataset
from torch.utils.data import IterableDataset

from fastvideo.configs.configs import (DatasetType, PreprocessConfig, VideoLoaderType)
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter, records_to_table)
from fastvideo.distributed.parallel_state import get_world_rank, get_world_size
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.pipelines.preprocess.preprocess_stages import resolve_file_backed_video_path
from fastvideo.workflow.preprocess.vidaforge_bucketing import (
    vidaforge_source_fps,
    vidaforge_source_resolution,
)
from fastvideo.workflow.preprocess.vidaforge_manifest import build_vidaforge_dataset

logger = init_logger(__name__)


class PreprocessingDataValidator:

    def __init__(self,
                 max_height: int = 1024,
                 max_width: int = 1024,
                 num_frames: int = 16,
                 train_fps: int = 24,
                 speed_factor: float = 1.0,
                 video_length_tolerance_range: float = 5.0,
                 drop_short_ratio: float = 0.0):
        self.max_height = max_height
        self.max_width = max_width
        self.num_frames = num_frames
        self.train_fps = train_fps
        self.speed_factor = speed_factor
        self.video_length_tolerance_range = video_length_tolerance_range
        self.drop_short_ratio = drop_short_ratio
        self.validators: dict[str, Callable[[dict[str, Any]], bool]] = {}
        self.filter_counts: dict[str, int] = {}

        self.num_items_before_filtering = 0
        self.num_items_after_filtering = 0

        self.register_validators()

    def register_validators(self) -> None:
        self.add_validator("data_type_validator", self._validate_data_type)
        self.add_validator("resolution_validator", self._validate_resolution)
        self.add_validator("frame_sampling_validator", self._validate_frame_sampling)

    def add_validator(self, name: str, validator: Callable[[dict[str, Any]], bool]) -> None:
        self.validators[name] = validator
        self.filter_counts[name] = 0

    def __call__(self, batch: dict[str, Any]) -> bool:
        """
        Validate whether the preprocessing data batch is valid.
        """
        self.num_items_before_filtering += 1

        for name, validator in self.validators.items():
            if not validator(batch):
                self.filter_counts[name] += 1
                return False

        self.num_items_after_filtering += 1
        return True

    def _validate_data_type(self, batch: dict[str, Any]) -> bool:
        """Validate basic validity of data items"""
        return not (batch["caption"] is None or batch["caption"] == "" or batch["fps"] is None or batch["fps"] <= 0
                    or batch["num_frames"] is None or batch["num_frames"] <= 0)

    def _validate_resolution(self, batch: dict[str, Any]) -> bool:
        """Validate resolution constraints"""

        if batch["resolution"] is not None:
            height = batch["resolution"].get("height", None)
            width = batch["resolution"].get("width", None)

        return not (height is None or width is None)

    def _validate_frame_sampling(self, batch: dict[str, Any]) -> bool:
        """Validate frame sampling constraints"""

        if (batch["num_frames"] / batch["fps"]
                > self.video_length_tolerance_range * (self.num_frames / self.train_fps * self.speed_factor)):
            return False

        frame_interval = batch["fps"] / self.train_fps
        start_frame_idx = 0
        frame_indices: np.ndarray = np.arange(start_frame_idx, batch["num_frames"], frame_interval).astype(int)
        return not (len(frame_indices) < self.num_frames and random.random() < self.drop_short_ratio)

    def log_validation_stats(self):
        info = ""
        for name, count in self.filter_counts.items():
            info += f"failed in {name}: {count}, "
        info += f"number of items before filtering: {self.num_items_before_filtering}, "
        info += f"number of items after filtering: {self.num_items_after_filtering}"

        logger.info(info)


class VidaForgeWanDataValidator(PreprocessingDataValidator):
    """Validate the decoded-frame contract used by VidaForge's Wan producer."""

    def __init__(self, *, num_frames: int, multi_bucket: bool = False) -> None:
        self.multi_bucket = bool(multi_bucket)
        super().__init__(num_frames=num_frames)

    def register_validators(self) -> None:
        self.add_validator("data_type_validator", self._validate_data_type)
        self.add_validator("resolution_validator", self._validate_resolution)
        if not self.multi_bucket:
            self.add_validator("decoded_frame_validator", self._validate_decoded_frame_count)

    def _validate_decoded_frame_count(self, batch: dict[str, Any]) -> bool:
        from fastvideo.workflow.preprocess.vidaforge_manifest import (
            VIDAFORGE_WAN_MAX_TEMPORAL_REPEAT_PAD_FRAMES, )

        return int(batch["num_frames"]) >= self.num_frames - VIDAFORGE_WAN_MAX_TEMPORAL_REPEAT_PAD_FRAMES


class VideoForwardBatchBuilder:

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, batch: list) -> PreprocessBatch:
        source_resolutions = [vidaforge_source_resolution(item) for item in batch]
        source_fps_values = [vidaforge_source_fps(item) for item in batch]
        source_durations = [
            float(item["duration_sec"]) if item.get("duration_sec") is not None else float(item["num_frames"]) /
            source_fps_values[index] for index, item in enumerate(batch)
        ]
        bucket_values = [item.get("_vidaforge_bucket") for item in batch]
        if any(value is not None for value in bucket_values):
            if not all(isinstance(value, dict) for value in bucket_values):
                raise ValueError("VidaForge forward batch has incomplete bucket assignments")
            first_bucket = bucket_values[0]
            if any(value != first_bucket for value in bucket_values[1:]):
                raise ValueError("VidaForge forward batch must contain one homogeneous bucket")

        forward_batch = PreprocessBatch(
            video_loader=[item["video"] for item in batch],
            video_file_name=[item["name"] for item in batch],
            height=[item["resolution"]["height"] for item in batch],
            width=[item["resolution"]["width"] for item in batch],
            fps=[item["fps"] for item in batch],
            num_frames=[item["num_frames"] for item in batch],
            prompt=[item["caption"] for item in batch],
            prompt_attention_mask=[],
            data_type="video",
            generator=torch.Generator("cpu").manual_seed(self.seed),
        )
        forward_batch.extra["source_metadata"] = [{
            "clip_id":
            str(item.get("clip_id", item["name"])),
            "original_filename":
            str(item["name"]),
            "original_video_path":
            resolve_file_backed_video_path(item["video"]),
            "source_resolution": [
                source_resolutions[index][0],
                source_resolutions[index][1],
            ],
            "source_fps":
            source_fps_values[index],
            "source_frame_count":
            int(item["num_frames"]),
            "source_duration_sec":
            source_durations[index],
            "caption":
            str(item["caption"]),
            "source_fingerprint":
            item.get("_vidaforge_source_fingerprint"),
        } for index, item in enumerate(batch)]
        if bucket_values and bucket_values[0] is not None:
            forward_batch.extra["vidaforge_bucket"] = dict(bucket_values[0])
        return forward_batch


class ParquetDatasetSaver:
    """Component for saving and writing Parquet datasets using shared parquet_io."""

    def __init__(self, flush_frequency: int, samples_per_file: int, schema: pa.Schema,
                 record_creator: Callable[..., list[dict[str, Any]]]):
        self.flush_frequency = flush_frequency
        self.samples_per_file = samples_per_file
        self.schema = schema
        self.create_records_from_batch = record_creator
        self.num_processed_samples: int = 0
        self._writer: ParquetDatasetWriter | None = None

    def save_and_write_parquet_batch(self,
                                     batch: PreprocessBatch,
                                     output_dir: str,
                                     extra_features: dict[str, Any] | None = None) -> None:
        """
        Save and write Parquet dataset batch
        
        Args:
            batch: PreprocessBatch containing video and metadata information
            output_dir: Output directory
            extra_features: Extra features
            
        Returns:
            Number of processed samples
        """
        assert isinstance(batch.latents, torch.Tensor)
        assert isinstance(batch.prompt_embeds, list)
        assert isinstance(batch.prompt_attention_mask, list)

        # Process non-padded embeddings (if needed)
        if batch.prompt_attention_mask is not None:
            batch.prompt_embeds = self._process_non_padded_embeddings(batch.prompt_embeds[0],
                                                                      batch.prompt_attention_mask[0])
        else:
            raise ValueError("prompt_attention_mask is None")

        # Prepare batch data for Parquet dataset
        batch_data: list[dict[str, Any]] = []

        for key in dataclasses.fields(batch):
            value = getattr(batch, key.name)
            if isinstance(value, list):
                for idx in range(len(value)):
                    if isinstance(value[idx], torch.Tensor):
                        value[idx] = value[idx].cpu().numpy()
            elif isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
                setattr(batch, key.name, value)

        # Create record for Parquet dataset
        records = self.create_records_from_batch(batch)
        batch_data.extend(records)

        if batch_data:
            self.num_processed_samples += len(batch_data)
            table = records_to_table(batch_data, self.schema)
            if self._writer is None:
                os.makedirs(output_dir, exist_ok=True)
                self._writer = ParquetDatasetWriter(out_dir=output_dir, samples_per_file=self.samples_per_file)
            self._writer.append_table(table)
            logger.debug("Collected batch with %s samples", len(table))

        # If flush is needed
        if self.num_processed_samples >= self.flush_frequency:
            self.flush_tables()

    def _process_non_padded_embeddings(self, prompt_embeds: torch.Tensor,
                                       prompt_attention_mask: torch.Tensor) -> list[torch.Tensor]:
        """Process non-padded embeddings"""
        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(prompt_attention_mask, torch.Tensor)
        assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]

        # Get sequence lengths from attention masks (number of 1s)
        seq_lens = prompt_attention_mask.sum(dim=1)

        non_padded_embeds = []

        # Process each item in the batch
        for i in range(prompt_embeds.size(0)):
            seq_len = seq_lens[i].item()
            # Slice the embeddings and masks to keep only non-padding parts
            non_padded_embeds.append(prompt_embeds[i, :seq_len])

        return non_padded_embeds

    def flush_tables(self, write_remainder: bool = False):
        """Flush buffered records to disk.

        Args:
            output_dir: Directory where parquet files are written. Kept for API
                symmetry (writer already configured with this path).
            write_remainder: If True, also write any leftover rows smaller than
                ``samples_per_file`` as a final small file. Useful for the last flush.
        """
        if self._writer is None:
            return
        _ = self._writer.flush(write_remainder=write_remainder)
        # Reset processed sample count modulo samples_per_file
        remainder = self.num_processed_samples % self.samples_per_file
        self.num_processed_samples = 0 if write_remainder else remainder

    def clean_up(self) -> None:
        """Clean up all tables"""
        self.flush_tables(write_remainder=True)
        self._writer = None
        self.num_processed_samples = 0
        gc.collect()

    def __del__(self):
        self.clean_up()


def build_dataset(preprocess_config: PreprocessConfig, split: str,
                  validator: Callable[[dict[str, Any]], bool]) -> Dataset | IterableDataset:
    if preprocess_config.dataset_type == DatasetType.HF:
        dataset = load_dataset(preprocess_config.dataset_path, split=split)
        dataset = dataset.filter(validator)
        dataset = dataset.shard(num_shards=get_world_size(), index=get_world_rank())
    elif preprocess_config.dataset_type == DatasetType.MERGED:
        metadata_json_path = os.path.join(preprocess_config.dataset_path, "videos2caption.json")
        video_folder = os.path.join(preprocess_config.dataset_path, "videos")
        dataset = load_dataset("json", data_files=metadata_json_path, split=split)
        column_names = dataset.column_names
        # rename columns to match the schema
        if "cap" in column_names:
            dataset = dataset.rename_column("cap", "caption")
        if "path" in column_names:
            dataset = dataset.rename_column("path", "name")

        dataset = dataset.filter(validator)
        dataset = dataset.shard(num_shards=get_world_size(), index=get_world_rank())

        # add video column
        def add_video_column(item: dict[str, Any]) -> dict[str, Any]:
            item["video"] = os.path.join(video_folder, item["name"])
            return item

        dataset = dataset.map(add_video_column)
        if preprocess_config.video_loader_type == VideoLoaderType.TORCHCODEC:
            dataset = dataset.cast_column("video", Video())
    elif preprocess_config.dataset_type == DatasetType.VIDAFORGE:
        dataset = build_vidaforge_dataset(preprocess_config, split=split, validator=validator)
    else:
        raise ValueError(f"Invalid dataset type: {preprocess_config.dataset_type}")

    return dataset
