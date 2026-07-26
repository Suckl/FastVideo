# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig, )


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=data_config.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(data_config.seed or 0),
    ))
    return dataloader


def build_parquet_matrixgame2_train_dataloader(
    data_config: DataConfig,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for Matrix-Game 2.0 datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=float(data_config.training_cfg_rate or 0.0),
        drop_last=True,
        text_padding_length=512,
        seed=int(data_config.seed or 0),
    ))
    return dataloader


def build_vidaforge_automodel_train_dataloader(
    data_config: DataConfig,
    *,
    expected_model_name: str,
) -> Any:
    """Build a bucketed dataloader for VidaForge Stage 5 Wan caches."""

    if not isinstance(data_config.data_path, str):
        raise TypeError("VidaForge AutoModel data_path must be one dataset directory")
    from fastvideo.dataset import (
        build_vidaforge_automodel_dataloader, )

    _dataset, dataloader = build_vidaforge_automodel_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        data_config.dataloader_num_workers,
        cfg_rate=data_config.training_cfg_rate,
        seed=int(data_config.seed or 0),
        expected_model_name=expected_model_name,
        expected_vae_fingerprint=(data_config.vidaforge_vae_fingerprint or None),
        expected_text_encoder_fingerprint=(data_config.vidaforge_text_encoder_fingerprint or None),
        allow_unverified_model=(data_config.vidaforge_allow_unverified_model),
    )
    return dataloader
