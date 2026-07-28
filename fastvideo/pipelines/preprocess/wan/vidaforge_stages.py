# SPDX-License-Identifier: Apache-2.0
"""VidaForge-compatible text and video transforms for Wan Stage 5."""

from __future__ import annotations

import html
import math
import re
from typing import cast

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange

from fastvideo.configs.configs import VideoLoaderType
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, PreprocessBatch
from fastvideo.pipelines.stages import EncodingStage, TextEncodingStage
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.workflow.preprocess.vidaforge_manifest import VIDAFORGE_WAN_MAX_TEMPORAL_REPEAT_PAD_FRAMES


def clean_vidaforge_prompt(prompt: str) -> str:
    """Apply the prompt normalization used by VidaForge's Wan encoder."""
    text = str(prompt)
    from diffusers.utils import is_ftfy_available
    if is_ftfy_available():
        import ftfy
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


class VidaForgeTextTransformStage(PipelineStage):
    """Normalize captions without permanently applying CFG dropout."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        preprocess_batch = cast(PreprocessBatch, batch)
        if not isinstance(preprocess_batch.prompt, list) or not all(
                isinstance(prompt, str) for prompt in preprocess_batch.prompt):
            raise ValueError("VidaForge producer requires one string caption per sample")
        preprocess_batch.prompt = [clean_vidaforge_prompt(prompt) for prompt in preprocess_batch.prompt]
        return preprocess_batch


class VidaForgeTextEncodingStage(TextEncodingStage):
    """Encode cleaned prompts and retain their untruncated token lengths."""

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not isinstance(batch.prompt, list) or len(self.tokenizers) != 1:
            raise ValueError("VidaForge producer requires one tokenizer and a prompt list")
        if len(self.text_encoders) != 1:
            raise ValueError("VidaForge producer requires one text encoder")
        if batch.prompt_embeds or batch.prompt_attention_mask is None or batch.prompt_attention_mask:
            raise ValueError("VidaForge producer requires empty prompt embedding and attention-mask lists")

        prompt_embeds, prompt_masks = self.encode_text(
            batch.prompt,
            fastvideo_args,
            encoder_index=[0],
            return_attention_mask=True,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        batch.prompt_embeds.extend(prompt_embeds)
        batch.prompt_attention_mask.extend(prompt_masks)

        tokenized = self.tokenizers[0](
            batch.prompt,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = tokenized["input_ids"]
        batch.extra["caption_token_lengths"] = [len(token_ids) for token_ids in input_ids]
        return batch


class VidaForgeWanVideoTransformStage(PipelineStage):
    """Match VidaForge's full-clip temporal sampling and center crop."""

    def __init__(self, *, num_frames: int, max_height: int, max_width: int) -> None:
        self.num_frames = int(num_frames)
        self.max_height = int(max_height)
        self.max_width = int(max_width)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        preprocess_batch = cast(PreprocessBatch, batch)
        if preprocess_batch.data_type != "video":
            return preprocess_batch
        if not preprocess_batch.video_loader:
            raise ValueError("Video loader is not set")

        num_frames, target_height, target_width = self._target_geometry(preprocess_batch)
        videos: list[torch.Tensor] = []
        for loader in preprocess_batch.video_loader:
            if fastvideo_args.preprocess_config.video_loader_type == VideoLoaderType.TORCHCODEC:
                official_decode = getattr(loader, "get_vidaforge_wan_frames", None)
                if callable(official_decode):
                    video = official_decode(num_frames)
                else:
                    frame_indices = self._frame_indices(len(loader), num_frames=num_frames)
                    video = loader.get_frames_at(frame_indices).data
            elif fastvideo_args.preprocess_config.video_loader_type == VideoLoaderType.TORCHVISION:
                video, _, _ = torchvision.io.read_video(loader, output_format="TCHW")
                frame_indices = self._frame_indices(int(video.shape[0]), num_frames=num_frames)
                video = video[frame_indices]
            else:
                raise ValueError(f"Invalid video loader type: {fastvideo_args.preprocess_config.video_loader_type}")
            videos.append(self._center_crop_resize(
                video,
                target_height=target_height,
                target_width=target_width,
            ))

        pixel_values = rearrange(torch.stack(videos), "b t c h w -> b c t h w")
        preprocess_batch.latents = pixel_values.float() / 255.0
        batch_size = len(videos)
        preprocess_batch.num_frames = [num_frames] * batch_size
        preprocess_batch.height = [target_height] * batch_size
        preprocess_batch.width = [target_width] * batch_size
        return preprocess_batch

    def _target_geometry(self, batch: PreprocessBatch) -> tuple[int, int, int]:
        bucket = batch.extra.get("vidaforge_bucket")
        if bucket is None:
            return self.num_frames, self.max_height, self.max_width
        if not isinstance(bucket, dict):
            raise ValueError("VidaForge bucket assignment must be a mapping")
        try:
            num_frames = int(bucket["frame_count"])
            target_width = int(bucket["width"])
            target_height = int(bucket["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"VidaForge bucket assignment is invalid: {bucket!r}") from exc
        if (num_frames <= 0 or (num_frames - 1) % 4 != 0 or target_width <= 0 or target_height <= 0 or target_width % 16
                or target_height % 16):
            raise ValueError(f"VidaForge Wan bucket geometry is invalid: {bucket!r}")
        return num_frames, target_height, target_width

    def _frame_indices(self, input_frame_count: int, *, num_frames: int | None = None) -> list[int]:
        target_frame_count = self.num_frames if num_frames is None else int(num_frames)
        if (input_frame_count <= 0
                or target_frame_count - input_frame_count > VIDAFORGE_WAN_MAX_TEMPORAL_REPEAT_PAD_FRAMES):
            raise ValueError(
                f"VidaForge producer requires at least "
                f"{target_frame_count - VIDAFORGE_WAN_MAX_TEMPORAL_REPEAT_PAD_FRAMES} decoded frames for a "
                f"{target_frame_count}-frame bucket, got {input_frame_count}")
        return (torch.linspace(
            0,
            input_frame_count - 1,
            steps=target_frame_count,
            dtype=torch.float64,
        ).round().to(dtype=torch.int64).tolist())

    def _center_crop_resize(
        self,
        video: torch.Tensor,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"Expected decoded video in TCHW format, got {tuple(video.shape)}")
        output_height = self.max_height if target_height is None else int(target_height)
        output_width = self.max_width if target_width is None else int(target_width)
        source_height, source_width = int(video.shape[-2]), int(video.shape[-1])
        source_ratio = source_width / source_height
        target_ratio = output_width / output_height
        if source_ratio > target_ratio:
            crop_height = source_height
            crop_width = max(1, int(math.floor(source_height * target_ratio)))
            top = 0
            left = (source_width - crop_width) // 2
        else:
            crop_width = source_width
            crop_height = max(1, int(math.floor(source_width / target_ratio)))
            top = (source_height - crop_height) // 2
            left = 0
        cropped = video[..., top:top + crop_height, left:left + crop_width]
        return F.interpolate(
            cropped.float(),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )


class VidaForgeWanEncodingStage(EncodingStage):
    """Encode with VidaForge's explicit FP16 VAE input semantics."""

    vae: ParallelTiledVAE

    def __init__(self, vae: ParallelTiledVAE) -> None:
        super().__init__(vae)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.pipeline_config.vae_precision != "fp16":
            raise ValueError("VidaForge Wan encoding requires an FP16 VAE")
        if fastvideo_args.pipeline_config.vae_tiling:
            raise ValueError("VidaForge Wan encoding does not support VAE tiling")
        if batch.latents is None or not isinstance(batch.latents, torch.Tensor):
            raise ValueError("VidaForge Wan encoding requires a pixel tensor")

        device = get_local_torch_device()
        self.vae = self.vae.to(device)
        video_tensor = (batch.latents * 2.0 - 1.0).clamp(-1, 1).to(
            device=device,
            dtype=torch.float16,
        )
        batch.latents = self.vae.encode(video_tensor).mean

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch


__all__ = [
    "VidaForgeTextEncodingStage",
    "VidaForgeTextTransformStage",
    "VidaForgeWanEncodingStage",
    "VidaForgeWanVideoTransformStage",
    "clean_vidaforge_prompt",
]
