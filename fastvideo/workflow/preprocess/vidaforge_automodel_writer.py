# SPDX-License-Identifier: Apache-2.0
"""VidaForge Stage 5 compatible writer for FastVideo Wan preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch

logger = init_logger(__name__)

_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".safetensors"})
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_revision(model_root: Path, requested_revision: str | None) -> str:
    del requested_revision
    parts = model_root.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and _REVISION_PATTERN.fullmatch(parts[index + 1].lower()):
            return parts[index + 1].lower()
    return "local-content-addressed"


def _component_manifest(
    model_root: Path,
    *,
    component: str,
    component_dirs: tuple[str, ...],
    model_name: str,
    revision: str,
) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    weight_count = 0
    for directory_name in component_dirs:
        directory = model_root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {component} provenance directory: {directory}")
        for path in sorted(file_path for file_path in directory.rglob("*") if file_path.is_file()):
            suffix = path.suffix.lower()
            if suffix in _WEIGHT_SUFFIXES:
                weight_count += 1
            files.append({
                "path": path.relative_to(model_root).as_posix(),
                "sha256": _sha256_file(path),
            })
    if weight_count == 0:
        raise ValueError(f"No weight files found for {component} under {model_root}")
    return {
        "schema_version": _SCHEMA_VERSION,
        "model_name": model_name,
        "model_revision": revision,
        "component": component,
        "files": files,
    }


def build_model_provenance(
    model_root: str | Path,
    *,
    model_name: str,
    requested_revision: str | None = None,
) -> dict[str, Any]:
    """Hash the exact configs and weight files that produce cached tensors."""
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Resolved model directory does not exist: {root}")
    revision = _resolved_revision(root, requested_revision)
    vae_manifest = _component_manifest(
        root,
        component="vae",
        component_dirs=("vae", ),
        model_name=model_name,
        revision=revision,
    )
    text_manifest = _component_manifest(
        root,
        component="text_encoder",
        component_dirs=("text_encoder", "tokenizer"),
        model_name=model_name,
        revision=revision,
    )
    vae_fingerprint = hashlib.sha256(_canonical_json(vae_manifest)).hexdigest()
    text_fingerprint = hashlib.sha256(_canonical_json(text_manifest)).hexdigest()
    return {
        "schema_version": _SCHEMA_VERSION,
        "producer": "fastvideo",
        "model_type": "wan",
        "model_name": model_name,
        "model_revision": revision,
        "vae_fingerprint": vae_fingerprint,
        "text_encoder_fingerprint": text_fingerprint,
        "components": {
            "vae": vae_manifest,
            "text_encoder": text_manifest,
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_bytes(_canonical_json(value))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _safe_child(root: Path, relative_value: object, *, kind: str) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute():
        raise ValueError(f"{kind} path must be relative: {relative_value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{kind} path escapes output directory: {relative_value!r}")
    return resolved


def build_vidaforge_source_fingerprint(item: dict[str, Any]) -> str:
    """Bind one cache entry to its cleaned caption, decoded metadata, and media bytes."""
    from fastvideo.pipelines.preprocess.wan.vidaforge_stages import clean_vidaforge_prompt

    clip_id = str(item.get("clip_id", "")).strip()
    video = item.get("video")
    media_source = getattr(video, "source", video)
    if isinstance(media_source, bytes):
        media_sha256 = hashlib.sha256(media_source).hexdigest()
    elif isinstance(media_source, str):
        media_path = Path(media_source).expanduser().resolve()
        if not media_path.is_file():
            raise FileNotFoundError(f"VidaForge producer media path does not exist for {clip_id!r}: {media_path}")
        media_sha256 = _sha256_file(media_path)
    else:
        raise ValueError(f"VidaForge producer cannot fingerprint media for {clip_id!r}")

    resolution = item.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError(f"VidaForge producer input resolution is invalid for {clip_id!r}")
    source_fps = float(item.get("fps", 0))
    source_frame_count = int(item.get("num_frames", 0))
    source_duration = item.get("duration_sec")
    source_duration_sec = (float(source_duration) if source_duration is not None else source_frame_count /
                           source_fps if source_fps > 0 else 0.0)
    identity = {
        "schema_version": 1,
        "clip_id": clip_id,
        "caption": clean_vidaforge_prompt(str(item.get("caption", ""))),
        "media_sha256": media_sha256,
        "source_resolution": [
            int(resolution.get("width", 0)),
            int(resolution.get("height", 0)),
        ],
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "source_duration_sec": source_duration_sec,
    }
    if (not clip_id or not identity["caption"] or min(identity["source_resolution"]) <= 0 or identity["source_fps"] <= 0
            or identity["source_frame_count"] <= 0 or identity["source_duration_sec"] <= 0
            or not math.isfinite(identity["source_duration_sec"])):
        raise ValueError(f"VidaForge producer input identity is incomplete for {clip_id!r}")
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _wan_normalize(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    mean_value = getattr(vae, "latents_mean", None)
    std_value = getattr(vae, "latents_std", None)
    if mean_value is None or std_value is None:
        config = getattr(vae, "config", None)
        mean_value = getattr(config, "latents_mean", None)
        std_value = getattr(config, "latents_std", None)
    if mean_value is None or std_value is None:
        raise ValueError("Wan VAE must expose latents_mean and latents_std")
    official_latents = latents.to(dtype=torch.float16)
    mean = torch.as_tensor(mean_value, dtype=torch.float16, device=latents.device).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(std_value, dtype=torch.float16, device=latents.device).view(1, -1, 1, 1, 1)
    if mean.shape[1] != latents.shape[1] or std.shape[1] != latents.shape[1] or torch.any(std <= 0):
        raise ValueError("Wan VAE latent normalization statistics do not match encoded channels")
    return (official_latents - mean) / std


def _validated_text_mask(
    text_mask: torch.Tensor,
    *,
    sequence_length: int,
    caption_token_length: int,
    clip_id: str,
    pad_short: bool,
) -> torch.Tensor:
    mask = text_mask.to(device="cpu", dtype=torch.float32)
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError("VidaForge producer text mask must have shape [1, M]: "
                         f"clip_id={clip_id!r}, shape={tuple(mask.shape)}")
    mask_length = int(mask.shape[1])
    if mask_length > sequence_length:
        raise ValueError("VidaForge producer text mask cannot exceed the embedding sequence length: "
                         f"clip_id={clip_id!r}, mask_length={mask_length}, sequence_length={sequence_length}")
    if mask_length < sequence_length:
        if not pad_short:
            raise ValueError("Stored VidaForge text mask must match the embedding sequence length: "
                             f"clip_id={clip_id!r}, mask_length={mask_length}, sequence_length={sequence_length}")
        padded_mask = torch.zeros((1, sequence_length), dtype=mask.dtype)
        padded_mask[:, :mask_length] = mask
        mask = padded_mask
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"VidaForge producer text mask must contain only finite binary values: {clip_id!r}")
    expected_valid_length = min(caption_token_length, sequence_length)
    valid_length = int(mask.sum().item())
    if valid_length != expected_valid_length:
        raise ValueError("VidaForge producer text mask disagrees with caption_token_length: "
                         f"clip_id={clip_id!r}, mask_tokens={valid_length}, "
                         f"caption_token_length={caption_token_length}, sequence_length={sequence_length}")
    return mask


class VidaForgeAutoModelWriter:
    """Write portable Stage 5 cache files and publish an atomic shard index."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model_root: str | Path,
        model_name: str,
        requested_revision: str | None,
        producer_config: dict[str, Any],
        samples_per_shard: int,
        resume: bool,
        rank: int,
        world_size: int,
        generation: str,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("generation must be a 32-character lowercase hexadecimal ID")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.generation = generation
        self.samples_per_shard = int(samples_per_shard)
        self.producer_config = json.loads(_canonical_json(producer_config))
        producer_config_fingerprint = hashlib.sha256(_canonical_json(self.producer_config)).hexdigest()
        self.provenance = build_model_provenance(
            model_root,
            model_name=model_name,
            requested_revision=requested_revision,
        )
        self.provenance["requested_revision"] = requested_revision
        self.provenance["producer_config"] = self.producer_config
        self.provenance["producer_config_fingerprint"] = producer_config_fingerprint
        self._existing_items = self._load_existing_items(resume=resume)
        self._retained_items: dict[str, dict[str, Any]] = {}
        self._new_items: dict[str, dict[str, Any]] = {}
        self._failures: dict[str, dict[str, Any]] = {}

    @property
    def completed_count(self) -> int:
        return len(self._existing_items) + len(self._new_items)

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    def pending_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for item in items:
            clip_id = str(item.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError("VidaForge producer input is missing clip_id")
            source_fingerprint = build_vidaforge_source_fingerprint(item)
            item["_vidaforge_source_fingerprint"] = source_fingerprint
            existing = self._existing_items.get(clip_id)
            if existing is not None:
                if existing.get("source_fingerprint") != source_fingerprint:
                    raise ValueError(
                        f"Cannot resume VidaForge clip {clip_id!r}: caption, media, or source metadata changed")
                if clip_id not in self._retained_items:
                    self._validate_item(
                        existing,
                        where=self.output_dir / "metadata.json",
                        validate_payload=True,
                    )
                    self._retained_items[clip_id] = existing
            elif clip_id not in self._new_items:
                pending.append(item)
        return pending

    def record_failure(self, item: dict[str, Any], *, stage: str, error: BaseException | str) -> None:
        """Record one recoverable source failure for the published diagnostics."""
        clip_id = str(item.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("Cannot record a VidaForge producer failure without clip_id")
        if clip_id in self._new_items:
            raise ValueError(f"Cannot mark an already encoded VidaForge clip as failed: {clip_id!r}")
        source_fingerprint = str(item.get("_vidaforge_source_fingerprint", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint):
            source_fingerprint = build_vidaforge_source_fingerprint(item)
        message = str(error).strip()
        if not message:
            message = error.__class__.__name__ if isinstance(error, BaseException) else "unknown error"
        self._failures[clip_id] = {
            "clip_id": clip_id,
            "stage": str(stage).strip() or "unknown",
            "error": message[:2000],
            "source_fingerprint": source_fingerprint,
        }

    def save_batch(self, batch: PreprocessBatch, *, vae: Any) -> None:
        if not isinstance(batch.latents, torch.Tensor) or batch.latents.ndim != 5:
            raise ValueError("VidaForge producer requires batched 5D VAE latents")
        if len(batch.prompt_embeds) != 1 or not isinstance(batch.prompt_embeds[0], torch.Tensor):
            raise ValueError("VidaForge producer requires exactly one text embedding tensor")
        if not batch.prompt_attention_mask or len(batch.prompt_attention_mask) != 1:
            raise ValueError("VidaForge producer requires exactly one text attention mask")
        source_metadata = batch.extra.get("source_metadata")
        if not isinstance(source_metadata, list) or len(source_metadata) != batch.latents.shape[0]:
            raise ValueError("VidaForge producer source metadata does not match encoded batch size")
        if batch.prompt_embeds[0].shape[0] != batch.latents.shape[0]:
            raise ValueError("VidaForge producer text/video batch sizes differ")
        caption_token_lengths = batch.extra.get("caption_token_lengths")
        if not isinstance(caption_token_lengths, list) or len(caption_token_lengths) != batch.latents.shape[0]:
            raise ValueError("VidaForge producer requires one untruncated caption token length per sample")

        normalized_latents = _wan_normalize(batch.latents, vae)
        for index, source in enumerate(source_metadata):
            width = int(batch.width[index])
            height = int(batch.height[index])
            frame_count = int(batch.num_frames[index])
            expected_geometry = self._expected_geometry(source)
            actual_geometry = (frame_count, width, height)
            if actual_geometry != expected_geometry or (frame_count - 1) % 4 != 0:
                raise ValueError(
                    "Encoded Wan sample geometry does not match the producer contract: "
                    f"clip_id={source.get('clip_id')!r}, actual={actual_geometry}, expected={expected_geometry}")
            self._save_sample(
                source=source,
                latents=normalized_latents[index:index + 1],
                text_embeddings=batch.prompt_embeds[0][index:index + 1],
                text_mask=batch.prompt_attention_mask[0][index:index + 1],
                caption_token_length=int(caption_token_lengths[index]),
                width=width,
                height=height,
                frame_count=frame_count,
            )

    def _expected_geometry(self, source: dict[str, Any]) -> tuple[int, int, int]:
        bucket_policy = self.producer_config.get("bucket_policy")
        if bucket_policy is None:
            return (
                int(self.producer_config["num_frames"]),
                int(self.producer_config["max_width"]),
                int(self.producer_config["max_height"]),
            )
        if not isinstance(bucket_policy, dict):
            raise ValueError("VidaForge producer bucket_policy must be a mapping")
        mode = str(bucket_policy.get("mode", "")).strip()
        if mode == "fixed":
            return (
                int(bucket_policy["frame_count"]),
                int(bucket_policy["width"]),
                int(bucket_policy["height"]),
            )
        if mode != "multi":
            raise ValueError(f"Unsupported VidaForge producer bucket mode: {mode!r}")

        from fastvideo.workflow.preprocess.vidaforge_bucketing import (
            VIDAFORGE_BUCKETING_REFERENCE_REVISION,
            VidaForgeBucketPlanner,
        )

        if bucket_policy.get("reference_revision") != VIDAFORGE_BUCKETING_REFERENCE_REVISION:
            raise ValueError("VidaForge producer bucket policy has an unsupported reference revision")
        planner = VidaForgeBucketPlanner(
            resolution=str(bucket_policy["resolution"]),
            upscale=bool(bucket_policy["upscale"]),
            durations_sec=tuple(float(value) for value in bucket_policy["durations_sec"]),
            temporal_stride=int(bucket_policy["temporal_stride"]),
            input_size_multiple=int(bucket_policy["input_size_multiple"]),
            dynamic_forward_batch_size=int(bucket_policy["dynamic_forward_batch_size"]),
        )
        source_fps = float(source["source_fps"])
        source_frame_count = int(source["source_frame_count"])
        source_duration_sec = float(source.get("source_duration_sec", source_frame_count / source_fps))
        source_resolution = source["source_resolution"]
        expected = planner.bucket_for_item({
            "duration_sec": source_duration_sec,
            "fps": source_fps,
            "resolution": {
                "width": int(source_resolution[0]),
                "height": int(source_resolution[1]),
            },
        })
        return expected.key

    def _save_sample(
        self,
        *,
        source: dict[str, Any],
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        caption_token_length: int,
        width: int,
        height: int,
        frame_count: int,
    ) -> None:
        clip_id = str(source["clip_id"])
        digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
        relative_path = Path(f"{frame_count}f") / f"{width}x{height}" / digest[:2] / digest[2:4] / f"{digest}.meta"
        path = _safe_child(self.output_dir, relative_path, kind="cache file")
        latent_cpu = latents.to(device="cpu", dtype=torch.float16)
        embeddings_cpu = text_embeddings.to(device="cpu", dtype=torch.bfloat16)
        if not torch.isfinite(latent_cpu).all():
            raise ValueError(f"VidaForge producer VAE latents must be finite: {clip_id!r}")
        if not torch.isfinite(embeddings_cpu).all():
            raise ValueError(f"VidaForge producer text embeddings must be finite: {clip_id!r}")
        source_fingerprint = str(source.get("source_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint):
            raise ValueError(f"VidaForge producer source fingerprint is invalid: {clip_id!r}")
        if caption_token_length <= 0:
            raise ValueError(f"VidaForge producer caption token length must be positive: {clip_id!r}")
        if embeddings_cpu.ndim != 3 or embeddings_cpu.shape[0] != 1:
            raise ValueError("VidaForge producer text embeddings must have shape [1, L, D]: "
                             f"clip_id={clip_id!r}, shape={tuple(embeddings_cpu.shape)}")
        sequence_length = int(embeddings_cpu.shape[1])
        mask_cpu = _validated_text_mask(
            text_mask,
            sequence_length=sequence_length,
            caption_token_length=caption_token_length,
            clip_id=clip_id,
            pad_short=True,
        )
        metadata = {
            "producer":
            "fastvideo",
            "model_type":
            "wan",
            "model_name":
            self.model_name,
            "model_revision":
            self.provenance["model_revision"],
            "vae_fingerprint":
            self.provenance["vae_fingerprint"],
            "text_encoder_fingerprint":
            self.provenance["text_encoder_fingerprint"],
            "producer_config_fingerprint":
            self.provenance["producer_config_fingerprint"],
            "source_fingerprint":
            source_fingerprint,
            "clip_id":
            clip_id,
            "caption":
            str(source["caption"]),
            "caption_field":
            self.producer_config["caption_field"],
            "bucket_resolution": [width, height],
            "bucket_frame_count":
            frame_count,
            "caption_token_length":
            caption_token_length,
            "caption_token_max_length":
            sequence_length,
            "caption_token_truncated":
            caption_token_length > sequence_length,
            "source_resolution":
            list(source["source_resolution"]),
            "source_fps":
            float(source["source_fps"]),
            "source_frame_count":
            int(source["source_frame_count"]),
            "source_duration_sec":
            float(source.get("source_duration_sec",
                             int(source["source_frame_count"]) / float(source["source_fps"]))),
            "latent_shape":
            list(latent_cpu.shape),
        }
        payload = {
            "video_latents": latent_cpu,
            "text_embeddings": embeddings_cpu,
            "text_mask": mask_cpu,
            "metadata": metadata,
            "original_filename": str(source["original_filename"]),
            "original_video_path": source.get("original_video_path"),
            "num_frames": frame_count,
            "bucket_frame_count": frame_count,
        }
        _atomic_torch_save(path, payload)
        relative_posix = relative_path.as_posix()
        self._failures.pop(clip_id, None)
        self._new_items[clip_id] = {
            "cache_file": relative_posix,
            "bucket_resolution": [width, height],
            "bucket_frame_count": frame_count,
            "latent_shape": list(latent_cpu.shape),
            "clip_id": clip_id,
            "caption_token_length": caption_token_length,
            "vae_fingerprint": self.provenance["vae_fingerprint"],
            "text_encoder_fingerprint": self.provenance["text_encoder_fingerprint"],
            "producer_config_fingerprint": self.provenance["producer_config_fingerprint"],
            "source_fingerprint": source_fingerprint,
        }

    def write_rank_progress(self) -> Path:
        path = self.output_dir / ".vidaforge_progress" / f"{self.generation}-rank-{self.rank:05d}.json"
        current_items = dict(self._retained_items)
        current_items.update(self._new_items)
        _atomic_json(path, list(current_items.values()))
        failure_path = self.output_dir / ".vidaforge_progress" / f"{self.generation}-rank-{self.rank:05d}-failures.json"
        _atomic_json(failure_path, [self._failures[key] for key in sorted(self._failures)])
        return path

    def publish(self) -> int:
        """Merge this generation on rank zero and atomically publish metadata.json."""
        if self.rank != 0:
            raise RuntimeError("Only rank zero may publish VidaForge metadata")
        items_by_clip: dict[str, dict[str, Any]] = {}
        failures_by_clip: dict[str, dict[str, Any]] = {}
        progress_dir = self.output_dir / ".vidaforge_progress"
        for rank in range(self.world_size):
            path = progress_dir / f"{self.generation}-rank-{rank:05d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing VidaForge rank progress: {path}")
            rank_items = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rank_items, list):
                raise ValueError(f"VidaForge rank progress must contain a list: {path}")
            for item in rank_items:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid VidaForge rank progress item: {path}")
                self._validate_item(item, where=path)
                clip_id = str(item.get("clip_id", ""))
                if clip_id in failures_by_clip:
                    raise ValueError(f"VidaForge clip is both successful and failed: {clip_id!r}")
                previous = items_by_clip.get(clip_id)
                if previous is not None and previous != item:
                    raise ValueError(f"Conflicting VidaForge cache entries for clip_id={clip_id!r}")
                items_by_clip[clip_id] = item
            failure_path = progress_dir / f"{self.generation}-rank-{rank:05d}-failures.json"
            if not failure_path.is_file():
                raise FileNotFoundError(f"Missing VidaForge rank failure progress: {failure_path}")
            rank_failures = json.loads(failure_path.read_text(encoding="utf-8"))
            if not isinstance(rank_failures, list):
                raise ValueError(f"VidaForge rank failure progress must contain a list: {failure_path}")
            for failure in rank_failures:
                if not isinstance(failure, dict):
                    raise ValueError(f"Invalid VidaForge rank failure item: {failure_path}")
                clip_id = str(failure.get("clip_id", "")).strip()
                stage = str(failure.get("stage", "")).strip()
                error = str(failure.get("error", "")).strip()
                source_fingerprint = str(failure.get("source_fingerprint", ""))
                if (not clip_id or not stage or not error or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint)):
                    raise ValueError(f"Incomplete VidaForge rank failure item: {failure_path}")
                if clip_id in items_by_clip:
                    raise ValueError(f"VidaForge clip is both successful and failed: {clip_id!r}")
                previous = failures_by_clip.get(clip_id)
                if previous is not None and previous != failure:
                    raise ValueError(f"Conflicting VidaForge failure entries for clip_id={clip_id!r}")
                failures_by_clip[clip_id] = failure
        sorted_items = [items_by_clip[key] for key in sorted(items_by_clip)]
        shard_names: list[str] = []
        for index, start in enumerate(range(0, len(sorted_items), self.samples_per_shard)):
            relative = Path("shards") / f"metadata-{self.generation}-{index:06d}.json"
            _atomic_json(self.output_dir / relative, sorted_items[start:start + self.samples_per_shard])
            shard_names.append(relative.as_posix())
        _atomic_json(self.output_dir / "provenance.json", self.provenance)
        failures = [failures_by_clip[key] for key in sorted(failures_by_clip)]
        _atomic_json(self.output_dir / "failures.json", failures)
        bucket_counts: dict[str, int] = {}
        for item in sorted_items:
            width, height = (int(value) for value in item["bucket_resolution"])
            bucket_key = f"{int(item['bucket_frame_count'])}f/{width}x{height}"
            bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
        summary = {
            "schema_version": _SCHEMA_VERSION,
            "format": "vidaforge_automodel",
            "input_count": len(sorted_items) + len(failures),
            "ok_count": len(sorted_items),
            "failed_count": len(failures),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "failure_stage_counts": {
                stage: sum(1 for failure in failures if failure["stage"] == stage)
                for stage in sorted({str(failure["stage"])
                                     for failure in failures})
            },
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        _atomic_json(
            self.output_dir / "metadata.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "format": "vidaforge_automodel",
                "producer": "fastvideo",
                "model_name": self.model_name,
                "model_revision": self.provenance["model_revision"],
                "vae_fingerprint": self.provenance["vae_fingerprint"],
                "text_encoder_fingerprint": self.provenance["text_encoder_fingerprint"],
                "producer_config_fingerprint": self.provenance["producer_config_fingerprint"],
                "provenance_file": "provenance.json",
                "summary_file": "summary.json",
                "failures_file": "failures.json",
                "shards": shard_names,
            },
        )
        logger.info("Published %d VidaForge AutoModel samples under %s", len(sorted_items), self.output_dir)
        return len(sorted_items)

    def _load_existing_items(self, *, resume: bool) -> dict[str, dict[str, Any]]:
        metadata_path = self.output_dir / "metadata.json"
        if not metadata_path.is_file():
            orphan = next(self.output_dir.rglob("*.meta"), None)
            if orphan is not None and not resume:
                raise FileExistsError(f"Output contains an unindexed .meta file: {orphan}")
            return {}
        if not resume:
            raise FileExistsError(
                f"VidaForge output already exists at {metadata_path}; pass --preprocess.vidaforge-resume")

        provenance_path = self.output_dir / "provenance.json"
        if not provenance_path.is_file():
            raise ValueError(f"Cannot resume cache without provenance.json: {self.output_dir}")
        existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for field in (
                "model_name",
                "model_revision",
                "vae_fingerprint",
                "text_encoder_fingerprint",
                "producer_config_fingerprint",
        ):
            if existing_provenance.get(field) != self.provenance[field]:
                raise ValueError(f"Cannot resume VidaForge cache with different {field}: "
                                 f"existing={existing_provenance.get(field)!r}, current={self.provenance[field]!r}")

        root = json.loads(metadata_path.read_text(encoding="utf-8"))
        shard_names = root.get("shards") if isinstance(root, dict) else None
        if not isinstance(shard_names, list):
            raise ValueError(f"Invalid VidaForge metadata index: {metadata_path}")
        items: dict[str, dict[str, Any]] = {}
        for shard_name in shard_names:
            shard_path = _safe_child(self.output_dir, shard_name, kind="metadata shard")
            shard_items = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard_items, list):
                raise ValueError(f"VidaForge metadata shard must contain a list: {shard_path}")
            for item in shard_items:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid VidaForge metadata item in {shard_path}")
                self._validate_item(item, where=shard_path)
                clip_id = str(item["clip_id"]).strip()
                if clip_id in items:
                    raise ValueError(f"Duplicate clip_id in existing VidaForge cache: {clip_id!r}")
                items[clip_id] = item
        return items

    def _validate_item(self, item: dict[str, Any], *, where: Path, validate_payload: bool = False) -> None:
        clip_id = str(item.get("clip_id", "")).strip()
        cache_path = _safe_child(self.output_dir, item.get("cache_file", ""), kind="cache file")
        if not clip_id or cache_path.suffix != ".meta" or not cache_path.is_file():
            raise ValueError(f"Incomplete VidaForge metadata item in {where}: {item}")
        if item.get("vae_fingerprint") != self.provenance["vae_fingerprint"]:
            raise ValueError(f"VAE fingerprint mismatch in {where}")
        if item.get("text_encoder_fingerprint") != self.provenance["text_encoder_fingerprint"]:
            raise ValueError(f"Text encoder fingerprint mismatch in {where}")
        if item.get("producer_config_fingerprint") != self.provenance["producer_config_fingerprint"]:
            raise ValueError(f"Producer config fingerprint mismatch in {where}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("source_fingerprint", ""))):
            raise ValueError(f"Source fingerprint is invalid in {where}")
        if validate_payload:
            self._validate_payload(item, cache_path=cache_path, where=where)

    def _validate_payload(self, item: dict[str, Any], *, cache_path: Path, where: Path) -> None:
        try:
            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as exc:
            raise ValueError(f"Cannot load resumed VidaForge payload {cache_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Resumed VidaForge payload must be a dict: {cache_path}")
        latents = payload.get("video_latents")
        embeddings = payload.get("text_embeddings")
        text_mask = payload.get("text_mask")
        metadata = payload.get("metadata")
        if (not isinstance(latents, torch.Tensor) or not latents.is_floating_point() or latents.ndim != 5
                or tuple(latents.shape) != tuple(item.get("latent_shape", ()))):
            raise ValueError(f"Resumed VidaForge latent shape mismatch: {cache_path}")
        if latents.dtype != torch.float16:
            raise ValueError(f"Resumed VidaForge latents must be FP16 tensors: {cache_path}")
        if (not isinstance(embeddings, torch.Tensor) or not embeddings.is_floating_point() or embeddings.ndim != 3
                or embeddings.shape[0] != 1):
            raise ValueError(f"Resumed VidaForge text embeddings are invalid: {cache_path}")
        if embeddings.dtype != torch.bfloat16:
            raise ValueError(f"Resumed VidaForge text embeddings must be BF16 tensors: {cache_path}")
        if not isinstance(metadata, dict):
            raise ValueError(f"Resumed VidaForge metadata is invalid: {cache_path}")
        item_caption_token_length = item.get("caption_token_length")
        payload_caption_token_length = metadata.get("caption_token_length")
        if (isinstance(item_caption_token_length, bool) or not isinstance(item_caption_token_length, int)
                or item_caption_token_length <= 0):
            raise ValueError(f"Resumed VidaForge index caption_token_length is invalid: {where}")
        if (isinstance(payload_caption_token_length, bool) or not isinstance(payload_caption_token_length, int)
                or payload_caption_token_length != item_caption_token_length):
            raise ValueError(f"Resumed VidaForge payload caption_token_length mismatch in {where}")
        if not isinstance(text_mask, torch.Tensor):
            raise ValueError(f"Resumed VidaForge text mask is invalid: {cache_path}")
        _validated_text_mask(
            text_mask,
            sequence_length=int(embeddings.shape[1]),
            caption_token_length=payload_caption_token_length,
            clip_id=str(item["clip_id"]),
            pad_short=False,
        )
        expected_metadata = {
            "clip_id": str(item["clip_id"]),
            "model_name": self.model_name,
            "vae_fingerprint": self.provenance["vae_fingerprint"],
            "text_encoder_fingerprint": self.provenance["text_encoder_fingerprint"],
            "producer_config_fingerprint": self.provenance["producer_config_fingerprint"],
            "source_fingerprint": item["source_fingerprint"],
            "bucket_resolution": item["bucket_resolution"],
            "bucket_frame_count": item["bucket_frame_count"],
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise ValueError(f"Resumed VidaForge payload {field} mismatch in {where}: "
                                 f"payload={metadata.get(field)!r}, index={expected!r}")
        if int(payload.get("bucket_frame_count", 0)) != int(item["bucket_frame_count"]):
            raise ValueError(f"Resumed VidaForge payload frame count mismatch: {cache_path}")


__all__ = [
    "VidaForgeAutoModelWriter",
    "build_model_provenance",
    "build_vidaforge_source_fingerprint",
]
