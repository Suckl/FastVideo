# 🧱 Data Preprocessing

To save GPU memory during training, FastVideo precomputes text embeddings and VAE latents. This eliminates the need to load the text encoder and VAE during training.

## Quick Start

Download the sample dataset and run preprocessing:

```bash
# Download the crush-smol dataset
python scripts/huggingface/download_hf.py \
    --repo_id "wlsaidhi/crush-smol-merged" \
    --local_dir "data/crush-smol" \
    --repo_type "dataset"

# Run preprocessing
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.sh
```

## Preprocessing Pipeline

The new preprocessing pipeline supports multiple dataset formats and video loaders:

```bash
GPU_NUM=2
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATASET_PATH="data/crush-smol/"
OUTPUT_DIR="data/crush-smol_processed_t2v/"

torchrun --nproc_per_node=$GPU_NUM \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 2 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 5
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--workload_type` | Task type: `t2v` (text-to-video) or `i2v` (image-to-video) |
| `--preprocess.dataset_type` | Input format: `hf`, `merged`, or `vidaforge` |
| `--preprocess.dataset_path` | Path to dataset (HF repo ID or local folder) |
| `--preprocess.dataset_output_dir` | Output directory for Parquet files |
| `--preprocess.video_loader_type` | Video decoder: `torchcodec` or `torchvision` |
| `--preprocess.max_height` / `max_width` | Target resolution for videos |
| `--preprocess.num_frames` | Number of frames to extract per video |
| `--preprocess.train_fps` | Target FPS for frame extraction |

## Dataset Formats

### Merged Dataset (Local Folder)

Structure your dataset as follows:

```
your_dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── videos2caption.json
```

The `videos2caption.json` maps video filenames to captions:

```json
[
  {"path": "video_001.mp4", "cap": "A cat playing with yarn..."},
  {"path": "video_002.mp4", "cap": "Ocean waves at sunset..."}
]
```

### HuggingFace Dataset

Use `--preprocess.dataset_type hf` and point `--preprocess.dataset_path` to a HuggingFace dataset with `video` and `caption` columns.

### VidaForge Manifest

[VidaForge](https://github.com/GAIR-NLP/VidaForge) turns raw videos into
standardized, segmented, selected, and annotated clips. FastVideo can consume
either the local Parquet shards produced by VidaForge Stage 4 Caption or Tag,
or locally downloaded paired Parquet/indexed-TAR shards from the public
[VidaForge-3M dataset](https://huggingface.co/datasets/VidaForge/VidaForge-3M).
It does not use VidaForge Stage 5 encoded `.meta` files.

#### Stage 4 workspace

Point `dataset_path` at one Stage 4 Parquet file or at the run directory that
directly contains `clip-*.parquet`. VidaForge stores `clip_path` relative to
its `DATA_DIR`, so pass that directory as `vidaforge_data_root`:

```bash
torchrun --nproc_per_node=1 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model-path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --mode preprocess \
    --workload-type t2v \
    --preprocess.video-loader-type torchcodec \
    --preprocess.dataset-type vidaforge \
    --preprocess.dataset-path \
        "/data/meta/stage4_annotation/step2_caption/run_id_demo" \
    --preprocess.vidaforge-data-root "/data" \
    --preprocess.vidaforge-caption-field caption_level_3 \
    --preprocess.vidaforge-selection auto \
    --preprocess.dataset-output-dir "/data/fastvideo_processed" \
    --preprocess.max-height 480 \
    --preprocess.max-width 832 \
    --preprocess.num-frames 77 \
    --preprocess.train-fps 16
```

Stage 4 input requires:

- `clip_id`, `clip_path`, `clip_ok`
- `width`, `height`, `fps`, `duration_sec`
- `select_ok`, `select_pass`, `caption_ok`
- the column selected by `vidaforge_caption_field`

#### Public VidaForge-3M release

Download one paired shard for a smoke run:

```bash
hf download VidaForge/VidaForge-3M \
    data/shard-00000.tar \
    meta/shard-00000.parquet \
    --repo-type dataset \
    --local-dir "/data/VidaForge-3M"
```

Then preprocess that metadata shard:

```bash
torchrun --nproc_per_node=1 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model-path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --mode preprocess \
    --workload-type t2v \
    --preprocess.video-loader-type torchcodec \
    --preprocess.dataset-type vidaforge \
    --preprocess.dataset-path \
        "/data/VidaForge-3M/meta/shard-00000.parquet" \
    --preprocess.vidaforge-data-root "/data/VidaForge-3M" \
    --preprocess.vidaforge-caption-field caption_level_3 \
    --preprocess.vidaforge-selection auto \
    --preprocess.dataset-output-dir "/data/fastvideo_processed" \
    --preprocess.max-height 480 \
    --preprocess.max-width 832 \
    --preprocess.num-frames 77 \
    --preprocess.train-fps 16
```

Public release metadata includes `tar_path`, `tar_offset`, `filesize_bytes`,
and `sha256`. If `<vidaforge_data_root>/data/<clip_path>` already exists,
FastVideo uses and verifies that extracted clip. Otherwise the default
TorchCodec path lazily reads only the requested clip's byte range from the
paired uncompressed TAR, verifies its SHA-256, and decodes the verified bytes
without creating a persistent copy. The input is sharded across distributed
ranks before metadata filtering and integrity checks, then across DataLoader
workers before video bytes are read. Stage 4 inputs use the same lazy media
path: MP4 files are not probed until a DataLoader worker consumes the row.

Set `vidaforge_materialize_dir` to opt into a persistent, content-addressed
clip cache. The Torchvision loader requires files and therefore materializes
clips automatically; when no cache path is configured, it uses
`.vidaforge_clips` under `dataset_output_dir`. Cached files are written
atomically. A persistent cache can grow to roughly the size of the downloaded
TAR data, so place it on storage with enough space or omit it when using
TorchCodec.

LTX-2 audio preprocessing requires a file-backed video source. When using
`with_audio=True` with the public release and TorchCodec, set
`vidaforge_materialize_dir` (or pre-extract clips under
`<vidaforge_data_root>/data/<clip_path>`). The default in-memory byte source
supports video decoding but cannot be passed to Torchaudio for audio
extraction.

`vidaforge_selection` accepts `auto` (the default), `pass`, `reject`, or `all`.
For Stage 4, `auto` means `pass`. For the public release, whose selection
decisions are intentionally omitted, `auto` means `all`; `pass` and `reject`
are rejected rather than silently applying the wrong selection policy.
Stage 4 rows with `clip_ok != 1`, `caption_ok != 1`, `select_ok != 1`, or an
empty selected caption are excluded. Checking `select_ok` prevents selection
worker failures from being treated as ordinary rejected clips.

FastVideo probes each selected MP4 for its container-indexed width, height,
FPS, and frame count, decoding to count frames when the container omits that
count. It does not derive frame count from `fps * duration_sec`, because
VidaForge clip timing and container duration can differ by one frame.
The recommended `torchcodec` loader requires compatible FFmpeg shared
libraries; follow the
[TorchCodec installation guide](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)
if its native library cannot be loaded.

The contract is based on VidaForge revision
[`4562d3f`](https://github.com/GAIR-NLP/VidaForge/tree/4562d3fbcbd4861fc74c2859950c0237363681bb).
The public release contains about 2.2 TiB of HEVC video, so install an FFmpeg
build with HEVC decoding and start with one paired shard. A full persistent
run needs space for the downloaded TARs, the optional materialized cache, and
FastVideo's processed output; the default in-memory TorchCodec path avoids the
extra clip cache.

To verify the exact decoder path without downloading a full TAR shard, run
the opt-in smoke test on Linux. It fetches one pinned official metadata row
and only that HEVC clip's byte range. This is a component-level integration
test through `VideoTransformStage`; it does not run model encoders or write
the final processed dataset:

```bash
VIDAFORGE_RUN_OFFICIAL_HEVC_SMOKE=1 \
pytest fastvideo/tests/workflow/test_vidaforge_manifest.py \
    -k official_hevc_torchcodec_pipeline_smoke -vs
```

## Creating Your Own Dataset

If you have raw videos and captions in separate files, generate the `videos2caption.json`:

```bash
python scripts/dataset_preparation/prepare_json_file.py \
    --data_folder path/to/your_raw_data/ \
    --output path/to/output_folder
```

Your raw data folder should contain:

```
your_raw_data/
├── videos/
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── videos.txt    # list of video filenames
└── prompt.txt    # corresponding captions (one per line)
```

## Output Format

Preprocessing outputs Parquet files in the `combined_parquet_dataset/` subdirectory containing:

- `vae_latent_bytes` — VAE-encoded video latent
- `text_embedding_bytes` — text encoder output
- `clip_feature_bytes` — CLIP image features (I2V only)
- `first_frame_latent_bytes` — first frame latent (I2V only)
- Metadata: shapes, dtypes, and sample identifiers

## Examples

See ready-to-run preprocessing scripts in the training examples:

- **T2V**: `examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.sh`
- **I2V**: `examples/training/finetune/wan_i2v_14B_480p/crush_smol/preprocess_wan_data_i2v_new.sh`

**→ [Browse all training examples](examples/examples_training_index.md)**
