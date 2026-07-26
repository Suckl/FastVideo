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
standardized, segmented, selected, annotated, and training-ready clips.
FastVideo supports two integration points:

- Stage 4 Caption/Tag Parquet or the public
  [VidaForge-3M dataset](https://huggingface.co/datasets/VidaForge/VidaForge-3M)
  as preprocessing input.
- Stage 5 AutoModel Wan `.meta` caches as direct modular-training input.

Use Stage 4 when FastVideo should run the model encoders itself. Use Stage 5
when VidaForge already produced the Wan VAE latents and UMT5 embeddings.

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
ranks only after eligibility filtering and global uniqueness checks, then
across DataLoader workers before video bytes are read. In distributed runs,
rank zero prepares a shared eligible-row indices cache under
`<dataset_output_dir>/.vidaforge_metadata_cache`; other ranks reuse it before
sharding. Stage 4 inputs use the same lazy media path: MP4 files are not probed
until a DataLoader worker consumes the row.

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

FastVideo treats `clip_id` as data, not as an output path. Every ID is
deterministically encoded as `vidaforge-<sha256>` before processed artifacts
are written. Using one mapping for all IDs avoids extension, case-folding, and
reserved-name collisions across platforms. Empty IDs and IDs with leading or
trailing whitespace are rejected instead of normalized, so uniqueness checks
and output hashing use identical input. Output savers also verify that every
resolved path remains under its configured output directory.

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

#### Stage 5 AutoModel training cache

VidaForge Stage 5 AutoModel writes `metadata.json`, JSON metadata shards, and
one `.meta` tensor file per clip. Point the modular Wan trainer directly at
the Stage 5 output directory:

FastVideo can produce that cache itself from the Stage 4 or public-release
input described above. This runs FastVideo's Wan VAE and UMT5 encoder, applies
the Wan latent mean/std normalization exactly once, and writes a portable
content-addressed cache:

```bash
torchrun --nproc_per_node=1 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model-path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --revision "<immutable-Hugging-Face-commit>" \
    --mode preprocess \
    --workload-type t2v \
    --vae-precision fp16 \
    --text-encoder-precisions bf16 \
    --preprocess.dataset-type vidaforge \
    --preprocess.dataset-path "/data/VidaForge-3M/meta/shard-00000.parquet" \
    --preprocess.video-loader-type torchcodec \
    --preprocess.vidaforge-data-root "/data/VidaForge-3M" \
    --preprocess.vidaforge-caption-field caption_level_3 \
    --preprocess.vidaforge-selection auto \
    --preprocess.output-type vidaforge_automodel \
    --preprocess.vidaforge-model-name \
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --preprocess.dataset-output-dir "/data/vidaforge-stage5" \
    --preprocess.max-height 144 \
    --preprocess.max-width 256 \
    --preprocess.num-frames 17 \
    --preprocess.train-fps 16 \
    --preprocess.samples-per-file 256
```

This producer currently supports Wan text-to-video caches. The frame count
must be `4n+1`, both spatial dimensions must be divisible by 16,
`training_cfg_rate` must be zero, and temporal random sampling must be
disabled. It uses VidaForge's full-clip linspace sampling, prompt cleanup,
CUDA beta exact-seek TorchCodec decoding, center crop, and fp16 VAE/bf16 UMT5
precisions. Torchvision is not accepted for this parity output. CFG dropout
remains a per-epoch training decision in the Section 2 loader rather than being
permanently baked into cached text embeddings. As in VidaForge, a clip may be
at most three decoded frames short; linspace repeats boundary-near samples to
fill the `4n+1` bucket. Shorter inputs are rejected.

The output contains `provenance.json`, `metadata.json`, metadata shards, and
one atomic `.meta` file per clip. `provenance.json` records the resolved model
revision plus the path and SHA-256 of every VAE/text-encoder weight and
configuration file. Copy its `vae_fingerprint` and
`text_encoder_fingerprint` values into the training configuration below.
Tokenizer files, including `spiece.model`, are included in the text-encoder
identity because they also affect the resulting embeddings. The same file
records a `producer_config_fingerprint` over output-affecting settings such as
resolution, frame count, FPS, caption field, selection, manifest content,
component precision, and tokenizer sequence length. Each item also records a
fingerprint over its cleaned caption, media bytes, and decoded source metadata.

Interrupted runs do not publish partial `.meta` files. To continue a
previously published compatible cache, add
`--preprocess.vidaforge-resume`; FastVideo verifies the model provenance,
manifest/selection identity, and per-item source fingerprint before retaining a
completed clip. Changed inputs fail instead of silently reusing stale tensors,
and the new metadata generation contains only clips seen in the current
manifest. Distributed ranks write independent progress files and rank zero
publishes the merged index after all ranks finish.

```yaml
training:
  data:
    data_path: /data/vidaforge-stage5
    preprocessed_data_type: vidaforge_automodel
    vidaforge_model_name: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    vidaforge_vae_fingerprint: <64-character-sha256>
    vidaforge_text_encoder_fingerprint: <64-character-sha256>
    train_batch_size: 1
    dataloader_num_workers: 4
    training_cfg_rate: 0.0
    seed: 42
    num_height: 144
    num_width: 256
    num_latent_t: 5
```

The loader keeps each batch within one VidaForge temporal/resolution/latent
bucket and shards global bucket batches across data-parallel groups. All ranks
inside one FastVideo sequence-parallel group receive the same sample indices.
It lazily reads `.meta` files with `torch.load(weights_only=True)` and preserves
their floating-point dtype until the Wan model casts the batch to its training
dtype.

VidaForge's Wan Stage 5 encoder has already applied the VAE latent mean/std
normalization. FastVideo marks this data type explicitly and does not normalize
those latents a second time. Regular FastVideo `t2v` Parquet inputs retain the
existing runtime normalization.

The Stage 5 cache must use `model_type: wan`. FastVideo compares `model_name`
exactly with `vidaforge_model_name`, or with `training.model_path` when the
explicit name is omitted. Set `vidaforge_model_name` when training from a local
checkpoint path. FastVideo does not infer model identity from a local directory
basename or a Hugging Face snapshot path.

Verified caches must record both `vae_fingerprint` and
`text_encoder_fingerprint` in each payload's `metadata`. Each value is the
lowercase SHA-256 of a canonical component manifest that includes the immutable
model revision, component configuration, and the path and SHA-256 of every
weight file. Configure the two expected fingerprints above; FastVideo compares
them exactly before using the already-encoded tensors. FastVideo's Stage 5
producer creates these manifests and writes the resulting fingerprints
alongside every cache payload.

VidaForge revision `4562d3f` does not yet write component fingerprints. To use
one of those legacy caches, opt into name-and-shape validation explicitly:

```yaml
training:
  data:
    preprocessed_data_type: vidaforge_automodel
    vidaforge_model_name: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    vidaforge_allow_unverified_model: true
```

This compatibility switch cannot prove which VAE or text encoder produced the
cache. Keep it disabled for newly produced data. FastVideo derives the
attention mask from `text_mask` when present, or from `caption_token_length`
for the standard VidaForge Wan payload. Moving a complete Stage 5 output tree
is supported even though VidaForge records absolute `cache_file` paths;
FastVideo rebases missing paths from the bucket directory suffix. Metadata-only
split directories may continue to reference an existing absolute cache tree.

With `drop_last: true` training semantics, every bucket needs at least
`train_batch_size * data_parallel_group_count` samples. The loader fails early
with bucket sizes when no complete distributed batch can be formed.

The opt-in GPU smoke test runs the pinned VidaForge repository revision and Wan
model ID on one real
HEVC clip from VidaForge-3M, writes the official Stage 5 payload, and checks
that FastVideo loads its tensors bit-exactly. Because that VidaForge revision
does not emit component fingerprints, this test exercises the explicit legacy
compatibility path; it is not a cryptographic model-provenance test:

```bash
git clone https://github.com/GAIR-NLP/VidaForge /tmp/VidaForge
git -C /tmp/VidaForge checkout 4562d3fbcbd4861fc74c2859950c0237363681bb
VIDAFORGE_RUN_OFFICIAL_AUTOMODEL_SMOKE=1 \
VIDAFORGE_REFERENCE_DIR=/tmp/VidaForge \
pytest fastvideo/tests/workflow/test_vidaforge_automodel_official.py -vs
```

This smoke test downloads the Wan 2.1 1.3B VAE, UMT5 encoder, and one ranged
release clip, so run it on a CUDA machine with sufficient VRAM or on Modal.

The native-producer parity gate runs the same official HEVC clip through the
FastVideo producer, reloads the published cache through the modular training
loader, and compares its latent, text embedding, and attention mask with the
pinned VidaForge `WanAutoModelEncoder`:

```bash
VIDAFORGE_RUN_FASTVIDEO_PRODUCER_SMOKE=1 \
VIDAFORGE_REFERENCE_DIR=/tmp/VidaForge \
pytest \
    fastvideo/tests/workflow/test_vidaforge_fastvideo_producer_official.py \
    -vs
```

Run this gate after any change to the native producer, its Wan preprocessing
stages, or component precision. The earlier
`test_vidaforge_automodel_official.py` gate proves that FastVideo can read an
officially produced payload; it does not exercise FastVideo's producer.

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

The default `--preprocess.output-type parquet` writes Parquet files containing:

- `vae_latent_bytes` — VAE-encoded video latent
- `text_embedding_bytes` — text encoder output
- `clip_feature_bytes` — CLIP image features (I2V only)
- `first_frame_latent_bytes` — first frame latent (I2V only)
- Metadata: shapes, dtypes, and sample identifiers

`--preprocess.output-type vidaforge_automodel` instead writes the Stage 5
layout documented above directly under `dataset_output_dir`.

## Examples

See ready-to-run preprocessing scripts in the training examples:

- **T2V**: `examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.sh`
- **I2V**: `examples/training/finetune/wan_i2v_14B_480p/crush_smol/preprocess_wan_data_i2v_new.sh`

**→ [Browse all training examples](examples/examples_training_index.md)**
