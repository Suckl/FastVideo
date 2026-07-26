import dataclasses
from enum import Enum
from typing import Any, Optional

from fastvideo.configs.utils import update_config_from_args
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser, StoreBoolean

logger = init_logger(__name__)


class DatasetType(str, Enum):
    """
    Enumeration for different dataset types.
    """
    HF = "hf"
    MERGED = "merged"
    VIDAFORGE = "vidaforge"

    @classmethod
    def from_string(cls, value: str) -> "DatasetType":
        """Convert string to DatasetType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid dataset type: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [dataset_type.value for dataset_type in cls]


class VideoLoaderType(str, Enum):
    """
    Enumeration for different video loaders.
    """
    TORCHCODEC = "torchcodec"
    TORCHVISION = "torchvision"

    @classmethod
    def from_string(cls, value: str) -> "VideoLoaderType":
        """Convert string to VideoLoader enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid video loader: {value}. Must be one of: {', '.join([m.value for m in cls])}") from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [video_loader.value for video_loader in cls]


class PreprocessOutputType(str, Enum):
    """Output contracts supported by the preprocessing workflow."""

    PARQUET = "parquet"
    VIDAFORGE_AUTOMODEL = "vidaforge_automodel"

    @classmethod
    def from_string(cls, value: str) -> "PreprocessOutputType":
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid preprocess output type: {value}. "
                             f"Must be one of: {', '.join(member.value for member in cls)}") from None

    @classmethod
    def choices(cls) -> list[str]:
        return [output_type.value for output_type in cls]


@dataclasses.dataclass
class PreprocessConfig:
    """Configuration for preprocessing operations."""

    # Model and dataset configuration
    model_path: str = ""
    dataset_path: str = ""
    dataset_type: DatasetType = DatasetType.HF
    dataset_output_dir: str = "./output"
    vidaforge_data_root: str = ""
    vidaforge_materialize_dir: str = ""
    vidaforge_caption_field: str = "caption_level_3"
    vidaforge_selection: str = "auto"
    output_type: PreprocessOutputType = PreprocessOutputType.PARQUET
    vidaforge_model_name: str = ""
    vidaforge_resume: bool = False

    # Dataloader configuration
    dataloader_num_workers: int = 1
    preprocess_video_batch_size: int = 2

    # Saver configuration
    samples_per_file: int = 64
    flush_frequency: int = 256

    # Video processing parameters
    video_loader_type: VideoLoaderType = VideoLoaderType.TORCHCODEC
    max_height: int = 480
    max_width: int = 848
    num_frames: int = 163
    video_length_tolerance_range: float = 2.0
    train_fps: int = 30
    speed_factor: float = 1.0
    drop_short_ratio: float = 1.0
    do_temporal_sample: bool = False

    # Model configuration
    training_cfg_rate: float = 0.0
    with_audio: bool = False

    # framework configuration
    seed: int = 42

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser, prefix: str = "preprocess") -> FlexibleArgumentParser:
        """Add preprocessing configuration arguments to the parser."""
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""

        preprocess_args = parser.add_argument_group("Preprocessing Arguments")
        # Model & Dataset
        preprocess_args.add_argument(f"--{prefix_with_dot}model-path",
                                     type=str,
                                     default=PreprocessConfig.model_path,
                                     help="Path to the model for preprocessing")
        preprocess_args.add_argument(f"--{prefix_with_dot}dataset-path",
                                     type=str,
                                     default=PreprocessConfig.dataset_path,
                                     help="Path to the dataset directory for preprocessing")
        preprocess_args.add_argument(f"--{prefix_with_dot}dataset-type",
                                     type=str,
                                     choices=DatasetType.choices(),
                                     default=PreprocessConfig.dataset_type.value,
                                     help="Type of the dataset")
        preprocess_args.add_argument(f"--{prefix_with_dot}dataset-output-dir",
                                     type=str,
                                     default=PreprocessConfig.dataset_output_dir,
                                     help="The output directory where the dataset will be written.")
        preprocess_args.add_argument(
            f"--{prefix_with_dot}vidaforge-data-root",
            type=str,
            default=PreprocessConfig.vidaforge_data_root,
            help="VidaForge DATA_DIR or downloaded VidaForge-3M root used to resolve videos and TAR shards.")
        preprocess_args.add_argument(
            f"--{prefix_with_dot}vidaforge-materialize-dir",
            type=str,
            default=PreprocessConfig.vidaforge_materialize_dir,
            help="Optional persistent cache for clips read from VidaForge-3M indexed TAR shards.")
        preprocess_args.add_argument(f"--{prefix_with_dot}vidaforge-caption-field",
                                     type=str,
                                     default=PreprocessConfig.vidaforge_caption_field,
                                     help="VidaForge caption column to use as the training caption.")
        preprocess_args.add_argument(f"--{prefix_with_dot}vidaforge-selection",
                                     type=str,
                                     choices=["auto", "pass", "reject", "all"],
                                     default=PreprocessConfig.vidaforge_selection,
                                     help="VidaForge selection partition to preprocess.")
        preprocess_args.add_argument(f"--{prefix_with_dot}output-type",
                                     type=str,
                                     choices=PreprocessOutputType.choices(),
                                     default=PreprocessConfig.output_type.value,
                                     help="Processed dataset contract: Parquet or a Wan VidaForge AutoModel cache.")
        preprocess_args.add_argument(f"--{prefix_with_dot}vidaforge-model-name",
                                     type=str,
                                     default=PreprocessConfig.vidaforge_model_name,
                                     help="Canonical checkpoint name recorded in generated VidaForge AutoModel caches.")
        preprocess_args.add_argument(f"--{prefix_with_dot}vidaforge-resume",
                                     action=StoreBoolean,
                                     default=PreprocessConfig.vidaforge_resume,
                                     help="Resume a compatible VidaForge AutoModel cache and skip completed clip IDs.")

        # Dataloader
        preprocess_args.add_argument(
            f"--{prefix_with_dot}dataloader-num-workers",
            type=int,
            default=PreprocessConfig.dataloader_num_workers,
            help=
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
        preprocess_args.add_argument(f"--{prefix_with_dot}preprocess-video-batch-size",
                                     type=int,
                                     default=PreprocessConfig.preprocess_video_batch_size,
                                     help="Batch size (per device) for the training dataloader.")

        # Saver
        preprocess_args.add_argument(f"--{prefix_with_dot}samples-per-file",
                                     type=int,
                                     default=PreprocessConfig.samples_per_file,
                                     help="Number of samples per output file")
        preprocess_args.add_argument(f"--{prefix_with_dot}flush-frequency",
                                     type=int,
                                     default=PreprocessConfig.flush_frequency,
                                     help="How often to save to parquet files")

        # Video processing parameters
        preprocess_args.add_argument(f"--{prefix_with_dot}video-loader-type",
                                     type=str,
                                     choices=VideoLoaderType.choices(),
                                     default=PreprocessConfig.video_loader_type.value,
                                     help="Type of the video loader")
        preprocess_args.add_argument(f"--{prefix_with_dot}max-height",
                                     type=int,
                                     default=PreprocessConfig.max_height,
                                     help="Maximum height for video processing")
        preprocess_args.add_argument(f"--{prefix_with_dot}max-width",
                                     type=int,
                                     default=PreprocessConfig.max_width,
                                     help="Maximum width for video processing")
        preprocess_args.add_argument(f"--{prefix_with_dot}num-frames",
                                     type=int,
                                     default=PreprocessConfig.num_frames,
                                     help="Number of frames to process")
        preprocess_args.add_argument(f"--{prefix_with_dot}video-length-tolerance-range",
                                     type=float,
                                     default=PreprocessConfig.video_length_tolerance_range,
                                     help="Video length tolerance range")
        preprocess_args.add_argument(f"--{prefix_with_dot}train-fps",
                                     type=int,
                                     default=PreprocessConfig.train_fps,
                                     help="Training FPS")
        preprocess_args.add_argument(f"--{prefix_with_dot}speed-factor",
                                     type=float,
                                     default=PreprocessConfig.speed_factor,
                                     help="Speed factor for video processing")
        preprocess_args.add_argument(f"--{prefix_with_dot}drop-short-ratio",
                                     type=float,
                                     default=PreprocessConfig.drop_short_ratio,
                                     help="Ratio for dropping short videos")
        preprocess_args.add_argument(f"--{prefix_with_dot}do-temporal-sample",
                                     action=StoreBoolean,
                                     default=PreprocessConfig.do_temporal_sample,
                                     help="Whether to do temporal sampling")

        # Model Training configuration
        preprocess_args.add_argument(f"--{prefix_with_dot}training-cfg-rate",
                                     type=float,
                                     default=PreprocessConfig.training_cfg_rate,
                                     help="Training CFG rate")
        preprocess_args.add_argument(f"--{prefix_with_dot}with-audio",
                                     action=StoreBoolean,
                                     default=PreprocessConfig.with_audio,
                                     help="Whether to extract and encode audio")
        preprocess_args.add_argument(f"--{prefix_with_dot}seed",
                                     type=int,
                                     default=PreprocessConfig.seed,
                                     help="Seed for random number generator")

        return parser

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> Optional["PreprocessConfig"]:
        """Create PreprocessConfig from keyword arguments."""
        if 'dataset_type' in kwargs and isinstance(kwargs['dataset_type'], str):
            kwargs['dataset_type'] = DatasetType.from_string(kwargs['dataset_type'])
        if 'video_loader_type' in kwargs and isinstance(kwargs['video_loader_type'], str):
            kwargs['video_loader_type'] = VideoLoaderType.from_string(kwargs['video_loader_type'])
        if 'output_type' in kwargs and isinstance(kwargs['output_type'], str):
            kwargs['output_type'] = PreprocessOutputType.from_string(kwargs['output_type'])

        preprocess_config = cls()
        if not update_config_from_args(preprocess_config, kwargs, prefix="preprocess", pop_args=True):
            return None
        if isinstance(preprocess_config.dataset_type, str):
            preprocess_config.dataset_type = DatasetType.from_string(preprocess_config.dataset_type)
        if isinstance(preprocess_config.video_loader_type, str):
            preprocess_config.video_loader_type = VideoLoaderType.from_string(preprocess_config.video_loader_type)
        if isinstance(preprocess_config.output_type, str):
            preprocess_config.output_type = PreprocessOutputType.from_string(preprocess_config.output_type)
        return preprocess_config

    def check_preprocess_config(self) -> None:
        if self.dataset_path == "":
            raise ValueError("dataset_path must be set for preprocess mode")
        if self.dataset_type == DatasetType.VIDAFORGE:
            if not self.vidaforge_caption_field.strip():
                raise ValueError("vidaforge_caption_field must not be empty")
            if self.vidaforge_selection not in {"auto", "pass", "reject", "all"}:
                raise ValueError("vidaforge_selection must be one of: auto, pass, reject, all")
        if self.output_type == PreprocessOutputType.VIDAFORGE_AUTOMODEL:
            if self.dataset_type != DatasetType.VIDAFORGE:
                raise ValueError("vidaforge_automodel output currently requires dataset_type=vidaforge")
            if not self.vidaforge_model_name.strip():
                raise ValueError("vidaforge_model_name is required for vidaforge_automodel output")
            if self.training_cfg_rate != 0:
                raise ValueError("vidaforge_automodel output requires training_cfg_rate=0; "
                                 "apply CFG dropout in the training dataloader")
            if self.do_temporal_sample:
                raise ValueError("vidaforge_automodel output requires do_temporal_sample=false "
                                 "so resumed samples are deterministic")
            if self.drop_short_ratio != 1.0:
                raise ValueError("vidaforge_automodel output requires drop_short_ratio=1 "
                                 "so short clips cannot create undersized Wan buckets")
            if self.num_frames <= 0 or (self.num_frames - 1) % 4 != 0:
                raise ValueError("Wan vidaforge_automodel output requires num_frames=4n+1")
            if self.max_height <= 0 or self.max_width <= 0 or self.max_height % 16 or self.max_width % 16:
                raise ValueError("Wan vidaforge_automodel output requires max_height/max_width divisible by 16")
        if self.samples_per_file <= 0:
            raise ValueError("samples_per_file must be greater than 0")
        if self.flush_frequency <= 0:
            raise ValueError("flush_frequency must be greater than 0")
