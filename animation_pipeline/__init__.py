"""动画帧目录服务与真实显示帧数据集生产管线。"""

from .catalog import (
    CatalogValidationError,
    FrameCatalog,
    RawFrame,
)
from .compositor import (
    CheckpointCache,
    DisplayFrame,
    FramePlayer,
    effective_delay_centis,
    normalize_time,
    produce_dataset,
    reference_replay,
)
from .gif_io import GifFile, GifImage, frame_spec, parse_gif, write_gif

__all__ = [
    "CatalogValidationError",
    "FrameCatalog",
    "RawFrame",
    "CheckpointCache",
    "DisplayFrame",
    "FramePlayer",
    "effective_delay_centis",
    "normalize_time",
    "produce_dataset",
    "reference_replay",
    "GifFile",
    "GifImage",
    "frame_spec",
    "parse_gif",
    "write_gif",
]
