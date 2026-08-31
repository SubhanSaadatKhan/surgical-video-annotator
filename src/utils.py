"""Shared utility functions for the surgical video annotator."""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class BBox:
    """Bounding box with normalized and pixel coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    label: str
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_coco(self) -> list:
        """Convert to COCO format [x, y, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]

    def to_xyxy(self) -> list:
        """Convert to [x_min, y_min, x_max, y_max] format."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class FrameAnnotation:
    """All annotations for a single video frame."""
    frame_idx: int
    timestamp_sec: float
    image_path: str
    bboxes: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    quality_flags: list = field(default_factory=list)
    status: str = "pending"  # "pending", "approved", "rejected", "corrected"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Validate configuration values."""
    required_sections = ["video", "detection", "segmentation", "export"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if config["video"]["frame_extraction_fps"] <= 0:
        raise ValueError("frame_extraction_fps must be positive")

    if not 0 < config["detection"]["box_threshold"] < 1:
        raise ValueError("box_threshold must be between 0 and 1")

    if not 0 < config["detection"]["text_threshold"] < 1:
        raise ValueError("text_threshold must be between 0 and 1")

    logger.info("Configuration validated successfully")


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    if intersection == 0:
        return 0.0

    area_a = box_a.area
    area_b = box_b.area
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def setup_logger(log_dir: str = "logs") -> None:
    """Configure logging with file and console output."""
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "pipeline_{time}.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
    )
