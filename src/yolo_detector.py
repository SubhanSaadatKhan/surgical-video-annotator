"""YOLO detector: specialized surgical instrument and organ detector.

Uses a YOLOv8 model fine-tuned on CholecSeg8k for accurate detection of
surgical instruments and anatomical structures. This is the primary
detector in the pipeline, providing much higher accuracy than zero-shot
foundation models on surgical footage.

Interface matches GroundingDINODetector so it's a drop-in replacement.

Usage:
    detector = YOLODetector(config)
    detector.load_model()
    detections = detector.detect("frame.jpg")

Training a YOLO model on CholecSeg8k:
    See scripts/train_yolo.py
"""

import os
import numpy as np
from loguru import logger
from typing import Optional
from pathlib import Path

from src.utils import BBox


class YOLODetector:
    """Wrapper for YOLOv8 fine-tuned on surgical data.

    Provides accurate detection of surgical instruments and anatomy
    without the domain gap issues of zero-shot foundation models.
    """

    def __init__(self, config: dict):
        # Look for YOLO-specific config, with fallbacks
        yolo_config = config.get("yolo_detection", {})

        # Default to fine-tuned surgical model path
        self.weights_path = yolo_config.get(
            "weights_path", "yolo_models/surgical_yolov8/best.pt"
        )
        self.confidence_threshold = yolo_config.get("confidence_threshold", 0.25)
        self.iou_threshold = yolo_config.get("iou_threshold", 0.5)
        self.image_size = yolo_config.get("image_size", 640)
        self.device = yolo_config.get(
            "device", config.get("detection", {}).get("device", "cuda")
        )

        # Class name mapping (loaded from model)
        self.class_names = {}
        self.model = None

    def load_model(self) -> None:
        """Load the YOLOv8 model.

        The model must be trained first (see scripts/train_yolo.py) or
        the weights_path must point to a valid YOLOv8 .pt file.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
            raise

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found: {self.weights_path}\n"
                f"Train the model first with: python scripts/train_yolo.py"
            )

        logger.info(f"Loading YOLO model from {self.weights_path}...")
        self.model = YOLO(self.weights_path)

        # Extract class names from model
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        logger.info(
            f"YOLO loaded on {self.device} with {len(self.class_names)} classes: "
            f"{list(self.class_names.values())}"
        )

    def detect(
        self,
        image_path: str,
        custom_prompts: Optional[list] = None,
    ) -> list:
        """Run detection on a single image.

        Args:
            image_path: Path to the image file.
            custom_prompts: Ignored for YOLO (kept for interface compatibility).

        Returns:
            List of BBox objects with detected instruments/organs.
        """
        if self.model is None:
            self.load_model()

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Get bounding boxes in xyxy format
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(xyxy, scores, class_ids):
                x_min, y_min, x_max, y_max = box
                label = self.class_names.get(int(class_id), f"class_{class_id}")

                detections.append(BBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                    confidence=float(score),
                    label=label,
                ))

        logger.debug(f"YOLO detected {len(detections)} objects in {image_path}")
        return detections

    def detect_batch(self, image_paths: list) -> dict:
        """Run detection on multiple images (batched for efficiency).

        YOLO can process a batch of images more efficiently than one at a time.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dict mapping image_path to list of BBox detections.
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            source=image_paths,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        batch_results = {}

        for image_path, result in zip(image_paths, results):
            detections = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                for box, score, class_id in zip(xyxy, scores, class_ids):
                    x_min, y_min, x_max, y_max = box
                    label = self.class_names.get(int(class_id), f"class_{class_id}")

                    detections.append(BBox(
                        x_min=float(x_min),
                        y_min=float(y_min),
                        x_max=float(x_max),
                        y_max=float(y_max),
                        confidence=float(score),
                        label=label,
                    ))

            batch_results[image_path] = detections

        return batch_results


class EnsembleDetector:
    """Combines YOLO and Grounding DINO for robust detection.

    Strategy:
        - Primary: YOLO (fast, accurate on trained classes)
        - Fallback: Grounding DINO (open-vocabulary, catches novel items)
        - Merge results with NMS to remove duplicates

    Use when you want both accuracy on known classes AND coverage of
    unusual items that YOLO wasn't trained on.
    """

    def __init__(self, config: dict):
        from src.detector import GroundingDINODetector

        self.yolo = YOLODetector(config)
        self.dino = GroundingDINODetector(config)

        ensemble_config = config.get("ensemble", {})
        self.use_dino_fallback = ensemble_config.get("use_dino_fallback", True)
        self.dino_only_if_yolo_empty = ensemble_config.get(
            "dino_only_if_yolo_empty", False
        )
        self.merge_iou_threshold = ensemble_config.get("merge_iou_threshold", 0.5)

    def load_model(self) -> None:
        """Load both models."""
        self.yolo.load_model()
        if self.use_dino_fallback:
            self.dino.load_model()

    def detect(
        self,
        image_path: str,
        custom_prompts: Optional[list] = None,
    ) -> list:
        """Run both detectors and merge results.

        Returns:
            List of BBox objects from ensemble (deduplicated).
        """
        yolo_dets = self.yolo.detect(image_path)

        # Skip DINO if configured or if YOLO found things and dino_only_if_yolo_empty
        if not self.use_dino_fallback:
            return yolo_dets

        if self.dino_only_if_yolo_empty and yolo_dets:
            return yolo_dets

        dino_dets = self.dino.detect(image_path, custom_prompts)

        # Merge with NMS: prefer YOLO detections when they overlap DINO ones
        merged = self._merge_with_nms(yolo_dets, dino_dets)

        logger.debug(
            f"Ensemble: YOLO {len(yolo_dets)}, DINO {len(dino_dets)}, "
            f"merged {len(merged)}"
        )
        return merged

    def _merge_with_nms(self, primary: list, secondary: list) -> list:
        """Merge two sets of detections, preferring primary on overlap.

        Args:
            primary: Higher-priority detections (YOLO).
            secondary: Lower-priority detections (DINO).

        Returns:
            Merged list without duplicates.
        """
        merged = list(primary)

        for sec_det in secondary:
            # Check if secondary detection overlaps significantly with any primary
            is_duplicate = False
            for prim_det in primary:
                iou = self._compute_iou(prim_det, sec_det)
                if iou > self.merge_iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(sec_det)

        return merged

    @staticmethod
    def _compute_iou(box_a: BBox, box_b: BBox) -> float:
        """Compute IoU between two bounding boxes."""
        x_min = max(box_a.x_min, box_b.x_min)
        y_min = max(box_a.y_min, box_b.y_min)
        x_max = min(box_a.x_max, box_b.x_max)
        y_max = min(box_a.y_max, box_b.y_max)

        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        if intersection == 0:
            return 0.0

        union = box_a.area + box_b.area - intersection
        return intersection / union if union > 0 else 0.0
