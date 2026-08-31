"""Grounding DINO detector: detects surgical instruments from text prompts.

Grounding DINO is an open-set object detector that combines a text encoder
(BERT) with a visual encoder (Swin Transformer) to detect objects based on
natural language descriptions. This means we can detect "scalpel" or "forceps"
without training on those specific classes.

How it works:
    1. Text prompts ("scalpel. forceps. scissors.") are encoded by BERT
    2. Video frame is encoded by Swin Transformer
    3. Cross-attention fuses text and visual features
    4. Model outputs bounding boxes with confidence scores per prompt

This is the key innovation: no training needed for new instrument types.
Just add the name to the config.
"""

import numpy as np
from loguru import logger
from typing import Optional

from src.utils import BBox


class GroundingDINODetector:
    """Wrapper around Grounding DINO for surgical instrument detection."""

    def __init__(self, config: dict):
        self.model_id = config["detection"]["model_id"]
        self.prompts = config["detection"]["prompts"]
        self.box_threshold = config["detection"]["box_threshold"]
        self.text_threshold = config["detection"]["text_threshold"]
        self.nms_threshold = config["detection"]["nms_threshold"]
        self.device = config["detection"]["device"]

        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load the Grounding DINO model and processor.

        Uses HuggingFace transformers for easy model management.
        The model is ~700MB and will be cached after first download.
        """
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        logger.info(f"Loading Grounding DINO from {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

        logger.info(f"Grounding DINO loaded on {self.device}")

    def detect(
        self,
        image_path: str,
        custom_prompts: Optional[list] = None,
    ) -> list:
        """Run detection on a single frame.

        Grounding DINO expects text prompts separated by periods.
        For example: "scalpel . forceps . scissors ."

        Args:
            image_path: Path to the frame image.
            custom_prompts: Override default prompts for this frame.

        Returns:
            List of BBox objects with detected instruments.
        """
        import torch
        from PIL import Image

        if self.model is None:
            self.load_model()

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        prompts = custom_prompts or self.prompts
        # Grounding DINO expects a single string with labels separated by "."
        text_prompt = " . ".join(prompts) + " ."

        # Preprocess
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: convert outputs to bounding boxes
        # Handle both old and new transformers API (parameter was renamed)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            # Fallback for older transformers versions
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        # Handle labels: newer transformers returns integer indices in "labels"
        # and text strings in "text_labels". Older versions return strings in "labels".
        if "text_labels" in results:
            labels = results["text_labels"]
        else:
            labels = results["labels"]

        # Convert to strings if needed (handle tensors, ints, or strings)
        label_strings = []
        for label in labels:
            if hasattr(label, "item"):
                label = label.item()
            label_strings.append(str(label))

        for box, score, label in zip(boxes, scores, label_strings):
            x_min, y_min, x_max, y_max = box

            detections.append(BBox(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
                confidence=float(score),
                label=label.strip(),
            ))

        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = self._apply_nms(detections)

        logger.debug(
            f"Detected {len(detections)} instruments in {image_path}"
        )
        return detections

    def detect_batch(self, image_paths: list) -> dict:
        """Run detection on multiple frames.

        Args:
            image_paths: List of frame image paths.

        Returns:
            Dict mapping image_path to list of BBox detections.
        """
        results = {}
        for path in image_paths:
            try:
                results[path] = self.detect(path)
            except Exception as e:
                logger.error(f"Detection failed for {path}: {e}")
                results[path] = []
        return results

    def _apply_nms(self, detections: list) -> list:
        """Apply Non-Maximum Suppression to remove overlapping detections.

        When multiple prompts match the same object (e.g., "forceps" and
        "surgical clamp" both detect the same tool), NMS keeps only the
        highest confidence detection.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept = []
        suppressed = set()

        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            kept.append(det_i)

            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue

                iou = self._compute_iou(det_i, detections[j])
                if iou > self.nms_threshold:
                    suppressed.add(j)

        return kept

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
