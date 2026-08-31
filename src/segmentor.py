"""SAM 2 segmentor: generates pixel-level masks from bounding box detections.

SAM 2 (Segment Anything Model 2) extends Meta's original SAM to video.
It can take a bounding box or point prompt on a single frame and produce
precise segmentation masks, then propagate those masks across video frames.

The workflow in this project:
    1. Grounding DINO provides bounding boxes (rough location)
    2. SAM 2 takes those boxes as prompts and generates precise masks
    3. The masks tell us exactly which pixels belong to each instrument

Why two models instead of one?
    Grounding DINO is great at finding objects from text but gives rough boxes.
    SAM 2 is great at precise segmentation but needs a prompt (box/point).
    Together: text prompt -> precise pixel mask.
"""

import numpy as np
from loguru import logger
from typing import Optional

from src.utils import BBox


class SAM2Segmentor:
    """Wrapper around SAM 2 for surgical instrument segmentation."""

    def __init__(self, config: dict):
        self.model_id = config["segmentation"]["model_id"]
        self.multimask_output = config["segmentation"]["multimask_output"]
        self.mask_threshold = config["segmentation"]["mask_threshold"]
        self.device = config["segmentation"]["device"]

        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load the SAM 2 model and processor."""
        import torch
        from transformers import AutoProcessor, AutoModelForMaskGeneration

        logger.info(f"Loading SAM 2 from {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMaskGeneration.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

        logger.info(f"SAM 2 loaded on {self.device}")

    def segment_from_boxes(
        self,
        image_path: str,
        bboxes: list,
    ) -> list:
        """Generate segmentation masks from bounding box prompts.

        For each bounding box from Grounding DINO, SAM 2 produces a
        precise pixel-level mask showing exactly which pixels belong
        to that instrument.

        Args:
            image_path: Path to the frame image.
            bboxes: List of BBox objects from the detector.

        Returns:
            List of numpy arrays (binary masks), one per bounding box.
            Each mask has shape (H, W) with values 0 or 1.
        """
        import torch
        from PIL import Image

        if self.model is None:
            self.load_model()

        if not bboxes:
            return []

        image = Image.open(image_path).convert("RGB")

        # Convert BBox objects to the format SAM 2 expects: [[x1,y1,x2,y2], ...]
        input_boxes = [bbox.to_xyxy() for bbox in bboxes]

        # Preprocess
        inputs = self.processor(
            images=image,
            input_boxes=[input_boxes],
            return_tensors="pt",
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]

        # Convert to binary numpy arrays
        binary_masks = []
        for mask in masks:
            # mask shape: (num_masks, H, W) - take the best one
            if mask.dim() == 3:
                mask = mask[0]  # Take first (best) mask

            binary_mask = (mask.cpu().numpy() > self.mask_threshold).astype(np.uint8)
            binary_masks.append(binary_mask)

        logger.debug(
            f"Generated {len(binary_masks)} masks for {len(bboxes)} boxes"
        )
        return binary_masks

    def segment_frame(
        self,
        image_path: str,
        bboxes: list,
    ) -> list:
        """Segment all instruments in a frame and return paired results.

        Args:
            image_path: Path to the frame image.
            bboxes: List of BBox objects.

        Returns:
            List of dicts with keys: bbox, mask, mask_area, mask_quality
        """
        masks = self.segment_from_boxes(image_path, bboxes)

        results = []
        for bbox, mask in zip(bboxes, masks):
            mask_area = int(np.sum(mask))
            # Mask quality: ratio of mask area to bounding box area
            # A good mask should fill a reasonable portion of the bbox
            bbox_pixel_area = bbox.area
            mask_quality = mask_area / bbox_pixel_area if bbox_pixel_area > 0 else 0

            results.append({
                "bbox": bbox,
                "mask": mask,
                "mask_area": mask_area,
                "mask_quality": round(mask_quality, 3),
            })

        return results

    @staticmethod
    def mask_to_polygon(mask: np.ndarray) -> list:
        """Convert a binary mask to polygon coordinates (for COCO format).

        COCO annotations store segmentation as polygons, not pixel masks.
        This converts our binary mask into a list of polygon contours.

        Args:
            mask: Binary mask array of shape (H, W).

        Returns:
            List of polygon coordinates [[x1,y1,x2,y2,...], ...].
        """
        import cv2

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS,
        )

        polygons = []
        for contour in contours:
            # Filter out very small contours (noise)
            if cv2.contourArea(contour) < 50:
                continue

            # Flatten contour to [x1, y1, x2, y2, ...]
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:  # Need at least 3 points
                polygons.append(polygon)

        return polygons

    @staticmethod
    def mask_to_rle(mask: np.ndarray) -> dict:
        """Convert a binary mask to Run-Length Encoding (for compact storage).

        RLE stores masks efficiently by encoding runs of 0s and 1s.
        A mask with large contiguous regions compresses very well.

        Args:
            mask: Binary mask array of shape (H, W).

        Returns:
            Dict with keys: counts (list of run lengths), size [H, W].
        """
        pixels = mask.flatten(order="F")  # Column-major (COCO convention)
        runs = []
        current_run = 0

        for i in range(len(pixels)):
            if i == 0:
                current_run = 1
            elif pixels[i] == pixels[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)

        # RLE starts with background count
        if pixels[0] == 1:
            runs = [0] + runs

        return {
            "counts": runs,
            "size": list(mask.shape),
        }
