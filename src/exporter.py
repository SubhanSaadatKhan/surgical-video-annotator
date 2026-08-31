"""Annotation exporter: saves annotations in standard COCO JSON format.

COCO format is the industry standard for object detection and segmentation
datasets. Using COCO means the annotations can be directly loaded by any
ML framework (PyTorch, TensorFlow, Detectron2, MMDetection, etc.) without
conversion.

COCO JSON structure:
    {
        "info": {...},
        "images": [{id, file_name, width, height}, ...],
        "annotations": [{id, image_id, category_id, bbox, segmentation, ...}, ...],
        "categories": [{id, name}, ...]
    }
"""

import json
import os
import cv2
import numpy as np
from datetime import datetime
from loguru import logger

from src.utils import FrameAnnotation
from src.segmentor import SAM2Segmentor


class COCOExporter:
    """Exports annotations in COCO JSON format with optional visualizations."""

    def __init__(self, config: dict):
        export_config = config.get("export", {})
        self.output_dir = export_config.get("output_dir", "output")
        self.save_viz = export_config.get("save_visualizations", True)
        self.save_masks = export_config.get("save_masks", True)
        self.viz_alpha = export_config.get("visualization_alpha", 0.4)

        self.frame_width = config["video"]["resize_width"]
        self.frame_height = config["video"]["resize_height"]

        # Color palette for different instrument classes
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

    def export(
        self,
        annotations: list,
        masks_by_frame: dict,
        categories: list,
        video_name: str,
    ) -> str:
        """Export all annotations to COCO JSON.

        Args:
            annotations: List of FrameAnnotation objects.
            masks_by_frame: Dict mapping frame_idx to list of binary masks.
            categories: List of category names (instrument types).
            video_name: Name of the source video (for metadata).

        Returns:
            Path to the saved COCO JSON file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Build category mapping
        category_map = {
            name: idx + 1 for idx, name in enumerate(sorted(set(categories)))
        }

        coco = {
            "info": {
                "description": f"Surgical Video Annotations: {video_name}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
                "contributor": "Surgical Video Annotator (Foundation Model Pipeline)",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx, "name": name, "supercategory": "surgical_instrument"}
                for name, idx in category_map.items()
            ],
        }

        annotation_id = 1

        for ann in annotations:
            # Add image entry
            image_entry = {
                "id": ann.frame_idx,
                "file_name": os.path.basename(ann.image_path),
                "width": self.frame_width,
                "height": self.frame_height,
                "timestamp_sec": ann.timestamp_sec,
            }
            coco["images"].append(image_entry)

            # Get masks for this frame
            frame_masks = masks_by_frame.get(ann.frame_idx, [])

            # Add annotation entries
            for i, bbox in enumerate(ann.bboxes):
                ann_entry = {
                    "id": annotation_id,
                    "image_id": ann.frame_idx,
                    "category_id": category_map.get(bbox.label, 1),
                    "bbox": bbox.to_coco(),
                    "area": bbox.area,
                    "iscrowd": 0,
                    "score": bbox.confidence,
                    "track_id": bbox.track_id,
                    "quality_status": ann.status,
                }

                # Add segmentation if mask exists
                if i < len(frame_masks) and frame_masks[i] is not None:
                    polygons = SAM2Segmentor.mask_to_polygon(frame_masks[i])
                    if polygons:
                        ann_entry["segmentation"] = polygons
                        ann_entry["area"] = float(np.sum(frame_masks[i]))

                coco["annotations"].append(ann_entry)
                annotation_id += 1

        # Save COCO JSON
        output_path = os.path.join(self.output_dir, f"{video_name}_coco.json")
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            f"Exported COCO annotations: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations, "
            f"{len(coco['categories'])} categories -> {output_path}"
        )

        # Save visualizations
        if self.save_viz:
            self._save_visualizations(annotations, masks_by_frame, category_map)

        # Save individual masks
        if self.save_masks:
            self._save_mask_files(annotations, masks_by_frame)

        return output_path

    def _save_visualizations(
        self,
        annotations: list,
        masks_by_frame: dict,
        category_map: dict,
    ) -> None:
        """Save annotated frame images with bounding boxes and masks overlaid."""
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        for ann in annotations:
            frame = cv2.imread(ann.image_path)
            if frame is None:
                continue

            frame_masks = masks_by_frame.get(ann.frame_idx, [])

            for i, bbox in enumerate(ann.bboxes):
                # Get color for this category
                cat_idx = category_map.get(bbox.label, 1) - 1
                color = self.colors[cat_idx % len(self.colors)]

                # Draw bounding box
                pt1 = (int(bbox.x_min), int(bbox.y_min))
                pt2 = (int(bbox.x_max), int(bbox.y_max))
                cv2.rectangle(frame, pt1, pt2, color, 2)

                # Draw label with confidence and track ID
                label_text = f"{bbox.label} {bbox.confidence:.2f}"
                if bbox.track_id is not None:
                    label_text += f" [T{bbox.track_id}]"

                label_size = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )[0]
                cv2.rectangle(
                    frame,
                    (pt1[0], pt1[1] - label_size[1] - 5),
                    (pt1[0] + label_size[0], pt1[1]),
                    color, -1,
                )
                cv2.putText(
                    frame, label_text,
                    (pt1[0], pt1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

                # Overlay mask
                if i < len(frame_masks) and frame_masks[i] is not None:
                    mask = frame_masks[i]
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 0] = color
                    frame = cv2.addWeighted(
                        frame, 1.0, colored_mask, self.viz_alpha, 0
                    )

            # Add quality flags
            for flag in ann.quality_flags:
                flag_color = (0, 0, 255) if flag.severity == "error" else (0, 165, 255)
                cv2.putText(
                    frame, f"! {flag.flag_type}",
                    (10, 30 + ann.quality_flags.index(flag) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, flag_color, 1,
                )

            viz_path = os.path.join(
                viz_dir, f"annotated_{ann.frame_idx:06d}.jpg"
            )
            cv2.imwrite(viz_path, frame)

        logger.info(f"Saved {len(annotations)} visualizations to {viz_dir}")

    def _save_mask_files(
        self,
        annotations: list,
        masks_by_frame: dict,
    ) -> None:
        """Save individual mask files as PNG (for training segmentation models)."""
        mask_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        for ann in annotations:
            frame_masks = masks_by_frame.get(ann.frame_idx, [])
            if not frame_masks:
                continue

            # Create a combined class mask (each pixel gets a class ID)
            combined = np.zeros(
                (self.frame_height, self.frame_width), dtype=np.uint8
            )
            for i, mask in enumerate(frame_masks):
                if mask is not None:
                    # Class ID starts at 1 (0 is background)
                    combined[mask > 0] = i + 1

            mask_path = os.path.join(
                mask_dir, f"mask_{ann.frame_idx:06d}.png"
            )
            cv2.imwrite(mask_path, combined)

        logger.info(f"Saved mask files to {mask_dir}")
