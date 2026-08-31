"""Image-mode pipeline: runs annotation on a folder of images (not video).

This lets us evaluate the pipeline on labeled datasets like CholecSeg8k where
we have ground truth. It skips video-specific steps (temporal sampling, video
propagation) and runs detection + per-frame segmentation on each image.

Then it maps the pipeline's detected labels back to the dataset's ground truth
labels so we can compute mAP/precision/recall properly.

Usage:
    from src.image_pipeline import ImageModePipeline

    pipeline = ImageModePipeline("config.yaml")
    results = pipeline.run(
        image_dir="cholecseg8k_data/images",
        ground_truth_path="ground_truth_coco.json",
        prompts=["grasper", "hook", "liver", "gallbladder", "fat"],
    )
    print(results["evaluation"])
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from src.utils import load_config, setup_logger, FrameAnnotation
from src.detector import GroundingDINODetector
from src.segmentor import SAM2Segmentor
from src.validator import AnnotationValidator
from src.evaluator import AnnotationEvaluator
from src.active_learning import ActiveLearningOrchestrator
from src.pipeline import create_detector


class ImageModePipeline:
    """Runs the annotation pipeline on a folder of images instead of a video.

    Used for evaluation against ground-truth labeled datasets. Since images
    have no temporal relationship, video propagation and tracking are disabled.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        setup_logger()

        self.detector = create_detector(self.config)
        self.segmentor = SAM2Segmentor(self.config)
        self.validator = AnnotationValidator(self.config)
        self.evaluator = AnnotationEvaluator()
        self.active_learning = ActiveLearningOrchestrator(self.config)

    def run(
        self,
        image_dir: str,
        ground_truth_path: str = None,
        prompts: list = None,
        output_dir: str = "output_evaluation",
        max_images: int = None,
        include_masks: bool = True,
        label_mapping: dict = None,
    ) -> dict:
        """Run the pipeline on a folder of images.

        Args:
            image_dir: Directory containing images to process.
            ground_truth_path: Optional COCO JSON with ground truth annotations.
                If provided, evaluation metrics are computed.
            prompts: Override detection prompts for this run.
            output_dir: Where to save predictions and results.
            max_images: Cap the number of images processed (for quick tests).
            include_masks: Whether to run SAM 2 segmentation on detections.
            label_mapping: Dict mapping pipeline labels to ground truth class IDs.
                If not provided, uses fuzzy string matching.

        Returns:
            Dict with keys: predictions_path, evaluation, stats
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Override prompts if provided
        if prompts:
            self.detector.prompts = prompts
            logger.info(f"Using custom prompts: {prompts}")

        # Load images
        image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if max_images:
            image_paths = image_paths[:max_images]

        logger.info(f"Processing {len(image_paths)} images from {image_dir}")

        # =====================================================
        # Step 1: Detect on each image
        # =====================================================
        logger.info("Step 1/3: Running detection...")
        self.detector.load_model()

        all_annotations = []
        detections_by_image = {}

        for i, image_path in enumerate(tqdm(image_paths, desc="Detecting")):
            image_id = i  # Use index as ID (must match ground truth if provided)

            try:
                detections = self.detector.detect(image_path)
            except Exception as e:
                logger.warning(f"Detection failed for {image_path}: {e}")
                detections = []

            detections_by_image[image_id] = detections

            annotation = FrameAnnotation(
                frame_idx=image_id,
                timestamp_sec=0.0,
                image_path=image_path,
                bboxes=detections,
            )
            all_annotations.append(annotation)

        total_detections = sum(len(a.bboxes) for a in all_annotations)
        logger.info(f"Detection complete: {total_detections} total detections")

        # =====================================================
        # Step 2: Segment (optional)
        # =====================================================
        all_masks = {}

        if include_masks:
            logger.info("Step 2/3: Running per-image segmentation...")
            self.segmentor.load_model()

            for annotation in tqdm(all_annotations, desc="Segmenting"):
                if not annotation.bboxes:
                    all_masks[annotation.frame_idx] = []
                    continue

                try:
                    results = self.segmentor.segment_frame(
                        annotation.image_path, annotation.bboxes
                    )
                    all_masks[annotation.frame_idx] = [r["mask"] for r in results]
                except Exception as e:
                    logger.warning(f"Segmentation failed: {e}")
                    all_masks[annotation.frame_idx] = [None] * len(annotation.bboxes)
        else:
            logger.info("Step 2/3: Segmentation skipped (include_masks=False)")

        # =====================================================
        # Step 3: Export predictions as COCO
        # =====================================================
        logger.info("Step 3/3: Exporting predictions and evaluating...")
        predictions_path = self._export_predictions_coco(
            all_annotations, output_dir, ground_truth_path, label_mapping,
        )

        # =====================================================
        # Optional: Evaluate against ground truth
        # =====================================================
        evaluation_result = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            logger.info("Evaluating against ground truth...")
            evaluation_result = self.evaluator.evaluate(
                predictions_path=predictions_path,
                ground_truth_path=ground_truth_path,
            )
            eval_output = os.path.join(output_dir, "evaluation_report.json")
            self.evaluator.save_report(evaluation_result, eval_output)
            logger.info(f"Evaluation saved to {eval_output}")

        # =====================================================
        # Compile stats
        # =====================================================
        elapsed = time.time() - start_time
        stats = {
            "total_images": len(image_paths),
            "images_with_detections": sum(1 for a in all_annotations if a.bboxes),
            "total_detections": total_detections,
            "unique_labels": len(set(
                bbox.label for a in all_annotations for bbox in a.bboxes
            )),
            "execution_time_sec": round(elapsed, 1),
            "fps_processed": round(len(image_paths) / elapsed, 1) if elapsed > 0 else 0,
        }

        if evaluation_result:
            stats["evaluation"] = evaluation_result.to_dict()["summary"]

        logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return {
            "predictions_path": predictions_path,
            "annotations": all_annotations,
            "masks": all_masks,
            "evaluation": evaluation_result,
            "stats": stats,
        }

    def _export_predictions_coco(
        self,
        annotations: list,
        output_dir: str,
        ground_truth_path: str = None,
        label_mapping: dict = None,
    ) -> str:
        """Export predictions in COCO format, mapping labels to GT categories.

        The tricky part: our detector outputs text labels like "grasper", but
        COCO evaluation compares by category_id. We need to map text labels
        back to the ground truth's category IDs.
        """
        # Build label -> category_id mapping from ground truth
        category_map = {}  # lowercase label -> category_id

        if ground_truth_path and os.path.exists(ground_truth_path):
            with open(ground_truth_path, "r") as f:
                gt = json.load(f)
            for cat in gt["categories"]:
                category_map[cat["name"].lower()] = cat["id"]
        elif label_mapping:
            category_map = {k.lower(): v for k, v in label_mapping.items()}

        # Build predictions COCO
        pred_coco = {
            "info": {"description": "Pipeline predictions"},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        if ground_truth_path and os.path.exists(ground_truth_path):
            pred_coco["categories"] = gt["categories"]
        else:
            # Build categories from unique detected labels
            unique_labels = set(bbox.label for a in annotations for bbox in a.bboxes)
            for i, label in enumerate(sorted(unique_labels)):
                cat_id = i + 1
                category_map[label.lower()] = cat_id
                pred_coco["categories"].append({"id": cat_id, "name": label})

        annotation_id = 1

        for ann in annotations:
            pred_coco["images"].append({
                "id": ann.frame_idx,
                "file_name": os.path.basename(ann.image_path),
                "width": 1280,  # Placeholder; actual dims read by evaluator if needed
                "height": 720,
            })

            for bbox in ann.bboxes:
                # Map label to category_id via fuzzy matching
                cat_id = self._match_label_to_category(bbox.label, category_map)
                if cat_id is None:
                    continue  # Skip detections that don't match any known category

                pred_coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": ann.frame_idx,
                    "category_id": cat_id,
                    "bbox": bbox.to_coco(),
                    "area": bbox.area,
                    "iscrowd": 0,
                    "score": bbox.confidence,
                })
                annotation_id += 1

        pred_path = os.path.join(output_dir, "predictions_coco.json")
        with open(pred_path, "w") as f:
            json.dump(pred_coco, f, indent=2)

        logger.info(
            f"Exported {len(pred_coco['annotations'])} predictions "
            f"across {len(pred_coco['images'])} images -> {pred_path}"
        )
        return pred_path

    @staticmethod
    def _match_label_to_category(label: str, category_map: dict) -> int:
        """Fuzzy match a predicted label to a ground truth category ID.

        Grounding DINO returns labels like "grasper hook" (merged) or "l-hook".
        We need to figure out which GT category that maps to.

        Args:
            label: Predicted label string.
            category_map: Dict of lowercase category name -> category ID.

        Returns:
            Matched category ID or None.
        """
        label_lower = label.lower().strip()

        # Direct match
        if label_lower in category_map:
            return category_map[label_lower]

        # Try each word in the label (for merged labels like "grasper hook")
        words = label_lower.split()
        for word in words:
            if word in category_map:
                return category_map[word]

        # Substring matching
        for cat_name, cat_id in category_map.items():
            if cat_name in label_lower or label_lower in cat_name:
                return cat_id
            # Try matching individual words
            for cat_word in cat_name.split():
                if cat_word in words:
                    return cat_id

        return None
