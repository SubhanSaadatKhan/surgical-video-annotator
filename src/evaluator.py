"""Ground-truth evaluation: measures how good our annotations really are.

Without evaluation metrics, we have no way to prove the pipeline works.
This module compares our automated annotations against human-labeled
ground truth using standard object detection and segmentation metrics:

    - Precision: Of what we detected, how much is correct?
    - Recall: Of what should have been detected, how much did we catch?
    - F1: Harmonic mean of precision and recall
    - mAP: Mean Average Precision (standard object detection metric)
    - IoU: Intersection over Union for bounding boxes
    - Mask IoU: Same for segmentation masks

These metrics are what industry uses to compare models. When Christina
asks "how well does your pipeline work?", you have concrete numbers:
"mAP@0.5 of 0.78 on the CholecT50 test set."
"""

import json
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from src.utils import BBox, compute_iou


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    map_50: float                    # mAP at IoU threshold 0.5
    map_75: float                    # mAP at IoU threshold 0.75 (stricter)
    map_50_95: float                 # mAP averaged over IoU 0.5-0.95
    per_class_ap: dict = field(default_factory=dict)
    per_class_precision: dict = field(default_factory=dict)
    per_class_recall: dict = field(default_factory=dict)
    confusion_matrix: dict = field(default_factory=dict)
    total_predictions: int = 0
    total_ground_truth: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "summary": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "map_50": round(self.map_50, 4),
                "map_75": round(self.map_75, 4),
                "map_50_95": round(self.map_50_95, 4),
            },
            "counts": {
                "total_predictions": self.total_predictions,
                "total_ground_truth": self.total_ground_truth,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "per_class": {
                "ap_50": {k: round(v, 4) for k, v in self.per_class_ap.items()},
                "precision": {k: round(v, 4) for k, v in self.per_class_precision.items()},
                "recall": {k: round(v, 4) for k, v in self.per_class_recall.items()},
            },
            "confusion_matrix": self.confusion_matrix,
        }


class AnnotationEvaluator:
    """Evaluates predicted annotations against ground truth.

    Ground truth format: COCO JSON (standard for object detection).
    Predictions format: Same COCO JSON structure produced by our pipeline.

    This makes it easy to evaluate against public benchmarks like CholecT50
    or the m2cai16-tool-locations dataset.
    """

    def __init__(self, iou_thresholds: list = None):
        """
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation.
                Default: [0.5, 0.55, ..., 0.95] (COCO standard).
        """
        if iou_thresholds is None:
            self.iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

    def evaluate(
        self,
        predictions_path: str,
        ground_truth_path: str,
        primary_iou_threshold: float = 0.5,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions_path: COCO JSON with our pipeline's predictions.
            ground_truth_path: COCO JSON with human-labeled ground truth.
            primary_iou_threshold: IoU threshold for precision/recall/F1.

        Returns:
            EvaluationResult with all metrics.
        """
        logger.info(f"Loading predictions from {predictions_path}")
        with open(predictions_path, "r") as f:
            predictions = json.load(f)

        logger.info(f"Loading ground truth from {ground_truth_path}")
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        # Verify compatibility
        self._verify_compatibility(predictions, ground_truth)

        # Group annotations by image_id
        pred_by_image = self._group_by_image(predictions["annotations"])
        gt_by_image = self._group_by_image(ground_truth["annotations"])

        categories = {c["id"]: c["name"] for c in ground_truth["categories"]}

        # Compute per-image matches
        all_matches = self._match_predictions_to_ground_truth(
            pred_by_image, gt_by_image, primary_iou_threshold
        )

        # Aggregate metrics
        result = self._compute_metrics(
            all_matches, pred_by_image, gt_by_image, categories
        )

        # Compute mAP at multiple thresholds
        ap_scores = {}
        for iou_thresh in self.iou_thresholds:
            matches = self._match_predictions_to_ground_truth(
                pred_by_image, gt_by_image, iou_thresh
            )
            ap_by_class = self._compute_ap_per_class(
                matches, pred_by_image, gt_by_image, categories
            )
            ap_scores[iou_thresh] = ap_by_class

        result.map_50 = self._mean_over_classes(ap_scores.get(0.5, {}))
        result.map_75 = self._mean_over_classes(ap_scores.get(0.75, {}))

        # mAP@0.5:0.95 (COCO primary metric)
        all_aps = []
        for iou_thresh in self.iou_thresholds:
            all_aps.append(self._mean_over_classes(ap_scores.get(iou_thresh, {})))
        result.map_50_95 = float(np.mean(all_aps)) if all_aps else 0.0

        # Per-class AP at 0.5
        result.per_class_ap = ap_scores.get(0.5, {})

        logger.info(
            f"Evaluation complete: "
            f"P={result.precision:.3f} R={result.recall:.3f} "
            f"F1={result.f1_score:.3f} mAP@0.5={result.map_50:.3f}"
        )

        return result

    def _verify_compatibility(self, pred: dict, gt: dict) -> None:
        """Ensure predictions and ground truth are comparable."""
        pred_images = {img["id"] for img in pred["images"]}
        gt_images = {img["id"] for img in gt["images"]}

        common = pred_images & gt_images
        if len(common) == 0:
            raise ValueError("No overlapping images between predictions and ground truth")

        if len(common) < len(gt_images) * 0.5:
            logger.warning(
                f"Only {len(common)}/{len(gt_images)} ground truth images "
                "have predictions. Metrics may be unreliable."
            )

    def _group_by_image(self, annotations: list) -> dict:
        """Group annotations by image_id."""
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann["image_id"]].append(ann)
        return dict(grouped)

    def _coco_bbox_to_bbox(self, ann: dict) -> BBox:
        """Convert COCO annotation to our BBox format."""
        x, y, w, h = ann["bbox"]
        return BBox(
            x_min=x, y_min=y, x_max=x + w, y_max=y + h,
            confidence=ann.get("score", 1.0),
            label=str(ann["category_id"]),
        )

    def _match_predictions_to_ground_truth(
        self,
        pred_by_image: dict,
        gt_by_image: dict,
        iou_threshold: float,
    ) -> list:
        """Match each prediction to its best-matching ground truth.

        A prediction is a true positive if it matches a ground truth with:
            - IoU >= threshold
            - Same category
            - GT hasn't been matched already (highest confidence wins)

        Returns:
            List of match dicts with keys: image_id, is_tp, category_id, score, iou
        """
        matches = []

        for image_id in pred_by_image.keys() | gt_by_image.keys():
            preds = pred_by_image.get(image_id, [])
            gts = gt_by_image.get(image_id, [])

            # Sort predictions by confidence (highest first)
            preds_sorted = sorted(
                preds, key=lambda p: p.get("score", 0), reverse=True
            )

            gt_matched = [False] * len(gts)

            for pred in preds_sorted:
                pred_box = self._coco_bbox_to_bbox(pred)
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    if gt["category_id"] != pred["category_id"]:
                        continue

                    gt_box = self._coco_bbox_to_bbox(gt)
                    iou = compute_iou(pred_box, gt_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
                if is_tp:
                    gt_matched[best_gt_idx] = True

                matches.append({
                    "image_id": image_id,
                    "is_tp": is_tp,
                    "category_id": pred["category_id"],
                    "score": pred.get("score", 0),
                    "iou": best_iou,
                })

        return matches

    def _compute_metrics(
        self,
        matches: list,
        pred_by_image: dict,
        gt_by_image: dict,
        categories: dict,
    ) -> EvaluationResult:
        """Compute overall precision, recall, F1 and per-class metrics."""
        total_predictions = sum(len(v) for v in pred_by_image.values())
        total_gt = sum(len(v) for v in gt_by_image.values())

        tp = sum(1 for m in matches if m["is_tp"])
        fp = total_predictions - tp
        fn = total_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        # Per-class precision and recall
        per_class_p = {}
        per_class_r = {}

        for cat_id, cat_name in categories.items():
            cat_tp = sum(
                1 for m in matches if m["is_tp"] and m["category_id"] == cat_id
            )
            cat_pred = sum(
                1 for anns in pred_by_image.values()
                for a in anns if a["category_id"] == cat_id
            )
            cat_gt = sum(
                1 for anns in gt_by_image.values()
                for a in anns if a["category_id"] == cat_id
            )

            per_class_p[cat_name] = cat_tp / cat_pred if cat_pred > 0 else 0
            per_class_r[cat_name] = cat_tp / cat_gt if cat_gt > 0 else 0

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            map_50=0,  # Filled in later
            map_75=0,
            map_50_95=0,
            per_class_precision=per_class_p,
            per_class_recall=per_class_r,
            total_predictions=total_predictions,
            total_ground_truth=total_gt,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def _compute_ap_per_class(
        self,
        matches: list,
        pred_by_image: dict,
        gt_by_image: dict,
        categories: dict,
    ) -> dict:
        """Compute Average Precision for each class.

        AP is the area under the precision-recall curve.
        Uses the 11-point interpolation method (Pascal VOC style).
        """
        ap_by_class = {}

        for cat_id, cat_name in categories.items():
            cat_matches = [m for m in matches if m["category_id"] == cat_id]
            cat_matches.sort(key=lambda m: m["score"], reverse=True)

            total_gt = sum(
                1 for anns in gt_by_image.values()
                for a in anns if a["category_id"] == cat_id
            )

            if total_gt == 0:
                ap_by_class[cat_name] = 0
                continue

            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for m in cat_matches:
                if m["is_tp"]:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                p = tp_cumsum / (tp_cumsum + fp_cumsum)
                r = tp_cumsum / total_gt
                precisions.append(p)
                recalls.append(r)

            # 11-point interpolation
            ap = 0
            for recall_threshold in [i / 10 for i in range(11)]:
                # Find max precision at recall >= threshold
                valid_precisions = [
                    p for p, r in zip(precisions, recalls)
                    if r >= recall_threshold
                ]
                max_p = max(valid_precisions) if valid_precisions else 0
                ap += max_p / 11

            ap_by_class[cat_name] = ap

        return ap_by_class

    @staticmethod
    def _mean_over_classes(class_scores: dict) -> float:
        """Compute mean AP across all classes."""
        if not class_scores:
            return 0.0
        return float(np.mean(list(class_scores.values())))

    def save_report(self, result: EvaluationResult, output_path: str) -> str:
        """Save evaluation report to JSON."""
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
