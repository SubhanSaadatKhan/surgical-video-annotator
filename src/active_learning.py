"""Active learning: closes the loop so the system improves over time.

Traditional pipelines are static: run once, get output, done. Active learning
makes the system self-improving:

    1. Model annotates a batch of videos
    2. Uncertain predictions are flagged for human review
    3. Humans correct the flagged annotations
    4. Corrections are used to fine-tune the detection model
    5. Next batch benefits from improvements
    6. Repeat, getting better each iteration

The key insight: instead of annotating random frames for training, we
select frames where the model is MOST uncertain. These are the frames
that will teach the model the most per unit of human effort.

This module handles:
    - Uncertainty scoring (which predictions need review)
    - Sample selection (which frames to send to humans)
    - Correction storage (organized human feedback)
    - Fine-tuning data preparation (converting corrections to training data)
    - Iteration tracking (measuring improvement over cycles)
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.utils import BBox, FrameAnnotation


@dataclass
class UncertaintyScore:
    """Uncertainty measurement for a single prediction."""
    frame_idx: int
    bbox_idx: int
    label: str
    confidence: float
    uncertainty: float          # Higher = more uncertain
    uncertainty_type: str       # "low_confidence", "boundary", "disagreement"
    reason: str


@dataclass
class ActiveLearningIteration:
    """Metadata for one iteration of the active learning loop."""
    iteration_id: int
    timestamp: str
    videos_processed: list
    samples_reviewed: int
    corrections_received: int
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    improvement: dict = field(default_factory=dict)


class UncertaintySampler:
    """Selects the most informative samples for human review.

    Uses multiple uncertainty signals to identify predictions that would
    benefit most from human labeling:

        - Low confidence: Model is unsure about the classification
        - Boundary uncertainty: Prediction is near the decision threshold
        - Class disagreement: Multiple plausible classes for same object

    The output is a prioritized list of samples for humans to review,
    focusing effort where it will have the biggest impact.
    """

    def __init__(self, config: dict):
        al_config = config.get("active_learning", {})
        self.max_samples_per_iteration = al_config.get(
            "max_samples_per_iteration", 100
        )
        self.uncertainty_threshold = al_config.get("uncertainty_threshold", 0.4)
        self.boundary_margin = al_config.get("boundary_margin", 0.1)
        # Confidence range considered "uncertain"
        self.low_conf_threshold = al_config.get("low_conf_threshold", 0.5)
        self.high_conf_threshold = al_config.get("high_conf_threshold", 0.8)

    def score_predictions(
        self, annotations: list
    ) -> list:
        """Compute uncertainty scores for all predictions.

        Args:
            annotations: List of FrameAnnotation objects.

        Returns:
            List of UncertaintyScore, sorted by uncertainty (highest first).
        """
        scores = []

        for annotation in annotations:
            for i, bbox in enumerate(annotation.bboxes):
                # Signal 1: Absolute confidence
                # Very high (>0.8) or very low (<0.3) are certain,
                # middle range is uncertain
                if self.low_conf_threshold <= bbox.confidence <= self.high_conf_threshold:
                    # Peak uncertainty in the middle of this range
                    mid = (self.low_conf_threshold + self.high_conf_threshold) / 2
                    distance_from_mid = abs(bbox.confidence - mid)
                    range_half = (self.high_conf_threshold - self.low_conf_threshold) / 2
                    conf_uncertainty = 1.0 - (distance_from_mid / range_half)
                else:
                    conf_uncertainty = 0.0

                uncertainty = conf_uncertainty
                reason = f"Confidence {bbox.confidence:.2f} in uncertain range"
                unc_type = "boundary"

                # Signal 2: Very low confidence but still detected
                if bbox.confidence < self.low_conf_threshold:
                    uncertainty = max(uncertainty, 0.7)
                    reason = f"Very low confidence ({bbox.confidence:.2f})"
                    unc_type = "low_confidence"

                # Signal 3: Small or unusual bounding box
                bbox_area = bbox.area
                if bbox_area < 500 or bbox_area > 100000:
                    uncertainty = max(uncertainty, 0.5)
                    reason += f" + unusual size ({bbox_area:.0f}px)"

                if uncertainty >= self.uncertainty_threshold:
                    scores.append(UncertaintyScore(
                        frame_idx=annotation.frame_idx,
                        bbox_idx=i,
                        label=bbox.label,
                        confidence=bbox.confidence,
                        uncertainty=uncertainty,
                        uncertainty_type=unc_type,
                        reason=reason,
                    ))

        # Sort by uncertainty (highest first)
        scores.sort(key=lambda s: s.uncertainty, reverse=True)

        logger.info(
            f"Uncertainty scoring: {len(scores)} samples above threshold "
            f"(from {sum(len(a.bboxes) for a in annotations)} total predictions)"
        )

        return scores

    def select_for_review(
        self, scores: list
    ) -> list:
        """Select the top-N most uncertain samples for human review.

        Applies a diversity filter to avoid reviewing many similar samples.

        Args:
            scores: Sorted list of UncertaintyScore (highest first).

        Returns:
            List of selected samples, limited to max_samples_per_iteration.
        """
        if len(scores) <= self.max_samples_per_iteration:
            return scores

        # Diversity: try to include samples from different frames and classes
        selected = []
        frames_seen = {}   # frame_idx -> count
        classes_seen = {}  # label -> count
        max_per_frame = 3
        max_per_class = self.max_samples_per_iteration // 3

        for score in scores:
            if len(selected) >= self.max_samples_per_iteration:
                break

            frame_count = frames_seen.get(score.frame_idx, 0)
            class_count = classes_seen.get(score.label, 0)

            if frame_count < max_per_frame and class_count < max_per_class:
                selected.append(score)
                frames_seen[score.frame_idx] = frame_count + 1
                classes_seen[score.label] = class_count + 1

        # Fill remaining slots without diversity constraints if needed
        for score in scores:
            if len(selected) >= self.max_samples_per_iteration:
                break
            if score not in selected:
                selected.append(score)

        logger.info(
            f"Selected {len(selected)} samples for review "
            f"across {len(frames_seen)} frames and {len(classes_seen)} classes"
        )
        return selected


class CorrectionStore:
    """Manages human corrections for use in fine-tuning.

    Human corrections are stored in a structured format that can be
    directly converted into training data for fine-tuning the detector.

    Structure:
        corrections/
            iteration_001/
                corrections.json         # All corrections
                metadata.json            # When, who, what
                fine_tune_dataset.json   # Ready for training
            iteration_002/
                ...
    """

    def __init__(self, base_dir: str = "corrections"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_iteration_id(self) -> int:
        """Find the next iteration number."""
        existing = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        ]
        if not existing:
            return 1
        ids = [int(d.name.split("_")[1]) for d in existing]
        return max(ids) + 1

    def save_iteration(
        self,
        iteration: ActiveLearningIteration,
        corrections: list,
    ) -> str:
        """Save an iteration's corrections and metadata.

        Args:
            iteration: Iteration metadata.
            corrections: List of correction dicts with structure:
                {
                    "frame_idx": int,
                    "image_path": str,
                    "original_bbox": {...},
                    "corrected_bbox": {...} or None (if rejected),
                    "action": "approved" | "corrected" | "rejected",
                    "notes": str,
                }

        Returns:
            Path to iteration directory.
        """
        iter_dir = self.base_dir / f"iteration_{iteration.iteration_id:03d}"
        iter_dir.mkdir(exist_ok=True)

        # Save corrections
        corrections_path = iter_dir / "corrections.json"
        with open(corrections_path, "w") as f:
            json.dump(corrections, f, indent=2)

        # Save metadata
        metadata_path = iter_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "iteration_id": iteration.iteration_id,
                "timestamp": iteration.timestamp,
                "videos_processed": iteration.videos_processed,
                "samples_reviewed": iteration.samples_reviewed,
                "corrections_received": iteration.corrections_received,
                "metrics_before": iteration.metrics_before,
                "metrics_after": iteration.metrics_after,
                "improvement": iteration.improvement,
            }, f, indent=2)

        # Convert to fine-tuning dataset format
        self._create_fine_tune_dataset(corrections, iter_dir)

        logger.info(
            f"Saved iteration {iteration.iteration_id}: "
            f"{len(corrections)} corrections -> {iter_dir}"
        )
        return str(iter_dir)

    def _create_fine_tune_dataset(
        self, corrections: list, iter_dir: Path
    ) -> None:
        """Convert corrections into COCO-format training data.

        Only includes approved and corrected annotations (not rejections).
        This becomes the training set for the next fine-tuning cycle.
        """
        fine_tune_data = {
            "info": {
                "description": "Fine-tuning dataset from active learning",
                "created": datetime.now().isoformat(),
            },
            "images": [],
            "annotations": [],
            "categories": [],
        }

        images_seen = set()
        categories_seen = {}
        annotation_id = 1

        for corr in corrections:
            if corr["action"] == "rejected":
                continue

            frame_idx = corr["frame_idx"]
            image_path = corr["image_path"]

            # Add image if not already added
            if frame_idx not in images_seen:
                fine_tune_data["images"].append({
                    "id": frame_idx,
                    "file_name": os.path.basename(image_path),
                    "path": image_path,
                })
                images_seen.add(frame_idx)

            # Use corrected bbox if available, otherwise original
            bbox_data = corr.get("corrected_bbox") or corr.get("original_bbox")
            if not bbox_data:
                continue

            label = bbox_data.get("label", "unknown")

            # Track categories
            if label not in categories_seen:
                cat_id = len(categories_seen) + 1
                categories_seen[label] = cat_id
                fine_tune_data["categories"].append({
                    "id": cat_id,
                    "name": label,
                })

            fine_tune_data["annotations"].append({
                "id": annotation_id,
                "image_id": frame_idx,
                "category_id": categories_seen[label],
                "bbox": bbox_data["bbox"],
                "area": bbox_data.get("area", 0),
                "iscrowd": 0,
                "human_verified": True,
                "source_action": corr["action"],
            })
            annotation_id += 1

        fine_tune_path = iter_dir / "fine_tune_dataset.json"
        with open(fine_tune_path, "w") as f:
            json.dump(fine_tune_data, f, indent=2)

    def load_all_corrections(self) -> list:
        """Load corrections from all iterations (for cumulative training)."""
        all_corrections = []
        for iter_dir in sorted(self.base_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            corr_path = iter_dir / "corrections.json"
            if corr_path.exists():
                with open(corr_path, "r") as f:
                    all_corrections.extend(json.load(f))
        return all_corrections

    def load_iteration_history(self) -> list:
        """Load metadata for all past iterations (for tracking improvement)."""
        iterations = []
        for iter_dir in sorted(self.base_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            meta_path = iter_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    iterations.append(json.load(f))
        return iterations


class ActiveLearningOrchestrator:
    """Coordinates the full active learning loop.

    Usage:
        orchestrator = ActiveLearningOrchestrator(config)

        # After running the annotation pipeline:
        samples_for_review = orchestrator.get_review_queue(annotations)

        # After human review is complete:
        iteration = orchestrator.record_iteration(
            videos=["video1.mp4"],
            corrections=human_corrections,
            metrics_before={...},
            metrics_after={...},
        )
    """

    def __init__(self, config: dict):
        self.config = config
        self.sampler = UncertaintySampler(config)
        self.store = CorrectionStore(
            config.get("active_learning", {}).get(
                "correction_dir", "corrections"
            )
        )

    def get_review_queue(
        self, annotations: list
    ) -> list:
        """Analyze annotations and return prioritized review queue."""
        scores = self.sampler.score_predictions(annotations)
        selected = self.sampler.select_for_review(scores)
        return selected

    def record_iteration(
        self,
        videos: list,
        corrections: list,
        metrics_before: dict = None,
        metrics_after: dict = None,
    ) -> ActiveLearningIteration:
        """Record a completed active learning iteration.

        Args:
            videos: List of videos processed in this iteration.
            corrections: Human corrections from the review interface.
            metrics_before: Evaluation metrics before this iteration.
            metrics_after: Evaluation metrics after fine-tuning (if applicable).

        Returns:
            ActiveLearningIteration object.
        """
        iteration_id = self.store.get_next_iteration_id()

        # Compute improvement if we have before/after metrics
        improvement = {}
        if metrics_before and metrics_after:
            for key in ["precision", "recall", "f1_score", "map_50"]:
                if key in metrics_before.get("summary", {}) and key in metrics_after.get("summary", {}):
                    before_val = metrics_before["summary"][key]
                    after_val = metrics_after["summary"][key]
                    improvement[key] = {
                        "before": before_val,
                        "after": after_val,
                        "delta": round(after_val - before_val, 4),
                        "relative_change_pct": round(
                            ((after_val - before_val) / before_val * 100)
                            if before_val > 0 else 0, 2
                        ),
                    }

        iteration = ActiveLearningIteration(
            iteration_id=iteration_id,
            timestamp=datetime.now().isoformat(),
            videos_processed=videos,
            samples_reviewed=len(corrections),
            corrections_received=sum(
                1 for c in corrections if c["action"] in ("corrected", "rejected")
            ),
            metrics_before=metrics_before or {},
            metrics_after=metrics_after or {},
            improvement=improvement,
        )

        self.store.save_iteration(iteration, corrections)
        return iteration

    def get_improvement_summary(self) -> dict:
        """Summarize improvement across all iterations.

        Returns:
            Dict showing how metrics have improved over iterations.
        """
        history = self.store.load_iteration_history()

        if not history:
            return {"iterations_completed": 0, "message": "No iterations yet"}

        # Extract metrics timeline
        map_timeline = []
        f1_timeline = []
        for iter_data in history:
            metrics = iter_data.get("metrics_after", {}).get("summary", {})
            if "map_50" in metrics:
                map_timeline.append({
                    "iteration": iter_data["iteration_id"],
                    "map_50": metrics["map_50"],
                })
            if "f1_score" in metrics:
                f1_timeline.append({
                    "iteration": iter_data["iteration_id"],
                    "f1": metrics["f1_score"],
                })

        summary = {
            "iterations_completed": len(history),
            "total_corrections": sum(
                iter_data["corrections_received"] for iter_data in history
            ),
            "total_samples_reviewed": sum(
                iter_data["samples_reviewed"] for iter_data in history
            ),
            "map_timeline": map_timeline,
            "f1_timeline": f1_timeline,
        }

        # Overall improvement
        if len(map_timeline) >= 2:
            initial_map = map_timeline[0]["map_50"]
            final_map = map_timeline[-1]["map_50"]
            summary["overall_map_improvement"] = round(final_map - initial_map, 4)

        return summary
