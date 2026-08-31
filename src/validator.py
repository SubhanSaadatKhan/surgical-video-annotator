"""Annotation quality validator: flags problematic annotations for human review.

Foundation models are not perfect. SAM might produce a noisy mask.
Grounding DINO might hallucinate a detection. The tracker might
lose an instrument. This module catches those issues automatically
so the human reviewer can focus on fixing real problems instead of
checking every single annotation.

Quality checks include:
    - Low confidence detections
    - Abnormally small or large bounding boxes
    - Highly overlapping detections (likely duplicates)
    - Temporal inconsistency (instrument appears/disappears suddenly)
    - Empty or near-empty masks
"""

import numpy as np
from dataclasses import dataclass
from loguru import logger

from src.utils import BBox, FrameAnnotation, compute_iou


@dataclass
class QualityFlag:
    """A quality issue found in an annotation."""
    frame_idx: int
    flag_type: str        # "low_confidence", "small_bbox", "large_bbox",
                          # "overlap", "temporal_gap", "empty_mask"
    severity: str         # "warning", "error"
    message: str
    bbox_idx: int = -1    # Index of the problematic bbox (-1 if frame-level)


class AnnotationValidator:
    """Validates annotation quality and flags issues for review."""

    def __init__(self, config: dict):
        quality_config = config.get("quality", {})
        self.min_confidence = quality_config.get("min_confidence", 0.35)
        self.min_bbox_area = quality_config.get("min_bbox_area", 100)
        self.max_bbox_area_ratio = quality_config.get("max_bbox_area_ratio", 0.8)
        self.overlap_iou_threshold = quality_config.get("overlap_iou_threshold", 0.7)
        self.temporal_window = quality_config.get("temporal_consistency_window", 5)
        self.flag_disappearance = quality_config.get("flag_sudden_disappearance", True)

        self.frame_width = config["video"]["resize_width"]
        self.frame_height = config["video"]["resize_height"]
        self.frame_area = self.frame_width * self.frame_height

    def validate_frame(
        self,
        annotation: FrameAnnotation,
        masks: list = None,
    ) -> list:
        """Run all quality checks on a single frame's annotations.

        Args:
            annotation: FrameAnnotation with bboxes populated.
            masks: Optional list of binary masks corresponding to bboxes.

        Returns:
            List of QualityFlag objects for any issues found.
        """
        flags = []

        for i, bbox in enumerate(annotation.bboxes):
            flags.extend(self._check_confidence(annotation.frame_idx, i, bbox))
            flags.extend(self._check_bbox_size(annotation.frame_idx, i, bbox))

        flags.extend(self._check_overlaps(annotation.frame_idx, annotation.bboxes))

        if masks:
            flags.extend(
                self._check_masks(annotation.frame_idx, annotation.bboxes, masks)
            )

        return flags

    def validate_sequence(
        self, annotations: list
    ) -> list:
        """Check temporal consistency across a sequence of frames.

        Looks for instruments that suddenly appear or disappear, which
        usually indicates a detection failure rather than an actual event.

        Args:
            annotations: List of FrameAnnotation objects, sorted by frame_idx.

        Returns:
            List of QualityFlag objects for temporal issues.
        """
        if len(annotations) < 2:
            return []

        flags = []
        if not self.flag_disappearance:
            return flags

        # Build track presence map: {track_id: [frame indices where present]}
        track_presence = {}
        for ann in annotations:
            for bbox in ann.bboxes:
                if bbox.track_id is not None:
                    if bbox.track_id not in track_presence:
                        track_presence[bbox.track_id] = []
                    track_presence[bbox.track_id].append(ann.frame_idx)

        # Check for gaps in tracks
        for track_id, frames in track_presence.items():
            frames_sorted = sorted(frames)
            for i in range(1, len(frames_sorted)):
                gap = frames_sorted[i] - frames_sorted[i - 1]
                # If the gap is larger than expected but the track resumes,
                # the instrument was probably missed in between
                if 1 < gap <= self.temporal_window:
                    flags.append(QualityFlag(
                        frame_idx=frames_sorted[i - 1],
                        flag_type="temporal_gap",
                        severity="warning",
                        message=(
                            f"Track {track_id} has a {gap}-frame gap "
                            f"(frames {frames_sorted[i-1]} to {frames_sorted[i]}). "
                            f"Detection might have been missed in between."
                        ),
                    ))

        # Check for very short tracks (likely false positives)
        for track_id, frames in track_presence.items():
            if len(frames) == 1:
                flags.append(QualityFlag(
                    frame_idx=frames[0],
                    flag_type="single_frame_track",
                    severity="warning",
                    message=(
                        f"Track {track_id} appears in only 1 frame. "
                        f"Likely a false positive detection."
                    ),
                ))

        logger.info(f"Temporal validation: {len(flags)} issues across {len(annotations)} frames")
        return flags

    def _check_confidence(
        self, frame_idx: int, bbox_idx: int, bbox: BBox
    ) -> list:
        """Flag detections with low confidence."""
        flags = []
        if bbox.confidence < self.min_confidence:
            flags.append(QualityFlag(
                frame_idx=frame_idx,
                flag_type="low_confidence",
                severity="warning",
                message=(
                    f"{bbox.label} detected with low confidence "
                    f"({bbox.confidence:.2f} < {self.min_confidence})"
                ),
                bbox_idx=bbox_idx,
            ))
        return flags

    def _check_bbox_size(
        self, frame_idx: int, bbox_idx: int, bbox: BBox
    ) -> list:
        """Flag bounding boxes that are too small or too large."""
        flags = []
        area = bbox.area

        if area < self.min_bbox_area:
            flags.append(QualityFlag(
                frame_idx=frame_idx,
                flag_type="small_bbox",
                severity="warning",
                message=(
                    f"{bbox.label} bounding box is very small "
                    f"({area:.0f}px, min: {self.min_bbox_area}px). "
                    f"Might be noise."
                ),
                bbox_idx=bbox_idx,
            ))

        area_ratio = area / self.frame_area
        if area_ratio > self.max_bbox_area_ratio:
            flags.append(QualityFlag(
                frame_idx=frame_idx,
                flag_type="large_bbox",
                severity="error",
                message=(
                    f"{bbox.label} covers {area_ratio:.0%} of the frame. "
                    f"Likely a false detection."
                ),
                bbox_idx=bbox_idx,
            ))

        return flags

    def _check_overlaps(
        self, frame_idx: int, bboxes: list
    ) -> list:
        """Flag highly overlapping detections (likely duplicates)."""
        flags = []
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = compute_iou(bboxes[i], bboxes[j])
                if iou > self.overlap_iou_threshold:
                    flags.append(QualityFlag(
                        frame_idx=frame_idx,
                        flag_type="overlap",
                        severity="warning",
                        message=(
                            f"'{bboxes[i].label}' and '{bboxes[j].label}' "
                            f"overlap significantly (IoU: {iou:.2f}). "
                            f"Possible duplicate detection."
                        ),
                        bbox_idx=i,
                    ))
        return flags

    def _check_masks(
        self,
        frame_idx: int,
        bboxes: list,
        masks: list,
    ) -> list:
        """Flag masks that are empty or don't align with their bounding box."""
        flags = []
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            # Skip if mask is None (segmentation failed or wasn't produced)
            if mask is None:
                flags.append(QualityFlag(
                    frame_idx=frame_idx,
                    flag_type="missing_mask",
                    severity="warning",
                    message=f"{bbox.label} has no segmentation mask produced.",
                    bbox_idx=i,
                ))
                continue

            mask_area = np.sum(mask)

            if mask_area == 0:
                flags.append(QualityFlag(
                    frame_idx=frame_idx,
                    flag_type="empty_mask",
                    severity="error",
                    message=f"{bbox.label} has an empty segmentation mask.",
                    bbox_idx=i,
                ))
                continue

            # Check if mask fills a reasonable portion of the bbox
            fill_ratio = mask_area / bbox.area if bbox.area > 0 else 0
            if fill_ratio < 0.05:
                flags.append(QualityFlag(
                    frame_idx=frame_idx,
                    flag_type="sparse_mask",
                    severity="warning",
                    message=(
                        f"{bbox.label} mask fills only {fill_ratio:.1%} "
                        f"of its bounding box. Segmentation might be poor."
                    ),
                    bbox_idx=i,
                ))

        return flags

    def generate_report(
        self, all_flags: list
    ) -> dict:
        """Generate a summary report of all quality issues.

        Returns:
            Dict with counts by type, severity, and overall quality score.
        """
        if not all_flags:
            return {
                "total_issues": 0,
                "quality_score": 1.0,
                "by_type": {},
                "by_severity": {},
            }

        by_type = {}
        by_severity = {"warning": 0, "error": 0}

        for flag in all_flags:
            by_type[flag.flag_type] = by_type.get(flag.flag_type, 0) + 1
            by_severity[flag.severity] = by_severity.get(flag.severity, 0) + 1

        # Quality score: 1.0 = perfect, penalize errors more than warnings
        total = len(all_flags)
        error_penalty = by_severity["error"] * 0.1
        warning_penalty = by_severity["warning"] * 0.02
        quality_score = max(0, 1.0 - error_penalty - warning_penalty)

        report = {
            "total_issues": total,
            "quality_score": round(quality_score, 3),
            "by_type": by_type,
            "by_severity": by_severity,
        }

        logger.info(
            f"Quality report: {total} issues, "
            f"score: {quality_score:.2f}, "
            f"errors: {by_severity['error']}, "
            f"warnings: {by_severity['warning']}"
        )
        return report
