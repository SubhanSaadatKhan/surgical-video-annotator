"""Multi-object tracker: maintains consistent instrument IDs across video frames.

Without tracking, each frame's detections are independent. Frame 1 might detect
"scalpel" and frame 2 also detects "scalpel," but we don't know if it's the same
scalpel. Tracking solves this by assigning persistent IDs to detected objects and
matching them across frames using IoU (spatial overlap).

This uses a simple IoU-based tracker (similar to SORT), which works well for
surgical videos because:
    - Instruments move slowly between frames
    - We extract frames at low fps (1-5), so movement is minimal
    - Instruments rarely overlap each other
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.utils import BBox, compute_iou


@dataclass
class Track:
    """A tracked object across multiple frames."""
    track_id: int
    label: str
    bboxes: dict = field(default_factory=dict)    # frame_idx -> BBox
    age: int = 0                                   # Frames since last detection
    hits: int = 1                                  # Total detection count
    confirmed: bool = False                        # Met min_hits threshold

    @property
    def last_bbox(self) -> Optional[BBox]:
        """Get the most recent bounding box for this track."""
        if not self.bboxes:
            return None
        last_frame = max(self.bboxes.keys())
        return self.bboxes[last_frame]

    @property
    def first_frame(self) -> int:
        return min(self.bboxes.keys()) if self.bboxes else -1

    @property
    def last_frame(self) -> int:
        return max(self.bboxes.keys()) if self.bboxes else -1


class InstrumentTracker:
    """IoU-based multi-object tracker for surgical instruments.

    Matches detections across frames using spatial overlap (IoU).
    Assigns persistent track IDs so that "scalpel in frame 10" and
    "scalpel in frame 11" get the same ID if they overlap sufficiently.
    """

    def __init__(self, config: dict):
        tracking_config = config.get("tracking", {})
        self.iou_threshold = tracking_config.get("iou_threshold", 0.3)
        self.max_age = tracking_config.get("max_age", 15)
        self.min_hits = tracking_config.get("min_hits", 3)
        self.max_tracks = tracking_config.get("max_tracks", 50)

        self.tracks: list = []
        self.next_id: int = 1
        self.frame_count: int = 0

    def update(self, frame_idx: int, detections: list) -> list:
        """Match new detections to existing tracks and return updated detections.

        This is called once per frame with that frame's detections.

        Algorithm:
            1. Compute IoU between all current tracks and new detections
            2. Greedily match highest-IoU pairs above threshold
            3. Unmatched detections become new tracks
            4. Unmatched tracks increment their age (and get removed if too old)

        Args:
            frame_idx: Current frame index.
            detections: List of BBox objects detected in this frame.

        Returns:
            List of BBox objects with track_id assigned.
        """
        self.frame_count = frame_idx

        if not self.tracks:
            # First frame: create a new track for each detection
            return self._init_tracks(frame_idx, detections)

        if not detections:
            # No detections: age all tracks
            self._age_tracks()
            return []

        # Step 1: Build IoU cost matrix
        iou_matrix = self._compute_iou_matrix(detections)

        # Step 2: Greedy matching (highest IoU first)
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            self._greedy_match(iou_matrix)
        )

        # Step 3: Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            track = self.tracks[track_idx]
            det = detections[det_idx]
            det.track_id = track.track_id
            track.bboxes[frame_idx] = det
            track.age = 0
            track.hits += 1
            if track.hits >= self.min_hits:
                track.confirmed = True

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(self.tracks) < self.max_tracks:
                det = detections[det_idx]
                new_track = Track(
                    track_id=self.next_id,
                    label=det.label,
                    bboxes={frame_idx: det},
                )
                det.track_id = self.next_id
                self.tracks.append(new_track)
                self.next_id += 1

        # Step 5: Age unmatched tracks and remove stale ones
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1

        self._remove_stale_tracks()

        return detections

    def _init_tracks(self, frame_idx: int, detections: list) -> list:
        """Initialize tracks from first frame's detections."""
        for det in detections:
            track = Track(
                track_id=self.next_id,
                label=det.label,
                bboxes={frame_idx: det},
            )
            det.track_id = self.next_id
            self.tracks.append(track)
            self.next_id += 1
        return detections

    def _compute_iou_matrix(self, detections: list) -> np.ndarray:
        """Compute IoU matrix between all tracks and detections.

        Returns:
            Matrix of shape (num_tracks, num_detections) with IoU values.
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets))

        for t, track in enumerate(self.tracks):
            last_box = track.last_bbox
            if last_box is None:
                continue
            for d, det in enumerate(detections):
                # Only match if labels are compatible
                if track.label == det.label:
                    iou_matrix[t, d] = compute_iou(last_box, det)

        return iou_matrix

    def _greedy_match(
        self, iou_matrix: np.ndarray
    ) -> tuple:
        """Greedy matching: assign detections to tracks by highest IoU.

        Returns:
            matched_track_indices, matched_det_indices,
            unmatched_track_indices, unmatched_det_indices
        """
        matched_tracks = []
        matched_dets = []
        used_tracks = set()
        used_dets = set()

        # Flatten matrix and sort by IoU (descending)
        num_tracks, num_dets = iou_matrix.shape
        pairs = []
        for t in range(num_tracks):
            for d in range(num_dets):
                if iou_matrix[t, d] >= self.iou_threshold:
                    pairs.append((iou_matrix[t, d], t, d))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou, t, d in pairs:
            if t not in used_tracks and d not in used_dets:
                matched_tracks.append(t)
                matched_dets.append(d)
                used_tracks.add(t)
                used_dets.add(d)

        unmatched_tracks = [
            t for t in range(num_tracks) if t not in used_tracks
        ]
        unmatched_dets = [
            d for d in range(num_dets) if d not in used_dets
        ]

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    def _age_tracks(self) -> None:
        """Increment age for all tracks (called when frame has no detections)."""
        for track in self.tracks:
            track.age += 1
        self._remove_stale_tracks()

    def _remove_stale_tracks(self) -> None:
        """Remove tracks that haven't been seen for too long."""
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        removed = before - len(self.tracks)
        if removed > 0:
            logger.debug(f"Removed {removed} stale tracks")

    def get_confirmed_tracks(self) -> list:
        """Get all confirmed tracks (met min_hits threshold)."""
        return [t for t in self.tracks if t.confirmed]

    def get_track_summary(self) -> dict:
        """Get summary statistics for all tracks."""
        confirmed = self.get_confirmed_tracks()
        return {
            "total_tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed),
            "labels": list(set(t.label for t in confirmed)),
            "avg_track_length": (
                np.mean([t.hits for t in confirmed]) if confirmed else 0
            ),
        }

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
