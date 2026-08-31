"""Unit tests for the surgical video annotation pipeline.

These tests validate the non-ML components (utils, tracking, validation,
export) without requiring GPU or model weights. ML components (detector,
segmentor) are tested with mock outputs.
"""

import pytest
import numpy as np
import json
import os
import tempfile

from src.utils import BBox, FrameAnnotation, compute_iou, load_config
from src.tracker import InstrumentTracker, Track
from src.validator import AnnotationValidator, QualityFlag
from src.segmentor import SAM2Segmentor


# =====================================================
# BBox Tests
# =====================================================

class TestBBox:
    def test_basic_properties(self):
        bbox = BBox(x_min=10, y_min=20, x_max=110, y_max=120,
                    confidence=0.9, label="scalpel")
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.area == 10000
        assert bbox.center == (60.0, 70.0)

    def test_to_coco_format(self):
        bbox = BBox(x_min=10, y_min=20, x_max=110, y_max=120,
                    confidence=0.9, label="scalpel")
        coco = bbox.to_coco()
        assert coco == [10, 20, 100, 100]

    def test_to_xyxy_format(self):
        bbox = BBox(x_min=10, y_min=20, x_max=110, y_max=120,
                    confidence=0.9, label="scalpel")
        xyxy = bbox.to_xyxy()
        assert xyxy == [10, 20, 110, 120]

    def test_zero_area_bbox(self):
        bbox = BBox(x_min=50, y_min=50, x_max=50, y_max=50,
                    confidence=0.5, label="needle")
        assert bbox.area == 0


# =====================================================
# IoU Tests
# =====================================================

class TestIoU:
    def test_identical_boxes(self):
        box = BBox(0, 0, 100, 100, 0.9, "scalpel")
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        box_a = BBox(0, 0, 50, 50, 0.9, "scalpel")
        box_b = BBox(100, 100, 150, 150, 0.9, "forceps")
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = BBox(0, 0, 100, 100, 0.9, "scalpel")
        box_b = BBox(50, 50, 150, 150, 0.9, "scalpel")
        iou = compute_iou(box_a, box_b)
        # Intersection: 50*50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert abs(iou - 2500 / 17500) < 0.01

    def test_contained_box(self):
        outer = BBox(0, 0, 100, 100, 0.9, "scalpel")
        inner = BBox(25, 25, 75, 75, 0.9, "scalpel")
        iou = compute_iou(outer, inner)
        # Intersection = inner area = 2500, Union = 10000
        assert abs(iou - 2500 / 10000) < 0.01


# =====================================================
# Tracker Tests
# =====================================================

class TestTracker:
    def get_test_config(self):
        return {
            "tracking": {
                "iou_threshold": 0.3,
                "max_age": 5,
                "min_hits": 2,
                "max_tracks": 10,
            }
        }

    def test_first_frame_creates_tracks(self):
        tracker = InstrumentTracker(self.get_test_config())
        detections = [
            BBox(10, 10, 50, 50, 0.9, "scalpel"),
            BBox(100, 100, 150, 150, 0.8, "forceps"),
        ]
        result = tracker.update(0, detections)
        assert len(result) == 2
        assert result[0].track_id == 1
        assert result[1].track_id == 2

    def test_matching_across_frames(self):
        tracker = InstrumentTracker(self.get_test_config())

        # Frame 0
        det_f0 = [BBox(10, 10, 50, 50, 0.9, "scalpel")]
        tracker.update(0, det_f0)

        # Frame 1: same instrument, slightly moved
        det_f1 = [BBox(15, 15, 55, 55, 0.85, "scalpel")]
        result = tracker.update(1, det_f1)

        assert result[0].track_id == 1  # Same track ID

    def test_new_track_for_new_instrument(self):
        tracker = InstrumentTracker(self.get_test_config())

        det_f0 = [BBox(10, 10, 50, 50, 0.9, "scalpel")]
        tracker.update(0, det_f0)

        # Frame 1: different location, same label
        det_f1 = [
            BBox(15, 15, 55, 55, 0.9, "scalpel"),   # Matched
            BBox(200, 200, 250, 250, 0.8, "forceps"), # New
        ]
        result = tracker.update(1, det_f1)
        track_ids = {d.track_id for d in result}
        assert len(track_ids) == 2

    def test_stale_tracks_removed(self):
        config = self.get_test_config()
        config["tracking"]["max_age"] = 2
        tracker = InstrumentTracker(config)

        det_f0 = [BBox(10, 10, 50, 50, 0.9, "scalpel")]
        tracker.update(0, det_f0)

        # No detections for several frames
        tracker.update(1, [])
        tracker.update(2, [])
        tracker.update(3, [])

        assert len(tracker.tracks) == 0

    def test_empty_detections(self):
        tracker = InstrumentTracker(self.get_test_config())
        result = tracker.update(0, [])
        assert result == []

    def test_reset(self):
        tracker = InstrumentTracker(self.get_test_config())
        tracker.update(0, [BBox(10, 10, 50, 50, 0.9, "scalpel")])
        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1


# =====================================================
# Validator Tests
# =====================================================

class TestValidator:
    def get_test_config(self):
        return {
            "quality": {
                "min_confidence": 0.35,
                "min_bbox_area": 100,
                "max_bbox_area_ratio": 0.8,
                "overlap_iou_threshold": 0.7,
                "temporal_consistency_window": 5,
                "flag_sudden_disappearance": True,
            },
            "video": {
                "resize_width": 1280,
                "resize_height": 720,
            },
        }

    def test_low_confidence_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="test.jpg",
            bboxes=[BBox(10, 10, 50, 50, 0.2, "scalpel")],
        )
        flags = validator.validate_frame(ann)
        assert any(f.flag_type == "low_confidence" for f in flags)

    def test_high_confidence_not_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="test.jpg",
            bboxes=[BBox(10, 10, 110, 110, 0.95, "scalpel")],
        )
        flags = validator.validate_frame(ann)
        assert not any(f.flag_type == "low_confidence" for f in flags)

    def test_small_bbox_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="test.jpg",
            bboxes=[BBox(10, 10, 15, 15, 0.9, "needle")],  # 25px area
        )
        flags = validator.validate_frame(ann)
        assert any(f.flag_type == "small_bbox" for f in flags)

    def test_large_bbox_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="test.jpg",
            bboxes=[BBox(0, 0, 1200, 700, 0.9, "error")],  # Almost full frame
        )
        flags = validator.validate_frame(ann)
        assert any(f.flag_type == "large_bbox" for f in flags)

    def test_overlap_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="test.jpg",
            bboxes=[
                BBox(10, 10, 110, 110, 0.9, "scalpel"),
                BBox(15, 15, 115, 115, 0.85, "forceps"),  # 90% overlap
            ],
        )
        flags = validator.validate_frame(ann)
        assert any(f.flag_type == "overlap" for f in flags)

    def test_single_frame_track_flagged(self):
        validator = AnnotationValidator(self.get_test_config())
        ann1 = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="t.jpg",
            bboxes=[BBox(10, 10, 50, 50, 0.9, "scalpel", track_id=1)],
        )
        ann2 = FrameAnnotation(
            frame_idx=1, timestamp_sec=0.5, image_path="t.jpg",
            bboxes=[],
        )
        flags = validator.validate_sequence([ann1, ann2])
        assert any(f.flag_type == "single_frame_track" for f in flags)

    def test_quality_report(self):
        validator = AnnotationValidator(self.get_test_config())
        flags = [
            QualityFlag(0, "low_confidence", "warning", "test"),
            QualityFlag(0, "empty_mask", "error", "test"),
        ]
        report = validator.generate_report(flags)
        assert report["total_issues"] == 2
        assert report["by_severity"]["warning"] == 1
        assert report["by_severity"]["error"] == 1
        assert report["quality_score"] < 1.0

    def test_empty_report(self):
        validator = AnnotationValidator(self.get_test_config())
        report = validator.generate_report([])
        assert report["quality_score"] == 1.0
        assert report["total_issues"] == 0


# =====================================================
# Mask Conversion Tests
# =====================================================

class TestMaskConversion:
    def test_mask_to_polygon(self):
        # Create a simple circular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1  # Square region

        polygons = SAM2Segmentor.mask_to_polygon(mask)
        assert len(polygons) > 0
        assert all(len(p) >= 6 for p in polygons)  # Min 3 points

    def test_empty_mask_to_polygon(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = SAM2Segmentor.mask_to_polygon(mask)
        assert len(polygons) == 0

    def test_mask_to_rle(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1

        rle = SAM2Segmentor.mask_to_rle(mask)
        assert "counts" in rle
        assert "size" in rle
        assert rle["size"] == [10, 10]


# =====================================================
# Config Tests
# =====================================================

class TestConfig:
    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
video:
  frame_extraction_fps: 2
  max_frames: null
  resize_width: 1280
  resize_height: 720
  supported_formats: [".mp4"]
detection:
  model_id: "test"
  prompts: ["scalpel"]
  box_threshold: 0.3
  text_threshold: 0.25
  nms_threshold: 0.5
  device: "cpu"
segmentation:
  model_id: "test"
  multimask_output: false
  mask_threshold: 0.5
  device: "cpu"
export:
  format: "coco"
  output_dir: "output"
  save_visualizations: false
  save_masks: false
  visualization_alpha: 0.4
            """)
            f.flush()
            config = load_config(f.name)
            assert config["video"]["frame_extraction_fps"] == 2
            os.unlink(f.name)

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
