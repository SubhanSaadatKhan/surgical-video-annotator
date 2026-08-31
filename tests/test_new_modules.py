"""Unit tests for the new modules: temporal sampler, evaluator, active learning."""

import pytest
import json
import os
import tempfile
import numpy as np

from src.utils import BBox, FrameAnnotation
from src.temporal_sampler import TemporalSampler, FrameScore
from src.evaluator import AnnotationEvaluator
from src.active_learning import (
    UncertaintySampler,
    CorrectionStore,
    ActiveLearningOrchestrator,
    ActiveLearningIteration,
)


# =====================================================
# Temporal Sampler Tests
# =====================================================

class TestTemporalSampler:
    def get_config(self, mode="adaptive"):
        return {
            "video": {"frame_extraction_fps": 2},
            "temporal_sampling": {
                "mode": mode,
                "min_interval_sec": 0.2,
                "max_interval_sec": 2.0,
                "motion_weight": 0.6,
                "scene_change_weight": 0.4,
                "scene_change_threshold": 0.35,
                "motion_threshold": 0.05,
            },
        }

    def make_scores(self, count=100):
        """Generate synthetic frame scores."""
        scores = []
        for i in range(count):
            # Simulate: mostly quiet with occasional activity bursts
            if i % 20 == 0:
                motion = 0.3
                scene_change = 0.5
                is_boundary = True
            elif 10 <= i % 20 <= 15:
                motion = 0.15
                scene_change = 0.1
                is_boundary = False
            else:
                motion = 0.02
                scene_change = 0.05
                is_boundary = False

            scores.append(FrameScore(
                frame_idx=i,
                timestamp_sec=i / 30,
                motion_score=motion,
                scene_change_score=scene_change,
                combined_score=motion * 0.6 + scene_change * 0.4,
                is_scene_boundary=is_boundary,
            ))
        return scores

    def test_fixed_mode_regular_intervals(self):
        sampler = TemporalSampler(self.get_config("fixed"))
        scores = self.make_scores(60)
        # Fixed at 2 fps on 30 fps video -> every 15 frames
        selected = sampler.select_frames(scores, video_fps=30)
        # Should be around 60/15 = 4 frames
        assert 3 <= len(selected) <= 5

    def test_adaptive_mode_prefers_active_frames(self):
        sampler = TemporalSampler(self.get_config("adaptive"))
        scores = self.make_scores(100)
        selected = sampler.select_frames(scores, video_fps=30)

        # Should always include scene boundaries
        boundaries = [s for s in scores if s.is_scene_boundary]
        selected_boundaries = [s for s in selected if s.is_scene_boundary]
        assert len(selected_boundaries) == len(boundaries)

    def test_keyframe_mode_only_important_frames(self):
        sampler = TemporalSampler(self.get_config("keyframe"))
        scores = self.make_scores(100)
        selected = sampler.select_frames(scores, video_fps=30)

        # Should select fewer frames than adaptive
        adaptive_sampler = TemporalSampler(self.get_config("adaptive"))
        adaptive_selected = adaptive_sampler.select_frames(scores, video_fps=30)

        assert len(selected) <= len(adaptive_selected)

    def test_empty_scores(self):
        sampler = TemporalSampler(self.get_config("adaptive"))
        selected = sampler.select_frames([], video_fps=30)
        assert selected == []


# =====================================================
# Evaluator Tests
# =====================================================

class TestEvaluator:
    def make_coco(self, annotations_data):
        """Build a minimal COCO structure."""
        return {
            "info": {},
            "images": [
                {"id": 1, "file_name": "1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "2.jpg", "width": 640, "height": 480},
            ],
            "categories": [
                {"id": 1, "name": "scalpel"},
                {"id": 2, "name": "forceps"},
            ],
            "annotations": annotations_data,
        }

    def make_ann(self, ann_id, img_id, cat_id, bbox, score=1.0):
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "score": score,
        }

    def test_perfect_prediction(self):
        """When predictions match ground truth exactly, precision/recall = 1."""
        gt = self.make_coco([
            self.make_ann(1, 1, 1, [10, 10, 50, 50]),
            self.make_ann(2, 2, 2, [100, 100, 60, 60]),
        ])
        pred = self.make_coco([
            self.make_ann(1, 1, 1, [10, 10, 50, 50], score=0.9),
            self.make_ann(2, 2, 2, [100, 100, 60, 60], score=0.85),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.json")
            pred_path = os.path.join(tmp, "pred.json")
            with open(gt_path, "w") as f:
                json.dump(gt, f)
            with open(pred_path, "w") as f:
                json.dump(pred, f)

            evaluator = AnnotationEvaluator()
            result = evaluator.evaluate(pred_path, gt_path)

            assert result.precision == 1.0
            assert result.recall == 1.0
            assert result.f1_score == 1.0

    def test_all_false_positives(self):
        """Predictions that don't match anything -> precision = 0."""
        gt = self.make_coco([self.make_ann(1, 1, 1, [10, 10, 50, 50])])
        pred = self.make_coco([
            self.make_ann(1, 1, 1, [400, 400, 50, 50], score=0.9),  # Wrong location
        ])

        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.json")
            pred_path = os.path.join(tmp, "pred.json")
            with open(gt_path, "w") as f:
                json.dump(gt, f)
            with open(pred_path, "w") as f:
                json.dump(pred, f)

            evaluator = AnnotationEvaluator()
            result = evaluator.evaluate(pred_path, gt_path)

            assert result.precision == 0.0
            assert result.true_positives == 0
            assert result.false_positives == 1
            assert result.false_negatives == 1

    def test_partial_precision_recall(self):
        """Half predictions correct, half missed -> 50% both."""
        gt = self.make_coco([
            self.make_ann(1, 1, 1, [10, 10, 50, 50]),
            self.make_ann(2, 2, 2, [100, 100, 60, 60]),
        ])
        # Only one correct prediction, one wrong
        pred = self.make_coco([
            self.make_ann(1, 1, 1, [10, 10, 50, 50], score=0.9),   # TP
            self.make_ann(2, 1, 2, [500, 500, 10, 10], score=0.8), # FP
        ])

        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.json")
            pred_path = os.path.join(tmp, "pred.json")
            with open(gt_path, "w") as f:
                json.dump(gt, f)
            with open(pred_path, "w") as f:
                json.dump(pred, f)

            evaluator = AnnotationEvaluator()
            result = evaluator.evaluate(pred_path, gt_path)

            assert result.precision == 0.5    # 1 TP out of 2 predictions
            assert result.recall == 0.5       # 1 TP out of 2 ground truths


# =====================================================
# Active Learning Tests
# =====================================================

class TestUncertaintySampler:
    def get_config(self):
        return {
            "active_learning": {
                "max_samples_per_iteration": 10,
                "uncertainty_threshold": 0.4,
                "low_conf_threshold": 0.5,
                "high_conf_threshold": 0.8,
            }
        }

    def test_high_confidence_not_flagged(self):
        sampler = UncertaintySampler(self.get_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="t.jpg",
            bboxes=[BBox(10, 10, 100, 100, 0.95, "scalpel")],
        )
        scores = sampler.score_predictions([ann])
        assert len(scores) == 0  # High confidence, no flag

    def test_medium_confidence_flagged(self):
        sampler = UncertaintySampler(self.get_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="t.jpg",
            bboxes=[BBox(10, 10, 100, 100, 0.65, "scalpel")],  # Middle of uncertain range
        )
        scores = sampler.score_predictions([ann])
        assert len(scores) == 1
        assert scores[0].uncertainty_type == "boundary"

    def test_low_confidence_flagged(self):
        sampler = UncertaintySampler(self.get_config())
        ann = FrameAnnotation(
            frame_idx=0, timestamp_sec=0.0, image_path="t.jpg",
            bboxes=[BBox(10, 10, 100, 100, 0.3, "forceps")],
        )
        scores = sampler.score_predictions([ann])
        assert len(scores) == 1
        assert scores[0].uncertainty_type == "low_confidence"

    def test_diversity_in_selection(self):
        sampler = UncertaintySampler(self.get_config())
        annotations = []
        for frame_idx in range(20):
            bboxes = [
                BBox(10, 10, 100, 100, 0.6, "scalpel"),
                BBox(200, 200, 250, 250, 0.55, "forceps"),
            ]
            annotations.append(FrameAnnotation(
                frame_idx=frame_idx, timestamp_sec=frame_idx * 0.5,
                image_path=f"{frame_idx}.jpg", bboxes=bboxes,
            ))

        scores = sampler.score_predictions(annotations)
        selected = sampler.select_for_review(scores)

        assert len(selected) <= 10
        # Should include multiple frames (diversity)
        unique_frames = {s.frame_idx for s in selected}
        assert len(unique_frames) > 1


class TestCorrectionStore:
    def test_save_and_load_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CorrectionStore(base_dir=tmp)

            iteration = ActiveLearningIteration(
                iteration_id=1,
                timestamp="2026-08-13T10:00:00",
                videos_processed=["video1.mp4"],
                samples_reviewed=50,
                corrections_received=15,
            )
            corrections = [
                {
                    "frame_idx": 0,
                    "image_path": "frame_0.jpg",
                    "original_bbox": {
                        "bbox": [10, 10, 50, 50], "label": "scalpel", "area": 2500
                    },
                    "corrected_bbox": None,
                    "action": "approved",
                    "notes": "Correct detection",
                },
                {
                    "frame_idx": 5,
                    "image_path": "frame_5.jpg",
                    "original_bbox": {
                        "bbox": [100, 100, 60, 60], "label": "forceps", "area": 3600
                    },
                    "corrected_bbox": {
                        "bbox": [95, 95, 70, 70], "label": "forceps", "area": 4900
                    },
                    "action": "corrected",
                    "notes": "Bbox slightly off",
                },
            ]

            store.save_iteration(iteration, corrections)

            # Verify
            history = store.load_iteration_history()
            assert len(history) == 1
            assert history[0]["iteration_id"] == 1

            all_corrections = store.load_all_corrections()
            assert len(all_corrections) == 2

    def test_next_iteration_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CorrectionStore(base_dir=tmp)
            assert store.get_next_iteration_id() == 1

            # After saving one
            iteration = ActiveLearningIteration(
                iteration_id=1,
                timestamp="2026-08-13T10:00:00",
                videos_processed=[],
                samples_reviewed=0,
                corrections_received=0,
            )
            store.save_iteration(iteration, [])
            assert store.get_next_iteration_id() == 2


class TestActiveLearningOrchestrator:
    def get_config(self):
        return {
            "active_learning": {
                "enabled": True,
                "correction_dir": None,  # Set per test
                "max_samples_per_iteration": 5,
                "uncertainty_threshold": 0.4,
                "low_conf_threshold": 0.5,
                "high_conf_threshold": 0.8,
            }
        }

    def test_review_queue_generation(self):
        config = self.get_config()
        with tempfile.TemporaryDirectory() as tmp:
            config["active_learning"]["correction_dir"] = tmp
            orchestrator = ActiveLearningOrchestrator(config)

            annotations = [
                FrameAnnotation(
                    frame_idx=0, timestamp_sec=0.0, image_path="t.jpg",
                    bboxes=[
                        BBox(10, 10, 100, 100, 0.6, "scalpel"),   # Uncertain
                        BBox(200, 200, 300, 300, 0.95, "forceps"),  # Certain
                    ],
                )
            ]

            queue = orchestrator.get_review_queue(annotations)
            assert len(queue) == 1  # Only the uncertain one
            assert queue[0].label == "scalpel"

    def test_improvement_tracking(self):
        config = self.get_config()
        with tempfile.TemporaryDirectory() as tmp:
            config["active_learning"]["correction_dir"] = tmp
            orchestrator = ActiveLearningOrchestrator(config)

            # First iteration
            orchestrator.record_iteration(
                videos=["v1.mp4"],
                corrections=[],
                metrics_before={"summary": {"map_50": 0.6, "f1_score": 0.65}},
                metrics_after={"summary": {"map_50": 0.7, "f1_score": 0.72}},
            )

            # Second iteration
            orchestrator.record_iteration(
                videos=["v2.mp4"],
                corrections=[],
                metrics_before={"summary": {"map_50": 0.7, "f1_score": 0.72}},
                metrics_after={"summary": {"map_50": 0.78, "f1_score": 0.8}},
            )

            summary = orchestrator.get_improvement_summary()
            assert summary["iterations_completed"] == 2
            assert len(summary["map_timeline"]) == 2
            assert summary["overall_map_improvement"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
