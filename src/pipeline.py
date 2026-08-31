"""Main pipeline v2: end-to-end surgical video annotation with active learning.

Pipeline flow:
    Video
      -> Smart Temporal Sampling (adaptive frame selection)
      -> Frame Extraction (OpenCV)
      -> Grounding DINO Detection
      -> SAM 2 Segmentation (per-frame OR video propagation)
      -> Object Tracking (if not using video propagation)
      -> Quality Validation
      -> Confidence Filtering (auto-accept vs review)
      -> Uncertainty Sampling (active learning)
      -> COCO Export
      -> [Optional] Ground Truth Evaluation
      -> [Optional] Active Learning Iteration Recording
"""

import os
import time
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from src.utils import load_config, setup_logger, FrameAnnotation
from src.video_processor import VideoProcessor
from src.temporal_sampler import TemporalSampler
from src.detector import GroundingDINODetector
from src.segmentor import SAM2Segmentor
from src.tracker import InstrumentTracker
from src.validator import AnnotationValidator
from src.exporter import COCOExporter
from src.active_learning import ActiveLearningOrchestrator
from src.evaluator import AnnotationEvaluator


def create_detector(config: dict):
    """Factory function to create the right detector based on config.

    Supports:
        - "grounding_dino": Zero-shot Grounding DINO (baseline)
        - "yolo": Fine-tuned YOLOv8 (accurate, needs training)
        - "ensemble": Both models combined
    """
    detector_type = config.get("detector_type", "grounding_dino")

    if detector_type == "yolo":
        from src.yolo_detector import YOLODetector
        return YOLODetector(config)
    elif detector_type == "ensemble":
        from src.yolo_detector import EnsembleDetector
        return EnsembleDetector(config)
    else:
        return GroundingDINODetector(config)


class SurgicalAnnotationPipeline:
    """End-to-end pipeline with smart sampling, video propagation, and active learning."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        setup_logger()

        # Core components
        self.video_processor = VideoProcessor(self.config)
        self.temporal_sampler = TemporalSampler(self.config)
        self.detector = create_detector(self.config)
        self.segmentor = SAM2Segmentor(self.config)
        self.tracker = InstrumentTracker(self.config)
        self.validator = AnnotationValidator(self.config)
        self.exporter = COCOExporter(self.config)

        # Advanced components
        self.active_learning = ActiveLearningOrchestrator(self.config)
        self.evaluator = AnnotationEvaluator()

        # Video propagation is loaded lazily (heavy dependency)
        self.video_propagator = None

        # Config flags
        self.use_video_propagation = (
            self.config.get("video_propagation", {}).get("enabled", False)
        )
        self.use_smart_sampling = (
            self.config.get("temporal_sampling", {}).get("mode", "fixed") != "fixed"
        )
        self.active_learning_enabled = (
            self.config.get("active_learning", {}).get("enabled", False)
        )

        logger.info(
            f"Pipeline initialized | "
            f"smart_sampling={self.use_smart_sampling} | "
            f"video_propagation={self.use_video_propagation} | "
            f"active_learning={self.active_learning_enabled}"
        )

    def run(
        self,
        video_path: str,
        ground_truth_path: str = None,
    ) -> dict:
        """Run the full pipeline on a video.

        Args:
            video_path: Input surgical video.
            ground_truth_path: Optional COCO JSON with human ground truth
                (enables evaluation and active learning metric tracking).

        Returns:
            Dict with pipeline outputs and metrics.
        """
        start_time = time.time()
        video_name = Path(video_path).stem

        logger.info(f"Starting pipeline for: {video_name}")

        # =====================================================
        # Step 1: Smart temporal sampling
        # =====================================================
        if self.use_smart_sampling:
            logger.info("Step 1/6: Smart temporal sampling...")
            selected_frames_info = self.temporal_sampler.sample_video(video_path)
            frames_dir = os.path.join(
                self.config["export"]["output_dir"], video_name, "frames"
            )
            frame_info = self._extract_selected_frames(
                video_path, selected_frames_info, frames_dir
            )
        else:
            logger.info("Step 1/6: Fixed-rate frame extraction...")
            frames_dir = os.path.join(
                self.config["export"]["output_dir"], video_name, "frames"
            )
            frame_info = self.video_processor.extract_frames(video_path, frames_dir)

        logger.info(f"Extracted {len(frame_info)} frames")

        # =====================================================
        # Step 2: Detect instruments in each frame
        # =====================================================
        logger.info("Step 2/6: Detecting instruments (Grounding DINO)...")
        self.detector.load_model()

        all_annotations = []
        all_categories = set()
        detections_by_frame = {}

        for frame in tqdm(frame_info, desc="Detecting"):
            detections = self.detector.detect(frame["image_path"])
            detections_by_frame[frame["frame_idx"]] = detections

            annotation = FrameAnnotation(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame["timestamp_sec"],
                image_path=frame["image_path"],
                bboxes=detections,
            )
            all_annotations.append(annotation)
            for det in detections:
                all_categories.add(det.label)

        # =====================================================
        # Step 3: Segmentation
        # =====================================================
        all_masks = {}

        if self.use_video_propagation:
            logger.info("Step 3/6: SAM 2 video propagation (native tracking)...")
            all_masks = self._run_video_propagation(
                frames_dir, detections_by_frame, all_annotations
            )
        else:
            logger.info("Step 3/6: Per-frame segmentation + IoU tracking...")
            all_masks = self._run_per_frame_segmentation(all_annotations)

        # =====================================================
        # Step 4: Quality validation
        # =====================================================
        logger.info("Step 4/6: Validating annotation quality...")
        all_flags = []

        for annotation in all_annotations:
            frame_masks = all_masks.get(annotation.frame_idx, [])
            flags = self.validator.validate_frame(annotation, frame_masks)
            annotation.quality_flags = flags
            all_flags.extend(flags)

            error_count = sum(1 for f in flags if f.severity == "error")
            if error_count > 0:
                annotation.status = "needs_review"
            elif flags:
                annotation.status = "warning"
            else:
                annotation.status = "auto_approved"

        temporal_flags = self.validator.validate_sequence(all_annotations)
        all_flags.extend(temporal_flags)
        quality_report = self.validator.generate_report(all_flags)

        # =====================================================
        # Step 5: Active learning sample selection
        # =====================================================
        review_queue = []
        if self.active_learning_enabled:
            logger.info("Step 5/6: Selecting uncertain samples for review...")
            review_queue = self.active_learning.get_review_queue(all_annotations)
        else:
            logger.info("Step 5/6: Active learning disabled, skipping")

        # =====================================================
        # Step 6: Export
        # =====================================================
        logger.info("Step 6/6: Exporting annotations (COCO format)...")
        coco_path = self.exporter.export(
            annotations=all_annotations,
            masks_by_frame=all_masks,
            categories=list(all_categories),
            video_name=video_name,
        )

        # =====================================================
        # Optional: Ground truth evaluation
        # =====================================================
        evaluation_result = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            logger.info("Bonus: Evaluating against ground truth...")
            evaluation_result = self.evaluator.evaluate(
                predictions_path=coco_path,
                ground_truth_path=ground_truth_path,
            )
            eval_output = os.path.join(
                self.config["export"]["output_dir"],
                f"{video_name}_evaluation.json",
            )
            self.evaluator.save_report(evaluation_result, eval_output)

        # =====================================================
        # Pipeline stats
        # =====================================================
        elapsed = time.time() - start_time
        stats = {
            "video": video_name,
            "total_frames_in_video": self.video_processor.validate_video(video_path)[
                "total_frames"
            ],
            "extracted_frames": len(frame_info),
            "total_detections": sum(len(a.bboxes) for a in all_annotations),
            "unique_instruments": len(all_categories),
            "instrument_types": sorted(all_categories),
            "quality_score": quality_report["quality_score"],
            "total_quality_issues": quality_report["total_issues"],
            "auto_approved_frames": sum(
                1 for a in all_annotations if a.status == "auto_approved"
            ),
            "needs_review_frames": sum(
                1 for a in all_annotations if a.status in ("needs_review", "warning")
            ),
            "samples_flagged_for_active_learning": len(review_queue),
            "used_smart_sampling": self.use_smart_sampling,
            "used_video_propagation": self.use_video_propagation,
            "execution_time_sec": round(elapsed, 1),
            "fps_processed": round(len(frame_info) / elapsed, 1) if elapsed > 0 else 0,
        }

        if evaluation_result:
            stats["evaluation"] = evaluation_result.to_dict()["summary"]

        logger.info(
            f"Pipeline complete in {elapsed:.1f}s | "
            f"{stats['total_detections']} detections | "
            f"Quality: {quality_report['quality_score']:.2f}"
        )

        return {
            "coco_path": coco_path,
            "annotations": all_annotations,
            "masks": all_masks,
            "quality_report": quality_report,
            "review_queue": review_queue,
            "evaluation": evaluation_result,
            "stats": stats,
        }

    def _extract_selected_frames(
        self, video_path: str, selected_info: list, output_dir: str
    ) -> list:
        """Extract only the frames selected by the temporal sampler."""
        import cv2
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        selected_indices = {s["frame_idx"]: s for s in selected_info}

        extracted = []
        frame_idx = 0
        w = self.config["video"]["resize_width"]
        h = self.config["video"]["resize_height"]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in selected_indices:
                frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame_resized)

                info = selected_indices[frame_idx]
                extracted.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": info["timestamp_sec"],
                    "image_path": frame_path,
                    "motion_score": info.get("motion_score", 0),
                    "is_scene_boundary": info.get("is_scene_boundary", False),
                })

            frame_idx += 1

        cap.release()
        return extracted

    def _run_video_propagation(
        self, frames_dir: str, detections_by_frame: dict, annotations: list
    ) -> dict:
        """Use SAM 2 native video propagation for segmentation and tracking."""
        if self.video_propagator is None:
            from src.video_propagator import SAM2VideoPropagator
            self.video_propagator = SAM2VideoPropagator(self.config)

        # SAM 2 video propagation handles both segmentation and tracking
        propagation_results = self.video_propagator.annotate_with_detections(
            frames_dir=frames_dir,
            detections_by_frame=detections_by_frame,
        )

        # Assign object_id as track_id in our annotations
        all_masks = {}
        for annotation in annotations:
            frame_results = propagation_results.get(annotation.frame_idx, [])
            frame_masks = []
            for i, bbox in enumerate(annotation.bboxes):
                if i < len(frame_results):
                    result = frame_results[i]
                    bbox.track_id = result["object_id"]
                    frame_masks.append(result["mask"])
                else:
                    frame_masks.append(None)
            all_masks[annotation.frame_idx] = frame_masks

        return all_masks

    def _run_per_frame_segmentation(self, annotations: list) -> dict:
        """Fallback: per-frame SAM 2 + IoU tracker."""
        self.segmentor.load_model()
        all_masks = {}

        for annotation in tqdm(annotations, desc="Segmenting"):
            # Tracking
            if annotation.bboxes:
                annotation.bboxes = self.tracker.update(
                    annotation.frame_idx, annotation.bboxes
                )

            # Segmentation
            if not annotation.bboxes:
                all_masks[annotation.frame_idx] = []
                continue

            results = self.segmentor.segment_frame(
                annotation.image_path, annotation.bboxes
            )
            all_masks[annotation.frame_idx] = [r["mask"] for r in results]

        return all_masks


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Surgical Video Annotation Pipeline (with active learning)"
    )
    parser.add_argument("video", type=str, help="Path to the surgical video file")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Optional COCO JSON with ground truth annotations (enables evaluation)"
    )
    args = parser.parse_args()

    pipeline = SurgicalAnnotationPipeline(args.config)
    results = pipeline.run(args.video, ground_truth_path=args.ground_truth)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    for key, value in results["stats"].items():
        print(f"  {key}: {value}")
    print(f"\n  COCO annotations: {results['coco_path']}")
    if results.get("evaluation"):
        print(f"  Evaluation available in stats['evaluation']")
    print("=" * 60)


if __name__ == "__main__":
    main()
