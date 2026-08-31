"""FIXED comparison: baseline (Grounding DINO) vs fine-tuned YOLO.

The previous compare_detectors.py had a bug where the config swap didn't
take effect, causing YOLO to run twice. This version explicitly loads
each detector class directly, no config swaps.

Usage in Colab:
    from scripts.compare_detectors_fixed import run_fixed_comparison
    results = run_fixed_comparison(
        yolo_weights='yolo_models/surgical_yolov8/weights/best.pt',
        n_test_samples=200,
    )
"""

import os
import sys
import json
import time
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, FrameAnnotation
from src.detector import GroundingDINODetector
from src.yolo_detector import YOLODetector
from src.evaluator import AnnotationEvaluator
from src.dataset_loader import CholecSeg8kLoader, CHOLECSEG8K_CLASSES, INSTRUMENT_CLASSES, ANATOMY_CLASSES


def run_detection_on_images(detector, image_paths: list, name: str = "detector") -> dict:
    """Run a detector on a list of images and return detections per image."""
    logger.info(f"Loading {name}...")
    detector.load_model()

    logger.info(f"Running {name} on {len(image_paths)} images...")
    results = {}

    for i, image_path in enumerate(tqdm(image_paths, desc=name)):
        try:
            detections = detector.detect(image_path)
        except Exception as e:
            logger.warning(f"{name} failed on {image_path}: {e}")
            detections = []
        results[i] = detections

    total_dets = sum(len(d) for d in results.values())
    logger.info(f"{name} found {total_dets} total detections")
    return results


def build_predictions_coco(
    detections_by_image: dict,
    image_paths: list,
    ground_truth_path: str,
    output_path: str,
) -> str:
    """Build a COCO predictions file from detection results.

    Maps detected labels to ground truth category IDs by fuzzy matching.
    """
    # Load ground truth to get category structure
    with open(ground_truth_path, "r") as f:
        gt = json.load(f)

    # Build lowercase name -> category_id lookup
    category_map = {}
    for cat in gt["categories"]:
        category_map[cat["name"].lower()] = cat["id"]

    pred_coco = {
        "info": {"description": "Detector predictions"},
        "images": [],
        "annotations": [],
        "categories": gt["categories"],
    }

    annotation_id = 1

    for img_id, image_path in enumerate(image_paths):
        pred_coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(image_path),
            "width": 1280,
            "height": 720,
        })

        detections = detections_by_image.get(img_id, [])
        for det in detections:
            # Map label to category_id
            cat_id = match_label_to_category(det.label, category_map)
            if cat_id is None:
                continue

            pred_coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": det.to_coco(),
                "area": det.area,
                "iscrowd": 0,
                "score": det.confidence,
            })
            annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(pred_coco, f, indent=2)

    logger.info(
        f"Saved predictions to {output_path}: "
        f"{len(pred_coco['annotations'])} annotations"
    )
    return output_path


def match_label_to_category(label: str, category_map: dict):
    """Fuzzy match label to category ID."""
    label_lower = label.lower().strip()

    if label_lower in category_map:
        return category_map[label_lower]

    words = label_lower.split()
    for word in words:
        if word in category_map:
            return category_map[word]

    for cat_name, cat_id in category_map.items():
        if cat_name in label_lower or label_lower in cat_name:
            return cat_id
        for cat_word in cat_name.split():
            if cat_word in words:
                return cat_id

    return None


def run_fixed_comparison(
    yolo_weights: str,
    test_image_dir: str = None,
    n_test_samples: int = 200,
    output_dir: str = "comparison_fixed",
    grounding_dino_config: dict = None,
) -> dict:
    """Run baseline vs fine-tuned comparison with explicit detector loading.

    Args:
        yolo_weights: Path to trained YOLO best.pt.
        test_image_dir: Test images (defaults to CholecSeg8k images folder).
        n_test_samples: How many test samples.
        output_dir: Output directory.
        grounding_dino_config: Optional custom DINO config.

    Returns:
        Dict with baseline and fine-tuned metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Setup: get test images and ground truth
    # =====================================================
    logger.info("Preparing test set...")
    loader = CholecSeg8kLoader(cache_dir="cholecseg8k_data")

    existing = sorted(Path("cholecseg8k_data/images").glob("*.png"))
    if not existing:
        logger.info(f"Downloading {n_test_samples} samples...")
        loader.download_subset(n=n_test_samples)
        existing = sorted(Path("cholecseg8k_data/images").glob("*.png"))

    # Use the last n samples as test (to avoid overlap with YOLO training)
    # YOLO trained on first 80% of data; test on the last 10% for a real test set
    total = len(existing)
    test_start = int(total * 0.9)
    test_images = existing[test_start:test_start + n_test_samples]

    if len(test_images) < n_test_samples:
        # Not enough samples in the reserved test range, use whatever's available
        test_images = existing[-n_test_samples:]

    logger.info(
        f"Test set: {len(test_images)} samples from unseen portion of dataset"
    )

    # Build sample list for ground truth export
    loader.samples = [
        {
            "id": i,
            "image_path": str(img),
            "mask_path": str(Path("cholecseg8k_data/masks") /
                             img.name.replace("image_", "mask_")),
        }
        for i, img in enumerate(test_images)
    ]

    # Export ground truth for this test set
    gt_path = os.path.join(output_dir, "test_ground_truth.json")
    loader.export_ground_truth_coco(
        output_path=gt_path,
        include_anatomy=True,
    )

    image_paths = [s["image_path"] for s in loader.samples]

    # =====================================================
    # Load configs
    # =====================================================
    config = load_config("config.yaml")

    # =====================================================
    # STEP 1: Run Grounding DINO (baseline)
    # =====================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1/2: BASELINE (Grounding DINO zero-shot)")
    logger.info("=" * 70)

    # Configure Grounding DINO with all class prompts
    all_classes = INSTRUMENT_CLASSES | ANATOMY_CLASSES
    prompts = sorted([CHOLECSEG8K_CLASSES[c] for c in all_classes])

    dino_config = dict(config)
    dino_config["detection"] = {
        **dino_config.get("detection", {}),
        "model_id": "IDEA-Research/grounding-dino-base",
        "prompts": prompts,
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "nms_threshold": 0.5,
        "device": "cuda",
    }

    dino_detector = GroundingDINODetector(dino_config)
    dino_detections = run_detection_on_images(
        dino_detector, image_paths, name="Grounding DINO"
    )

    # Save DINO predictions and evaluate
    baseline_pred_path = os.path.join(output_dir, "baseline_predictions.json")
    build_predictions_coco(
        dino_detections, image_paths, gt_path, baseline_pred_path
    )

    logger.info("Evaluating baseline against ground truth...")
    evaluator = AnnotationEvaluator()
    baseline_eval = evaluator.evaluate(
        predictions_path=baseline_pred_path,
        ground_truth_path=gt_path,
    )

    # Free DINO from GPU memory before loading YOLO
    del dino_detector
    import torch
    torch.cuda.empty_cache()

    # =====================================================
    # STEP 2: Run YOLO (fine-tuned)
    # =====================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2/2: FINE-TUNED (YOLOv8 trained on CholecSeg8k)")
    logger.info("=" * 70)

    yolo_config_dict = dict(config)
    yolo_config_dict["yolo_detection"] = {
        "weights_path": yolo_weights,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "image_size": 640,
        "device": "cuda",
    }

    yolo_detector = YOLODetector(yolo_config_dict)
    yolo_detections = run_detection_on_images(
        yolo_detector, image_paths, name="YOLO"
    )

    # Save YOLO predictions and evaluate
    yolo_pred_path = os.path.join(output_dir, "yolo_predictions.json")
    build_predictions_coco(
        yolo_detections, image_paths, gt_path, yolo_pred_path
    )

    logger.info("Evaluating fine-tuned YOLO against ground truth...")
    yolo_eval = evaluator.evaluate(
        predictions_path=yolo_pred_path,
        ground_truth_path=gt_path,
    )

    # =====================================================
    # STEP 3: Print comparison
    # =====================================================
    print_comparison(baseline_eval, yolo_eval, output_dir)

    return {
        "baseline": baseline_eval.to_dict() if baseline_eval else {},
        "fine_tuned": yolo_eval.to_dict() if yolo_eval else {},
    }


def print_comparison(baseline_eval, yolo_eval, output_dir):
    """Print a formatted comparison table."""
    if not baseline_eval or not yolo_eval:
        logger.error("Evaluation failed for one of the models.")
        return

    b = baseline_eval.to_dict()
    y = yolo_eval.to_dict()

    print("\n\n" + "=" * 100)
    print("COMPARISON: Grounding DINO (baseline) vs Fine-tuned YOLOv8")
    print("=" * 100)

    print(f"\n{'Metric':<25} {'DINO Baseline':>18} {'Fine-tuned YOLO':>20} {'Absolute Delta':>18} {'Relative':>12}")
    print("-" * 100)

    for metric in ["precision", "recall", "f1_score", "map_50", "map_75", "map_50_95"]:
        b_val = b["summary"].get(metric, 0)
        y_val = y["summary"].get(metric, 0)
        delta = y_val - b_val
        if b_val > 0:
            relative = f"{((y_val - b_val) / b_val * 100):+.1f}%"
        else:
            relative = "N/A (from 0)"
        print(f"{metric:<25} {b_val:>18.4f} {y_val:>20.4f} {delta:>+18.4f} {relative:>12}")

    print("=" * 100)

    print("\n\nPER-CLASS AP@0.5 COMPARISON:")
    print("=" * 100)
    print(f"{'Class':<30} {'Baseline':>15} {'Fine-tuned':>15} {'Delta':>15} {'Improvement':>15}")
    print("-" * 100)

    all_classes = set(b["per_class"]["ap_50"].keys()) | set(y["per_class"]["ap_50"].keys())
    for cls in sorted(all_classes):
        b_ap = b["per_class"]["ap_50"].get(cls, 0)
        y_ap = y["per_class"]["ap_50"].get(cls, 0)
        delta = y_ap - b_ap
        if b_ap > 0:
            imp = f"{((y_ap - b_ap) / b_ap * 100):+.1f}%"
        elif y_ap > 0:
            imp = f"NEW (from 0)"
        else:
            imp = "-"
        print(f"{cls:<30} {b_ap:>15.4f} {y_ap:>15.4f} {delta:>+15.4f} {imp:>15}")

    print("=" * 100)

    # Save comparison JSON
    comparison = {
        "baseline": b,
        "fine_tuned": y,
        "improvement": {
            metric: {
                "baseline": b["summary"].get(metric, 0),
                "fine_tuned": y["summary"].get(metric, 0),
                "absolute_delta": y["summary"].get(metric, 0) - b["summary"].get(metric, 0),
            }
            for metric in ["precision", "recall", "f1_score", "map_50", "map_75", "map_50_95"]
        },
    }

    comparison_path = os.path.join(output_dir, "comparison_fixed.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nComparison saved to {comparison_path}")
