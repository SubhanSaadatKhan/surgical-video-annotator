"""Fine-tune YOLOv8 on CholecSeg8k for surgical instrument and organ detection.

This script:
    1. Downloads CholecSeg8k (or uses cached version)
    2. Converts to YOLO training format
    3. Trains YOLOv8 on surgical instruments and anatomy
    4. Saves the best weights for use in the pipeline

Expected results after training:
    - Instrument mAP@0.5: 60-80%
    - Organ mAP@0.5: 40-60%
    - Combined mAP@0.5: 50-70%

Compare to zero-shot Grounding DINO baseline: ~4.5% mAP

Usage:
    python scripts/train_yolo.py --n-samples 2000 --epochs 50

    Then in config.yaml, point yolo_detection.weights_path to the trained model.
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import CholecSeg8kLoader


def train_yolo(
    n_samples: int = 2000,
    epochs: int = 50,
    model_size: str = "n",  # 'n' (nano), 's' (small), 'm' (medium), 'l', 'x'
    batch_size: int = 16,
    image_size: int = 640,
    output_dir: str = "yolo_models",
    dataset_dir: str = "yolo_dataset",
    include_anatomy: bool = True,
    resume: bool = False,
) -> str:
    """Train YOLOv8 on CholecSeg8k data.

    Args:
        n_samples: Number of CholecSeg8k images to use for training.
        epochs: Number of training epochs.
        model_size: YOLO model variant ('n' fastest, 'x' most accurate).
        batch_size: Training batch size (reduce if OOM).
        image_size: Input image size for training.
        output_dir: Where to save trained weights.
        dataset_dir: Where to build the YOLO dataset structure.
        include_anatomy: If True, include organ classes.
        resume: If True, resume from previous checkpoint.

    Returns:
        Path to the best trained weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics not installed. Install with: pip install ultralytics"
        )
        raise

    # =====================================================
    # Step 1: Prepare dataset
    # =====================================================
    logger.info(f"Preparing CholecSeg8k dataset with {n_samples} samples...")

    loader = CholecSeg8kLoader(cache_dir="cholecseg8k_data")

    # Check if we already have enough samples cached
    existing = list(Path("cholecseg8k_data/images").glob("*.png"))
    if len(existing) >= n_samples:
        logger.info(f"Using {len(existing)} cached samples")
        loader.samples = [
            {
                "id": i,
                "image_path": str(img),
                "mask_path": str(Path("cholecseg8k_data/masks") / img.name.replace("image_", "mask_")),
            }
            for i, img in enumerate(sorted(existing)[:n_samples])
        ]
    else:
        loader.download_subset(n=n_samples)

    # =====================================================
    # Step 2: Convert to YOLO format
    # =====================================================
    logger.info("Converting dataset to YOLO format...")

    yolo_config = loader.export_yolo_dataset(
        output_dir=dataset_dir,
        train_split=0.8,
        val_split=0.1,
        include_anatomy=include_anatomy,
    )

    logger.info(f"Dataset ready:")
    logger.info(f"  Classes: {yolo_config['num_classes']}")
    logger.info(f"  Class names: {yolo_config['class_names']}")
    logger.info(f"  Splits: {yolo_config['splits']}")

    # =====================================================
    # Step 3: Train YOLOv8
    # =====================================================
    logger.info(f"Training YOLOv8{model_size} for {epochs} epochs...")

    os.makedirs(output_dir, exist_ok=True)

    # Load pretrained YOLOv8 (starts from COCO weights, adapts to surgical data)
    model_name = f"yolov8{model_size}.pt"
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=yolo_config["data_yaml"],
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=output_dir,
        name="surgical_yolov8",
        exist_ok=True,
        resume=resume,
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Regularization
        patience=15,  # Early stopping if no improvement
        # Validation
        val=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    # =====================================================
    # Step 4: Report results
    # =====================================================
    best_weights = Path(output_dir) / "surgical_yolov8" / "weights" / "best.pt"

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Best weights: {best_weights}")
    logger.info(f"\nFinal metrics on validation set:")
    if hasattr(results, "results_dict"):
        for key, val in results.results_dict.items():
            logger.info(f"  {key}: {val}")

    # =====================================================
    # Step 5: Run on test set
    # =====================================================
    logger.info("\nEvaluating on test set...")

    trained_model = YOLO(str(best_weights))
    test_results = trained_model.val(
        data=yolo_config["data_yaml"],
        split="test",
        imgsz=image_size,
        batch=batch_size,
        verbose=True,
    )

    logger.info(f"\nTest set metrics:")
    if hasattr(test_results, "box"):
        logger.info(f"  mAP@0.5: {test_results.box.map50:.4f}")
        logger.info(f"  mAP@0.5:0.95: {test_results.box.map:.4f}")
        logger.info(f"  Precision: {test_results.box.mp:.4f}")
        logger.info(f"  Recall: {test_results.box.mr:.4f}")

    logger.info(f"\nTo use this model in the pipeline, update config.yaml:")
    logger.info(f"  yolo_detection:")
    logger.info(f"    weights_path: {best_weights}")

    return str(best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on CholecSeg8k for surgical annotation"
    )
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--no-anatomy", action="store_true",
                        help="Skip anatomy classes (instruments only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous checkpoint")

    args = parser.parse_args()

    train_yolo(
        n_samples=args.n_samples,
        epochs=args.epochs,
        model_size=args.model_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_anatomy=not args.no_anatomy,
        resume=args.resume,
    )
