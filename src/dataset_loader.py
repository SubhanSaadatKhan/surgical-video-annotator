"""CholecSeg8k dataset loader with ground truth conversion.

CholecSeg8k is a semantic segmentation dataset for laparoscopic
cholecystectomy with 8,080 pixel-annotated images. It contains 13
classes covering both surgical instruments and anatomy.

Class mapping (from the dataset):
    0: Black Background
    1: Abdominal Wall
    2: Liver
    3: Gastrointestinal Tract
    4: Fat
    5: Grasper (instrument)
    6: Connective Tissue
    7: Blood
    8: Cystic Duct
    9: L-hook Electrocautery (instrument)
    10: Gallbladder
    11: Hepatic Vein
    12: Liver Ligament

This module:
    1. Loads CholecSeg8k from HuggingFace
    2. Extracts ground truth bounding boxes from segmentation masks
    3. Converts to COCO format for use with the evaluator
    4. Provides samples for pipeline testing

Usage:
    loader = CholecSeg8kLoader()
    loader.download_subset(n=200)
    gt_coco_path = loader.export_ground_truth_coco()
    image_dir = loader.get_image_dir()
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger
from typing import Optional


# CholecSeg8k class definitions
CHOLECSEG8K_CLASSES = {
    0: "background",
    1: "abdominal wall",
    2: "liver",
    3: "gastrointestinal tract",
    4: "fat",
    5: "grasper",
    6: "connective tissue",
    7: "blood",
    8: "cystic duct",
    9: "hook",  # L-hook Electrocautery
    10: "gallbladder",
    11: "hepatic vein",
    12: "liver ligament",
}

# Which classes are instruments vs anatomy
INSTRUMENT_CLASSES = {5, 9}  # grasper, hook
ANATOMY_CLASSES = {1, 2, 3, 4, 6, 7, 8, 10, 11, 12}
BACKGROUND_CLASS = 0

# Color mapping used in CholecSeg8k color masks
# (R, G, B) -> class_id
COLOR_TO_CLASS = {
    (127, 127, 127): 0,   # Black Background
    (210, 140, 140): 1,   # Abdominal Wall
    (255, 114, 114): 2,   # Liver
    (231, 70, 156): 3,    # Gastrointestinal Tract
    (186, 183, 75): 4,    # Fat
    (170, 255, 0): 5,     # Grasper
    (255, 85, 0): 6,      # Connective Tissue
    (255, 0, 0): 7,       # Blood
    (255, 255, 0): 8,     # Cystic Duct
    (169, 255, 184): 9,   # L-hook
    (255, 160, 165): 10,  # Gallbladder
    (0, 50, 128): 11,     # Hepatic Vein
    (111, 74, 0): 12,     # Liver Ligament
}


class CholecSeg8kLoader:
    """Loads CholecSeg8k dataset and prepares it for pipeline testing."""

    def __init__(self, cache_dir: str = "cholecseg8k_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = self.cache_dir / "images"
        self.mask_dir = self.cache_dir / "masks"
        self.image_dir.mkdir(exist_ok=True)
        self.mask_dir.mkdir(exist_ok=True)

        self.samples = []

    def download_subset(self, n: int = 200, seed: int = 42) -> list:
        """Download a subset of CholecSeg8k directly from HuggingFace Hub.

        The dataset is stored as a zip file. We download it, extract only what
        we need, and process the images. This avoids the deprecated datasets
        library loading script.

        Args:
            n: Number of samples to extract.
            seed: Random seed for reproducibility (unused currently).

        Returns:
            List of sample dicts with image_path and mask_path.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            )
            raise

        import zipfile
        import shutil

        logger.info("Downloading CholecSeg8k zip from HuggingFace Hub...")
        logger.info("Note: this downloads ~3GB, first run only. Cached afterwards.")

        # Download the zip file (cached automatically)
        zip_path = hf_hub_download(
            repo_id="minwoosun/CholecSeg8k",
            filename="data/CholecSeg8k.zip",
            repo_type="dataset",
        )
        logger.info(f"Zip file: {zip_path}")

        # Extract to cache dir
        extract_dir = self.cache_dir / "extracted"
        if not extract_dir.exists():
            logger.info("Extracting zip file (first run only)...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            logger.info(f"Extracted to {extract_dir}")
        else:
            logger.info(f"Using existing extraction at {extract_dir}")

        # Find all image files. CholecSeg8k structure:
        # extracted/CholecSeg8k/video01/video01_00080/frame_80_endo.png
        # The color mask is named: frame_80_endo_color_mask.png
        image_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith("_endo.png") and not f.endswith("_color_mask.png") and "watershed" not in f and "annotation" not in f:
                    full_path = os.path.join(root, f)
                    # Check that the color mask exists
                    mask_name = f.replace("_endo.png", "_endo_color_mask.png")
                    mask_path = os.path.join(root, mask_name)
                    if os.path.exists(mask_path):
                        image_files.append((full_path, mask_path))

        logger.info(f"Found {len(image_files)} image/mask pairs in dataset")

        if not image_files:
            raise RuntimeError(
                f"No image/mask pairs found. Extract dir: {extract_dir}\n"
                f"Check the folder structure."
            )

        # Take first n samples (or all if n exceeds available)
        selected = image_files[:min(n, len(image_files))]

        # Copy to our organized image_dir and mask_dir
        samples = []
        logger.info(f"Copying {len(selected)} samples to organized folders...")

        for i, (img_src, mask_src) in enumerate(selected):
            image_path = self.image_dir / f"image_{i:05d}.png"
            mask_path = self.mask_dir / f"mask_{i:05d}.png"

            if not image_path.exists():
                shutil.copy2(img_src, image_path)
            if not mask_path.exists():
                shutil.copy2(mask_src, mask_path)

            samples.append({
                "id": i,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            })

            if (i + 1) % 50 == 0:
                logger.info(f"Prepared {i + 1}/{len(selected)} samples...")

        self.samples = samples
        logger.info(f"Prepared {len(samples)} samples in {self.cache_dir}")
        return samples

    def mask_to_bboxes(self, mask_path: str) -> list:
        """Extract bounding boxes from a color-coded segmentation mask.

        Args:
            mask_path: Path to the color mask PNG.

        Returns:
            List of dicts with keys: class_id, class_name, bbox [x, y, w, h],
            and pixel_count.
        """
        mask = np.array(Image.open(mask_path).convert("RGB"))
        h, w = mask.shape[:2]

        bboxes = []

        for color, class_id in COLOR_TO_CLASS.items():
            if class_id == BACKGROUND_CLASS:
                continue

            # Find all pixels matching this color (with small tolerance)
            r_match = np.abs(mask[:, :, 0].astype(int) - color[0]) < 15
            g_match = np.abs(mask[:, :, 1].astype(int) - color[1]) < 15
            b_match = np.abs(mask[:, :, 2].astype(int) - color[2]) < 15
            class_mask = r_match & g_match & b_match

            pixel_count = int(np.sum(class_mask))
            if pixel_count < 50:  # Skip tiny regions (noise)
                continue

            # Find connected components and get bounding boxes
            try:
                from scipy import ndimage
                labeled, num_features = ndimage.label(class_mask)

                for component_id in range(1, num_features + 1):
                    component = (labeled == component_id)
                    component_pixels = int(np.sum(component))

                    if component_pixels < 100:  # Skip small components
                        continue

                    ys, xs = np.where(component)
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())

                    bboxes.append({
                        "class_id": class_id,
                        "class_name": CHOLECSEG8K_CLASSES[class_id],
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "pixel_count": component_pixels,
                        "mask": component.astype(np.uint8),
                    })
            except ImportError:
                # Fallback without scipy: single bbox per class
                ys, xs = np.where(class_mask)
                if len(xs) == 0:
                    continue
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bboxes.append({
                    "class_id": class_id,
                    "class_name": CHOLECSEG8K_CLASSES[class_id],
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "pixel_count": pixel_count,
                    "mask": class_mask.astype(np.uint8),
                })

        return bboxes

    def export_ground_truth_coco(
        self,
        output_path: str = "ground_truth_coco.json",
        include_anatomy: bool = True,
    ) -> str:
        """Convert dataset ground truth to COCO format.

        Args:
            output_path: Where to save the COCO JSON.
            include_anatomy: If False, only include instrument classes.

        Returns:
            Path to the saved COCO JSON file.
        """
        if not self.samples:
            raise RuntimeError("Call download_subset() first")

        # Filter classes based on what we want to evaluate
        if include_anatomy:
            valid_classes = INSTRUMENT_CLASSES | ANATOMY_CLASSES
        else:
            valid_classes = INSTRUMENT_CLASSES

        coco = {
            "info": {
                "description": "CholecSeg8k ground truth for pipeline evaluation",
                "version": "1.0",
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": cid, "name": name}
                for cid, name in CHOLECSEG8K_CLASSES.items()
                if cid in valid_classes
            ],
        }

        annotation_id = 1

        for sample in self.samples:
            # Get image dimensions
            image = Image.open(sample["image_path"])
            w, h = image.size

            coco["images"].append({
                "id": sample["id"],
                "file_name": os.path.basename(sample["image_path"]),
                "width": w,
                "height": h,
            })

            # Extract bounding boxes from mask
            bboxes = self.mask_to_bboxes(sample["mask_path"])

            for bbox_info in bboxes:
                if bbox_info["class_id"] not in valid_classes:
                    continue

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": sample["id"],
                    "category_id": bbox_info["class_id"],
                    "bbox": bbox_info["bbox"],
                    "area": bbox_info["bbox"][2] * bbox_info["bbox"][3],
                    "iscrowd": 0,
                })
                annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            f"Exported COCO ground truth: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations, "
            f"{len(coco['categories'])} categories -> {output_path}"
        )
        return output_path

    def get_image_dir(self) -> str:
        """Return the directory containing downloaded images."""
        return str(self.image_dir)

    def get_prompts_for_evaluation(self, include_anatomy: bool = True) -> list:
        """Return the class name prompts to use for detection.

        These are the labels the pipeline should try to detect.

        Args:
            include_anatomy: If True, include organ/tissue prompts.

        Returns:
            List of prompt strings.
        """
        if include_anatomy:
            classes = INSTRUMENT_CLASSES | ANATOMY_CLASSES
        else:
            classes = INSTRUMENT_CLASSES

        return sorted([
            CHOLECSEG8K_CLASSES[cid]
            for cid in classes
        ])

    def get_class_id_by_name(self, name: str) -> Optional[int]:
        """Get class ID from name (case-insensitive fuzzy match)."""
        name_lower = name.lower().strip()
        for cid, cname in CHOLECSEG8K_CLASSES.items():
            if cname.lower() == name_lower:
                return cid
        # Fuzzy match
        for cid, cname in CHOLECSEG8K_CLASSES.items():
            if name_lower in cname.lower() or cname.lower() in name_lower:
                return cid
        return None

    def export_yolo_dataset(
        self,
        output_dir: str = "yolo_dataset",
        train_split: float = 0.8,
        val_split: float = 0.1,
        include_anatomy: bool = True,
        seed: int = 42,
    ) -> dict:
        """Convert dataset to YOLO training format.

        YOLO expects:
            dataset/
                images/train/*.jpg
                images/val/*.jpg
                images/test/*.jpg
                labels/train/*.txt  (one line per box: class_id cx cy w h, normalized)
                labels/val/*.txt
                labels/test/*.txt
                data.yaml (config file)

        Args:
            output_dir: Where to build the YOLO dataset structure.
            train_split: Fraction for training set.
            val_split: Fraction for validation set (rest goes to test).
            include_anatomy: If False, only include instrument classes.
            seed: Random seed for split.

        Returns:
            Dict with paths and class info.
        """
        import random
        import shutil

        if not self.samples:
            raise RuntimeError("Call download_subset() first")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create YOLO directory structure
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Filter classes
        if include_anatomy:
            valid_classes = INSTRUMENT_CLASSES | ANATOMY_CLASSES
        else:
            valid_classes = INSTRUMENT_CLASSES

        # Build class mapping: original CholecSeg8k ID -> YOLO 0-indexed ID
        sorted_classes = sorted(valid_classes)
        cholecseg_to_yolo = {cid: i for i, cid in enumerate(sorted_classes)}
        yolo_class_names = [CHOLECSEG8K_CLASSES[cid] for cid in sorted_classes]

        # Split samples
        random.seed(seed)
        shuffled = list(self.samples)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:]

        splits = {"train": train_samples, "val": val_samples, "test": test_samples}

        logger.info(
            f"Splitting dataset: {n_train} train / {len(val_samples)} val / "
            f"{len(test_samples)} test"
        )

        # Process each split
        for split_name, samples in splits.items():
            logger.info(f"Processing {split_name} split ({len(samples)} samples)...")

            for sample in samples:
                # Copy image
                image_dst = output_dir / "images" / split_name / os.path.basename(
                    sample["image_path"]
                )
                if not image_dst.exists():
                    shutil.copy2(sample["image_path"], image_dst)

                # Get image dimensions
                img = Image.open(sample["image_path"])
                img_w, img_h = img.size

                # Extract bboxes from mask and write YOLO labels
                bboxes = self.mask_to_bboxes(sample["mask_path"])

                label_lines = []
                for bbox_info in bboxes:
                    if bbox_info["class_id"] not in valid_classes:
                        continue

                    yolo_class = cholecseg_to_yolo[bbox_info["class_id"]]
                    x, y, w, h = bbox_info["bbox"]

                    # YOLO format: normalized center_x, center_y, width, height
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h

                    # Clip to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    label_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                # Write label file (even if empty, YOLO needs it)
                label_name = os.path.basename(sample["image_path"]).rsplit(".", 1)[0] + ".txt"
                label_path = output_dir / "labels" / split_name / label_name
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

        # Write data.yaml
        data_yaml = {
            "path": str(output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(yolo_class_names),
            "names": yolo_class_names,
        }

        import yaml as pyyaml
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            pyyaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(
            f"YOLO dataset ready at {output_dir}\n"
            f"  Classes: {len(yolo_class_names)}\n"
            f"  Data config: {yaml_path}"
        )

        return {
            "dataset_dir": str(output_dir),
            "data_yaml": str(yaml_path),
            "num_classes": len(yolo_class_names),
            "class_names": yolo_class_names,
            "cholecseg_to_yolo": cholecseg_to_yolo,
            "splits": {k: len(v) for k, v in splits.items()},
        }
