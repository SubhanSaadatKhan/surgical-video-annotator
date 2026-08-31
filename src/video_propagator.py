"""SAM 2 video propagation: uses SAM 2's built-in video tracking.

Previously we ran SAM 2 per-frame and used our custom IoU tracker to
maintain identities. That works but wastes SAM 2's most powerful feature:
it was designed for video from the ground up.

SAM 2 video propagation:
    1. Initialize a video predictor with the full video
    2. Prompt the model on one or more frames (bounding boxes or points)
    3. SAM 2 automatically:
        - Generates precise masks on the prompted frame
        - Propagates those masks across all other frames
        - Maintains consistent object identities
        - Handles occlusion and reappearance

This is significantly better than per-frame segmentation because SAM 2
uses temporal features to keep masks consistent across frames, even when
the object briefly disappears or is partially occluded.

We fall back to per-frame segmentation when video propagation is not
appropriate (e.g., adaptive sampling with large gaps between frames).
"""

import numpy as np
from loguru import logger
from typing import Optional

from src.utils import BBox


class SAM2VideoPropagator:
    """Wrapper for SAM 2 native video segmentation and propagation."""

    def __init__(self, config: dict):
        propagation_config = config.get("video_propagation", {})
        self.enabled = propagation_config.get("enabled", True)
        self.model_id = config["segmentation"]["model_id"]
        self.device = config["segmentation"]["device"]
        self.prompt_interval = propagation_config.get("prompt_interval", 30)
        self.mask_threshold = config["segmentation"]["mask_threshold"]

        self.predictor = None
        self.inference_state = None

    def load_model(self) -> None:
        """Load the SAM 2 video predictor.

        Note: This uses the SAM 2 library directly, not HuggingFace,
        because HuggingFace transformers doesn't expose the video
        propagation API yet. We download the checkpoint from HuggingFace
        Hub, then pass the local file path to SAM 2's builder.
        """
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error(
                "SAM 2 not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )
            raise

        logger.info(f"Loading SAM 2 video predictor from {self.model_id}...")

        # Map model_id to (repo_id, checkpoint_filename, config_name)
        # SAM 2's config files are bundled with the sam2 package.
        model_map = {
            "facebook/sam2.1-hiera-large": (
                "facebook/sam2.1-hiera-large",
                "sam2.1_hiera_large.pt",
                "configs/sam2.1/sam2.1_hiera_l.yaml",
            ),
            "facebook/sam2.1-hiera-base-plus": (
                "facebook/sam2.1-hiera-base-plus",
                "sam2.1_hiera_base_plus.pt",
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
            ),
            "facebook/sam2.1-hiera-small": (
                "facebook/sam2.1-hiera-small",
                "sam2.1_hiera_small.pt",
                "configs/sam2.1/sam2.1_hiera_s.yaml",
            ),
            "facebook/sam2.1-hiera-tiny": (
                "facebook/sam2.1-hiera-tiny",
                "sam2.1_hiera_tiny.pt",
                "configs/sam2.1/sam2.1_hiera_t.yaml",
            ),
        }

        if self.model_id not in model_map:
            # Fallback: assume it's a HF repo with a .pt checkpoint at root
            repo_id = self.model_id
            ckpt_filename = self.model_id.split("/")[-1] + ".pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            repo_id, ckpt_filename, model_cfg = model_map[self.model_id]

        # Download the checkpoint from HuggingFace Hub
        logger.info(f"Downloading SAM 2 checkpoint: {repo_id}/{ckpt_filename}")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
        logger.info(f"Checkpoint downloaded to: {ckpt_path}")

        self.predictor = build_sam2_video_predictor(
            model_cfg, ckpt_path, device=self.device
        )

        logger.info(f"SAM 2 video predictor loaded on {self.device}")

    def initialize_video(self, frames_dir: str) -> dict:
        """Initialize the predictor with a directory of video frames.

        SAM 2 has strict requirements for frame naming:
            - Files must be named as pure integers: 0.jpg, 1.jpg, 2.jpg, ...
            - Must be sequential starting from 0

        Our smart sampling produces non-sequential frames like
        frame_000000.jpg, frame_000013.jpg. We create a temporary
        directory with symlinks (or copies) using SAM 2's naming.

        Args:
            frames_dir: Directory containing extracted frames.

        Returns:
            Dict mapping SAM 2's sequential index (0, 1, 2, ...) to our
            original frame index (0, 13, 55, ...). Needed to map results back.
        """
        import os
        import shutil

        if self.predictor is None:
            self.load_model()

        # Find all frame files and sort them by frame index
        all_files = os.listdir(frames_dir)
        frame_files = sorted([
            f for f in all_files
            if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))
        ], key=lambda f: int("".join(c for c in f if c.isdigit()) or "0"))

        if not frame_files:
            raise ValueError(f"No JPG frames found in {frames_dir}")

        # Create a SAM 2-compatible directory with renamed frames
        sam2_frames_dir = os.path.join(frames_dir, "_sam2_frames")
        # Clean up any previous run
        if os.path.exists(sam2_frames_dir):
            shutil.rmtree(sam2_frames_dir)
        os.makedirs(sam2_frames_dir)

        # Copy/symlink frames with SAM 2's expected naming (0.jpg, 1.jpg, ...)
        # And build the index mapping so we can translate results back.
        self.sam2_to_original_idx = {}
        for sam2_idx, original_filename in enumerate(frame_files):
            # Extract original frame index from filename (e.g. "frame_000013.jpg" -> 13)
            original_idx = int("".join(c for c in original_filename if c.isdigit()) or "0")
            self.sam2_to_original_idx[sam2_idx] = original_idx

            src = os.path.join(frames_dir, original_filename)
            dst = os.path.join(sam2_frames_dir, f"{sam2_idx}.jpg")
            # Use copy to be safe across filesystems
            shutil.copy2(src, dst)

        # Also build reverse mapping for prompt frames
        self.original_to_sam2_idx = {
            v: k for k, v in self.sam2_to_original_idx.items()
        }

        logger.info(
            f"Prepared {len(frame_files)} frames for SAM 2 "
            f"(renamed to sequential 0.jpg to {len(frame_files) - 1}.jpg)"
        )

        logger.info(f"Initializing SAM 2 video state from {sam2_frames_dir}...")
        self.inference_state = self.predictor.init_state(video_path=sam2_frames_dir)
        logger.info("Video state initialized")

        return self.sam2_to_original_idx

    def add_prompt(
        self,
        frame_idx: int,
        object_id: int,
        bbox: Optional[BBox] = None,
        points: Optional[list] = None,
        labels: Optional[list] = None,
    ) -> np.ndarray:
        """Add a prompt for an object on a specific frame.

        SAM 2 will use this to segment the object on this frame AND
        propagate that segmentation to other frames.

        Args:
            frame_idx: Frame index (within the extracted frames sequence).
            object_id: Unique ID for this object (persists across frames).
            bbox: Bounding box prompt (from Grounding DINO).
            points: Optional point prompts [[x, y], ...].
            labels: Optional labels for points (1=foreground, 0=background).

        Returns:
            Binary mask for the object on this frame.
        """
        import torch

        if self.inference_state is None:
            raise RuntimeError("Call initialize_video() first")

        # SAM 2 uses BFloat16 internally on GPU, so we need autocast to avoid dtype mismatches
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

        with autocast_context, torch.inference_mode():
            if bbox is not None:
                box_input = np.array(bbox.to_xyxy(), dtype=np.float32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    box=box_input,
                )
            elif points is not None:
                points_arr = np.array(points, dtype=np.float32)
                labels_arr = np.array(labels or [1] * len(points), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    points=points_arr,
                    labels=labels_arr,
                )
            else:
                raise ValueError("Must provide either bbox or points")

        # Extract mask for this object
        for i, obj_id in enumerate(out_obj_ids):
            if obj_id == object_id:
                mask = (out_mask_logits[i] > self.mask_threshold).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                return mask.astype(np.uint8)

        # Fallback: empty mask
        logger.warning(f"Could not extract mask for object {object_id}")
        return np.zeros((1, 1), dtype=np.uint8)

    def propagate(self) -> dict:
        """Propagate all prompted objects across the entire video.

        This is the core operation: after adding prompts, this runs SAM 2's
        temporal propagation to segment all objects in all frames.

        Returns:
            Dict mapping frame_idx to dict of {object_id: mask}.
        """
        import torch

        if self.inference_state is None:
            raise RuntimeError("Call initialize_video() first")

        logger.info("Propagating masks across video...")

        # SAM 2 needs BFloat16 autocast on GPU to match internal computation dtype
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

        video_segments = {}
        with autocast_context, torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state
            ):
                frame_masks = {}
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > self.mask_threshold).cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask[0]
                    frame_masks[int(obj_id)] = mask.astype(np.uint8)
                video_segments[int(out_frame_idx)] = frame_masks

        logger.info(f"Propagated masks to {len(video_segments)} frames")
        return video_segments

    def annotate_with_detections(
        self,
        frames_dir: str,
        detections_by_frame: dict,
    ) -> dict:
        """End-to-end: initialize video, add detection prompts, propagate.

        This is the main entry point for using SAM 2 video propagation
        with Grounding DINO detections.

        Strategy:
            - Every Nth frame (prompt_interval), use detections as prompts
            - SAM 2 propagates masks to all frames in between
            - Consistent object IDs are maintained across the whole video

        Args:
            frames_dir: Directory with extracted frames.
            detections_by_frame: Dict mapping frame_idx to list of BBox.

        Returns:
            Dict mapping frame_idx to list of (object_id, label, mask) tuples.
        """
        # initialize_video builds sam2_to_original_idx and original_to_sam2_idx maps
        self.initialize_video(frames_dir)

        # Track object IDs across the whole video
        next_object_id = 1
        object_labels = {}  # object_id -> label

        # Sort frames and pick prompt frames (using our original indices)
        sorted_frames = sorted(detections_by_frame.keys())
        prompt_frames = sorted_frames[::self.prompt_interval]
        if sorted_frames and sorted_frames[-1] not in prompt_frames:
            prompt_frames.append(sorted_frames[-1])

        logger.info(
            f"Using {len(prompt_frames)}/{len(sorted_frames)} frames as prompts"
        )

        # Add prompts for each detection on prompt frames
        # Translate our frame indices to SAM 2's sequential indices
        for original_frame_idx in prompt_frames:
            if original_frame_idx not in self.original_to_sam2_idx:
                logger.warning(
                    f"Frame {original_frame_idx} not in SAM 2 mapping, skipping"
                )
                continue

            sam2_frame_idx = self.original_to_sam2_idx[original_frame_idx]
            detections = detections_by_frame.get(original_frame_idx, [])

            for det in detections:
                object_labels[next_object_id] = det.label
                try:
                    self.add_prompt(
                        frame_idx=sam2_frame_idx,  # Use SAM 2's sequential index
                        object_id=next_object_id,
                        bbox=det,
                    )
                except Exception as e:
                    logger.warning(f"Failed to add prompt: {e}")
                next_object_id += 1

        # Propagate across full video (returns SAM 2's sequential indices)
        video_segments = self.propagate()

        # Convert to output format, translating SAM 2 indices back to our indices
        results = {}
        for sam2_frame_idx, obj_masks in video_segments.items():
            # Map back to our original frame index
            original_frame_idx = self.sam2_to_original_idx.get(sam2_frame_idx)
            if original_frame_idx is None:
                continue

            frame_annotations = []
            for obj_id, mask in obj_masks.items():
                label = object_labels.get(obj_id, "unknown")
                frame_annotations.append({
                    "object_id": obj_id,
                    "label": label,
                    "mask": mask,
                })
            results[original_frame_idx] = frame_annotations

        return results

    def reset(self) -> None:
        """Clear the inference state (for processing a new video)."""
        if self.predictor is not None and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        self.inference_state = None
