"""Video processing module: extracts frames from surgical videos at configurable rates."""

import cv2
import os
from pathlib import Path
from loguru import logger


class VideoProcessor:
    """Handles video loading and frame extraction.

    Surgical videos are typically long (30min-2hrs) but instrument movement
    is relatively slow, so extracting 1-5 frames per second captures all
    relevant information without wasting compute.
    """

    def __init__(self, config: dict):
        self.fps = config["video"]["frame_extraction_fps"]
        self.max_frames = config["video"].get("max_frames")
        self.resize_w = config["video"]["resize_width"]
        self.resize_h = config["video"]["resize_height"]
        self.supported_formats = config["video"]["supported_formats"]

    def validate_video(self, video_path: str) -> dict:
        """Validate video file and return metadata.

        Returns:
            dict with keys: total_frames, fps, duration_sec, width, height
        """
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {self.supported_formats}"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        metadata = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        metadata["duration_sec"] = metadata["total_frames"] / metadata["fps"]
        cap.release()

        logger.info(
            f"Video validated: {path.name} | "
            f"{metadata['total_frames']} frames | "
            f"{metadata['fps']:.1f} fps | "
            f"{metadata['duration_sec']:.1f}s | "
            f"{metadata['width']}x{metadata['height']}"
        )
        return metadata

    def extract_frames(self, video_path: str, output_dir: str) -> list:
        """Extract frames from video at the configured rate.

        Args:
            video_path: Path to the surgical video file.
            output_dir: Directory to save extracted frames.

        Returns:
            List of dicts with keys: frame_idx, timestamp_sec, image_path
        """
        metadata = self.validate_video(video_path)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = metadata["fps"]

        # Calculate frame interval: if video is 30fps and we want 2fps,
        # we take every 15th frame
        frame_interval = max(1, int(video_fps / self.fps))

        extracted = []
        frame_count = 0
        extracted_count = 0

        logger.info(
            f"Extracting frames at {self.fps} fps "
            f"(every {frame_interval} frames from {video_fps:.1f} fps source)"
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Resize frame
                frame_resized = cv2.resize(
                    frame, (self.resize_w, self.resize_h),
                    interpolation=cv2.INTER_AREA
                )

                # Save frame
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame_resized)

                timestamp = frame_count / video_fps

                extracted.append({
                    "frame_idx": frame_count,
                    "timestamp_sec": round(timestamp, 3),
                    "image_path": frame_path,
                })
                extracted_count += 1

                if extracted_count % 100 == 0:
                    logger.info(f"Extracted {extracted_count} frames...")

                if self.max_frames and extracted_count >= self.max_frames:
                    logger.info(f"Reached max_frames limit: {self.max_frames}")
                    break

            frame_count += 1

        cap.release()
        logger.info(
            f"Extraction complete: {extracted_count} frames "
            f"from {frame_count} total"
        )
        return extracted

    def load_frame(self, image_path: str):
        """Load a single frame from disk.

        Returns:
            BGR image as numpy array, or None if loading fails.
        """
        if not os.path.exists(image_path):
            logger.warning(f"Frame not found: {image_path}")
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning(f"Could not read frame: {image_path}")
        return frame

    def create_video_from_frames(
        self, frame_paths: list, output_path: str, fps: float = 5.0
    ) -> str:
        """Create a video from annotated frames (for review/demo purposes).

        Args:
            frame_paths: Ordered list of frame image paths.
            output_path: Path for the output video.
            fps: Playback frame rate.

        Returns:
            Path to the created video.
        """
        if not frame_paths:
            raise ValueError("No frames provided")

        first_frame = cv2.imread(frame_paths[0])
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is not None:
                writer.write(frame)

        writer.release()
        logger.info(f"Created review video: {output_path} ({len(frame_paths)} frames)")
        return output_path
