"""
Video and frame utility functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from PIL import Image

from utils.logging_utils import get_logger

log = get_logger(__name__)


def get_video_metadata(video_path: str) -> dict:
    """
    Extract video metadata using OpenCV.

    Args:
        video_path: Path to video file.

    Returns:
        Dict with keys: fps, total_frames, duration, width, height.

    Raises:
        FileNotFoundError: If video file does not exist.
        RuntimeError: If video cannot be opened.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def iter_frames_fixed_fps(
    video_path: str,
    target_fps: float = 1.0,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Yield frames at a fixed FPS from a video.

    Args:
        video_path: Path to video file.
        target_fps: Target extraction rate (frames per second).

    Yields:
        Tuple of (frame_idx, timestamp_seconds, frame_bgr_array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    frame_interval = max(1, int(round(video_fps / target_fps)))
    frame_idx = 0
    extracted = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                yield frame_idx, timestamp, frame
                extracted += 1

            frame_idx += 1
    finally:
        cap.release()

    log.debug(f"Extracted {extracted} frames at {target_fps} fps from {video_path}")


def iter_frames_every_n(
    video_path: str,
    every_n: int = 30,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Yield every N-th frame from a video.

    Args:
        video_path: Path to video file.
        every_n:    Extract one frame every N frames.

    Yields:
        Tuple of (frame_idx, timestamp_seconds, frame_bgr_array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                timestamp = frame_idx / video_fps
                yield frame_idx, timestamp, frame

            frame_idx += 1
    finally:
        cap.release()


def iter_frames_by_indices(
    video_path: str,
    target_indices: list[int],
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Yield specific frames by their frame indices from a video.

    Args:
        video_path:     Path to video file.
        target_indices: List of 0-based frame indices to extract.

    Yields:
        Tuple of (frame_idx, timestamp_seconds, frame_bgr_array).
    """
    if not target_indices:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    target_set = set(target_indices)
    frame_idx = 0
    extracted = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in target_set:
                timestamp = frame_idx / video_fps
                yield frame_idx, timestamp, frame
                extracted += 1
                if extracted >= len(target_set):
                    break

            frame_idx += 1
    finally:
        cap.release()


def save_frame(
    frame: np.ndarray,
    output_path: str,
    quality: int = 85,
    max_dimension: Optional[int] = None,
) -> str:
    """
    Save a BGR numpy frame as JPEG/PNG/WebP.

    Args:
        frame:         BGR numpy array from OpenCV.
        output_path:   Destination path (extension determines format).
        quality:       JPEG/WebP quality [1-100].
        max_dimension: If set, resize so longest side ≤ max_dimension.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Optionally resize
    if max_dimension is not None:
        h, w = frame.shape[:2]
        scale = max_dimension / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".webp":
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(output_path, frame)

    return os.path.abspath(output_path)


def load_image_rgb(image_path: str) -> Image.Image:
    """
    Load an image as PIL RGB Image.

    Args:
        image_path: Path to image file.

    Returns:
        PIL Image in RGB mode.
    """
    img = Image.open(image_path).convert("RGB")
    return img


def build_frame_id(video_id: str, frame_idx: int) -> str:
    """
    Build a globally unique frame ID string.

    Format: "{video_id}_frame_{frame_idx:06d}"
    Zero-padded to 6 digits for lexicographic ordering.

    Args:
        video_id:  Video identifier string.
        frame_idx: 0-based frame index.

    Returns:
        Frame ID string.
    """
    return f"{video_id}_frame_{frame_idx:06d}"
