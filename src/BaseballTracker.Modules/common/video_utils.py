"""
Video utility functions for frame extraction and preprocessing.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_frames(video_path: str) -> List[np.ndarray]:
    """Extract all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def get_video_info(video_path: str) -> dict:
    """Get video metadata (resolution, fps, frame count, duration)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
    }

    cap.release()
    return info


def extract_frame_range(video_path: str, start_frame: int, end_frame: int) -> List[np.ndarray]:
    """Extract a range of frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    logger.info(f"Extracted frames {start_frame}-{start_frame + len(frames)} from {video_path}")
    return frames


def save_frame(frame: np.ndarray, output_path: str) -> None:
    """Save a single frame as an image file."""
    cv2.imwrite(output_path, frame)
    logger.info(f"Saved frame to {output_path}")
