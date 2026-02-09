"""
Machine-exit ball detector: finds the pitched ball in the region
near the pitching machine using frame differencing.

YOLO struggles to detect the ball near the machine because:
  - The ball is only ~10-15px at 60ft from camera
  - It moves slowly in pixel space (~2-6 px/frame)
  - It partially overlaps with the machine in early frames

This detector uses background subtraction in a small search region
around the machine exit point. The ball appears as a bright blob
(intensity 150-230) against a dark background (intensity 5-80),
producing strong frame-difference signal.

Ground-truth analysis shows that the max-diff location in the
machine exit region tracks the ball position with high accuracy.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MachineExitConfig:
    """Configuration for the machine-exit ball detector."""
    # Machine bounding box [x1, y1, x2, y2]
    machine_bbox: List[int]
    # How far right of machine to search (px)
    search_extend_right: int = 250
    # How far left of machine to include (px)
    search_extend_left: int = 20
    # Vertical padding above/below machine (px)
    search_extend_vertical: int = 40
    # Minimum frame-difference intensity to consider as ball motion
    min_diff_intensity: float = 80.0
    # How many background frames to accumulate before starting detection
    bg_warmup_frames: int = 30
    # Exponential moving average rate for background model
    bg_alpha: float = 0.02
    # Minimum blob area (pixels) to accept as ball
    min_blob_area: int = 4
    # Maximum blob area (pixels) - ball shouldn't be huge
    max_blob_area: int = 400
    # Maximum blob aspect ratio (ball should be roughly round)
    max_aspect_ratio: float = 3.0
    # Minimum confidence score to accept a detection
    min_confidence: float = 0.75


class MachineExitDetector:
    """
    Detects the ball in the machine exit region using frame differencing.

    Uses a running background model and looks for small bright blobs
    that appear in the search region near the machine exit.
    """

    def __init__(self, config: MachineExitConfig, frame_height: int, frame_width: int):
        self.config = config

        mx1, my1, mx2, my2 = config.machine_bbox
        # Define search region (clamp to frame bounds)
        self.search_x1 = max(0, mx1 - config.search_extend_left)
        self.search_y1 = max(0, my1 - config.search_extend_vertical)
        self.search_x2 = min(frame_width, mx2 + config.search_extend_right)
        self.search_y2 = min(frame_height, my2 + config.search_extend_vertical)

        self._bg_model: Optional[np.ndarray] = None
        self._frame_count = 0
        self._prev_gray: Optional[np.ndarray] = None
        # Trajectory tracking for consistency filtering
        self._recent_positions: List[Tuple[float, float]] = []  # last N detected positions
        self._max_trajectory_history = 5
        self._consecutive_misses = 0  # frames since last detection
        self._consecutive_detections = 0  # consecutive frames with detection

    @property
    def search_region(self) -> Tuple[int, int, int, int]:
        """Return the search region as (x1, y1, x2, y2)."""
        return (self.search_x1, self.search_y1, self.search_x2, self.search_y2)

    def _extract_region(self, gray: np.ndarray) -> np.ndarray:
        """Extract the search region from a grayscale frame."""
        return gray[self.search_y1:self.search_y2, self.search_x1:self.search_x2]

    def update(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Process one frame. Returns detected ball as (cx, cy, radius, confidence, diff_val)
        in full-frame coordinates, or None if no ball detected.

        confidence is 0.0-1.0 based on how well the blob matches expected ball properties.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        region = self._extract_region(gray)
        self._frame_count += 1

        # Build background model
        if self._bg_model is None:
            self._bg_model = region.copy()
            self._prev_gray = gray.copy()
            return None

        # Update background with exponential moving average
        # (slow update so the ball doesn't get absorbed into background)
        cv2.accumulateWeighted(region, self._bg_model, self.config.bg_alpha)

        # Need warmup period for background to stabilize
        if self._frame_count < self.config.bg_warmup_frames:
            self._prev_gray = gray.copy()
            return None

        # Frame-to-frame difference (catches motion)
        prev_region = self._extract_region(self._prev_gray)
        frame_diff = cv2.absdiff(region, prev_region)

        # Background subtraction (catches objects that differ from background)
        bg_diff = cv2.absdiff(region, self._bg_model)

        # Combine: use bg_diff for detection (more stable than frame_diff)
        # The ball is bright against dark background, so bg_diff is strong
        diff = bg_diff

        # Threshold to binary
        _, binary = cv2.threshold(
            diff.astype(np.uint8),
            int(self.config.min_diff_intensity),
            255,
            cv2.THRESH_BINARY
        )

        # Morphological cleanup: remove noise, connect nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours (blobs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._prev_gray = gray.copy()

        if not contours:
            return None

        # Score each blob and pick the best candidate
        best_score = 0.0
        best_result = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_blob_area or area > self.config.max_blob_area:
                continue

            # Bounding rect for aspect ratio check
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect > self.config.max_aspect_ratio:
                continue

            # Centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            local_cx = M["m10"] / M["m00"]
            local_cy = M["m01"] / M["m00"]

            # Convert to full-frame coordinates
            abs_cx = local_cx + self.search_x1
            abs_cy = local_cy + self.search_y1

            # Intensity of the blob in the original frame (should be bright for ball)
            mask = np.zeros_like(region, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            blob_intensity = cv2.mean(region, mask=mask)[0]

            # Diff intensity (how much the blob differs from background)
            diff_intensity = cv2.mean(diff, mask=mask)[0]

            # Radius estimate
            radius = np.sqrt(area / np.pi)

            # Score: combine diff intensity, roundness, and brightness
            roundness = 1.0 / aspect  # 1.0 = perfectly square, lower = elongated
            size_score = 1.0 - abs(area - 50) / 200  # prefer ~50px area (7px radius)
            size_score = max(0.0, min(1.0, size_score))

            intensity_score = min(1.0, blob_intensity / 200.0)  # bright is good
            diff_score = min(1.0, diff_intensity / 150.0)  # strong diff is good

            score = (diff_score * 0.4 + intensity_score * 0.3 +
                     roundness * 0.2 + size_score * 0.1)

            if score > best_score:
                best_score = score
                best_result = (abs_cx, abs_cy, radius, score, diff_intensity)

        if best_result is None:
            self._consecutive_misses += 1
            self._consecutive_detections = 0
            # Reset trajectory after a gap
            if self._consecutive_misses > 5:
                self._recent_positions.clear()
            return None

        cx, cy, radius, score, diff_val = best_result

        # Confidence threshold
        if score < self.config.min_confidence:
            self._consecutive_misses += 1
            self._consecutive_detections = 0
            if self._consecutive_misses > 5:
                self._recent_positions.clear()
            return None

        # Trajectory consistency check: if we have prior positions,
        # reject detections that jump too far (likely noise, not ball)
        if len(self._recent_positions) >= 2:
            # Predict next position from last two
            px1, py1 = self._recent_positions[-2]
            px2, py2 = self._recent_positions[-1]
            pred_x = px2 + (px2 - px1)
            pred_y = py2 + (py2 - py1)
            dist = np.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)
            # Allow up to 30px deviation from predicted position
            if dist > 30:
                self._consecutive_detections = 0
                return None

        self._consecutive_detections += 1
        self._consecutive_misses = 0

        # Require at least 2 consecutive detections before reporting
        # (prevents single-frame noise from being reported as ball)
        if self._consecutive_detections < 2 and len(self._recent_positions) == 0:
            self._recent_positions.append((cx, cy))
            return None

        # Update trajectory history
        self._recent_positions.append((cx, cy))
        if len(self._recent_positions) > self._max_trajectory_history:
            self._recent_positions.pop(0)

        return best_result
