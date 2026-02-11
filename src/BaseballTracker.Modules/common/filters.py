"""
Detection filter module for the smart detection pipeline.

Each filter takes a list of detections and returns (kept, rejected) tuples,
where each rejected detection includes a reason string for visual debugging.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Detection:
    """A single YOLO detection with metadata."""
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_idx: int = 0
    # Detection source for handoff tracking
    # "yolo" = standard YOLO detection, "machine_exit" = ME background subtraction
    source: str = "yolo"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class FilterResult:
    """Result of a single detection through a filter."""
    detection: Detection
    passed: bool
    reason: str = ""


def filter_by_roi(detections: List[Detection],
                  polygon: List[List[int]]) -> Tuple[List[Detection], List[FilterResult]]:
    """
    Keep only detections whose center falls inside the ROI polygon.
    If polygon is empty, all detections pass (no ROI configured).
    """
    if not polygon:
        return detections, []

    poly_np = np.array(polygon, dtype=np.int32)
    kept = []
    rejected = []

    for det in detections:
        # pointPolygonTest returns positive if inside, 0 on edge, negative if outside
        result = cv2.pointPolygonTest(poly_np, (det.cx, det.cy), measureDist=False)
        if result >= 0:
            kept.append(det)
        else:
            rejected.append(FilterResult(
                detection=det,
                passed=False,
                reason=f"Outside ROI ({det.cx:.0f},{det.cy:.0f})"
            ))

    return kept, rejected


def filter_by_class(detections: List[Detection],
                    allowed_classes: List[int]) -> Tuple[List[Detection], List[FilterResult]]:
    """Keep only detections of allowed COCO classes."""
    kept = []
    rejected = []

    for det in detections:
        if det.cls_id in allowed_classes:
            kept.append(det)
        else:
            rejected.append(FilterResult(
                detection=det,
                passed=False,
                reason=f"Class {det.cls_name} not tracked"
            ))

    return kept, rejected


def filter_by_size(detections: List[Detection],
                   min_sizes: Dict[int, Tuple[float, float]]) -> Tuple[List[Detection], List[FilterResult]]:
    """
    Reject detections that are too small for their class.
    min_sizes maps class_id -> (min_width, min_height).
    """
    kept = []
    rejected = []

    for det in detections:
        if det.cls_id in min_sizes:
            min_w, min_h = min_sizes[det.cls_id]
            if det.width < min_w or det.height < min_h:
                rejected.append(FilterResult(
                    detection=det,
                    passed=False,
                    reason=f"Too small: {det.width:.0f}x{det.height:.0f} < {min_w:.0f}x{min_h:.0f}"
                ))
                continue
        kept.append(det)

    return kept, rejected


def filter_by_confidence(detections: List[Detection],
                         thresholds: Dict[int, float]) -> Tuple[List[Detection], List[FilterResult]]:
    """
    Apply per-class confidence thresholds.
    thresholds maps class_id -> minimum confidence.
    Detections for classes not in the map use conf=0.0 (pass all).
    """
    kept = []
    rejected = []

    for det in detections:
        min_conf = thresholds.get(det.cls_id, 0.0)
        if det.conf >= min_conf:
            kept.append(det)
        else:
            rejected.append(FilterResult(
                detection=det,
                passed=False,
                reason=f"Low confidence: {det.conf:.0%} < {min_conf:.0%}"
            ))

    return kept, rejected


class StationaryFilter:
    """
    Track detection positions across frames and reject those that don't move.

    A detection cluster is considered stationary if its center position
    variance over the history window is below the threshold.

    Key design: a ball is only rejected as stationary if it has been
    consistently detected near the same cluster position for several
    consecutive frames. A ball that is merely passing through a stationary
    position (e.g., a pitched ball flying past a ball on the ground)
    will NOT be rejected — it hasn't dwelled long enough.
    """

    def __init__(self, history_frames: int = 10, max_variance_px: float = 8.0,
                 cluster_radius: float = 30.0, dwell_frames: int = 3):
        self.history_frames = history_frames
        self.max_variance_px = max_variance_px
        self.cluster_radius = cluster_radius
        self.dwell_frames = dwell_frames  # how many consecutive frames near a cluster = stationary
        # List of known stationary positions: [(cx, cy, hit_count)]
        self._stationary_clusters: List[List[float]] = []
        # Recent ball detection positions for variance calculation
        self._position_history: List[List[Tuple[float, float]]] = []
        self._frame_count = 0
        # Per-cluster dwell tracker: cluster_idx -> consecutive frames a
        # detection has been near it. Reset when no detection is nearby.
        self._cluster_dwell: Dict[int, int] = {}

    def update(self, detections: List[Detection]) -> Tuple[List[Detection], List[FilterResult]]:
        """
        Process one frame of ball detections.
        Returns (moving_detections, stationary_results).
        """
        self._frame_count += 1

        # Only filter sports ball detections (class 32)
        ball_dets = [d for d in detections if d.cls_id == 32]
        other_dets = [d for d in detections if d.cls_id != 32]

        kept = list(other_dets)  # non-ball detections pass through
        rejected = []

        # Track which clusters have a detection nearby this frame
        clusters_hit_this_frame = set()

        for det in ball_dets:
            cluster_idx = self._nearest_stationary_cluster(det.cx, det.cy)
            if cluster_idx is not None:
                clusters_hit_this_frame.add(cluster_idx)
                # Check dwell count: only reject if seen near this cluster
                # for enough consecutive frames
                dwell = self._cluster_dwell.get(cluster_idx, 0) + 1
                self._cluster_dwell[cluster_idx] = dwell

                if dwell >= self.dwell_frames:
                    # This detection has been sitting here for several frames
                    rejected.append(FilterResult(
                        detection=det,
                        passed=False,
                        reason="Stationary object"
                    ))
                else:
                    # Near a known cluster but hasn't dwelled long enough —
                    # could be a ball passing through. Let it through.
                    kept.append(det)
            else:
                kept.append(det)

        # Reset dwell count for clusters that had no detection this frame
        for idx in list(self._cluster_dwell.keys()):
            if idx not in clusters_hit_this_frame:
                self._cluster_dwell[idx] = 0

        # Update stationary cluster model with this frame's ball positions
        self._update_clusters(ball_dets)

        return kept, rejected

    def _nearest_stationary_cluster(self, cx: float, cy: float) -> Optional[int]:
        """
        Return the index of the nearest stationary cluster within radius,
        or None if no cluster is close enough.
        """
        for i, cluster in enumerate(self._stationary_clusters):
            dx = cx - cluster[0]
            dy = cy - cluster[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < self.cluster_radius:
                return i
        return None

    def _update_clusters(self, ball_dets: List[Detection]):
        """
        Build up stationary cluster knowledge over time.
        After seeing enough frames, positions that repeat become stationary.
        """
        current_positions = [(d.cx, d.cy) for d in ball_dets]
        self._position_history.append(current_positions)

        # Keep only recent history
        if len(self._position_history) > self.history_frames:
            self._position_history.pop(0)

        # Need enough history to determine stationarity
        if len(self._position_history) < self.history_frames:
            return

        # Collect all positions from recent history
        all_positions = []
        for frame_positions in self._position_history:
            all_positions.extend(frame_positions)

        if not all_positions:
            return

        # Simple clustering: for each position, count how many nearby positions exist
        # Positions that appear in many frames at similar locations are stationary
        for px, py in all_positions:
            # Already known?
            if self._nearest_stationary_cluster(px, py) is not None:
                continue

            # Count appearances near this position across history frames
            frames_with_nearby = 0
            for frame_positions in self._position_history:
                for fx, fy in frame_positions:
                    dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)
                    if dist < self.cluster_radius:
                        frames_with_nearby += 1
                        break

            # If detected in most frames at this position -> stationary
            if frames_with_nearby >= self.history_frames * 0.7:
                self._stationary_clusters.append([px, py, frames_with_nearby])

    def get_stationary_positions(self) -> List[Tuple[float, float]]:
        """Return known stationary cluster centers for visualization."""
        return [(c[0], c[1]) for c in self._stationary_clusters]
