"""
Simple frame-to-frame object tracker using nearest-neighbor matching.

Designed for single-ball, single-bat scenarios in a batting cage.
Not a full MOT tracker — intentionally simple and debuggable.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackedObject:
    """A tracked object with position history."""
    track_id: int
    cls_id: int
    cls_name: str
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        if self.positions:
            return self.positions[-1]
        return None

    @property
    def last_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if self.bboxes:
            return self.bboxes[-1]
        return None

    @property
    def age(self) -> int:
        """Number of frames this track has existed."""
        return len(self.positions)

    def velocity_px(self) -> Optional[Tuple[float, float]]:
        """
        Compute pixel velocity from the last two positions.
        Returns (vx, vy) in pixels/frame, or None if not enough data.
        """
        if len(self.positions) < 2:
            return None
        x1, y1 = self.positions[-2]
        x2, y2 = self.positions[-1]
        return (x2 - x1, y2 - y1)

    def speed_px_per_frame(self) -> float:
        """Scalar speed in pixels/frame from last two positions."""
        v = self.velocity_px()
        if v is None:
            return 0.0
        return np.sqrt(v[0] ** 2 + v[1] ** 2)

    def speed_px_per_sec(self, fps: float) -> float:
        """Scalar speed in pixels/second."""
        return self.speed_px_per_frame() * fps

    def smoothed_speed_px_per_sec(self, fps: float, window: int = 3) -> float:
        """
        Smoothed speed using median of recent frame-to-frame speeds.
        Filters out single-frame tracker jumps that produce fake spikes.
        """
        if len(self.positions) < 2:
            return 0.0
        recent = self.positions[-min(window + 1, len(self.positions)):]
        speeds = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            speeds.append(np.sqrt(dx * dx + dy * dy) * fps)
        if not speeds:
            return 0.0
        return float(np.median(speeds))

    def total_displacement(self) -> float:
        """Total pixel displacement from first to last position."""
        if len(self.positions) < 2:
            return 0.0
        x1, y1 = self.positions[0]
        x2, y2 = self.positions[-1]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_trail(self, max_points: int = 20) -> List[Tuple[int, int]]:
        """Get recent positions as integer tuples for drawing a trail."""
        recent = self.positions[-max_points:]
        return [(int(x), int(y)) for x, y in recent]


class ObjectTracker:
    """
    Simple nearest-neighbor multi-object tracker.

    Matches new detections to existing tracks by Euclidean distance.
    Creates new tracks for unmatched detections, removes stale tracks.
    """

    def __init__(self, max_match_distance: float = 150.0,
                 max_missed_frames: int = 5):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def update(self, detections: list, frame_idx: int) -> List[TrackedObject]:
        """
        Match detections to existing tracks and return active tracks.

        detections: list of Detection objects (from filters.py)
        frame_idx: current frame number

        Returns list of all active TrackedObject instances.
        """
        # Group detections by class for independent matching
        from common.filters import Detection
        det_by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            det_by_class.setdefault(det.cls_id, []).append(det)

        matched_track_ids = set()

        # Match each class independently
        for cls_id, cls_dets in det_by_class.items():
            # Get existing tracks for this class
            cls_tracks = {tid: t for tid, t in self._tracks.items()
                          if t.cls_id == cls_id and t.missed_frames < self.max_missed_frames}

            if not cls_tracks and not cls_dets:
                continue

            # Build cost matrix: distance from each track to each detection
            track_ids = list(cls_tracks.keys())
            if track_ids and cls_dets:
                costs = np.zeros((len(track_ids), len(cls_dets)))
                for i, tid in enumerate(track_ids):
                    track_pos = cls_tracks[tid].last_position
                    if track_pos is None:
                        costs[i, :] = float('inf')
                        continue
                    for j, det in enumerate(cls_dets):
                        dx = det.cx - track_pos[0]
                        dy = det.cy - track_pos[1]
                        costs[i, j] = np.sqrt(dx * dx + dy * dy)

                # Greedy matching: assign closest pairs first
                used_dets = set()
                pairs = []
                flat_indices = np.argsort(costs.ravel())
                for flat_idx in flat_indices:
                    i = flat_idx // len(cls_dets)
                    j = flat_idx % len(cls_dets)
                    if i >= len(track_ids) or j >= len(cls_dets):
                        continue
                    if track_ids[i] in matched_track_ids or j in used_dets:
                        continue
                    if costs[i, j] > self.max_match_distance:
                        break
                    pairs.append((track_ids[i], j))
                    matched_track_ids.add(track_ids[i])
                    used_dets.add(j)

                # Update matched tracks
                for tid, det_idx in pairs:
                    det = cls_dets[det_idx]
                    track = self._tracks[tid]
                    track.positions.append((det.cx, det.cy))
                    track.frame_indices.append(frame_idx)
                    track.confidences.append(det.conf)
                    track.bboxes.append((det.x1, det.y1, det.x2, det.y2))
                    track.missed_frames = 0

                # Create new tracks for unmatched detections
                for j, det in enumerate(cls_dets):
                    if j not in used_dets:
                        self._create_track(det, frame_idx)
            else:
                # No existing tracks for this class: create all new
                for det in cls_dets:
                    self._create_track(det, frame_idx)

        # Increment missed frames for unmatched tracks
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.missed_frames += 1

        # Remove dead tracks
        dead_ids = [tid for tid, t in self._tracks.items()
                    if t.missed_frames >= self.max_missed_frames]
        for tid in dead_ids:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _create_track(self, det, frame_idx: int):
        """Create a new track from an unmatched detection."""
        track = TrackedObject(
            track_id=self._next_id,
            cls_id=det.cls_id,
            cls_name=det.cls_name,
            positions=[(det.cx, det.cy)],
            frame_indices=[frame_idx],
            confidences=[det.conf],
            bboxes=[(det.x1, det.y1, det.x2, det.y2)],
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    def get_ball_tracks(self) -> List[TrackedObject]:
        """Get active ball tracks (class 32)."""
        return [t for t in self._tracks.values()
                if t.cls_id == 32 and t.missed_frames == 0]

    def get_bat_tracks(self) -> List[TrackedObject]:
        """Get active bat tracks (class 34)."""
        return [t for t in self._tracks.values()
                if t.cls_id == 34 and t.missed_frames == 0]

    def get_best_ball_track(self) -> Optional[TrackedObject]:
        """Get the ball track with the most positions (likely the real ball)."""
        ball_tracks = self.get_ball_tracks()
        if not ball_tracks:
            return None
        return max(ball_tracks, key=lambda t: t.age)

    def get_best_bat_track(self) -> Optional[TrackedObject]:
        """Get the bat track with the most positions."""
        bat_tracks = self.get_bat_tracks()
        if not bat_tracks:
            return None
        return max(bat_tracks, key=lambda t: t.age)


def detect_contact(ball_track: Optional[TrackedObject],
                   bat_track: Optional[TrackedObject],
                   distance_threshold: float = 60.0) -> Optional[int]:
    """
    Detect if bat and ball are close enough for contact.

    Returns the frame index of contact, or None.
    Looks at the most recent positions of both tracks.
    """
    if ball_track is None or bat_track is None:
        return None

    if not ball_track.positions or not bat_track.positions:
        return None

    # Check if both tracks have a position in the same frame
    ball_pos = ball_track.last_position
    bat_pos = bat_track.last_position

    if ball_pos is None or bat_pos is None:
        return None

    # Must be in the same frame (or within 1 frame)
    if ball_track.frame_indices and bat_track.frame_indices:
        frame_diff = abs(ball_track.frame_indices[-1] - bat_track.frame_indices[-1])
        if frame_diff > 1:
            return None

    dx = ball_pos[0] - bat_pos[0]
    dy = ball_pos[1] - bat_pos[1]
    distance = np.sqrt(dx * dx + dy * dy)

    if distance < distance_threshold:
        return ball_track.frame_indices[-1]

    return None
