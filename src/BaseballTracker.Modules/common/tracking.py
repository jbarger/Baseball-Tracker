"""
Frame-to-frame object tracker with Kalman-style velocity estimation.

Designed for single-ball, single-bat scenarios in a batting cage.
Not a full MOT tracker -- intentionally simple and debuggable.

Key improvements over basic nearest-neighbor:
  - Kalman-inspired position/velocity state for smooth estimates
  - Detection source tracking (YOLO vs machine-exit) for handoff
  - Velocity-gated matching to prevent tracker jumps at handoffs
  - 3D-aware speed computation via CageGeometry
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackedObject:
    """A tracked object with position history and velocity estimation."""
    track_id: int
    cls_id: int
    cls_name: str
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    missed_frames: int = 0

    # --- Kalman-style state ---
    # Smoothed position estimate (px)
    _est_x: float = 0.0
    _est_y: float = 0.0
    # Smoothed velocity estimate (px/frame)
    _est_vx: float = 0.0
    _est_vy: float = 0.0
    # Whether the Kalman state has been initialized
    _state_initialized: bool = False
    # Smoothing factor: 0 = trust prediction, 1 = trust measurement
    # Start high (trust measurements) and decrease as track matures
    _alpha: float = 0.6
    _alpha_min: float = 0.3

    # --- 3D state (populated when geometry is available) ---
    world_positions: List[Optional[Tuple[float, float, float]]] = field(
        default_factory=list
    )
    # Cached 3D speeds (mph) for each frame transition
    speeds_3d_mph: List[Optional[float]] = field(default_factory=list)

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
    def last_source(self) -> str:
        if self.sources:
            return self.sources[-1]
        return "unknown"

    @property
    def age(self) -> int:
        """Number of frames this track has existed."""
        return len(self.positions)

    @property
    def estimated_position(self) -> Tuple[float, float]:
        """Current smoothed position estimate."""
        if self._state_initialized:
            return (self._est_x, self._est_y)
        if self.positions:
            return self.positions[-1]
        return (0.0, 0.0)

    @property
    def estimated_velocity_px_per_frame(self) -> Tuple[float, float]:
        """Current smoothed velocity in px/frame."""
        if self._state_initialized:
            return (self._est_vx, self._est_vy)
        return (0.0, 0.0)

    def predicted_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """Predict where this object will be N frames from now."""
        ex, ey = self.estimated_position
        vx, vy = self.estimated_velocity_px_per_frame
        return (ex + vx * frames_ahead, ey + vy * frames_ahead)

    def _init_state(self, x: float, y: float):
        """Initialize the Kalman state from first measurement."""
        self._est_x = x
        self._est_y = y
        self._est_vx = 0.0
        self._est_vy = 0.0
        self._state_initialized = True
        self._alpha = 0.6

    def _update_state(self, measured_x: float, measured_y: float,
                      source_changed: bool = False):
        """
        Kalman-style predict + update step.

        If detection source changed (ME -> YOLO or vice versa),
        we increase trust in prediction to smooth the handoff.
        """
        if not self._state_initialized:
            self._init_state(measured_x, measured_y)
            return

        # --- Predict step ---
        pred_x = self._est_x + self._est_vx
        pred_y = self._est_y + self._est_vy

        # --- Measurement residual ---
        residual_x = measured_x - pred_x
        residual_y = measured_y - pred_y
        residual_dist = np.sqrt(residual_x ** 2 + residual_y ** 2)

        # --- Adaptive alpha ---
        # If source changed OR residual is large relative to velocity,
        # trust prediction more (lower alpha) to smooth the jump.
        alpha = self._alpha

        if source_changed:
            # At handoff, strongly trust the velocity model
            alpha = min(alpha, 0.2)

        # Also reduce alpha for large residuals (likely tracker artifact)
        expected_move = np.sqrt(self._est_vx ** 2 + self._est_vy ** 2)
        if expected_move > 0.5 and residual_dist > expected_move * 3.0:
            # Residual is 3x larger than expected - likely a jump
            alpha = min(alpha, 0.15)

        # --- Update step ---
        self._est_x = pred_x + alpha * residual_x
        self._est_y = pred_y + alpha * residual_y

        # Update velocity: use the smoothed position change
        new_vx = self._est_x - (pred_x - self._est_vx)  # = est_vx + alpha * residual_x
        new_vy = self._est_y - (pred_y - self._est_vy)

        # Blend velocity update (even more conservative)
        vel_alpha = alpha * 0.7
        self._est_vx = (1 - vel_alpha) * self._est_vx + vel_alpha * (new_vx - pred_x + self._est_vx)
        self._est_vy = (1 - vel_alpha) * self._est_vy + vel_alpha * (new_vy - pred_y + self._est_vy)

        # Decay alpha toward minimum as track matures (more trust in model)
        self._alpha = max(self._alpha_min, self._alpha * 0.98)

    def velocity_px(self) -> Optional[Tuple[float, float]]:
        """
        Smoothed pixel velocity from Kalman state.
        Returns (vx, vy) in pixels/frame, or None if not enough data.
        """
        if not self._state_initialized or self.age < 2:
            return None
        return (self._est_vx, self._est_vy)

    def speed_px_per_frame(self) -> float:
        """Smoothed scalar speed in pixels/frame."""
        v = self.velocity_px()
        if v is None:
            return 0.0
        return np.sqrt(v[0] ** 2 + v[1] ** 2)

    def speed_px_per_sec(self, fps: float) -> float:
        """Smoothed scalar speed in pixels/second."""
        return self.speed_px_per_frame() * fps

    def smoothed_speed_px_per_sec(self, fps: float, window: int = 5) -> float:
        """
        Smoothed speed using median of recent Kalman-estimated speeds.

        Uses the smoothed position estimates rather than raw measurements,
        which filters out handoff jumps.
        """
        if len(self.positions) < 2:
            return 0.0

        # Use estimated velocity directly (already smoothed by Kalman)
        # But also compute a windowed median for extra robustness
        n = min(window + 1, len(self.positions))
        recent_pos = self.positions[-n:]
        recent_frames = self.frame_indices[-n:]

        speeds = []
        for i in range(1, len(recent_pos)):
            dx = recent_pos[i][0] - recent_pos[i - 1][0]
            dy = recent_pos[i][1] - recent_pos[i - 1][1]
            df = recent_frames[i] - recent_frames[i - 1]
            if df > 0:
                px_per_frame = np.sqrt(dx * dx + dy * dy) / df
                speeds.append(px_per_frame * fps)

        if not speeds:
            return self.speed_px_per_sec(fps)

        # Use the Kalman velocity as one "vote" alongside the windowed median
        kalman_speed = self.speed_px_per_sec(fps)
        if kalman_speed > 0:
            speeds.append(kalman_speed)

        return float(np.median(speeds))

    def speed_3d_mph(self, window: int = 5) -> Optional[float]:
        """
        Get the smoothed 3D speed in mph from recent history.

        Uses the pre-computed 3D speeds (set by the pipeline when
        geometry is available). Falls back to None.
        """
        valid = [s for s in self.speeds_3d_mph[-window:] if s is not None]
        if not valid:
            return None
        return float(np.median(valid))

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
    Nearest-neighbor multi-object tracker with Kalman smoothing.

    Matches new detections to existing tracks using predicted positions.
    Detection source changes (ME -> YOLO) trigger handoff smoothing
    to prevent speed spikes.
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
        from common.filters import Detection
        det_by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            det_by_class.setdefault(det.cls_id, []).append(det)

        matched_track_ids = set()

        for cls_id, cls_dets in det_by_class.items():
            cls_tracks = {tid: t for tid, t in self._tracks.items()
                          if t.cls_id == cls_id and t.missed_frames < self.max_missed_frames}

            if not cls_tracks and not cls_dets:
                continue

            track_ids = list(cls_tracks.keys())
            if track_ids and cls_dets:
                costs = np.zeros((len(track_ids), len(cls_dets)))
                for i, tid in enumerate(track_ids):
                    track = cls_tracks[tid]
                    # Use PREDICTED position for matching (not last raw position)
                    pred_pos = track.predicted_position(
                        frames_ahead=track.missed_frames + 1
                    )
                    for j, det in enumerate(cls_dets):
                        dx = det.cx - pred_pos[0]
                        dy = det.cy - pred_pos[1]
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

                    # Detect source change for handoff smoothing
                    source_changed = (
                        len(track.sources) > 0 and
                        track.sources[-1] != det.source
                    )

                    # Update Kalman state (smooths handoff jumps)
                    track._update_state(det.cx, det.cy,
                                        source_changed=source_changed)

                    # Store raw measurement
                    track.positions.append((det.cx, det.cy))
                    track.frame_indices.append(frame_idx)
                    track.confidences.append(det.conf)
                    track.bboxes.append((det.x1, det.y1, det.x2, det.y2))
                    track.sources.append(det.source)
                    track.missed_frames = 0

                # Create new tracks for unmatched detections
                for j, det in enumerate(cls_dets):
                    if j not in used_dets:
                        self._create_track(det, frame_idx)
            else:
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
            sources=[getattr(det, 'source', 'yolo')],
        )
        track._init_state(det.cx, det.cy)
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
