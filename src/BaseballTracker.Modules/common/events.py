"""
Event detection state machine for baseball tracking.

Detects three event types from tracker state:
  1. PitchEvent   — ball released from machine area
  2. PlateCrossingEvent — ball reaches home plate zone
  3. HitEvent     — hit (bat contact / direction change) or miss

State machine: IDLE → PITCH_DETECTED → APPROACHING_PLATE → RESOLVED → IDLE
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class PitchEvent:
    """Ball detected leaving the machine area."""
    frame_idx: int
    timestamp_s: float
    release_position_px: Tuple[float, float]


@dataclass
class PlateCrossingEvent:
    """Ball reached the home plate zone."""
    frame_idx: int
    timestamp_s: float
    position_px: Tuple[float, float]
    speed_mph: Optional[float] = None


@dataclass
class HitEvent:
    """Hit or miss determination after plate crossing."""
    frame_idx: int
    timestamp_s: float
    position_px: Tuple[float, float]
    is_hit: bool
    contact_speed_mph: Optional[float] = None


class EventDetector:
    """
    State-machine event detector that consumes tracker state each frame.

    Detects pitch release, plate crossing, and hit/miss events.
    Manages display TTL for overlay rendering.
    """

    # States
    IDLE = "IDLE"
    PITCH_DETECTED = "PITCH_DETECTED"
    APPROACHING = "APPROACHING_PLATE"
    RESOLVED = "RESOLVED"

    def __init__(self, config: dict, fps: float):
        """
        Args:
            config: Full cage_config dict (needs calibration, event_detection sections)
            fps: Video frame rate
        """
        self.fps = fps

        # Machine bbox for pitch detection
        cal = config.get("calibration", {})
        self.machine_bbox = cal.get("machine_bbox_px")
        self.home_plate_px = cal.get("home_plate_px")

        # Event detection config
        evt = config.get("event_detection", {})
        self.home_plate_radius = evt.get("home_plate_radius_px", 80)
        self.hit_check_frames = evt.get("hit_check_frames", 15)
        self.velocity_reversal_threshold = evt.get(
            "hit_velocity_reversal_threshold", 5.0
        )
        self.display_duration = evt.get("display_duration_frames", 60)
        self.speed_display_duration = evt.get("speed_display_duration_frames", 120)

        # Tracking config for contact detection
        self.contact_distance = config.get("tracking", {}).get(
            "contact_distance_px", 60
        )

        # Machine search region (expanded bbox for pitch detection)
        self.machine_search_margin = 50  # px margin around machine bbox

        # State machine
        self._state = self.IDLE
        self._pitch_event: Optional[PitchEvent] = None
        self._pitch_track_id: Optional[int] = None
        self._crossing_event: Optional[PlateCrossingEvent] = None
        self._crossing_frame: Optional[int] = None
        self._resolve_deadline: Optional[int] = None

        # All detected events (log)
        self.events: List[Any] = []

        # Display queue: {"event": obj, "type": str, "ttl": int, "max_ttl": int}
        self.pending_display: List[Dict] = []

        # Track velocity history for direction reversal detection
        self._prev_velocities: List[Tuple[float, float]] = []

    def reset(self):
        """Reset state machine for new session."""
        self._state = self.IDLE
        self._pitch_event = None
        self._pitch_track_id = None
        self._crossing_event = None
        self._crossing_frame = None
        self._resolve_deadline = None
        self.events.clear()
        self.pending_display.clear()
        self._prev_velocities.clear()

    def update(self, tracker, frame_idx: int) -> List[Any]:
        """
        Called each frame. Examines tracker state and emits events.

        Args:
            tracker: ObjectTracker instance
            frame_idx: Current frame index

        Returns:
            List of newly emitted events this frame.
        """
        from common.tracking import detect_contact

        new_events = []
        timestamp = frame_idx / self.fps if self.fps > 0 else 0

        best_ball = tracker.get_best_ball_track()
        best_bat = tracker.get_best_bat_track()

        # --- Update display TTLs ---
        self.pending_display = [
            {**d, "ttl": d["ttl"] - 1}
            for d in self.pending_display
            if d["ttl"] > 1
        ]

        # --- State machine ---
        if self._state == self.IDLE:
            # Look for a new ball track near the machine
            if best_ball and best_ball.age >= 2 and best_ball.missed_frames == 0:
                if self._is_near_machine(best_ball.positions[0]):
                    event = PitchEvent(
                        frame_idx=best_ball.frame_indices[0],
                        timestamp_s=best_ball.frame_indices[0] / self.fps,
                        release_position_px=best_ball.positions[0],
                    )
                    self._pitch_event = event
                    self._pitch_track_id = best_ball.track_id
                    self._state = self.PITCH_DETECTED
                    self._prev_velocities.clear()
                    new_events.append(event)
                    self.events.append(event)

        elif self._state == self.PITCH_DETECTED:
            # Transition to APPROACHING once ball is moving toward plate
            if best_ball and best_ball.track_id == self._pitch_track_id:
                if best_ball.missed_frames == 0 and best_ball.age >= 3:
                    self._state = self.APPROACHING
            elif best_ball is None or best_ball.track_id != self._pitch_track_id:
                # Lost the pitch track — reset
                self._state = self.IDLE

        elif self._state == self.APPROACHING:
            if best_ball and best_ball.track_id == self._pitch_track_id:
                if best_ball.missed_frames == 0:
                    pos = best_ball.last_position

                    # Track velocity for direction reversal detection
                    vel = best_ball.velocity_px()
                    if vel:
                        self._prev_velocities.append(vel)
                        # Keep last 10
                        if len(self._prev_velocities) > 10:
                            self._prev_velocities.pop(0)

                    # Check if ball reached plate zone
                    if self._is_near_plate(pos):
                        speed = best_ball.speed_3d_mph(window=5)
                        event = PlateCrossingEvent(
                            frame_idx=frame_idx,
                            timestamp_s=timestamp,
                            position_px=pos,
                            speed_mph=speed,
                        )
                        self._crossing_event = event
                        self._crossing_frame = frame_idx
                        self._resolve_deadline = frame_idx + self.hit_check_frames
                        new_events.append(event)
                        self.events.append(event)

                        # Add speed display
                        self.pending_display.append({
                            "event": event,
                            "type": "speed",
                            "ttl": self.speed_display_duration,
                            "max_ttl": self.speed_display_duration,
                        })

                        self._state = self.RESOLVED
                        # Immediately check for contact
                        self._check_hit(
                            best_ball, best_bat, frame_idx, timestamp,
                            new_events
                        )

                elif best_ball.missed_frames >= 5:
                    # Lost track before plate — probably went out of frame
                    # Still try to get speed from last known data
                    self._state = self.IDLE
            else:
                # Track switched to a different ball — lost pitch
                self._state = self.IDLE

        elif self._state == self.RESOLVED:
            # Wait for hit check deadline
            if self._resolve_deadline and frame_idx >= self._resolve_deadline:
                # If no hit was already detected, emit a miss
                has_hit = any(
                    isinstance(e, HitEvent) and e.frame_idx >= self._crossing_frame
                    for e in self.events
                )
                if not has_hit:
                    pos = (self._crossing_event.position_px
                           if self._crossing_event else (0, 0))
                    miss = HitEvent(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp,
                        position_px=pos,
                        is_hit=False,
                    )
                    new_events.append(miss)
                    self.events.append(miss)
                    self.pending_display.append({
                        "event": miss,
                        "type": "hit_miss",
                        "ttl": self.display_duration,
                        "max_ttl": self.display_duration,
                    })
                self._state = self.IDLE
            else:
                # Still checking for hit
                if best_ball and best_bat:
                    self._check_hit(
                        best_ball, best_bat, frame_idx, timestamp,
                        new_events
                    )

        return new_events

    def _check_hit(self, ball_track, bat_track, frame_idx: int,
                   timestamp: float, new_events: list):
        """Check for hit via proximity or velocity reversal."""
        from common.tracking import detect_contact

        # Already resolved?
        has_hit = any(
            isinstance(e, HitEvent) and e.frame_idx >= (self._crossing_frame or 0)
            for e in self.events
        )
        if has_hit:
            return

        # Method 1: Proximity-based contact
        contact_frame = detect_contact(ball_track, bat_track, self.contact_distance)
        if contact_frame is not None:
            speed = ball_track.speed_3d_mph(window=5)
            hit = HitEvent(
                frame_idx=frame_idx,
                timestamp_s=timestamp,
                position_px=ball_track.last_position or (0, 0),
                is_hit=True,
                contact_speed_mph=speed,
            )
            new_events.append(hit)
            self.events.append(hit)
            self.pending_display.append({
                "event": hit,
                "type": "hit_miss",
                "ttl": self.display_duration,
                "max_ttl": self.display_duration,
            })
            return

        # Method 2: Velocity direction reversal
        if len(self._prev_velocities) >= 4:
            recent = self._prev_velocities[-3:]
            older = self._prev_velocities[-6:-3] if len(self._prev_velocities) >= 6 else self._prev_velocities[:3]

            # Check for sign flip in x or y velocity component
            avg_old_vx = np.mean([v[0] for v in older])
            avg_old_vy = np.mean([v[1] for v in older])
            avg_new_vx = np.mean([v[0] for v in recent])
            avg_new_vy = np.mean([v[1] for v in recent])

            old_speed = np.sqrt(avg_old_vx**2 + avg_old_vy**2)
            new_speed = np.sqrt(avg_new_vx**2 + avg_new_vy**2)

            # Direction reversal: significant velocity AND sign change
            if old_speed > self.velocity_reversal_threshold:
                # Check if dominant velocity component reversed
                vx_reversed = (avg_old_vx * avg_new_vx < 0
                               and abs(avg_new_vx) > 1.0)
                vy_reversed = (avg_old_vy * avg_new_vy < 0
                               and abs(avg_new_vy) > 1.0)

                if (vx_reversed or vy_reversed) and new_speed > 2.0:
                    speed = ball_track.speed_3d_mph(window=5)
                    hit = HitEvent(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp,
                        position_px=ball_track.last_position or (0, 0),
                        is_hit=True,
                        contact_speed_mph=speed,
                    )
                    new_events.append(hit)
                    self.events.append(hit)
                    self.pending_display.append({
                        "event": hit,
                        "type": "hit_miss",
                        "ttl": self.display_duration,
                        "max_ttl": self.display_duration,
                    })

    def _is_near_machine(self, pos: Tuple[float, float]) -> bool:
        """Check if position is near the machine bbox."""
        if not self.machine_bbox or len(self.machine_bbox) != 4:
            # Without machine bbox, accept any ball track
            return True
        mx1, my1, mx2, my2 = self.machine_bbox
        m = self.machine_search_margin
        return (mx1 - m <= pos[0] <= mx2 + m and
                my1 - m <= pos[1] <= my2 + m)

    def _is_near_plate(self, pos: Tuple[float, float]) -> bool:
        """Check if position is near home plate."""
        if not self.home_plate_px:
            # Without home plate configured, use bottom-center heuristic:
            # Consider "near plate" if ball is in the lower 30% of frame
            # and has traveled significant distance from machine
            return False
        hx, hy = self.home_plate_px
        dx = pos[0] - hx
        dy = pos[1] - hy
        return np.sqrt(dx * dx + dy * dy) < self.home_plate_radius

    def get_event_log(self, since_frame: int = 0) -> List[Dict]:
        """Return events as serializable dicts, optionally filtered by frame."""
        result = []
        for e in self.events:
            if e.frame_idx < since_frame:
                continue
            entry = {
                "frame_idx": e.frame_idx,
                "timestamp_s": round(e.timestamp_s, 3),
            }
            if isinstance(e, PitchEvent):
                entry["type"] = "pitch"
                entry["position_px"] = list(e.release_position_px)
            elif isinstance(e, PlateCrossingEvent):
                entry["type"] = "plate_crossing"
                entry["position_px"] = list(e.position_px)
                entry["speed_mph"] = (round(e.speed_mph, 1)
                                      if e.speed_mph else None)
            elif isinstance(e, HitEvent):
                entry["type"] = "hit" if e.is_hit else "miss"
                entry["position_px"] = list(e.position_px)
                entry["is_hit"] = e.is_hit
                entry["speed_mph"] = (round(e.contact_speed_mph, 1)
                                      if e.contact_speed_mph else None)
            result.append(entry)
        return result
