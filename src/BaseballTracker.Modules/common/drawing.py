"""
Shared drawing functions for the baseball detection pipeline.

Provides all visual overlay drawing (bounding boxes, trails, HUD, events)
used by both smart_detect.py (offline video) and calibration_server.py (live).
All functions accept a DrawConfig for configurable font settings.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------- COCO class IDs ----------
BALL_CLASS = 32
BAT_CLASS = 34
PERSON_CLASS = 0

# ---------- Drawing colours (BGR) ----------
COLOR_BALL_ACTIVE = (0, 255, 0)       # Bright green
COLOR_BALL_TRAIL = (0, 200, 0)        # Dimmer green
COLOR_BAT = (0, 165, 255)             # Orange
COLOR_PERSON = (0, 255, 255)          # Yellow
COLOR_STATIONARY = (100, 100, 100)    # Dim gray
COLOR_REJECTED = (0, 0, 180)          # Dark red
COLOR_ROI = (255, 255, 255)           # White
COLOR_CONTACT = (0, 255, 255)         # Yellow flash
COLOR_HUD_BG = (0, 0, 0)             # Black
COLOR_HUD_TEXT = (255, 255, 255)      # White
COLOR_MACHINE_EXIT = (255, 200, 0)    # Cyan-ish for machine-exit detections
COLOR_SEARCH_REGION = (80, 80, 80)    # Dim gray for search region outline
COLOR_3D_INFO = (200, 200, 100)       # Light blue-ish for 3D data
COLOR_HIT = (0, 255, 0)              # Green for HIT
COLOR_MISS = (0, 0, 255)             # Red for MISS
COLOR_SPEED = (255, 220, 100)         # Light cyan for speed display

ACTIVE_COLORS = {
    BALL_CLASS: COLOR_BALL_ACTIVE,
    BAT_CLASS: COLOR_BAT,
    PERSON_CLASS: COLOR_PERSON,
}

# OpenCV font constants lookup
FONT_MAP = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "script_simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script_complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}


@dataclass
class DrawConfig:
    """Configurable font and drawing parameters for the HUD."""
    font_family: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.55
    font_thickness: int = 1
    hud_font_scale: float = 0.5
    hud_line_height: int = 22

    @classmethod
    def from_config(cls, config: dict) -> "DrawConfig":
        """Create DrawConfig from cage_config.json hud section."""
        hud = config.get("hud", {})
        font_name = hud.get("font_family", "simplex")
        font_family = FONT_MAP.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)
        return cls(
            font_family=font_family,
            font_scale=hud.get("font_scale", 0.55),
            font_thickness=hud.get("font_thickness", 1),
            hud_font_scale=hud.get("hud_font_scale", 0.5),
            hud_line_height=hud.get("line_height", 22),
        )


def draw_roi(frame: np.ndarray, polygon: list, dc: DrawConfig = None):
    """Draw ROI polygon as dashed white outline."""
    if not polygon or len(polygon) < 3:
        return
    pts = np.array(polygon, dtype=np.int32)
    for i in range(len(pts)):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % len(pts)])
        cv2.line(frame, p1, p2, COLOR_ROI, 1, cv2.LINE_AA)


def draw_stationary_markers(frame: np.ndarray, positions: list,
                             dc: DrawConfig = None):
    """Draw dim circles at known stationary detection positions."""
    dc = dc or DrawConfig()
    for (cx, cy) in positions:
        cv2.circle(frame, (int(cx), int(cy)), 12, COLOR_STATIONARY, 1)
        cv2.putText(frame, "S", (int(cx) - 4, int(cy) + 4),
                    dc.font_family, 0.35, COLOR_STATIONARY, 1)


def draw_rejected(frame: np.ndarray, results: list, dc: DrawConfig = None):
    """Draw rejected detections as dim indicators."""
    for fr in results:
        det = fr.detection
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cx, cy = int(det.cx), int(det.cy)

        if "Stationary" in fr.reason:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_STATIONARY, 1)
        else:
            size = 6
            cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size),
                     COLOR_REJECTED, 1)
            cv2.line(frame, (cx - size, cy + size), (cx + size, cy - size),
                     COLOR_REJECTED, 1)


def draw_machine_exit_region(frame: np.ndarray, search_region,
                              detection=None, dc: DrawConfig = None):
    """Draw the machine-exit search region and any detection from it."""
    dc = dc or DrawConfig()
    x1, y1, x2, y2 = search_region
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SEARCH_REGION, 1)
    cv2.putText(frame, "MACHINE-EXIT", (x1, y1 - 5),
                dc.font_family, 0.35, COLOR_SEARCH_REGION, 1)

    if detection is not None:
        cx, cy, radius, conf, diff_val = detection
        cx, cy = int(cx), int(cy)
        r = max(int(radius * 2), 6)
        cv2.circle(frame, (cx, cy), r, COLOR_MACHINE_EXIT, 2)
        cv2.line(frame, (cx - r - 4, cy), (cx + r + 4, cy), COLOR_MACHINE_EXIT, 1)
        cv2.line(frame, (cx, cy - r - 4), (cx, cy + r + 4), COLOR_MACHINE_EXIT, 1)
        cv2.putText(frame, f"ME {conf:.0%}", (cx + r + 4, cy - 2),
                    dc.font_family, 0.4, COLOR_MACHINE_EXIT, 1, cv2.LINE_AA)


def draw_tracked_objects(frame: np.ndarray, tracks: list, fps: float,
                          calibration, dc: DrawConfig = None):
    """Draw bounding boxes, labels, trails for tracked objects."""
    dc = dc or DrawConfig()

    for track in tracks:
        if track.missed_frames > 0:
            continue

        bbox = track.last_bbox
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = ACTIVE_COLORS.get(track.cls_id, COLOR_HUD_TEXT)
        cx, cy = int(track.last_position[0]), int(track.last_position[1])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- Speed label ---
        speed_label = ""
        depth_label = ""

        if track.cls_id == BALL_CLASS:
            speed_3d = track.speed_3d_mph(window=5)
            if speed_3d is not None and 5.0 < speed_3d < 150.0:
                speed_label = f"BALL {speed_3d:.0f}mph"
            else:
                speed_px = track.smoothed_speed_px_per_sec(fps, window=5)
                if calibration and calibration.is_calibrated:
                    flat_mph = calibration.to_mph(speed_px)
                    if flat_mph is not None and flat_mph < 120:
                        speed_label = f"BALL ~{flat_mph:.0f}mph"
                    else:
                        speed_label = f"BALL {speed_px:.0f}px/s"
                else:
                    speed_label = f"BALL {speed_px:.0f}px/s"

            # Depth annotation (3D)
            if calibration:
                depth = calibration.get_depth_ft(
                    track.last_position[0], track.last_position[1]
                )
                if depth is not None:
                    depth_label = f"{depth:.0f}ft"

            # Source indicator
            src = track.last_source
            if src == "machine_exit":
                speed_label += " [ME]"

        elif track.cls_id == BAT_CLASS:
            speed_px = track.smoothed_speed_px_per_sec(fps, window=5)
            if calibration and calibration.is_calibrated:
                flat_mph = calibration.to_mph(speed_px)
                if flat_mph is not None and flat_mph < 120:
                    speed_label = f"BAT {flat_mph:.0f}mph"
                else:
                    speed_label = f"BAT {speed_px:.0f}px/s"
            else:
                speed_label = f"BAT {speed_px:.0f}px/s"
        else:
            speed_label = f"{track.cls_name}"

        conf = track.confidences[-1] if track.confidences else 0
        speed_label += f" {conf:.0%}"

        # Draw label background + text
        (tw, th), _ = cv2.getTextSize(speed_label, dc.font_family,
                                       dc.hud_font_scale, dc.font_thickness)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, speed_label, (x1 + 2, y1 - 4),
                    dc.font_family, dc.hud_font_scale, (0, 0, 0),
                    dc.font_thickness, cv2.LINE_AA)

        # Depth label
        if depth_label:
            cv2.putText(frame, depth_label, (x1 + 2, y2 + 14),
                        dc.font_family, 0.4, COLOR_3D_INFO, 1, cv2.LINE_AA)

        # Center dot for ball
        if track.cls_id == BALL_CLASS:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Trail line
            trail = track.get_trail(max_points=30)
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, trail[i - 1], trail[i],
                             COLOR_BALL_TRAIL, thickness, cv2.LINE_AA)


def draw_contact_flash(frame: np.ndarray, dc: DrawConfig = None):
    """Flash frame border yellow to indicate bat-ball contact."""
    dc = dc or DrawConfig()
    h, w = frame.shape[:2]
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CONTACT, thickness)
    cv2.putText(frame, "CONTACT!", (w // 2 - 80, 50),
                dc.font_family, 1.2, COLOR_CONTACT, 3, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, frame_idx: int, fps: float,
             ball_count: int, active_tracks: int,
             contact_detected: bool, calibration,
             best_ball=None, dc: DrawConfig = None,
             detection_fps: float = None):
    """Draw heads-up display with frame info and status."""
    dc = dc or DrawConfig()
    time_s = frame_idx / fps if fps > 0 else 0
    lines = [
        f"Frame {frame_idx} | {time_s:.2f}s",
    ]

    if detection_fps is not None:
        lines[0] = f"Frame {frame_idx} | Det: {detection_fps:.0f}fps"
    else:
        lines[0] = f"Frame {frame_idx} | {time_s:.2f}s"

    lines.append(f"Active tracks: {active_tracks} | Balls: {ball_count}")

    if best_ball and best_ball.age >= 2:
        speed_3d = best_ball.speed_3d_mph(window=5)
        if speed_3d is not None and 5.0 < speed_3d < 150.0:
            lines.append(f"Ball speed: {speed_3d:.0f} mph (3D)")
        else:
            speed_px = best_ball.smoothed_speed_px_per_sec(fps, window=5)
            if calibration and hasattr(calibration, 'to_mph'):
                flat_mph = calibration.to_mph(speed_px)
                if flat_mph is not None:
                    lines.append(f"Ball speed: {flat_mph:.0f} mph ({speed_px:.0f} px/s)")
                else:
                    lines.append(f"Ball speed: {speed_px:.0f} px/s")
            else:
                lines.append(f"Ball speed: {speed_px:.0f} px/s")

        # Show depth if available
        pos = best_ball.last_position
        if pos and calibration:
            depth = calibration.get_depth_ft(pos[0], pos[1])
            if depth is not None:
                lines.append(f"Ball depth: {depth:.0f} ft from plate")

        # Show detection source
        src = best_ball.last_source
        lines.append(f"Source: {src.upper()}")

    if contact_detected:
        lines.append(">> CONTACT DETECTED <<")

    # Draw HUD background
    max_width = max(cv2.getTextSize(l, dc.font_family, dc.font_scale,
                                     dc.font_thickness)[0][0] for l in lines)
    lh = dc.hud_line_height
    hud_h = len(lines) * lh + 16
    cv2.rectangle(frame, (5, 5), (max_width + 20, hud_h), COLOR_HUD_BG, -1)
    cv2.rectangle(frame, (5, 5), (max_width + 20, hud_h), (80, 80, 80), 1)

    y = 5 + lh - 2
    for line in lines:
        color = COLOR_CONTACT if "CONTACT" in line else COLOR_HUD_TEXT
        cv2.putText(frame, line, (12, y),
                    dc.font_family, dc.font_scale, color,
                    dc.font_thickness, cv2.LINE_AA)
        y += lh

    # Calibration info (bottom-left)
    if calibration and hasattr(calibration, 'get_info_lines'):
        cal_lines = calibration.get_info_lines()
        y_cal = frame.shape[0] - 15
        for line in reversed(cal_lines):
            cv2.putText(frame, line, (10, y_cal),
                        dc.font_family, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            y_cal -= 18


def draw_event_overlays(frame: np.ndarray, pending_events: list,
                         dc: DrawConfig = None):
    """
    Draw event overlays: HIT!/MISS text and ball speed in lower-left.

    pending_events is a list of (event, ttl) tuples where ttl counts down
    each frame. Events have attributes: type ('hit'|'miss'|'speed'),
    and associated data.
    """
    dc = dc or DrawConfig()
    h, w = frame.shape[:2]

    for event_info in pending_events:
        event = event_info["event"]
        ttl = event_info["ttl"]
        event_type = event_info["type"]

        if event_type == "hit_miss":
            is_hit = event.is_hit
            text = "HIT!" if is_hit else "MISS"
            color = COLOR_HIT if is_hit else COLOR_MISS

            # Scale for large centered text
            scale = dc.font_scale * 4.0
            thickness = max(3, dc.font_thickness * 3)

            # Fade based on TTL (0-1)
            max_ttl = event_info.get("max_ttl", 60)
            alpha = min(1.0, ttl / (max_ttl * 0.5))

            # Fade color toward black
            faded = tuple(int(c * alpha) for c in color)

            (tw, th), _ = cv2.getTextSize(text, dc.font_family, scale, thickness)
            tx = (w - tw) // 2
            ty = (h + th) // 2

            # Draw shadow + text
            cv2.putText(frame, text, (tx + 2, ty + 2),
                        dc.font_family, scale, (0, 0, 0), thickness + 1,
                        cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty),
                        dc.font_family, scale, faded, thickness,
                        cv2.LINE_AA)

        elif event_type == "speed":
            speed_mph = event.speed_mph
            if speed_mph is not None:
                text = f"{speed_mph:.0f} MPH"
                scale = dc.font_scale * 3.0
                thickness = max(2, dc.font_thickness * 2)

                (tw, th), _ = cv2.getTextSize(text, dc.font_family,
                                               scale, thickness)
                tx = 20
                ty = h - 30

                # Background
                cv2.rectangle(frame, (tx - 5, ty - th - 10),
                              (tx + tw + 10, ty + 10), COLOR_HUD_BG, -1)
                cv2.putText(frame, text, (tx, ty),
                            dc.font_family, scale, COLOR_SPEED, thickness,
                            cv2.LINE_AA)
