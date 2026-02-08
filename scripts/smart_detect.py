"""
Smart Detection Pipeline: filters, tracks, and annotates batting cage video.

Applies a multi-stage filter pipeline to YOLO detections:
  1. ROI mask (inside cage only)
  2. Class filter (ball, bat, person only)
  3. Size filter (reject tiny false positives like red lights)
  4. Per-class confidence thresholds
  5. Stationary object filter (ignore balls that don't move)
  6. Frame-to-frame tracking with trajectory trail

Outputs an annotated MP4 with color-coded detections and a HUD.

Usage (inside Docker):
    python /app/scripts/smart_detect.py /videos/clip.mov /output/smart.mp4
    python /app/scripts/smart_detect.py /videos/clip.mov /output/smart.mp4 --config /app/config/cage_config.json
"""
import sys
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Add common module path
sys.path.insert(0, "/app")

from common.filters import (
    Detection, FilterResult, StationaryFilter,
    filter_by_roi, filter_by_class, filter_by_size, filter_by_confidence,
)
from common.tracking import ObjectTracker, TrackedObject, detect_contact
from common.calibration import load_calibration, CageCalibration

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

ACTIVE_COLORS = {
    BALL_CLASS: COLOR_BALL_ACTIVE,
    BAT_CLASS: COLOR_BAT,
    PERSON_CLASS: COLOR_PERSON,
}


def load_config(config_path: str) -> dict:
    """Load cage configuration from JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


def yolo_to_detections(results, model, frame_idx: int) -> list:
    """Convert YOLO results to our Detection dataclass list."""
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(Detection(
                cls_id=cls_id,
                cls_name=model.names[cls_id],
                conf=float(box.conf[0]),
                x1=float(box.xyxy[0][0]),
                y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]),
                y2=float(box.xyxy[0][3]),
                frame_idx=frame_idx,
            ))
    return detections


def draw_roi(frame: np.ndarray, polygon: list):
    """Draw ROI polygon as dashed white outline."""
    if not polygon or len(polygon) < 3:
        return
    pts = np.array(polygon, dtype=np.int32)
    # Draw dashed-like effect with short line segments
    for i in range(len(pts)):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % len(pts)])
        cv2.line(frame, p1, p2, COLOR_ROI, 1, cv2.LINE_AA)


def draw_stationary_markers(frame: np.ndarray, positions: list):
    """Draw dim circles at known stationary detection positions."""
    for (cx, cy) in positions:
        cv2.circle(frame, (int(cx), int(cy)), 12, COLOR_STATIONARY, 1)
        # Small 'S' label
        cv2.putText(frame, "S", (int(cx) - 4, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_STATIONARY, 1)


def draw_rejected(frame: np.ndarray, results: list):
    """Draw rejected detections as dim indicators."""
    for fr in results:
        det = fr.detection
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cx, cy = int(det.cx), int(det.cy)

        if "Stationary" in fr.reason:
            # Dim gray box for stationary objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_STATIONARY, 1)
        else:
            # Small red X for other rejections
            size = 6
            cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size),
                     COLOR_REJECTED, 1)
            cv2.line(frame, (cx - size, cy + size), (cx + size, cy - size),
                     COLOR_REJECTED, 1)


def draw_tracked_objects(frame: np.ndarray, tracks: list, fps: float,
                         calibration: CageCalibration):
    """Draw bounding boxes, labels, trails for tracked objects."""
    for track in tracks:
        if track.missed_frames > 0:
            continue  # Don't draw objects not seen this frame

        bbox = track.last_bbox
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = ACTIVE_COLORS.get(track.cls_id, COLOR_HUD_TEXT)
        cx, cy = int(track.last_position[0]), int(track.last_position[1])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        speed_px = track.speed_px_per_sec(fps)
        speed_mph = calibration.to_mph(speed_px) if calibration.is_calibrated else None

        if track.cls_id == BALL_CLASS:
            if speed_mph is not None:
                label = f"BALL {speed_mph:.0f}mph"
            else:
                label = f"BALL {speed_px:.0f}px/s"
        elif track.cls_id == BAT_CLASS:
            if speed_mph is not None:
                label = f"BAT {speed_mph:.0f}mph"
            else:
                label = f"BAT {speed_px:.0f}px/s"
        else:
            label = f"{track.cls_name}"

        conf = track.confidences[-1] if track.confidences else 0
        label += f" {conf:.0%}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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


def draw_contact_flash(frame: np.ndarray):
    """Flash frame border yellow to indicate bat-ball contact."""
    h, w = frame.shape[:2]
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CONTACT, thickness)
    cv2.putText(frame, "CONTACT!", (w // 2 - 80, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_CONTACT, 3, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, frame_idx: int, fps: float,
             ball_count: int, active_tracks: int,
             contact_detected: bool, calibration: CageCalibration,
             best_ball: TrackedObject = None):
    """Draw heads-up display with frame info and status."""
    time_s = frame_idx / fps
    lines = [
        f"Frame {frame_idx} | {time_s:.2f}s",
        f"Active tracks: {active_tracks} | Balls: {ball_count}",
    ]

    if best_ball and best_ball.age >= 2:
        speed_px = best_ball.speed_px_per_sec(fps)
        speed_mph = calibration.to_mph(speed_px)
        if speed_mph is not None:
            lines.append(f"Ball speed: {speed_mph:.0f} mph ({speed_px:.0f} px/s)")
        else:
            lines.append(f"Ball speed: {speed_px:.0f} px/s")

    if contact_detected:
        lines.append(">> CONTACT DETECTED <<")

    # Draw HUD background
    max_width = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                    for l in lines)
    hud_h = len(lines) * 24 + 16
    cv2.rectangle(frame, (5, 5), (max_width + 20, hud_h), COLOR_HUD_BG, -1)
    cv2.rectangle(frame, (5, 5), (max_width + 20, hud_h), (80, 80, 80), 1)

    # Draw text
    y = 25
    for line in lines:
        color = COLOR_CONTACT if "CONTACT" in line else COLOR_HUD_TEXT
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24

    # Calibration info (bottom-left)
    cal_lines = calibration.get_info_lines()
    y_cal = frame.shape[0] - 15
    for line in reversed(cal_lines):
        cv2.putText(frame, line, (10, y_cal),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        y_cal -= 18


def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_detect.py <input_video> <output_video> [--config path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Config paths (defaults for Docker volume mounts)
    config_path = "/app/config/cage_config.json"
    machines_path = "/app/config/machines.json"

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    # Load configuration
    print(f"=== Smart Detection Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")

    config = load_config(config_path)
    calibration = load_calibration(config_path, machines_path)

    print(f"\nMachine: {calibration.machine_spec.name}")
    print(f"Distance: {calibration.machine_distance_ft} ft")
    print(f"Calibrated: {'YES' if calibration.is_calibrated else 'NO (pixel speed only)'}")

    # Detection config
    det_config = config["detection"]
    model_name = det_config["model"]
    allowed_classes = det_config.get("allowed_classes", [0, 32, 34])
    conf_thresholds = {}
    for name, threshold in det_config["confidence_thresholds"].items():
        # Map config names to COCO class IDs
        name_to_id = {"sports_ball": 32, "baseball_bat": 34, "person": 0}
        if name in name_to_id:
            conf_thresholds[name_to_id[name]] = threshold

    # Size filter config: class_id -> (min_width, min_height)
    min_sizes = {}
    size_config = det_config.get("min_size_px", {})
    if "person_height" in size_config:
        min_sizes[PERSON_CLASS] = (
            size_config.get("person_width", 0),
            size_config.get("person_height", 0)
        )

    # Stationary filter
    stat_config = config["stationary_filter"]
    stationary_filter = StationaryFilter(
        history_frames=stat_config["history_frames"],
        max_variance_px=stat_config["max_variance_px"],
    )

    # Tracker
    track_config = config["tracking"]
    tracker = ObjectTracker(
        max_match_distance=track_config["max_match_distance_px"],
    )
    contact_distance = track_config.get("contact_distance_px", 60)

    # ROI polygon
    roi_polygon = config.get("roi_polygon", [])

    # Open video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps, {total_frames} frames ({duration:.1f}s)")
    print(f"ROI polygon: {len(roi_polygon)} points {'(active)' if roi_polygon else '(disabled - full frame)'}")
    print(f"Filters: class={allowed_classes}, conf={conf_thresholds}, size={min_sizes}")
    print(f"Stationary: history={stat_config['history_frames']}f, var<{stat_config['max_variance_px']}px")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load YOLO
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)

    print(f"Processing frames...\n")

    # Stats
    frame_idx = 0
    total_detections = 0
    total_filtered = 0
    total_tracked_ball_frames = 0
    contact_frames = []
    max_ball_speed_px = 0.0
    max_ball_speed_mph = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLO inference ---
        results = model(frame, verbose=False, conf=0.10)  # low conf, we filter later
        raw_detections = yolo_to_detections(results, model, frame_idx)
        total_detections += len(raw_detections)

        # --- Filter pipeline ---
        all_rejected = []

        # Stage 1: ROI
        kept, rejected = filter_by_roi(raw_detections, roi_polygon)
        all_rejected.extend(rejected)

        # Stage 2: Class filter
        kept, rejected = filter_by_class(kept, allowed_classes)
        all_rejected.extend(rejected)

        # Stage 3: Size filter
        kept, rejected = filter_by_size(kept, min_sizes)
        all_rejected.extend(rejected)

        # Stage 4: Confidence filter
        kept, rejected = filter_by_confidence(kept, conf_thresholds)
        all_rejected.extend(rejected)

        # Stage 5: Stationary filter
        kept, rejected = stationary_filter.update(kept)
        all_rejected.extend(rejected)

        total_filtered += len(all_rejected)

        # --- Tracking ---
        active_tracks = tracker.update(kept, frame_idx)
        best_ball = tracker.get_best_ball_track()
        best_bat = tracker.get_best_bat_track()

        ball_count = len([t for t in active_tracks
                          if t.cls_id == BALL_CLASS and t.missed_frames == 0])
        if ball_count > 0:
            total_tracked_ball_frames += 1

        # Track max speed
        if best_ball and best_ball.age >= 2:
            speed_px = best_ball.speed_px_per_sec(fps)
            if speed_px > max_ball_speed_px:
                max_ball_speed_px = speed_px
                mph = calibration.to_mph(speed_px)
                if mph is not None:
                    max_ball_speed_mph = mph

        # --- Contact detection ---
        contact_frame = detect_contact(best_ball, best_bat, contact_distance)
        if contact_frame is not None and contact_frame not in contact_frames:
            contact_frames.append(contact_frame)

        is_contact = contact_frame is not None

        # --- Draw annotations ---
        annotated = frame.copy()

        # ROI boundary
        draw_roi(annotated, roi_polygon)

        # Stationary position markers
        draw_stationary_markers(annotated, stationary_filter.get_stationary_positions())

        # Rejected detections (dim)
        draw_rejected(annotated, all_rejected)

        # Active tracked objects
        draw_tracked_objects(annotated, active_tracks, fps, calibration)

        # Contact flash
        if is_contact:
            draw_contact_flash(annotated)

        # HUD
        draw_hud(annotated, frame_idx, fps, ball_count, len(active_tracks),
                 is_contact, calibration, best_ball)

        writer.write(annotated)
        frame_idx += 1

        # Progress
        if frame_idx % 60 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)"
                  f"  | tracked ball in {total_tracked_ball_frames} frames"
                  f"  | filtered {total_filtered} detections")

    cap.release()
    writer.release()

    # --- Summary ---
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"SMART DETECTION RESULTS")
    print(f"{'=' * 50}")
    print(f"Frames processed:   {frame_idx}")
    print(f"Total detections:   {total_detections}")
    print(f"Filtered out:       {total_filtered} ({total_filtered / max(total_detections, 1) * 100:.0f}%)")
    print(f"Ball tracked in:    {total_tracked_ball_frames} frames")
    print(f"Stationary clusters: {len(stationary_filter.get_stationary_positions())}")
    print(f"Contact frames:     {contact_frames if contact_frames else 'None detected'}")

    print(f"\nMax ball speed:     {max_ball_speed_px:.0f} px/s", end="")
    if max_ball_speed_mph > 0:
        print(f" ({max_ball_speed_mph:.0f} mph)")
    else:
        print(f" (calibrate machine bbox for mph)")

    print(f"\nOutput: {output_path} ({file_size_mb:.1f} MB)")
    print(f"\nStationary positions found:")
    for i, (cx, cy) in enumerate(stationary_filter.get_stationary_positions()):
        print(f"  Cluster {i + 1}: ({cx:.0f}, {cy:.0f})")


if __name__ == "__main__":
    main()
