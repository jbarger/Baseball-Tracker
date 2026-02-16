"""
Smart Detection Pipeline: filters, tracks, and annotates batting cage video.

Applies a multi-stage filter pipeline to YOLO detections:
  1. ROI mask (inside cage only)
  2. Class filter (ball, bat, person only)
  3. Size filter (reject tiny false positives like red lights)
  4. Per-class confidence thresholds
  5. Stationary object filter (ignore balls that don't move)
  6. Frame-to-frame tracking with Kalman smoothing & trajectory trail

3D geometry model projects pixel detections onto the pitch line to
compute true speed (correctly handles depth-dominant trajectory where
the ball moves mostly toward the camera).

Outputs an annotated MP4 with color-coded detections, 3D speed, and a HUD.
Optionally exports 3D trajectory data as JSON for visualization.

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
from common.machine_exit_detector import MachineExitDetector, MachineExitConfig
from common.drawing import (
    DrawConfig, BALL_CLASS, BAT_CLASS, PERSON_CLASS,
    draw_roi, draw_stationary_markers, draw_rejected,
    draw_machine_exit_region, draw_tracked_objects,
    draw_contact_flash, draw_hud, draw_event_overlays,
)
from common.events import EventDetector


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
                source="yolo",
            ))
    return detections


def machine_exit_to_detection(me_result, frame_idx: int) -> Detection:
    """Convert machine-exit detector result to a Detection object."""
    cx, cy, radius, conf, diff_val = me_result
    r = max(radius, 5)
    return Detection(
        cls_id=BALL_CLASS,
        cls_name="sports ball",
        conf=conf,
        x1=cx - r,
        y1=cy - r,
        x2=cx + r,
        y2=cy + r,
        frame_idx=frame_idx,
        source="machine_exit",
    )


def compute_3d_speeds_for_track(track: TrackedObject, calibration: CageCalibration,
                                 fps: float):
    """
    Compute and cache 3D speed for the latest position in a ball track.

    Called each frame after tracking update. Uses the 3D geometry model
    to convert the last two pixel positions into a true 3D speed along
    the pitch line. This correctly accounts for the depth-dominant
    trajectory.
    """
    if track.cls_id != BALL_CLASS:
        return
    if not calibration.has_3d:
        return
    if len(track.positions) < 2:
        track.speeds_3d_mph.append(None)
        track.world_positions.append(None)
        return

    # Get the two most recent positions
    u1, v1 = track.positions[-2]
    u2, v2 = track.positions[-1]
    f1 = track.frame_indices[-2]
    f2 = track.frame_indices[-1]
    dt = (f2 - f1) / fps

    # Compute 3D speed
    speed = calibration.to_mph_3d(u1, v1, u2, v2, dt)

    # Sanity check: reject obviously wrong speeds
    if speed is not None and (speed < 0 or speed > 200):
        speed = None

    track.speeds_3d_mph.append(speed)

    # Also store world position
    world_pos = calibration.get_world_position(u2, v2)
    track.world_positions.append(world_pos)


def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_detect.py <input_video> <output_video> [--config path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Config paths (defaults for Docker volume mounts)
    config_path = "/app/config/cage_config.json"
    machines_path = "/app/config/machines.json"
    export_3d = False

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        if arg == "--export-3d":
            export_3d = True

    # Load configuration
    print(f"=== Smart Detection Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")

    config = load_config(config_path)
    cameras_path = os.path.join(os.path.dirname(config_path), "camera_models.json")
    calibration = load_calibration(config_path, machines_path, cameras_path)

    print(f"\nMachine: {calibration.machine_spec.name}")
    print(f"Mound: {calibration.machine_distance_ft} ft")
    print(f"Ball: {calibration.ball_type} ({calibration.ball_diameter_inches}\" dia)")
    if calibration.sign_speed_mph:
        print(f"Sign speed: {calibration.sign_speed_mph} mph")
    print(f"Calibrated: {calibration.calibration_mode}")
    print(f"3D geometry: {'ENABLED' if calibration.has_3d else 'DISABLED'}")

    # Detection config
    det_config = config["detection"]
    model_name = det_config["model"]
    allowed_classes = det_config.get("allowed_classes", [0, 32, 34])
    conf_thresholds = {}
    for name, threshold in det_config["confidence_thresholds"].items():
        name_to_id = {"sports_ball": 32, "baseball_bat": 34, "person": 0}
        if name in name_to_id:
            conf_thresholds[name_to_id[name]] = threshold

    # Size filter config
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
    max_missed = track_config.get("max_missed_frames", 10)
    tracker = ObjectTracker(
        max_match_distance=track_config["max_match_distance_px"],
        max_missed_frames=max_missed,
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

    # Machine-exit ball detector
    me_detector = None
    machine_bbox = config.get("calibration", {}).get("machine_bbox_px")
    if machine_bbox and len(machine_bbox) == 4:
        me_config_data = config.get("machine_exit_detector", {})

        # Adjust ME detector threshold for ball type:
        # Yellow dimple balls are dimmer than white regulation balls
        default_intensity = 80.0
        ball_type = config.get("ball_type", "yellow_dimple")
        if ball_type == "yellow_dimple" and "min_diff_intensity" not in me_config_data:
            default_intensity = 60.0  # lower threshold for yellow balls

        me_config = MachineExitConfig(
            machine_bbox=machine_bbox,
            search_extend_right=me_config_data.get("search_extend_right", 250),
            search_extend_left=me_config_data.get("search_extend_left", 20),
            search_extend_vertical=me_config_data.get("search_extend_vertical", 40),
            min_diff_intensity=me_config_data.get("min_diff_intensity", default_intensity),
            bg_warmup_frames=me_config_data.get("bg_warmup_frames", 30),
            bg_alpha=me_config_data.get("bg_alpha", 0.02),
            min_blob_area=me_config_data.get("min_blob_area", 4),
            max_blob_area=me_config_data.get("max_blob_area", 400),
            max_aspect_ratio=me_config_data.get("max_aspect_ratio", 3.0),
            min_confidence=me_config_data.get("min_confidence", 0.75),
        )
        me_detector = MachineExitDetector(me_config, height, width)
        sr = me_detector.search_region
        print(f"Machine-exit detector: search region ({sr[0]},{sr[1]})-({sr[2]},{sr[3]})")
    else:
        print("Machine-exit detector: DISABLED (no machine_bbox_px in config)")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load YOLO
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)

    # ================================================================
    # PASS 1: If sign-speed calibration is available, do a quick
    # analysis pass to find the pitched ball's average pixel speed.
    # Then calibrate, and do Pass 2 with correct mph overlay.
    # If 3D geometry is enabled, this pass also validates the model.
    # ================================================================
    need_calibration_pass = (calibration.sign_speed_mph is not None
                             and calibration._empirical_factor is None)

    if need_calibration_pass:
        print(f"\n--- CALIBRATION PASS: measuring pitch speed in pixels ---")
        cal_stationary = StationaryFilter(
            history_frames=stat_config["history_frames"],
            max_variance_px=stat_config["max_variance_px"],
        )
        cal_tracker = ObjectTracker(
            max_match_distance=track_config["max_match_distance_px"],
            max_missed_frames=max_missed,
        )
        cal_me_detector = None
        if machine_bbox and len(machine_bbox) == 4:
            cal_me_detector = MachineExitDetector(me_config, height, width)
        ball_speeds_px = []
        ball_speeds_3d = []  # 3D speeds for validation
        cal_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False, conf=0.10)
            dets = yolo_to_detections(results, model, cal_frame_idx)
            kept, _ = filter_by_roi(dets, roi_polygon)
            kept, _ = filter_by_class(kept, allowed_classes)
            kept, _ = filter_by_size(kept, min_sizes)
            kept, _ = filter_by_confidence(kept, conf_thresholds)
            kept, _ = cal_stationary.update(kept)

            if cal_me_detector is not None:
                me_result = cal_me_detector.update(frame)
                yolo_has_ball = any(d.cls_id == BALL_CLASS for d in kept)
                if me_result is not None and not yolo_has_ball:
                    kept.append(machine_exit_to_detection(me_result, cal_frame_idx))

            cal_tracker.update(kept, cal_frame_idx)
            ball = cal_tracker.get_best_ball_track()
            if ball and ball.age >= 2:
                # Compute 3D speed for the ball track
                compute_3d_speeds_for_track(ball, calibration, fps)

                spd = ball.speed_px_per_sec(fps)
                if 200 < spd < 2000:
                    ball_speeds_px.append(spd)

                # Collect 3D speeds for comparison
                s3d = ball.speed_3d_mph(window=3)
                if s3d is not None and 10 < s3d < 120:
                    ball_speeds_3d.append(s3d)

            cal_frame_idx += 1

        if ball_speeds_px:
            speeds_arr = np.array(ball_speeds_px)
            median_speed = float(np.median(speeds_arr))
            calibration.calibrate_from_pitch(median_speed)
            print(f"  Measured {len(ball_speeds_px)} ball speed samples")
            print(f"  Speed distribution: min={speeds_arr.min():.0f}, "
                  f"median={median_speed:.0f}, max={speeds_arr.max():.0f} px/s")
            print(f"  Sign speed: {calibration.sign_speed_mph} mph")
            print(f"  Empirical factor: {calibration._empirical_factor:.6f} mph/(px/s)")
            print(f"  Calibration mode: {calibration.calibration_mode}")
        else:
            print(f"  WARNING: No moving ball detected in calibration pass.")

        if ball_speeds_3d:
            speeds_3d_arr = np.array(ball_speeds_3d)
            print(f"\n  3D speed samples: {len(ball_speeds_3d)}")
            print(f"  3D speed distribution: min={speeds_3d_arr.min():.0f}, "
                  f"median={np.median(speeds_3d_arr):.0f}, "
                  f"max={speeds_3d_arr.max():.0f} mph")
            print(f"  (compare to sign speed: {calibration.sign_speed_mph} mph)")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ================================================================
    # PASS 2 (or single pass): Process + annotate
    # ================================================================
    print(f"\nProcessing frames...\n")

    # Reset filters and tracker for clean render pass
    stationary_filter = StationaryFilter(
        history_frames=stat_config["history_frames"],
        max_variance_px=stat_config["max_variance_px"],
    )
    tracker = ObjectTracker(
        max_match_distance=track_config["max_match_distance_px"],
        max_missed_frames=max_missed,
    )
    if machine_bbox and len(machine_bbox) == 4:
        me_detector = MachineExitDetector(me_config, height, width)

    # Stats
    frame_idx = 0
    total_detections = 0
    total_filtered = 0
    total_tracked_ball_frames = 0
    total_me_detections = 0
    contact_frames = []
    max_ball_speed_3d = 0.0
    max_ball_speed_px = 0.0
    max_ball_speed_mph = 0.0
    handoff_count = 0

    # For 3D trajectory export
    trajectory_data = [] if export_3d else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLO inference ---
        results = model(frame, verbose=False, conf=0.10)
        raw_detections = yolo_to_detections(results, model, frame_idx)
        total_detections += len(raw_detections)

        # --- Filter pipeline ---
        all_rejected = []

        kept, rejected = filter_by_roi(raw_detections, roi_polygon)
        all_rejected.extend(rejected)

        kept, rejected = filter_by_class(kept, allowed_classes)
        all_rejected.extend(rejected)

        kept, rejected = filter_by_size(kept, min_sizes)
        all_rejected.extend(rejected)

        kept, rejected = filter_by_confidence(kept, conf_thresholds)
        all_rejected.extend(rejected)

        kept, rejected = stationary_filter.update(kept)
        all_rejected.extend(rejected)

        total_filtered += len(all_rejected)

        # --- Machine-exit detector ---
        me_result = None
        if me_detector is not None:
            me_result = me_detector.update(frame)
            yolo_has_ball = any(d.cls_id == BALL_CLASS for d in kept)
            if me_result is not None and not yolo_has_ball:
                kept.append(machine_exit_to_detection(me_result, frame_idx))
                total_me_detections += 1

        # --- Tracking ---
        active_tracks = tracker.update(kept, frame_idx)
        best_ball = tracker.get_best_ball_track()
        best_bat = tracker.get_best_bat_track()

        ball_count = len([t for t in active_tracks
                          if t.cls_id == BALL_CLASS and t.missed_frames == 0])
        if ball_count > 0:
            total_tracked_ball_frames += 1

        # --- 3D speed computation for ball tracks ---
        for track in active_tracks:
            if track.cls_id == BALL_CLASS and track.missed_frames == 0:
                compute_3d_speeds_for_track(track, calibration, fps)

        # Track max speed
        if best_ball and best_ball.age >= 2:
            # Prefer 3D speed
            speed_3d = best_ball.speed_3d_mph(window=5)
            if speed_3d is not None and speed_3d > max_ball_speed_3d and speed_3d < 150:
                max_ball_speed_3d = speed_3d

            speed_px = best_ball.smoothed_speed_px_per_sec(fps, window=5)
            if speed_px > max_ball_speed_px:
                max_ball_speed_px = speed_px
                mph = calibration.to_mph(speed_px)
                if mph is not None:
                    max_ball_speed_mph = mph

            # Track handoffs
            if len(best_ball.sources) >= 2:
                if best_ball.sources[-1] != best_ball.sources[-2]:
                    handoff_count += 1

        # --- 3D trajectory export ---
        if trajectory_data is not None and best_ball and best_ball.last_position:
            pos = best_ball.last_position
            world = calibration.get_world_position(pos[0], pos[1])
            speed_3d = best_ball.speed_3d_mph(window=3)
            trajectory_data.append({
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 4),
                "pixel": {"u": round(pos[0], 1), "v": round(pos[1], 1)},
                "world_ft": {
                    "x": round(world[0], 3),
                    "y": round(world[1], 3),
                    "z": round(world[2], 3),
                } if world else None,
                "depth_ft": round(world[2], 2) if world else None,
                "speed_3d_mph": round(speed_3d, 1) if speed_3d else None,
                "source": best_ball.last_source,
            })

        # --- Contact detection ---
        contact_frame = detect_contact(best_ball, best_bat, contact_distance)
        if contact_frame is not None and contact_frame not in contact_frames:
            contact_frames.append(contact_frame)

        is_contact = contact_frame is not None

        # --- Draw annotations ---
        annotated = frame.copy()

        draw_roi(annotated, roi_polygon)
        draw_stationary_markers(annotated, stationary_filter.get_stationary_positions())

        if me_detector is not None:
            draw_machine_exit_region(annotated, me_detector.search_region, me_result)

        draw_rejected(annotated, all_rejected)
        draw_tracked_objects(annotated, active_tracks, fps, calibration)

        if is_contact:
            draw_contact_flash(annotated)

        draw_hud(annotated, frame_idx, fps, ball_count, len(active_tracks),
                 is_contact, calibration, best_ball)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 60 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)"
                  f"  | tracked ball in {total_tracked_ball_frames} frames"
                  f"  | filtered {total_filtered} detections")

    cap.release()
    writer.release()

    # --- Export 3D trajectory data ---
    if trajectory_data:
        trajectory_path = output_path.rsplit(".", 1)[0] + "_trajectory_3d.json"
        with open(trajectory_path, "w") as f:
            json.dump({
                "video": os.path.basename(input_path),
                "fps": fps,
                "calibration_mode": calibration.calibration_mode,
                "cage_geometry": {
                    "machine_distance_ft": calibration.machine_distance_ft,
                    "sign_speed_mph": calibration.sign_speed_mph,
                    "has_3d": calibration.has_3d,
                },
                "trajectory": trajectory_data,
            }, f, indent=2)
        print(f"\n3D trajectory exported: {trajectory_path}")

    # --- Summary ---
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print(f"SMART DETECTION RESULTS")
    print(f"{'=' * 50}")
    print(f"Frames processed:   {frame_idx}")
    print(f"Total detections:   {total_detections}")
    print(f"Filtered out:       {total_filtered} ({total_filtered / max(total_detections, 1) * 100:.0f}%)")
    print(f"Ball tracked in:    {total_tracked_ball_frames} frames")
    print(f"Machine-exit dets:  {total_me_detections} frames (supplemented YOLO)")
    print(f"ME->YOLO handoffs:  {handoff_count}")
    print(f"Stationary clusters: {len(stationary_filter.get_stationary_positions())}")
    print(f"Contact frames:     {contact_frames if contact_frames else 'None detected'}")
    print(f"Calibration:        {calibration.calibration_mode}")

    # Speed summary
    print(f"\n--- Speed Summary ---")
    if max_ball_speed_3d > 0:
        print(f"Max ball speed (3D): {max_ball_speed_3d:.0f} mph")
    if max_ball_speed_mph > 0:
        print(f"Max ball speed (2D): {max_ball_speed_mph:.0f} mph ({max_ball_speed_px:.0f} px/s)")

    MAX_REALISTIC_MPH = 120
    if max_ball_speed_mph > MAX_REALISTIC_MPH:
        print(f"  NOTE: 2D speed {max_ball_speed_mph:.0f} mph is a tracker artifact "
              f"(3D model gives {max_ball_speed_3d:.0f} mph)")

    print(f"\nOutput: {output_path} ({file_size_mb:.1f} MB)")
    print(f"\nStationary positions found:")
    for i, (cx, cy) in enumerate(stationary_filter.get_stationary_positions()):
        print(f"  Cluster {i + 1}: ({cx:.0f}, {cy:.0f})")


if __name__ == "__main__":
    main()
