"""
Ball Tracking Module
Detects and tracks baseball through video to calculate launch metrics.

Uses the full CV pipeline:
  - YOLO detection (sports ball, class 32)
  - MachineExitDetector for near-machine ball detection
  - Detection filter pipeline (ROI, confidence, size, stationary)
  - ObjectTracker with Kalman-style smoothing
  - CageCalibration for 3D speed (exit velocity) and world position
"""
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

from common.calibration import load_calibration
from common.filters import (
    Detection,
    StationaryFilter,
    filter_by_confidence,
    filter_by_roi,
    filter_by_size,
)
from common.machine_exit_detector import MachineExitConfig, MachineExitDetector
from common.tracking import ObjectTracker
from common.video_utils import extract_frames, get_video_info

logger = logging.getLogger(__name__)

# Config directory: walk up from ball_tracking/ until we find a config/ dir
# with cage_config.json. Works in both Docker (/app/config) and dev checkout
# (quirky-carson/config), regardless of how many intermediate dirs exist.
_MODULE_DIR = Path(__file__).parent


def _find_config_dir() -> Path:
    """Walk up from the module directory to find the config/ directory."""
    candidate = _MODULE_DIR
    for _ in range(6):  # max 6 levels up
        config_dir = candidate / "config"
        if (config_dir / "cage_config.json").exists():
            return config_dir
        candidate = candidate.parent
    # Fallback: conventional location relative to module (will produce a clear error)
    return _MODULE_DIR.parent.parent.parent / "config"


_CONFIG_DIR = _find_config_dir()
_CAGE_CONFIG_PATH = _CONFIG_DIR / "cage_config.json"
_MACHINES_PATH = _CONFIG_DIR / "machines.json"
_CAMERAS_PATH = _CONFIG_DIR / "camera_models.json"

# YOLO class IDs (COCO)
_CLS_SPORTS_BALL = 32
_CLS_BASEBALL_BAT = 34
_CLS_PERSON = 0

_CLS_NAMES = {
    _CLS_SPORTS_BALL: "sports ball",
    _CLS_BASEBALL_BAT: "baseball bat",
    _CLS_PERSON: "person",
}

# Module-level YOLO singleton — loaded once, reused across requests
_yolo_model = None


def _get_yolo_model(model_name: str):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {model_name}")
        _yolo_model = YOLO(model_name)
        logger.info("YOLO model loaded")
    return _yolo_model


def _load_cage_config() -> dict:
    if not _CAGE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Cage config not found at {_CAGE_CONFIG_PATH}. "
            "Expected config/cage_config.json at repo root."
        )
    with open(_CAGE_CONFIG_PATH) as f:
        return json.load(f)


def _yolo_results_to_detections(results, frame_idx: int, allowed_classes: list) -> list:
    """Convert ultralytics YOLO result to list of Detection objects."""
    detections = []
    if not results or len(results) == 0:
        return detections
    result = results[0]
    if result.boxes is None:
        return detections
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id not in allowed_classes:
            continue
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append(Detection(
            cls_id=cls_id,
            cls_name=_CLS_NAMES.get(cls_id, str(cls_id)),
            conf=conf,
            x1=x1, y1=y1, x2=x2, y2=y2,
            frame_idx=frame_idx,
            source="yolo",
        ))
    return detections


def _machine_exit_to_detection(me_result, frame_idx: int) -> Detection:
    cx, cy, radius, confidence, _ = me_result
    r = radius if radius > 0 else 5.0
    return Detection(
        cls_id=_CLS_SPORTS_BALL,
        cls_name="sports ball",
        conf=confidence,
        x1=cx - r, y1=cy - r, x2=cx + r, y2=cy + r,
        frame_idx=frame_idx,
        source="machine_exit",
    )


def _compute_launch_angle(track, calibration) -> float:
    """Vertical launch angle off the bat in degrees (positive = upward)."""
    positions = track.positions
    if len(positions) < 2:
        return 0.0
    n = min(5, len(positions))
    p_start = positions[-n]
    p_end = positions[-1]
    if calibration is not None and calibration.has_3d:
        world_start = calibration.get_world_position(p_start[0], p_start[1])
        world_end = calibration.get_world_position(p_end[0], p_end[1])
        if world_start and world_end:
            dx = world_end[0] - world_start[0]
            dy = world_end[1] - world_start[1]
            dz = world_end[2] - world_start[2]
            horizontal = math.sqrt(dx * dx + dz * dz)
            if horizontal > 0.001:
                return math.degrees(math.atan2(dy, horizontal))
    # Pixel-space fallback (image y is flipped: up = smaller v)
    du = p_end[0] - p_start[0]
    dv = p_end[1] - p_start[1]
    horizontal_px = abs(du)
    if horizontal_px > 0.5:
        return math.degrees(math.atan2(-dv, horizontal_px))
    return 0.0


def _compute_spray_angle(track, calibration, frame_width: int) -> float:
    """Horizontal spray angle in degrees (negative = pull, positive = opposite)."""
    positions = track.positions
    if len(positions) < 2:
        return 0.0
    n = min(5, len(positions))
    p_start = positions[-n]
    p_end = positions[-1]
    if calibration is not None and calibration.has_3d:
        world_start = calibration.get_world_position(p_start[0], p_start[1])
        world_end = calibration.get_world_position(p_end[0], p_end[1])
        if world_start and world_end:
            dx = world_end[0] - world_start[0]
            dz = world_end[2] - world_start[2]
            if abs(dz) > 0.001:
                return math.degrees(math.atan2(dx, dz))
    # Pixel-space fallback
    du = p_end[0] - p_start[0]
    dv = p_end[1] - p_start[1]
    depth_proxy = abs(dv) if abs(dv) > 1.0 else 1.0
    return math.degrees(math.atan2(du, depth_proxy))


def _estimate_contact_frame(track) -> int:
    """Estimate contact frame as the frame of peak pixel speed."""
    positions = track.positions
    frames = track.frame_indices
    if len(positions) < 3:
        return frames[-1] if frames else 0
    speeds = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        df = frames[i] - frames[i - 1]
        speeds.append(math.sqrt(dx * dx + dy * dy) / max(df, 1))
    peak_idx = int(np.argmax(speeds))
    return frames[min(peak_idx + 1, len(frames) - 1)]


class BallTracker:
    """
    Ball tracking using the full computer vision pipeline.

    Loads cage calibration from config/cage_config.json, runs YOLO
    detection + MachineExitDetector on every frame, filters and tracks
    the ball, then computes exit velocity, launch angle, and spray angle
    using the 3D geometry model.
    """

    def __init__(self):
        logger.info("BallTracker initialized")

    def process_video(self, video_path: str):
        # Import inside method to avoid circular import
        from api import BallTrackingResult, Point3D

        logger.info(f"Processing video: {video_path}")

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # --- Load cage config ---
        try:
            cage_cfg = _load_cage_config()
        except FileNotFoundError as e:
            logger.warning(f"Cage config not found, using defaults: {e}")
            cage_cfg = {}

        det_cfg = cage_cfg.get("detection", {})
        model_name = det_cfg.get("model", "yolov8n.pt")
        allowed_classes = det_cfg.get("allowed_classes", [_CLS_SPORTS_BALL])
        roi_polygon = cage_cfg.get("roi_polygon", [])
        track_cfg = cage_cfg.get("tracking", {})
        stat_cfg = cage_cfg.get("stationary_filter", {})
        me_cfg_raw = cage_cfg.get("machine_exit_detector", {})
        cal_bbox = cage_cfg.get("calibration", {}).get("machine_bbox_px")

        conf_thresholds_raw = det_cfg.get("confidence_thresholds", {})
        conf_thresholds = {
            _CLS_SPORTS_BALL: conf_thresholds_raw.get("sports_ball", 0.15),
            _CLS_BASEBALL_BAT: conf_thresholds_raw.get("baseball_bat", 0.25),
            _CLS_PERSON: conf_thresholds_raw.get("person", 0.7),
        }

        min_size_px = det_cfg.get("min_size_px", {})
        min_sizes = {}
        if "person_height" in min_size_px and "person_width" in min_size_px:
            min_sizes[_CLS_PERSON] = (
                min_size_px["person_width"],
                min_size_px["person_height"],
            )

        min_traj_frames = track_cfg.get("min_trajectory_frames", 2)
        max_match_dist = track_cfg.get("max_match_distance_px", 150)
        max_missed = track_cfg.get("max_missed_frames", 10)
        min_speed_px = track_cfg.get("min_speed_px_per_frame", 3.0)

        # --- Load calibration ---
        calibration = None
        if _CAGE_CONFIG_PATH.exists() and _MACHINES_PATH.exists():
            try:
                calibration = load_calibration(
                    str(_CAGE_CONFIG_PATH),
                    str(_MACHINES_PATH),
                    str(_CAMERAS_PATH) if _CAMERAS_PATH.exists() else None,
                )
                logger.info(f"Calibration loaded: {calibration.calibration_mode}")
            except Exception as e:
                logger.warning(f"Could not load calibration: {e}")

        # --- Load video ---
        video_info = get_video_info(video_path)
        fps = video_info["fps"] or 30.0
        frame_width = video_info["width"]
        frame_height = video_info["height"]
        logger.info(
            f"Video: {frame_width}x{frame_height} @ {fps:.1f} fps, "
            f"{video_info['frame_count']} frames"
        )

        frames = extract_frames(video_path)
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # --- Initialize pipeline components ---
        model = _get_yolo_model(model_name)
        obj_tracker = ObjectTracker(
            max_match_distance=max_match_dist,
            max_missed_frames=max_missed,
        )
        stat_filter = StationaryFilter(
            history_frames=stat_cfg.get("history_frames", 10),
            max_variance_px=stat_cfg.get("max_variance_px", 8.0),
        )

        me_detector = None
        if cal_bbox and len(cal_bbox) == 4:
            me_config = MachineExitConfig(
                machine_bbox=cal_bbox,
                search_extend_right=me_cfg_raw.get("search_extend_right", 250),
                search_extend_left=me_cfg_raw.get("search_extend_left", 20),
                search_extend_vertical=me_cfg_raw.get("search_extend_vertical", 40),
                min_diff_intensity=me_cfg_raw.get("min_diff_intensity", 80),
                bg_warmup_frames=me_cfg_raw.get("bg_warmup_frames", 30),
                bg_alpha=me_cfg_raw.get("bg_alpha", 0.02),
                min_blob_area=me_cfg_raw.get("min_blob_area", 4),
                max_blob_area=me_cfg_raw.get("max_blob_area", 400),
                max_aspect_ratio=me_cfg_raw.get("max_aspect_ratio", 3.0),
                min_confidence=me_cfg_raw.get("min_confidence", 0.75),
            )
            me_detector = MachineExitDetector(me_config, frame_height, frame_width)
            logger.info("MachineExitDetector enabled")

        # --- Per-frame detection and tracking loop ---
        logger.info(f"Processing {len(frames)} frames...")
        for frame_idx, frame in enumerate(frames):
            # YOLO detection
            yolo_results = model(frame, conf=0.05, verbose=False)
            detections = _yolo_results_to_detections(yolo_results, frame_idx, allowed_classes)

            # Machine-exit detector (near-machine ball detection)
            if me_detector is not None:
                me_result = me_detector.update(frame)
                if me_result is not None:
                    detections.append(_machine_exit_to_detection(me_result, frame_idx))

            # Filter pipeline
            detections, _ = filter_by_roi(detections, roi_polygon)
            detections, _ = filter_by_confidence(detections, conf_thresholds)
            detections, _ = filter_by_size(detections, min_sizes)
            detections, _ = stat_filter.update(detections)

            # Update tracker
            active_tracks = obj_tracker.update(detections, frame_idx)

            # Compute 3D speed for each active ball track
            if calibration is not None:
                for track in active_tracks:
                    if track.cls_id != _CLS_SPORTS_BALL:
                        continue
                    if len(track.positions) < 2:
                        track.speeds_3d_mph.append(None)
                        continue
                    p1 = track.positions[-2]
                    p2 = track.positions[-1]
                    f1 = track.frame_indices[-2]
                    f2 = track.frame_indices[-1]
                    dt = (f2 - f1) / fps if fps > 0 else 0.0
                    speed = calibration.to_mph_3d(p1[0], p1[1], p2[0], p2[1], dt)
                    track.speeds_3d_mph.append(speed)

        # --- Extract results from best ball track ---
        # Include both active tracks and completed (expired) tracks
        all_ball_tracks = [
            t for t in list(obj_tracker._tracks.values()) + obj_tracker._completed_tracks
            if t.cls_id == _CLS_SPORTS_BALL
        ]

        def _track_avg_speed(track) -> float:
            """Average pixel displacement per frame over entire track."""
            if len(track.positions) < 2:
                return 0.0
            total_disp = 0.0
            for i in range(1, len(track.positions)):
                dx = track.positions[i][0] - track.positions[i - 1][0]
                dy = track.positions[i][1] - track.positions[i - 1][1]
                df = max(track.frame_indices[i] - track.frame_indices[i - 1], 1)
                total_disp += math.sqrt(dx * dx + dy * dy) / df
            return total_disp / (len(track.positions) - 1)

        def _track_linearity(track) -> float:
            """
            How linear/straight the track is (0=chaotic, 1=perfectly straight).
            Pitched ball should be highly linear; bounces are erratic.
            """
            if len(track.positions) < 3:
                return 1.0  # can't tell with 2 points
            pts = track.positions
            x0, y0 = pts[0]
            xn, yn = pts[-1]
            total_len = math.sqrt((xn - x0) ** 2 + (yn - y0) ** 2)
            if total_len < 1.0:
                return 0.0  # stationary
            # Sum of actual step distances
            step_len = sum(
                math.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2)
                for i in range(1, len(pts))
            )
            # Linearity: ratio of straight-line distance to path length
            # 1.0 = perfectly straight, approaches 0 for chaotic path
            return total_len / max(step_len, 1.0)

        def _track_best_window_score(track) -> float:
            """
            Score a track by its best consecutive window of linear motion.
            This handles tracks that have a good start (pitched ball) but
            later pick up false detections (bounces, re-detections).
            Uses a sliding window of up to 8 frames, scoring by speed × linearity.
            """
            pos = track.positions
            frames = track.frame_indices
            if len(pos) < 2:
                return 0.0
            window = min(8, len(pos))
            best = 0.0
            for start in range(len(pos) - window + 1):
                seg_pos = pos[start:start + window]
                seg_frames = frames[start:start + window]
                spd = _track_avg_speed_seg(seg_pos, seg_frames)
                lin = _track_linearity_seg(seg_pos)
                score = spd * lin * len(seg_pos)
                if score > best:
                    best = score
            return best

        def _track_avg_speed_seg(positions, frame_indices) -> float:
            if len(positions) < 2:
                return 0.0
            total = 0.0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i - 1][0]
                dy = positions[i][1] - positions[i - 1][1]
                df = max(frame_indices[i] - frame_indices[i - 1], 1)
                total += math.sqrt(dx * dx + dy * dy) / df
            return total / (len(positions) - 1)

        def _track_linearity_seg(positions) -> float:
            if len(positions) < 3:
                return 1.0
            x0, y0 = positions[0]
            xn, yn = positions[-1]
            total_len = math.sqrt((xn - x0) ** 2 + (yn - y0) ** 2)
            if total_len < 1.0:
                return 0.0
            step_len = sum(
                math.sqrt((positions[i][0] - positions[i-1][0])**2 +
                          (positions[i][1] - positions[i-1][1])**2)
                for i in range(1, len(positions))
            )
            return total_len / max(step_len, 1.0)

        def _track_score(track) -> float:
            """
            Score a track for likelihood of being the pitched ball.
            Uses best sliding window to ignore false-match tails.
            """
            return _track_best_window_score(track)

        # Prefer tracks that are long AND fast AND linear (pitched ball characteristics)
        # Filter: length >= min, speed in plausible range
        # Also require minimum 3 frames to avoid trivially "linear" 2-point tracks
        max_speed_px = 100.0
        qualifying = [
            t for t in all_ball_tracks
            if len(t.positions) >= max(min_traj_frames, 3)
            and min_speed_px <= _track_avg_speed(t) <= max_speed_px
        ]

        if not qualifying:
            # Fall back to 2-frame tracks if nothing better found
            qualifying = [
                t for t in all_ball_tracks
                if len(t.positions) >= min_traj_frames
                and min_speed_px <= _track_avg_speed(t) <= max_speed_px
            ]

        best_track = None
        if qualifying:
            # Among qualifying tracks, prefer the most linear and fast one
            best_track = max(qualifying, key=_track_score)
        elif all_ball_tracks:
            # Fall back to longest track even if below min_traj_frames
            best_track = max(all_ball_tracks, key=lambda t: len(t.positions))

        if best_track is None or len(best_track.positions) < min_traj_frames:
            count = len(best_track.positions) if best_track else 0
            raise ValueError(
                f"No ball detected (trajectory too short: {count} frames, "
                f"need {min_traj_frames}). Check video path and calibration."
            )

        logger.info(
            f"Best ball track: {len(best_track.positions)} frames, "
            f"sources: {set(best_track.sources)}"
        )

        # --- Compute metrics ---
        exit_velocity = best_track.speed_3d_mph(window=5)
        if exit_velocity is None and calibration is not None:
            px_speed = best_track.smoothed_speed_px_per_sec(fps)
            exit_velocity = calibration.to_mph(px_speed)
        if exit_velocity is None:
            exit_velocity = 0.0

        launch_angle = _compute_launch_angle(best_track, calibration)
        spray_angle = _compute_spray_angle(best_track, calibration, frame_width)
        contact_frame = _estimate_contact_frame(best_track)

        # Build trajectory (at most 20 sample points)
        trajectory_points = []
        positions = best_track.positions
        step = max(1, len(positions) // 20)
        for i in range(0, len(positions), step):
            u, v = positions[i]
            world = calibration.get_world_position(u, v) if calibration else None
            if world:
                trajectory_points.append(Point3D(x=world[0], y=world[1], z=world[2]))
            else:
                trajectory_points.append(Point3D(
                    x=round(u / frame_width * 100, 2),
                    y=round((1.0 - v / frame_height) * 100, 2),
                    z=0.0,
                ))

        mean_conf = float(np.mean(best_track.confidences)) if best_track.confidences else 0.5
        if calibration is None or not calibration.has_3d:
            mean_conf = min(mean_conf, 0.6)

        result = BallTrackingResult(
            exit_velocity_mph=round(exit_velocity, 1),
            launch_angle_degrees=round(launch_angle, 1),
            spray_angle_degrees=round(spray_angle, 1),
            contact_frame=contact_frame,
            trajectory_points=trajectory_points,
            confidence=round(mean_conf, 3),
        )

        logger.info(
            f"Ball tracking complete: {result.exit_velocity_mph} mph, "
            f"launch={result.launch_angle_degrees}°, "
            f"spray={result.spray_angle_degrees}°, "
            f"contact_frame={result.contact_frame}, "
            f"confidence={result.confidence:.2f}"
        )
        return result
