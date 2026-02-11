"""
FastAPI calibration server for the batting cage camera setup.

Serves the annotation/calibration UI and provides API endpoints for
reading/writing cage configuration, running auto-calibration, and
live camera streaming (USB webcam or RTMP).

Usage:
    pip install fastapi uvicorn
    python -m uvicorn scripts.calibration_server:app --port 8765 --reload

Or inside Docker:
    python -m uvicorn scripts.calibration_server:app --host 0.0.0.0 --port 8765

For USB webcam streaming in Docker (Linux):
    docker run --device=/dev/video0 -p 8765:8765 ...

For USB webcam on Windows/Docker Desktop:
    Requires usbipd-win to attach USB camera to WSL2, or use RTMP instead.
"""
import json
import os
import sys
import shutil
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, "/app/src/BaseballTracker.Modules")

# Paths (Docker container layout)
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/app/config")
TOOLS_DIR = os.environ.get("TOOLS_DIR", "/app/tools")
CONFIG_PATH = os.path.join(CONFIG_DIR, "cage_config.json")
CAMERAS_PATH = os.path.join(CONFIG_DIR, "camera_models.json")
MACHINES_PATH = os.path.join(CONFIG_DIR, "machines.json")

app = FastAPI(title="Baseball Tracker Calibration", version="1.0")


# ==============================================================
# LIVE CAMERA STREAMING
# ==============================================================

class CameraStream:
    """Background video capture with thread-safe frame access."""

    def __init__(self):
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.frame = None  # Latest BGR frame from OpenCV
        self.source_type = None  # "usb" or "rtmp"
        self.source_info = ""  # device index or URL
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0  # Total frames captured
        self._fps_timer = 0
        self._fps_count = 0
        self._measured_fps = 0

    def start(self, source_type: str, source: str):
        """Start capturing from a USB device index or RTMP URL."""
        self.stop()  # Stop any existing capture

        if source_type == "usb":
            device_idx = int(source)
            self.cap = cv2.VideoCapture(device_idx)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"Cannot open USB camera at index {device_idx}")
            # Try to set 1080p
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.source_info = f"USB device {device_idx}"
        elif source_type == "rtmp":
            url = source.strip()
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"Cannot open RTMP stream: {url}")
            self.source_info = url
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        self.source_type = source_type
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.running = True
        self.frame_count = 0
        self._fps_timer = time.time()
        self._fps_count = 0
        self._measured_fps = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the capture thread and release the camera."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.thread = None
        self.source_type = None
        with self.lock:
            self.frame = None

    def _capture_loop(self):
        """Background loop: continuously read frames from the camera."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "rtmp":
                    # RTMP streams can drop — try to reconnect
                    time.sleep(0.5)
                    continue
                else:
                    break
            with self.lock:
                self.frame = frame
            self.frame_count += 1
            self._fps_count += 1

            # Measure actual FPS every second
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._measured_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_timer = now

        self.running = False

    def get_frame(self):
        """Get the latest frame (BGR numpy array) or None."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_jpeg(self, quality=80):
        """Get the latest frame as JPEG bytes, or None."""
        frame = self.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def status(self):
        """Return current stream status dict."""
        return {
            "active": self.running,
            "source_type": self.source_type,
            "source_info": self.source_info,
            "width": self.width,
            "height": self.height,
            "nominal_fps": self.fps,
            "measured_fps": round(self._measured_fps, 1),
            "frames_captured": self.frame_count,
        }


# Global camera stream instance
camera_stream = CameraStream()


def _detect_usb_cameras(max_index=5):
    """Probe USB camera indices 0..max_index and return list of available ones."""
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({"index": idx, "resolution": f"{w}x{h}"})
            cap.release()
    return available


# ==============================================================
# REAL-TIME DETECTION PIPELINE
# ==============================================================

# Color constants (BGR for OpenCV) — match smart_detect.py
COLOR_BALL = (0, 255, 0)        # Bright green
COLOR_BALL_TRAIL = (0, 200, 0)  # Dimmer green
COLOR_BAT = (0, 165, 255)       # Orange
COLOR_PERSON = (0, 255, 255)    # Yellow
COLOR_STATIONARY = (100, 100, 100)
COLOR_REJECTED = (0, 0, 180)    # Dark red
COLOR_ROI = (255, 255, 255)     # White
COLOR_HUD_BG = (0, 0, 0)
COLOR_HUD_TEXT = (255, 255, 255)
COLOR_3D_INFO = (200, 200, 100)

BALL_CLASS = 32
BAT_CLASS = 34
PERSON_CLASS = 0

ACTIVE_COLORS = {
    BALL_CLASS: COLOR_BALL,
    BAT_CLASS: COLOR_BAT,
    PERSON_CLASS: COLOR_PERSON,
}


class DetectionPipeline:
    """
    Runs YOLO detection + tracking on live camera frames in a background thread.

    Architecture: The capture thread runs at full camera speed. This pipeline
    reads the latest raw frame, runs YOLO + filters + tracker, draws overlays
    onto a copy, and stores it as the "annotated frame". The MJPEG feed serves
    the annotated frame when detection is enabled.
    """

    def __init__(self):
        self.enabled = False
        self.model = None
        self.tracker = None
        self.calibration = None
        self.stationary_filter = None
        self.config = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.annotated_frame = None
        self.frame_idx = 0
        self._fps_timer = 0
        self._fps_count = 0
        self._measured_fps = 0
        self._model_loaded = False

    def _load_model_and_config(self):
        """Load YOLO model, config, calibration, and filter instances."""
        if self._model_loaded:
            return

        from ultralytics import YOLO
        from common.filters import StationaryFilter
        from common.tracking import ObjectTracker
        from common.calibration import load_calibration

        # Load config
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Load YOLO model
        model_name = self.config.get("detection", {}).get("model", "yolov8n.pt")
        self.model = YOLO(model_name)

        # Load calibration for 3D speed
        try:
            self.calibration = load_calibration(
                config_path=CONFIG_PATH,
                machines_path=MACHINES_PATH,
                cameras_path=CAMERAS_PATH,
            )
        except Exception as e:
            print(f"[DetectionPipeline] Calibration load failed: {e}")
            self.calibration = None

        # Create tracker
        tracking_cfg = self.config.get("tracking", {})
        self.tracker = ObjectTracker(
            max_match_distance=tracking_cfg.get("max_match_distance_px", 150),
            max_missed_frames=tracking_cfg.get("max_missed_frames", 10),
        )

        # Create stationary filter
        stat_cfg = self.config.get("stationary_filter", {})
        self.stationary_filter = StationaryFilter(
            history_frames=stat_cfg.get("history_frames", 10),
            max_variance_px=stat_cfg.get("max_variance_px", 8.0),
        )

        self._model_loaded = True
        print(f"[DetectionPipeline] Model loaded: {model_name}")

    def start(self, camera_stream):
        """Start the detection thread, reading frames from camera_stream."""
        self._load_model_and_config()
        self.running = True
        self.enabled = True
        self.frame_idx = 0
        self._fps_timer = time.time()
        self._fps_count = 0
        self._measured_fps = 0

        # Reset tracker and stationary filter for new stream session
        tracking_cfg = self.config.get("tracking", {}) if self.config else {}
        from common.tracking import ObjectTracker
        from common.filters import StationaryFilter
        self.tracker = ObjectTracker(
            max_match_distance=tracking_cfg.get("max_match_distance_px", 150),
            max_missed_frames=tracking_cfg.get("max_missed_frames", 10),
        )
        stat_cfg = self.config.get("stationary_filter", {}) if self.config else {}
        self.stationary_filter = StationaryFilter(
            history_frames=stat_cfg.get("history_frames", 10),
            max_variance_px=stat_cfg.get("max_variance_px", 8.0),
        )

        self.thread = threading.Thread(
            target=self._detection_loop, args=(camera_stream,), daemon=True
        )
        self.thread.start()

    def stop(self):
        """Stop the detection thread."""
        self.running = False
        self.enabled = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        with self.lock:
            self.annotated_frame = None

    def _detection_loop(self, camera_stream):
        """Background loop: grab frames, run YOLO, draw overlays."""
        from common.filters import (
            Detection, filter_by_roi, filter_by_class,
            filter_by_size, filter_by_confidence,
        )

        # Parse config for filter params
        det_cfg = self.config.get("detection", {})
        roi_polygon = self.config.get("roi_polygon", [])
        allowed_classes = det_cfg.get("allowed_classes", [0, 32, 34])

        # Confidence thresholds: config uses string names, we need int class IDs
        conf_thresh_raw = det_cfg.get("confidence_thresholds", {})
        conf_thresholds = {}
        name_to_id = {"sports_ball": 32, "baseball_bat": 34, "person": 0}
        for name, thresh in conf_thresh_raw.items():
            if name in name_to_id:
                conf_thresholds[name_to_id[name]] = thresh

        # Size thresholds
        size_cfg = det_cfg.get("min_size_px", {})
        min_sizes = {}
        if "person_height" in size_cfg:
            min_sizes[0] = (
                size_cfg.get("person_width", 40),
                size_cfg.get("person_height", 80),
            )

        fps = camera_stream.fps or 30

        while self.running and camera_stream.running:
            if not self.enabled:
                time.sleep(0.1)
                continue

            frame = camera_stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_idx += 1
            annotated = frame.copy()

            try:
                # Run YOLO inference
                results = self.model(frame, verbose=False, conf=0.10)

                # Convert to Detection objects
                detections = []
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        detections.append(Detection(
                            cls_id=cls_id,
                            cls_name=self.model.names[cls_id],
                            conf=float(box.conf[0]),
                            x1=float(box.xyxy[0][0]),
                            y1=float(box.xyxy[0][1]),
                            x2=float(box.xyxy[0][2]),
                            y2=float(box.xyxy[0][3]),
                            frame_idx=self.frame_idx,
                            source="yolo",
                        ))

                # Apply filter chain
                all_rejected = []
                kept = detections

                kept, rejected = filter_by_roi(kept, roi_polygon)
                all_rejected.extend(rejected)

                kept, rejected = filter_by_class(kept, allowed_classes)
                all_rejected.extend(rejected)

                kept, rejected = filter_by_size(kept, min_sizes)
                all_rejected.extend(rejected)

                kept, rejected = filter_by_confidence(kept, conf_thresholds)
                all_rejected.extend(rejected)

                kept, rejected = self.stationary_filter.update(kept)
                all_rejected.extend(rejected)

                # Update tracker
                self.tracker.update(kept, self.frame_idx)

                # Compute 3D speeds for ball tracks
                if self.calibration:
                    for track in self.tracker._tracks.values():
                        if (track.cls_id == BALL_CLASS and track.age >= 2
                                and len(track.positions) >= 2):
                            self._compute_3d_speed(track, fps)

                # --- Draw overlays ---

                # ROI polygon
                if roi_polygon and len(roi_polygon) >= 3:
                    pts = np.array(roi_polygon, dtype=np.int32)
                    for i in range(len(pts)):
                        p1 = tuple(pts[i])
                        p2 = tuple(pts[(i + 1) % len(pts)])
                        cv2.line(annotated, p1, p2, COLOR_ROI, 1, cv2.LINE_AA)

                # Rejected detections (dim X marks)
                for fr in all_rejected:
                    det = fr.detection
                    cx, cy = int(det.cx), int(det.cy)
                    if "Stationary" in fr.reason:
                        cv2.rectangle(annotated,
                                      (int(det.x1), int(det.y1)),
                                      (int(det.x2), int(det.y2)),
                                      COLOR_STATIONARY, 1)
                    else:
                        s = 6
                        cv2.line(annotated, (cx-s, cy-s), (cx+s, cy+s), COLOR_REJECTED, 1)
                        cv2.line(annotated, (cx-s, cy+s), (cx+s, cy-s), COLOR_REJECTED, 1)

                # Stationary markers
                for (sx, sy) in self.stationary_filter.get_stationary_positions():
                    cv2.circle(annotated, (int(sx), int(sy)), 12, COLOR_STATIONARY, 1)
                    cv2.putText(annotated, "S", (int(sx)-4, int(sy)+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_STATIONARY, 1)

                # Tracked objects (bounding boxes, labels, trails)
                active_tracks = [t for t in self.tracker._tracks.values()
                                 if t.missed_frames == 0]
                for track in active_tracks:
                    bbox = track.last_bbox
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = ACTIVE_COLORS.get(track.cls_id, COLOR_HUD_TEXT)
                    cx, cy = int(track.last_position[0]), int(track.last_position[1])

                    # Bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Speed label
                    speed_label = ""
                    depth_label = ""

                    if track.cls_id == BALL_CLASS:
                        speed_3d = track.speed_3d_mph(window=5)
                        if speed_3d is not None and 5.0 < speed_3d < 150.0:
                            speed_label = f"BALL {speed_3d:.0f}mph"
                        else:
                            speed_px = track.smoothed_speed_px_per_sec(fps, window=5)
                            if self.calibration and self.calibration.is_calibrated:
                                flat_mph = self.calibration.to_mph(speed_px)
                                if flat_mph is not None and flat_mph < 120:
                                    speed_label = f"BALL ~{flat_mph:.0f}mph"
                                else:
                                    speed_label = f"BALL {speed_px:.0f}px/s"
                            else:
                                speed_label = f"BALL {speed_px:.0f}px/s"

                        # Depth
                        if self.calibration:
                            depth = self.calibration.get_depth_ft(
                                track.last_position[0], track.last_position[1])
                            if depth is not None:
                                depth_label = f"{depth:.0f}ft"

                    elif track.cls_id == BAT_CLASS:
                        speed_px = track.smoothed_speed_px_per_sec(fps, window=5)
                        if self.calibration and self.calibration.is_calibrated:
                            flat_mph = self.calibration.to_mph(speed_px)
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

                    # Draw label
                    (tw, th), _ = cv2.getTextSize(
                        speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 6),
                                  (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, speed_label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                                cv2.LINE_AA)

                    if depth_label:
                        cv2.putText(annotated, depth_label, (x1 + 2, y2 + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_3D_INFO,
                                    1, cv2.LINE_AA)

                    # Ball center dot + trail
                    if track.cls_id == BALL_CLASS:
                        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                        trail = track.get_trail(max_points=30)
                        if len(trail) >= 2:
                            for i in range(1, len(trail)):
                                alpha = i / len(trail)
                                thickness = max(1, int(alpha * 3))
                                cv2.line(annotated, trail[i-1], trail[i],
                                         COLOR_BALL_TRAIL, thickness, cv2.LINE_AA)

                # HUD
                ball_tracks = [t for t in active_tracks if t.cls_id == BALL_CLASS]
                best_ball = self.tracker.get_best_ball_track() if hasattr(self.tracker, 'get_best_ball_track') else None
                hud_lines = [
                    f"Frame {self.frame_idx} | Det: {self._measured_fps:.0f}fps",
                    f"Tracks: {len(active_tracks)} | Balls: {len(ball_tracks)}",
                ]
                if best_ball and best_ball.age >= 2:
                    speed_3d = best_ball.speed_3d_mph(window=5)
                    if speed_3d is not None and 5.0 < speed_3d < 150.0:
                        hud_lines.append(f"Ball: {speed_3d:.0f} mph (3D)")
                    else:
                        spx = best_ball.smoothed_speed_px_per_sec(fps, window=5)
                        hud_lines.append(f"Ball: {spx:.0f} px/s")

                max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                            for l in hud_lines)
                hud_h = len(hud_lines) * 22 + 12
                cv2.rectangle(annotated, (5, 5), (max_w + 18, hud_h),
                              COLOR_HUD_BG, -1)
                cv2.rectangle(annotated, (5, 5), (max_w + 18, hud_h),
                              (80, 80, 80), 1)
                y_text = 22
                for line in hud_lines:
                    cv2.putText(annotated, line, (10, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_TEXT,
                                1, cv2.LINE_AA)
                    y_text += 22

            except Exception as e:
                # Draw error on frame so user sees it
                cv2.putText(annotated, f"Detection error: {str(e)[:60]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)

            # Store annotated frame
            with self.lock:
                self.annotated_frame = annotated

            # Measure detection FPS
            self._fps_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._measured_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_timer = now

        self.running = False

    def _compute_3d_speed(self, track, fps):
        """Compute 3D speed for the last pair of positions in a track."""
        if not self.calibration or len(track.positions) < 2:
            return
        try:
            p1 = track.positions[-2]
            p2 = track.positions[-1]
            f1 = track.frame_indices[-2] if hasattr(track, 'frame_indices') else 0
            f2 = track.frame_indices[-1] if hasattr(track, 'frame_indices') else 1
            dt = (f2 - f1) / fps if f2 > f1 else 1.0 / fps

            speed = self.calibration.to_mph_3d(
                float(p1[0]), float(p1[1]),
                float(p2[0]), float(p2[1]),
                dt,
            )
            if speed is not None:
                if not hasattr(track, '_speed_3d_history'):
                    track._speed_3d_history = []
                track._speed_3d_history.append(speed)
                # Keep only last 10
                if len(track._speed_3d_history) > 10:
                    track._speed_3d_history.pop(0)
        except Exception:
            pass

    def get_annotated_frame(self):
        """Get the latest annotated frame, or None."""
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def status(self):
        """Return detection pipeline status."""
        return {
            "enabled": self.enabled,
            "model_loaded": self._model_loaded,
            "detection_fps": round(self._measured_fps, 1),
            "frame_idx": self.frame_idx,
        }


# Global detection pipeline instance
detection_pipeline = DetectionPipeline()


class StreamStartRequest(BaseModel):
    source: str  # "usb" or "rtmp"
    device: int = 0  # USB device index (for source="usb")
    url: str = ""  # RTMP URL (for source="rtmp")
    quality: int = 80  # JPEG quality 1-100


@app.post("/api/stream/start")
async def stream_start(req: StreamStartRequest):
    """Start video capture from USB camera or RTMP stream."""
    try:
        if req.source == "usb":
            camera_stream.start("usb", str(req.device))
        elif req.source == "rtmp":
            if not req.url:
                raise HTTPException(400, "RTMP URL is required")
            camera_stream.start("rtmp", req.url)
        else:
            raise HTTPException(400, f"Unknown source: {req.source}")
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    # Wait briefly for first frame
    for _ in range(20):
        if camera_stream.get_frame() is not None:
            break
        time.sleep(0.05)

    # Auto-start detection pipeline
    try:
        detection_pipeline.start(camera_stream)
    except Exception as e:
        print(f"[WARNING] Detection pipeline failed to start: {e}")

    return camera_stream.status()


@app.post("/api/stream/stop")
async def stream_stop():
    """Stop the active video capture and detection pipeline."""
    detection_pipeline.stop()
    camera_stream.stop()
    return {"status": "stopped"}


@app.get("/api/stream/status")
async def stream_status():
    """Return current stream state, detection status, and available USB cameras."""
    status = camera_stream.status()
    status["detection"] = detection_pipeline.status()
    # Only probe USB cameras if not currently streaming (probing can be slow)
    if not camera_stream.running:
        status["usb_cameras"] = _detect_usb_cameras()
    return status


@app.get("/api/stream/feed")
async def stream_feed(quality: int = 80):
    """MJPEG streaming response — serves annotated frames when detection is on."""

    def generate():
        while camera_stream.running:
            frame = None

            # Use annotated frame if detection is enabled
            if detection_pipeline.enabled:
                frame = detection_pipeline.get_annotated_frame()

            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, quality])
                jpeg = buf.tobytes()
            else:
                # Fall back to raw camera frame
                jpeg = camera_stream.get_jpeg(quality)

            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.01)
            # Throttle to ~30fps max to limit bandwidth
            time.sleep(0.03)

    if not camera_stream.running:
        raise HTTPException(400, "No active stream — call /api/stream/start first")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class DetectionToggleRequest(BaseModel):
    enabled: bool = True


@app.post("/api/stream/detection")
async def toggle_detection(req: DetectionToggleRequest):
    """Toggle real-time detection overlay on/off."""
    if req.enabled and not camera_stream.running:
        raise HTTPException(400, "Cannot enable detection without an active stream")

    if req.enabled:
        if not detection_pipeline.running:
            try:
                detection_pipeline.start(camera_stream)
            except Exception as e:
                raise HTTPException(500, f"Failed to start detection: {e}")
        detection_pipeline.enabled = True
    else:
        detection_pipeline.enabled = False
        # Clear annotated frame so feed falls back to raw
        with detection_pipeline.lock:
            detection_pipeline.annotated_frame = None

    return detection_pipeline.status()


@app.get("/api/stream/snapshot")
async def stream_snapshot(quality: int = 90):
    """Return a single JPEG frame from the active stream."""
    jpeg = camera_stream.get_jpeg(quality)
    if jpeg is None:
        raise HTTPException(400, "No frame available — is the stream active?")
    return StreamingResponse(
        iter([jpeg]),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=snapshot.jpg"},
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the annotation/calibration HTML tool."""
    html_path = os.path.join(TOOLS_DIR, "annotate.html")
    if not os.path.exists(html_path):
        raise HTTPException(404, f"annotate.html not found at {html_path}")
    with open(html_path) as f:
        return f.read()


@app.get("/api/config")
async def get_config():
    """Read current cage configuration."""
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(404, "cage_config.json not found")
    with open(CONFIG_PATH) as f:
        return json.load(f)


class ConfigUpdate(BaseModel):
    config: dict


@app.post("/api/config")
async def save_config(update: ConfigUpdate):
    """
    Save cage configuration with timestamped backup.

    Accepts a full or partial config dict. Merges with existing config.
    """
    # Read existing config
    existing = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            existing = json.load(f)

        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = CONFIG_PATH + f".bak.{timestamp}"
        shutil.copy2(CONFIG_PATH, backup_path)

    # Deep merge: update existing with new values
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    merged = deep_merge(existing, update.config)

    with open(CONFIG_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    return {"status": "ok", "backup": backup_path if os.path.exists(CONFIG_PATH) else None}


@app.get("/api/cameras")
async def get_cameras():
    """Return camera models database."""
    if not os.path.exists(CAMERAS_PATH):
        raise HTTPException(404, "camera_models.json not found")
    with open(CAMERAS_PATH) as f:
        return json.load(f)


@app.get("/api/machines")
async def get_machines():
    """Return pitching machines database."""
    if not os.path.exists(MACHINES_PATH):
        raise HTTPException(404, "machines.json not found")
    with open(MACHINES_PATH) as f:
        return json.load(f)


class CalibrateRequest(BaseModel):
    ground_truth: dict  # { "ball_positions": { "frame": [u, v], ... }, "fps": 59.94 }
    config_overrides: Optional[dict] = None  # Optional cage_config overrides


@app.post("/api/calibrate")
async def run_calibration(request: CalibrateRequest):
    """
    Run auto-calibration optimizer with ground truth data.

    Returns optimized parameters and per-frame speed analysis.
    """
    try:
        from scipy.optimize import differential_evolution
        from common.geometry3d import (
            CageGeometry, CameraModel, PitchTrajectory,
            FEET_PER_MILE, SECONDS_PER_HOUR
        )
        import numpy as np
    except ImportError as e:
        raise HTTPException(500, f"Missing dependency: {e}")

    # Load base config
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(404, "cage_config.json not found")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Apply overrides
    if request.config_overrides:
        for key, value in request.config_overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

    # Parse ground truth
    gt = request.ground_truth
    fps = gt.get("fps", 59.94)
    positions = gt.get("ball_positions", {})
    if len(positions) < 3:
        raise HTTPException(400, "Need at least 3 ground truth positions")

    gt_frames = sorted([int(f) for f in positions.keys()])
    gt_positions = [positions[str(f)] for f in gt_frames]

    # Config params
    machine_distance = config["machine_distance_ft"]
    cal_config = config.get("calibration", {})
    sign_speed = cal_config.get("known_pitch_speed_mph", 47.5)
    machine_bbox = cal_config.get("machine_bbox_px")
    cage_geo = config.get("cage_geometry", {})
    video_width = cage_geo.get("video_width", 1920)
    video_height = cage_geo.get("video_height", 1080)
    release_height = cage_geo.get("release_height_ft", 3.5)
    strike_zone_height = cage_geo.get("strike_zone_height_ft", 2.5)

    # Camera specs for initial focal length
    initial_focal = 1200.0
    camera_model_key = config.get("camera_model")
    if camera_model_key and os.path.exists(CAMERAS_PATH):
        with open(CAMERAS_PATH) as f:
            cam_db = json.load(f)
        cam_specs = cam_db.get("cameras", {}).get(camera_model_key)
        if cam_specs:
            sensor = cam_specs.get("sensor_size_mm")
            focal_mm = cam_specs.get("focal_length_mm")
            if sensor and focal_mm and sensor[0] > 0:
                initial_focal = focal_mm * video_width / sensor[0]

    # Import the objective function from auto_calibrate
    from scripts.auto_calibrate import build_geometry, compute_objective

    # Stage 1: Position + Aim + Focal Length
    bounds = [
        (-3.0, 3.0),
        (2.0, 10.0),
        (-10.0, 0.0),
        (-200.0, 200.0),
        (-200.0, 200.0),
        (initial_focal * 0.6, initial_focal * 1.4),
    ]

    result = differential_evolution(
        compute_objective,
        bounds,
        args=(gt_frames, gt_positions, fps, sign_speed,
              machine_bbox, machine_distance,
              video_width, video_height,
              release_height, strike_zone_height, False),
        seed=42, maxiter=300, tol=1e-8, popsize=20,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )

    # Build geometry with best params
    best = result.x
    geo = build_geometry(
        best, video_width, video_height,
        machine_distance, release_height, strike_zone_height,
        sign_speed, fit_distortion=False,
    )

    # Per-frame analysis
    frame_results = []
    speeds_3d = []
    for i in range(len(gt_frames) - 1):
        f1, f2 = gt_frames[i], gt_frames[i + 1]
        u1, v1 = gt_positions[i]
        u2, v2 = gt_positions[i + 1]
        dt = (f2 - f1) / fps

        speed = geo.pixel_speed_to_3d_speed_on_pitch_line(
            float(u1), float(v1), float(u2), float(v2), dt
        )
        depth = geo.pixel_to_depth_on_pitch_line(float(u1), float(v1))

        entry = {"frame": f1, "depth_ft": round(depth, 2) if depth else None}
        if speed:
            speeds_3d.append(speed)
            entry["speed_mph"] = round(speed, 1)
            entry["error_pct"] = round((speed - sign_speed) / sign_speed * 100, 1)
        frame_results.append(entry)

    import numpy as np
    arr = np.array(speeds_3d) if speeds_3d else np.array([0])
    mean_err = abs(arr.mean() - sign_speed) / sign_speed * 100 if speeds_3d else 100

    return {
        "status": "ok",
        "params": {
            "camera_x_ft": round(float(best[0]), 4),
            "camera_height_ft": round(float(best[1]), 4),
            "camera_behind_plate_ft": round(-float(best[2]), 4),
            "aim_offset_x": round(float(best[3]), 2),
            "aim_offset_y": round(float(best[4]), 2),
            "focal_length_px": round(float(best[5]), 2),
        },
        "summary": {
            "mean_speed_mph": round(float(arr.mean()), 1) if speeds_3d else None,
            "median_speed_mph": round(float(np.median(arr)), 1) if speeds_3d else None,
            "std_speed_mph": round(float(arr.std()), 1) if speeds_3d else None,
            "mean_error_pct": round(float(mean_err), 1),
            "target_speed_mph": sign_speed,
            "objective": round(float(result.fun), 4),
        },
        "frames": frame_results,
    }


@app.post("/api/preview-projection")
async def preview_projection(params: dict):
    """
    Compute projected overlay points for given camera parameters.

    Returns projected pitch line points and machine bbox outline
    for preview rendering in the UI.
    """
    try:
        from common.geometry3d import (
            CageGeometry, CameraModel, PitchTrajectory,
            FEET_PER_MILE, SECONDS_PER_HOUR
        )
    except ImportError as e:
        raise HTTPException(500, f"Missing dependency: {e}")

    camera = CameraModel(
        image_width=params.get("video_width", 1920),
        image_height=params.get("video_height", 1080),
        focal_length_px=params.get("focal_length_px", 1200),
        camera_x_ft=params.get("camera_x_ft", 0),
        camera_y_ft=params.get("camera_height_ft", 5),
        camera_z_ft=-params.get("camera_behind_plate_ft", 2),
        aim_offset_x=params.get("aim_offset_x", 0),
        aim_offset_y=params.get("aim_offset_y", 0),
        k1=params.get("k1", 0),
        k2=params.get("k2", 0),
    )

    release_height = params.get("release_height_ft", 3.5)
    strike_zone_height = params.get("strike_zone_height_ft", 2.5)
    machine_distance = params.get("machine_distance_ft", 60)

    pitch = PitchTrajectory(
        release_y_ft=release_height,
        release_z_ft=machine_distance,
        target_y_ft=strike_zone_height,
    )

    # Project pitch line at 10 evenly-spaced depths
    pitch_line_points = []
    for i in range(11):
        z = machine_distance * (1 - i / 10.0)
        pos = pitch.position_at_z(z)
        px = camera.world_to_pixel(*pos)
        if px:
            pitch_line_points.append({
                "u": round(px[0], 1), "v": round(px[1], 1),
                "z_ft": round(z, 1),
            })

    # Project machine outline (4 corners)
    machine_height_ft = params.get("machine_height_inches", 57) / 12.0
    machine_width_ft = params.get("machine_width_inches", 44) / 12.0
    machine_corners = []
    for dx, dy in [(-0.5, 0), (0.5, 0), (0.5, 1), (-0.5, 1)]:
        wx = dx * machine_width_ft
        wy = dy * machine_height_ft
        wz = machine_distance
        px = camera.world_to_pixel(wx, wy, wz)
        if px:
            machine_corners.append({"u": round(px[0], 1), "v": round(px[1], 1)})

    return {
        "pitch_line": pitch_line_points,
        "machine_outline": machine_corners,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
