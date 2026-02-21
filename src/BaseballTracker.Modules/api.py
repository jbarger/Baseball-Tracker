"""
Baseball Tracker - Python CV API
"""
import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Baseball Tracker CV API",
    description="Computer vision services for baseball swing analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models (defined here to avoid circular imports)
# ============================================================================

class Point3D(BaseModel):
    x: float
    y: float
    z: float

class BallTrackingResult(BaseModel):
    exit_velocity_mph: float
    launch_angle_degrees: float
    spray_angle_degrees: float
    contact_frame: int
    trajectory_points: List[Point3D]
    confidence: float
    overlay_video_path: Optional[str] = None

class BatTrackingResult(BaseModel):
    bat_speed_mph: float
    attack_angle_degrees: float
    contact_point: Point3D
    contact_frame: int
    swing_path_points: List[Point3D]
    confidence: float

class TrackingRequest(BaseModel):
    video_path: str
    camera_id: Optional[str] = "default"

# Import trackers AFTER models are defined
from ball_tracking.tracker import BallTracker
from bat_tracking.tracker import BatTracker

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Baseball Tracker CV API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "baseball-tracker-cv"
    }

@app.post("/track/ball", response_model=BallTrackingResult)
async def track_ball(request: TrackingRequest):
    try:
        logger.info(f"Processing ball tracking for: {request.video_path}")
        tracker = BallTracker()
        result = tracker.process_video(request.video_path)
        logger.info(f"Ball tracking complete: {result.exit_velocity_mph} mph")
        return result
    except Exception as e:
        logger.error(f"Ball tracking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track/bat", response_model=BatTrackingResult)
async def track_bat(request: TrackingRequest):
    try:
        logger.info(f"Processing bat tracking for: {request.video_path}")
        tracker = BatTracker()
        result = tracker.process_video(request.video_path)
        logger.info(f"Bat tracking complete: {result.bat_speed_mph} mph")
        return result
    except Exception as e:
        logger.error(f"Bat tracking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _warmup_yolo():
    """
    Export yolov8n.pt → yolov8n.onnx on first run (saved to /app/models/ volume so it
    persists across container restarts), then load the ONNX model and swap it into the
    module singleton. ONNX Runtime on CPU is ~2x faster than PyTorch inference.

    Also warms up the model with a dummy inference to trigger any JIT/compile work,
    so the first real request runs at full speed.
    """
    import shutil
    from pathlib import Path
    from ball_tracking.tracker import _get_yolo_model, _load_cage_config, _set_yolo_model

    try:
        cage_cfg = _load_cage_config()
        model_name = cage_cfg.get("detection", {}).get("model", "yolov8n.pt")
    except Exception:
        model_name = "yolov8n.pt"

    models_dir = Path("/app/models")
    onnx_stem = Path(model_name).stem
    onnx_path = models_dir / f"{onnx_stem}.onnx"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)

        if onnx_path.exists():
            logger.info(f"Loading cached ONNX model: {onnx_path}")
            from ultralytics import YOLO as _YOLO
            onnx_model = _YOLO(str(onnx_path))
        else:
            logger.info(f"Loading YOLO model ({model_name}) for ONNX export...")
            pt_model = _get_yolo_model(model_name)
            logger.info("Exporting YOLO to ONNX for faster CPU inference...")
            exported = pt_model.export(format="onnx", imgsz=640)
            shutil.copy(str(exported), str(onnx_path))
            logger.info(f"ONNX model saved to {onnx_path}")
            from ultralytics import YOLO as _YOLO
            onnx_model = _YOLO(str(onnx_path))

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        onnx_model(dummy, conf=0.05, verbose=False)
        _set_yolo_model(onnx_model)
        logger.info("YOLO warmup complete (ONNX) — first request will run at full speed")

    except Exception as e:
        logger.warning(f"ONNX export/load failed ({e}), falling back to PT model warmup")
        model = _get_yolo_model(model_name)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, conf=0.05, verbose=False)
        logger.info("YOLO warmup complete (PT fallback) — first request will run at full speed")


@app.on_event("startup")
async def startup_event():
    logger.info("Baseball Tracker CV API starting up...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warmup_yolo)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Baseball Tracker CV API shutting down...")