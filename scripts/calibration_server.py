"""
FastAPI calibration server for the batting cage camera setup.

Serves the annotation/calibration UI and provides API endpoints for
reading/writing cage configuration and running auto-calibration.

Usage:
    pip install fastapi uvicorn
    python -m uvicorn scripts.calibration_server:app --port 8765 --reload

Or inside Docker:
    python -m uvicorn scripts.calibration_server:app --host 0.0.0.0 --port 8765
"""
import json
import os
import sys
import shutil
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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
