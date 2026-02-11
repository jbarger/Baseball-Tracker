"""
Auto-calibrate camera parameters from ground truth ball positions.

Uses scipy.optimize.differential_evolution to find camera position,
aim, focal length, and distortion coefficients that produce consistent
3D speed estimates matching the known sign speed across all ground
truth frames.

The key insight: the ball travels at a constant speed (e.g., 47.5 mph),
so any set of camera parameters that recovers that speed at every frame
— regardless of the 11x perspective variation in pixel speed — must be
correct (or very close to correct).

Two-stage optimization:
  Stage 1: Fit position + aim + focal length (freeze distortion)
  Stage 2: Refine all params including distortion

Usage (inside Docker):
    python /app/scripts/auto_calibrate.py
    python /app/scripts/auto_calibrate.py --gt /app/config/ground_truth_00351348.json
    python /app/scripts/auto_calibrate.py --output /app/config/cage_config.json
"""
import sys
import json
import os
import copy
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, "/app/src/BaseballTracker.Modules")

from scipy.optimize import differential_evolution
from common.geometry3d import (
    CageGeometry, CameraModel, PitchTrajectory,
    FEET_PER_MILE, SECONDS_PER_HOUR
)


def build_geometry(params, image_width, image_height, machine_distance_ft,
                   release_height_ft, strike_zone_height_ft,
                   sign_speed_mph, fit_distortion=False):
    """
    Build a CageGeometry from a parameter vector.

    Stage 1 (6 params): camera_x, camera_y, camera_z, aim_x, aim_y, focal_length
    Stage 2 (8 params): + k1, k2
    """
    camera_x = params[0]
    camera_y = params[1]
    camera_z = params[2]
    aim_x = params[3]
    aim_y = params[4]
    focal_length = params[5]

    k1 = params[6] if fit_distortion and len(params) > 6 else 0.0
    k2 = params[7] if fit_distortion and len(params) > 7 else 0.0

    camera = CameraModel(
        image_width=image_width,
        image_height=image_height,
        focal_length_px=focal_length,
        camera_x_ft=camera_x,
        camera_y_ft=camera_y,
        camera_z_ft=camera_z,
        aim_offset_x=aim_x,
        aim_offset_y=aim_y,
        k1=k1,
        k2=k2,
    )

    pitch = PitchTrajectory(
        release_x_ft=0.0,
        release_y_ft=release_height_ft,
        release_z_ft=machine_distance_ft,
        target_x_ft=0.0,
        target_y_ft=strike_zone_height_ft,
        target_z_ft=0.0,
        speed_ft_per_sec=sign_speed_mph * FEET_PER_MILE / SECONDS_PER_HOUR,
    )

    geo = CageGeometry(camera=camera, pitch=pitch)
    geo._calibrated = True
    return geo


def compute_objective(params, gt_frames, gt_positions, fps, sign_speed_mph,
                      machine_bbox, machine_distance_ft,
                      image_width, image_height,
                      release_height_ft, strike_zone_height_ft,
                      fit_distortion=False):
    """
    Objective function: sum of squared speed errors across all GT frame pairs.

    Also includes a penalty for machine bbox center projection mismatch.
    """
    geo = build_geometry(
        params, image_width, image_height,
        machine_distance_ft, release_height_ft, strike_zone_height_ft,
        sign_speed_mph, fit_distortion=fit_distortion,
    )

    target_speed = sign_speed_mph
    total_error = 0.0
    n_pairs = 0

    # Speed consistency error: each consecutive pair should give sign_speed
    for i in range(len(gt_frames) - 1):
        f1, f2 = gt_frames[i], gt_frames[i + 1]
        u1, v1 = gt_positions[i]
        u2, v2 = gt_positions[i + 1]

        dt = (f2 - f1) / fps
        if dt <= 0:
            continue

        speed = geo.pixel_speed_to_3d_speed_on_pitch_line(
            float(u1), float(v1), float(u2), float(v2), dt
        )

        if speed is not None and speed > 0:
            # Squared error in mph
            err = speed - target_speed
            total_error += err * err
            n_pairs += 1
        else:
            # Penalize failed projections heavily
            total_error += 10000.0
            n_pairs += 1

    if n_pairs == 0:
        return 1e6

    # Average speed error
    avg_speed_err = total_error / n_pairs

    # Machine bbox center penalty: the projected machine center should
    # match the observed bbox center
    if machine_bbox and len(machine_bbox) == 4:
        mx1, my1, mx2, my2 = machine_bbox
        bbox_cx = (mx1 + mx2) / 2.0
        bbox_cy = (my1 + my2) / 2.0

        # Project machine mid-height position
        machine_y = release_height_ft / 2.0  # approximate mid-height
        projected = geo.camera.world_to_pixel(0.0, machine_y, machine_distance_ft)

        if projected is not None:
            du = projected[0] - bbox_cx
            dv = projected[1] - bbox_cy
            bbox_penalty = (du * du + dv * dv) * 0.01  # weight factor
            avg_speed_err += bbox_penalty
        else:
            avg_speed_err += 1000.0

    # Depth ordering penalty: first GT frame should be farther than last
    first_depth = geo.pixel_to_depth_on_pitch_line(
        float(gt_positions[0][0]), float(gt_positions[0][1]))
    last_depth = geo.pixel_to_depth_on_pitch_line(
        float(gt_positions[-1][0]), float(gt_positions[-1][1]))
    if first_depth is not None and last_depth is not None:
        if first_depth < last_depth:
            # Ball should be moving toward plate (decreasing Z)
            avg_speed_err += (last_depth - first_depth) * 10.0

        # Depth should be reasonable (ball in cage, not behind camera)
        if first_depth < 0 or first_depth > 100:
            avg_speed_err += 5000.0
        if last_depth < 0 or last_depth > 100:
            avg_speed_err += 5000.0

    return avg_speed_err


def main():
    config_path = "/app/config/cage_config.json"
    machines_path = "/app/config/machines.json"
    cameras_path = "/app/config/camera_models.json"
    gt_path = "/app/config/ground_truth_00351348.json"
    output_path = None

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        if arg == "--gt" and i + 1 < len(sys.argv):
            gt_path = sys.argv[i + 1]
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load ground truth
    with open(gt_path) as f:
        gt_data = json.load(f)

    fps = gt_data.get("fps", 59.94)
    gt_positions_dict = gt_data["ball_positions"]
    gt_frames = sorted([int(f) for f in gt_positions_dict.keys()])
    gt_positions = [gt_positions_dict[str(f)] for f in gt_frames]

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

    # Camera specs for initial focal length estimate
    camera_model_key = config.get("camera_model")
    initial_focal = 1200.0  # fallback
    initial_k1 = 0.0
    initial_k2 = 0.0

    if camera_model_key and os.path.exists(cameras_path):
        with open(cameras_path) as f:
            cam_db = json.load(f)
        cam_specs = cam_db.get("cameras", {}).get(camera_model_key)
        if cam_specs:
            sensor = cam_specs.get("sensor_size_mm")
            focal_mm = cam_specs.get("focal_length_mm")
            if sensor and focal_mm and sensor[0] > 0:
                initial_focal = focal_mm * video_width / sensor[0]
                print(f"Camera specs focal length: {initial_focal:.1f} px")
            dist = cam_specs.get("distortion_coefficients", {})
            initial_k1 = dist.get("k1", 0.0)
            initial_k2 = dist.get("k2", 0.0)

    print(f"=== AUTO-CALIBRATION ===")
    print(f"Ground truth: {len(gt_frames)} frames ({gt_frames[0]}-{gt_frames[-1]})")
    print(f"Sign speed: {sign_speed} mph")
    print(f"Machine distance: {machine_distance} ft")
    print(f"Initial focal length estimate: {initial_focal:.0f} px")
    print(f"Machine bbox: {machine_bbox}")
    print()

    # === Stage 1: Position + Aim + Focal Length ===
    print("--- Stage 1: Fitting position, aim, focal length ---")
    t0 = time.time()

    # Parameter bounds
    #   camera_x: ±3 ft (lateral offset)
    #   camera_y: 2-10 ft (height)
    #   camera_z: -10 to 0 ft (behind plate)
    #   aim_x: ±200 px
    #   aim_y: ±200 px
    #   focal_length: initial ±40%
    bounds_stage1 = [
        (-3.0, 3.0),           # camera_x_ft
        (2.0, 10.0),           # camera_y_ft
        (-10.0, 0.0),          # camera_z_ft
        (-200.0, 200.0),       # aim_offset_x
        (-200.0, 200.0),       # aim_offset_y
        (initial_focal * 0.6, initial_focal * 1.4),  # focal_length_px
    ]

    result1 = differential_evolution(
        compute_objective,
        bounds_stage1,
        args=(gt_frames, gt_positions, fps, sign_speed,
              machine_bbox, machine_distance,
              video_width, video_height,
              release_height, strike_zone_height,
              False),  # fit_distortion=False
        seed=42,
        maxiter=500,
        tol=1e-8,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
    )

    t1 = time.time()
    print(f"  Stage 1 complete in {t1-t0:.1f}s")
    print(f"  Objective: {result1.fun:.4f}")
    print(f"  Params: cam=({result1.x[0]:.2f}, {result1.x[1]:.2f}, {result1.x[2]:.2f})ft, "
          f"aim=({result1.x[3]:.1f}, {result1.x[4]:.1f})px, "
          f"focal={result1.x[5]:.1f}px")
    print()

    # === Stage 2: Refine with distortion ===
    print("--- Stage 2: Refining with lens distortion ---")
    t2 = time.time()

    # Start from Stage 1 result, add distortion params
    x0_stage2 = list(result1.x) + [initial_k1, initial_k2]

    bounds_stage2 = list(bounds_stage1) + [
        (-0.6, 0.0),    # k1 (barrel distortion, always negative for wide-angle)
        (-0.1, 0.3),    # k2 (correction term)
    ]

    result2 = differential_evolution(
        compute_objective,
        bounds_stage2,
        args=(gt_frames, gt_positions, fps, sign_speed,
              machine_bbox, machine_distance,
              video_width, video_height,
              release_height, strike_zone_height,
              True),  # fit_distortion=True
        seed=42,
        maxiter=500,
        tol=1e-8,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.9,
        x0=x0_stage2,
        polish=True,
    )

    t3 = time.time()
    print(f"  Stage 2 complete in {t3-t2:.1f}s")
    print(f"  Objective: {result2.fun:.4f}")
    print(f"  Params: cam=({result2.x[0]:.2f}, {result2.x[1]:.2f}, {result2.x[2]:.2f})ft, "
          f"aim=({result2.x[3]:.1f}, {result2.x[4]:.1f})px, "
          f"focal={result2.x[5]:.1f}px, "
          f"k1={result2.x[6]:.4f}, k2={result2.x[7]:.4f}")
    print()

    # Use whichever stage produced a lower objective
    if result2.fun < result1.fun:
        best_result = result2
        best_stage = 2
        fit_distortion = True
    else:
        best_result = result1
        best_stage = 1
        fit_distortion = False
        print("  Note: Stage 2 didn't improve — using Stage 1 result.")

    # === Evaluate best result ===
    print(f"{'='*60}")
    print(f"BEST RESULT (Stage {best_stage})")
    print(f"{'='*60}")

    best_params = best_result.x
    geo = build_geometry(
        best_params, video_width, video_height,
        machine_distance, release_height, strike_zone_height,
        sign_speed, fit_distortion=fit_distortion,
    )

    print(f"\nCamera position: ({best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}) ft")
    print(f"Aim offset: ({best_params[3]:.1f}, {best_params[4]:.1f}) px")
    print(f"Focal length: {best_params[5]:.1f} px")
    if fit_distortion:
        print(f"Distortion: k1={best_params[6]:.5f}, k2={best_params[7]:.5f}")
    print()

    # Per-frame speed analysis
    print(f"{'Frame':>5} {'px_spd':>7} {'3d_mph':>7} {'err%':>6} {'depth':>7}")
    print("-" * 50)

    speeds_3d = []
    for i in range(len(gt_frames) - 1):
        f1, f2 = gt_frames[i], gt_frames[i + 1]
        u1, v1 = gt_positions[i]
        u2, v2 = gt_positions[i + 1]
        dt = (f2 - f1) / fps

        # Pixel speed
        dx = u2 - u1
        dy = v2 - v1
        px_per_sec = np.sqrt(dx**2 + dy**2) / dt

        speed_3d = geo.pixel_speed_to_3d_speed_on_pitch_line(
            float(u1), float(v1), float(u2), float(v2), dt
        )

        depth = geo.pixel_to_depth_on_pitch_line(float(u1), float(v1))

        if speed_3d is not None:
            speeds_3d.append(speed_3d)
            err_pct = (speed_3d - sign_speed) / sign_speed * 100
            depth_str = f"{depth:.1f}" if depth is not None else "N/A"
            print(f"{f1:5d} {px_per_sec:7.0f} {speed_3d:7.1f} {err_pct:+5.1f}% {depth_str:>7}")

    if speeds_3d:
        arr = np.array(speeds_3d)
        mean_err = abs(arr.mean() - sign_speed) / sign_speed * 100
        print(f"\n3D Speed Summary:")
        print(f"  Mean:   {arr.mean():.1f} mph  (target: {sign_speed})")
        print(f"  Median: {np.median(arr):.1f} mph")
        print(f"  Std:    {arr.std():.1f} mph")
        print(f"  Min:    {arr.min():.1f} mph")
        print(f"  Max:    {arr.max():.1f} mph")
        print(f"  Mean error: {mean_err:.1f}%")

        # Depth range
        first_depth = geo.pixel_to_depth_on_pitch_line(
            float(gt_positions[0][0]), float(gt_positions[0][1]))
        last_depth = geo.pixel_to_depth_on_pitch_line(
            float(gt_positions[-1][0]), float(gt_positions[-1][1]))
        if first_depth and last_depth:
            print(f"\nDepth range: {first_depth:.1f}ft → {last_depth:.1f}ft")

        if mean_err < 10:
            print(f"\n>>> CALIBRATION SUCCESSFUL (mean error {mean_err:.1f}%)")
        elif mean_err < 25:
            print(f"\n>>> CALIBRATION FAIR (mean error {mean_err:.1f}%)")
        else:
            print(f"\n>>> CALIBRATION NEEDS WORK (mean error {mean_err:.1f}%)")

    # === Save results ===
    save_path = output_path or config_path
    print(f"\nSaving optimized config to: {save_path}")

    # Read fresh config to preserve all other fields
    with open(config_path) as f:
        out_config = json.load(f)

    # Update cage_geometry with optimized camera params
    if "cage_geometry" not in out_config:
        out_config["cage_geometry"] = {}

    out_config["cage_geometry"]["camera_x_ft"] = round(float(best_params[0]), 4)
    out_config["cage_geometry"]["camera_height_ft"] = round(float(best_params[1]), 4)
    out_config["cage_geometry"]["camera_behind_plate_ft"] = round(-float(best_params[2]), 4)
    out_config["cage_geometry"]["aim_offset_x"] = round(float(best_params[3]), 2)
    out_config["cage_geometry"]["aim_offset_y"] = round(float(best_params[4]), 2)
    out_config["cage_geometry"]["focal_length_px"] = round(float(best_params[5]), 2)

    if fit_distortion:
        out_config["cage_geometry"]["k1"] = round(float(best_params[6]), 6)
        out_config["cage_geometry"]["k2"] = round(float(best_params[7]), 6)

    # Metadata
    out_config["cage_geometry"]["calibration_meta"] = {
        "method": "auto_calibrate_differential_evolution",
        "stage": best_stage,
        "objective": round(float(best_result.fun), 6),
        "mean_speed_error_pct": round(float(mean_err), 2) if speeds_3d else None,
        "ground_truth_file": os.path.basename(gt_path),
        "ground_truth_frames": len(gt_frames),
        "timestamp": datetime.now().isoformat(),
    }

    # Create backup before overwriting
    if os.path.exists(save_path) and save_path == config_path:
        backup_path = save_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            with open(save_path) as src:
                f.write(src.read())
        print(f"  Backup saved: {backup_path}")

    with open(save_path, 'w') as f:
        json.dump(out_config, f, indent=2)
    print(f"  Config saved: {save_path}")


if __name__ == "__main__":
    main()
