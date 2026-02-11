"""
Validate the 3D geometry model against hand-annotated ground truth.

For each annotated ball position, the 3D model projects the pixel
coordinates onto the pitch line to estimate depth and speed. Since
the ball travels at a known constant speed (~47.5 mph), the 3D model
should recover that speed at every frame regardless of the ball's
apparent pixel speed (which varies 11x due to perspective).

This script reports:
  1. Estimated depth at each frame (should decrease linearly)
  2. Computed 3D speed at each frame (should be ~47.5 mph everywhere)
  3. Comparison of pixel speed vs 3D speed (shows perspective correction)
  4. Error metrics vs known sign speed

Usage (inside Docker):
    python /app/scripts/validate_3d_model.py
    python /app/scripts/validate_3d_model.py --gt /app/config/ground_truth_00351348.json
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/app/src/BaseballTracker.Modules")

from common.calibration import load_calibration
from common.geometry3d import CageGeometry, FEET_PER_MILE, SECONDS_PER_HOUR


def main():
    config_path = "/app/config/cage_config.json"
    machines_path = "/app/config/machines.json"
    cameras_path = "/app/config/camera_models.json"
    gt_path = "/app/config/ground_truth_00351348.json"

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        if arg == "--gt" and i + 1 < len(sys.argv):
            gt_path = sys.argv[i + 1]

    # Load calibration (sets up 3D geometry)
    calibration = load_calibration(config_path, machines_path, cameras_path)

    if not calibration.has_3d:
        print("ERROR: 3D geometry not calibrated. Check cage_config.json.")
        sys.exit(1)

    geo = calibration.geometry
    sign_speed = calibration.sign_speed_mph or 47.5
    fps = 59.94

    print(f"=== 3D MODEL VALIDATION ===")
    print(f"Sign speed: {sign_speed} mph")
    print(f"Focal length: {geo.camera.focal_length_px:.1f} px")
    print(f"Camera: ({geo.camera.camera_x_ft:.1f}, {geo.camera.camera_y_ft:.1f}, {geo.camera.camera_z_ft:.1f}) ft")
    print(f"Pitch: ({geo.pitch.release_x_ft:.1f}, {geo.pitch.release_y_ft:.1f}, {geo.pitch.release_z_ft:.1f}) -> "
          f"({geo.pitch.target_x_ft:.1f}, {geo.pitch.target_y_ft:.1f}, {geo.pitch.target_z_ft:.1f})")

    # Load ground truth
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_positions = gt_data["ball_positions"]
    frames = sorted([int(f) for f in gt_positions.keys()])

    print(f"\nGround truth: {len(frames)} frames ({frames[0]}-{frames[-1]})")
    print()

    # --- Test 1: Depth estimation at each frame ---
    print(f"{'Frame':>5} {'px_X':>6} {'px_Y':>6} | {'depth_ft':>8} {'world_X':>8} {'world_Y':>8} {'world_Z':>8} | "
          f"{'px_spd':>7} {'3d_mph':>7} {'err%':>6}")
    print("-" * 100)

    prev_frame = None
    prev_u, prev_v = None, None
    prev_world = None
    speeds_3d = []
    speeds_px = []
    depths = []

    for f in frames:
        u, v = gt_positions[str(f)]

        # Project to 3D on pitch line
        world = geo.pixel_to_world_on_pitch_line(float(u), float(v))
        depth = geo.pixel_to_depth_on_pitch_line(float(u), float(v))

        depth_str = f"{depth:.1f}" if depth is not None else "N/A"
        world_str = (f"{world[0]:8.2f} {world[1]:8.2f} {world[2]:8.2f}"
                     if world is not None else "      N/A      N/A      N/A")

        if depth is not None:
            depths.append((f, depth))

        # Speed computation
        px_spd_str = ""
        speed_3d_str = ""
        err_str = ""

        if prev_frame is not None:
            df = f - prev_frame
            dt = df / fps

            # Pixel speed
            dx = u - prev_u
            dy = v - prev_v
            px_dist = np.sqrt(dx**2 + dy**2)
            px_per_sec = px_dist / dt
            speeds_px.append(px_per_sec)
            px_spd_str = f"{px_per_sec:7.0f}"

            # 3D speed
            speed_3d = calibration.to_mph_3d(
                float(prev_u), float(prev_v),
                float(u), float(v), dt
            )
            if speed_3d is not None:
                speeds_3d.append(speed_3d)
                speed_3d_str = f"{speed_3d:7.1f}"
                err_pct = (speed_3d - sign_speed) / sign_speed * 100
                err_str = f"{err_pct:+5.1f}%"
        else:
            px_spd_str = "     --"
            speed_3d_str = "     --"

        print(f"{f:5d} {u:6d} {v:6d} | {depth_str:>8} {world_str} | {px_spd_str} {speed_3d_str} {err_str}")

        prev_frame = f
        prev_u, prev_v = u, v
        prev_world = world

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    if speeds_3d:
        arr = np.array(speeds_3d)
        print(f"\n3D Speed (should be ~{sign_speed} mph everywhere):")
        print(f"  Mean:   {arr.mean():.1f} mph")
        print(f"  Median: {np.median(arr):.1f} mph")
        print(f"  Std:    {arr.std():.1f} mph")
        print(f"  Min:    {arr.min():.1f} mph (frame with lowest)")
        print(f"  Max:    {arr.max():.1f} mph (frame with highest)")
        print(f"  Mean error: {(arr.mean() - sign_speed)/sign_speed*100:+.1f}%")

    if speeds_px:
        arr_px = np.array(speeds_px)
        print(f"\nPixel Speed (shows perspective effect):")
        print(f"  Min:  {arr_px.min():.0f} px/s (near machine)")
        print(f"  Max:  {arr_px.max():.0f} px/s (near plate)")
        print(f"  Ratio: {arr_px.max()/arr_px.min():.1f}x (perspective acceleration)")

    if depths:
        first_depth = depths[0][1]
        last_depth = depths[-1][1]
        print(f"\nDepth range:")
        print(f"  Frame {depths[0][0]}: {first_depth:.1f} ft from plate")
        print(f"  Frame {depths[-1][0]}: {last_depth:.1f} ft from plate")
        print(f"  Expected: ~60ft -> ~12ft")

    # --- Verdict ---
    if speeds_3d:
        mean_err = abs(np.mean(speeds_3d) - sign_speed) / sign_speed * 100
        if mean_err < 10:
            print(f"\n>>> 3D MODEL LOOKS GOOD (mean error {mean_err:.1f}%)")
        elif mean_err < 25:
            print(f"\n>>> 3D MODEL NEEDS TUNING (mean error {mean_err:.1f}%)")
            print(f"    Try adjusting camera_height_ft and camera_behind_plate_ft in config")
        else:
            print(f"\n>>> 3D MODEL NEEDS SIGNIFICANT WORK (mean error {mean_err:.1f}%)")
            print(f"    The camera geometry parameters may be far off.")
            print(f"    Consider measuring camera position more precisely.")


if __name__ == "__main__":
    main()
