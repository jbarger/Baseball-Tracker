"""
Analyze user-provided ground-truth ball positions to understand
ball characteristics and design a supplementary detector.

Ground truth from user annotation:
  Frame 97:  (988, 450)
  Frame 100: (993, 445)
  Frame 105: (1005, 443)
  Frame 110: (1019, 445)
  Frame 115: (1039, 451)
  Frame 120: (1068, 463)

Machine bbox: [972, 441, 1035, 522]
"""
import cv2
import numpy as np
import json
import sys
import os

# Ground truth
GT = {
    97:  (988, 450),
    100: (993, 445),
    105: (1005, 443),
    110: (1019, 445),
    115: (1039, 451),
    120: (1068, 463),
}

MACHINE_BBOX = [972, 441, 1035, 522]  # x1, y1, x2, y2

def interpolate_positions(gt):
    """Linearly interpolate positions for missing frames."""
    frames = sorted(gt.keys())
    all_pos = {}
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        x1, y1 = gt[f1]
        x2, y2 = gt[f2]
        for f in range(f1, f2 + 1):
            t = (f - f1) / (f2 - f1)
            all_pos[f] = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return all_pos


def main():
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/output"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps")
    print(f"Machine bbox: {MACHINE_BBOX}")
    print()

    # --- Trajectory analysis from ground truth ---
    frames = sorted(GT.keys())
    print("=== TRAJECTORY ANALYSIS ===")
    print(f"Ground truth positions: {len(GT)} frames ({frames[0]}-{frames[-1]})")
    print()

    for i in range(1, len(frames)):
        f1, f2 = frames[i-1], frames[i]
        x1, y1 = GT[f1]
        x2, y2 = GT[f2]
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy)
        dt_frames = f2 - f1
        px_per_frame = dist / dt_frames
        px_per_sec = px_per_frame * fps
        angle = np.degrees(np.arctan2(dy, dx))
        print(f"  Frame {f1}->{f2}: ({x1},{y1})->({x2},{y2})")
        print(f"    dx={dx:.0f} dy={dy:.0f} dist={dist:.1f}px over {dt_frames} frames")
        print(f"    {px_per_frame:.2f} px/frame = {px_per_sec:.0f} px/s, angle={angle:.1f}deg")

    # Overall stats
    x0, y0 = GT[frames[0]]
    xn, yn = GT[frames[-1]]
    total_dist = np.sqrt((xn-x0)**2 + (yn-y0)**2)
    total_frames = frames[-1] - frames[0]
    avg_px_per_frame = total_dist / total_frames
    avg_px_per_sec = avg_px_per_frame * fps

    print(f"\n  Overall: {total_dist:.1f}px in {total_frames} frames")
    print(f"  Average: {avg_px_per_frame:.2f} px/frame = {avg_px_per_sec:.0f} px/s")
    print(f"  Direction: mostly rightward with slight downward curve")

    # Interpolate all positions
    all_pos = interpolate_positions(GT)

    # --- Examine ball appearance at each GT position ---
    print("\n=== BALL APPEARANCE ANALYSIS ===")
    os.makedirs(f"{output_dir}/gt_crops", exist_ok=True)

    # Read a background frame (before ball appears) for reference
    cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
    ret, bg_frame = cap.read()
    if not ret:
        print("ERROR: Can't read background frame")
        return
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Collect stats about the ball's pixel appearance
    ball_intensities = []
    ball_sizes = []
    bg_intensities = []

    crop_size = 40  # Extract 80x80 crop centered on ball

    for frame_idx in sorted(all_pos.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        bx, by = all_pos[frame_idx]
        bx, by = int(round(bx)), int(round(by))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract crop around ball position
        y1 = max(0, by - crop_size)
        y2 = min(height, by + crop_size)
        x1 = max(0, bx - crop_size)
        x2 = min(width, bx + crop_size)

        crop = frame[y1:y2, x1:x2]
        crop_gray = gray[y1:y2, x1:x2]
        bg_crop_gray = bg_gray[y1:y2, x1:x2]

        # Frame differencing in the crop
        diff = cv2.absdiff(crop_gray, bg_crop_gray)

        # Ball pixel intensity at the annotated position
        if 0 <= by < height and 0 <= bx < width:
            ball_val = gray[by, bx]
            # Get background intensity at same position
            bg_val = bg_gray[by, bx]
            ball_intensities.append(float(ball_val))
            bg_intensities.append(float(bg_val))

        # Save crops for GT frames only
        if frame_idx in GT:
            # Draw crosshair at ball position on crop
            local_x = bx - x1
            local_y = by - y1
            crop_annotated = crop.copy()
            cv2.circle(crop_annotated, (local_x, local_y), 8, (0, 255, 0), 1)
            cv2.line(crop_annotated, (local_x - 12, local_y), (local_x + 12, local_y), (0, 255, 0), 1)
            cv2.line(crop_annotated, (local_x, local_y - 12), (local_x, local_y + 12), (0, 255, 0), 1)

            # Scale up 4x for visibility
            crop_big = cv2.resize(crop_annotated, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{output_dir}/gt_crops/frame_{frame_idx:03d}_crop.png", crop_big)

            # Also save the diff image
            diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            cv2.circle(diff_color, (local_x, local_y), 8, (255, 255, 255), 1)
            diff_big = cv2.resize(diff_color, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{output_dir}/gt_crops/frame_{frame_idx:03d}_diff.png", diff_big)

            # Compute diff stats around ball
            r = 10  # 10px radius
            ball_region = diff[max(0, local_y-r):local_y+r, max(0, local_x-r):local_x+r]
            if ball_region.size > 0:
                diff_mean = ball_region.mean()
                diff_max = ball_region.max()
                print(f"  Frame {frame_idx}: ball@({bx},{by}) "
                      f"intensity={gray[by,bx]:.0f} bg={bg_gray[by,bx]:.0f} "
                      f"diff_mean={diff_mean:.1f} diff_max={diff_max:.0f}")

    # Summary of ball appearance
    if ball_intensities:
        print(f"\n  Ball intensity: mean={np.mean(ball_intensities):.0f}, "
              f"std={np.std(ball_intensities):.0f}, "
              f"range=[{np.min(ball_intensities):.0f}, {np.max(ball_intensities):.0f}]")
        print(f"  BG intensity:   mean={np.mean(bg_intensities):.0f}, "
              f"std={np.std(bg_intensities):.0f}")
        print(f"  Ball-BG diff:   mean={np.mean(np.array(ball_intensities) - np.array(bg_intensities)):.0f}")

    # --- Examine machine exit region dynamics ---
    print("\n=== MACHINE EXIT REGION ANALYSIS ===")

    # Define the search region: extend machine bbox rightward (ball exits to the right)
    mx1, my1, mx2, my2 = MACHINE_BBOX
    # Search region: from machine right edge to +150px right, vertically centered
    search_x1 = mx1 - 20  # slightly left of machine
    search_y1 = my1 - 30  # above machine
    search_x2 = mx2 + 200  # well right of machine
    search_y2 = my2 + 30  # below machine
    print(f"  Search region: ({search_x1},{search_y1})-({search_x2},{search_y2})")
    print(f"  Size: {search_x2-search_x1}x{search_y2-search_y1}")

    # Analyze frame-by-frame what changes in this region
    print("\n  Frame-by-frame motion in search region:")
    prev_gray = None
    for frame_idx in range(90, 130):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        region = gray[search_y1:search_y2, search_x1:search_x2]

        if prev_gray is not None:
            prev_region = prev_gray[search_y1:search_y2, search_x1:search_x2]
            diff = cv2.absdiff(region, prev_region)
            mean_diff = diff.mean()
            max_diff = diff.max()

            # Find location of max diff
            max_loc = np.unravel_index(diff.argmax(), diff.shape)
            max_abs_y = max_loc[0] + search_y1
            max_abs_x = max_loc[1] + search_x1

            # Count pixels above threshold
            thresh = 15
            motion_pixels = (diff > thresh).sum()

            marker = ""
            if frame_idx in GT:
                gx, gy = GT[frame_idx]
                marker = f" <<< GT ball@({gx},{gy})"

            if mean_diff > 1.0 or frame_idx in GT:
                print(f"    F{frame_idx}: mean_diff={mean_diff:.1f} max={max_diff:.0f}@({max_abs_x},{max_abs_y}) "
                      f"motion_px={motion_pixels}{marker}")

        prev_gray = gray.copy()

    # --- Compute what detection approach would work ---
    print("\n=== DETECTION APPROACH ANALYSIS ===")
    print(f"Ball speed: ~{avg_px_per_frame:.1f} px/frame = ~{avg_px_per_sec:.0f} px/s")
    print(f"Ball size: likely 8-15px diameter (baseball at 60ft from camera)")
    print(f"YOLO misses because:")
    print(f"  - Ball is ~10-15px, near minimum YOLO grid cell size")
    print(f"  - Low contrast against cage background")
    print(f"  - Partially overlaps with machine in early frames")
    print()
    print(f"Candidate approaches:")
    print(f"  1. Background subtraction + blob detection in machine exit region")
    print(f"  2. Frame differencing with small ROI around machine exit")
    print(f"  3. Template matching (if ball has consistent appearance)")
    print(f"  4. Optical flow in machine exit region")

    cap.release()
    print("\nDone.")


if __name__ == "__main__":
    main()
