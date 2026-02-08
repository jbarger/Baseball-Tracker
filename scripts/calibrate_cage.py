"""
Cage calibration helper: extracts a reference frame and overlays calibration info.

Usage (inside Docker):
    # Step 1: Extract reference frame
    python /app/scripts/calibrate_cage.py /videos/TrainingClips00351348.mov /output/reference_frame.jpg

    # Step 2: After setting machine_bbox_px and roi_polygon in cage_config.json,
    #          run again to verify calibration overlay
    python /app/scripts/calibrate_cage.py /videos/TrainingClips00351348.mov /output/calibration_overlay.jpg --verify

Workflow:
    1. Run this script to get reference_frame.jpg with a pixel grid overlay
    2. Open the image and note:
       - Machine bounding box: [x1, y1, x2, y2] around the pitching machine
       - ROI polygon: list of [x, y] points tracing the cage net boundary
    3. Enter those values into config/cage_config.json
    4. Run again with --verify to see the calibration overlaid on the frame
"""
import sys
import os
import json
import cv2
import numpy as np

# Add common module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_reference_frame(video_path: str, frame_num: int = 0) -> np.ndarray:
    """Extract a single frame from the video for calibration reference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    if frame_num > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Cannot read frame {frame_num}")
        sys.exit(1)

    return frame


def draw_grid(frame: np.ndarray, spacing: int = 100) -> np.ndarray:
    """Draw a pixel coordinate grid on the frame for measurement reference."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Grid lines (subtle)
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), (128, 128, 128), 1)
        cv2.putText(overlay, str(x), (x + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), (128, 128, 128), 1)
        cv2.putText(overlay, str(y), (2, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Crosshair at center
    cx, cy = w // 2, h // 2
    cv2.line(overlay, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
    cv2.line(overlay, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
    cv2.putText(overlay, f"Center: ({cx}, {cy})", (cx + 25, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Resolution info
    cv2.putText(overlay, f"{w}x{h}", (w - 120, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Instructions
    instructions = [
        "CALIBRATION REFERENCE FRAME",
        "1. Note the pitching machine bounding box [x1, y1, x2, y2]",
        "2. Note the cage net boundary polygon [[x,y], [x,y], ...]",
        "3. Enter values in config/cage_config.json",
        "4. Re-run with --verify to check overlay",
    ]
    y = 30
    for line in instructions:
        cv2.putText(overlay, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22

    # Alpha blend grid onto original
    alpha = 0.6
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def draw_calibration_overlay(frame: np.ndarray, config: dict,
                             machines_db: dict) -> np.ndarray:
    """Draw calibration results (ROI polygon, machine bbox, scale) on frame."""
    overlay = frame.copy()

    # ROI polygon
    roi = config.get("roi_polygon", [])
    if roi and len(roi) >= 3:
        poly = np.array(roi, dtype=np.int32)
        cv2.polylines(overlay, [poly], isClosed=True, color=(255, 255, 255),
                      thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(overlay, "ROI BOUNDARY", (roi[0][0], roi[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(overlay, "ROI: Not configured (full frame)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # Machine bounding box
    cal = config.get("calibration", {})
    bbox = cal.get("machine_bbox_px")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Machine info
        machine_type = config.get("machine_type", "unknown")
        machine_info = machines_db.get("machines", {}).get(machine_type, {})
        machine_name = machine_info.get("name", machine_type)
        dims = machine_info.get("dimensions_inches", {})

        box_h_px = abs(y2 - y1)
        box_w_px = abs(x2 - x1)
        machine_h_in = dims.get("height", 0)

        label_lines = [
            f"MACHINE: {machine_name}",
            f"Box: {box_w_px}x{box_h_px} px",
        ]

        if machine_h_in > 0:
            ppi = box_h_px / machine_h_in
            label_lines.append(f"Scale: {ppi:.2f} px/in")
            label_lines.append(f"Real: {machine_h_in}\"H x {dims.get('width', '?')}\"W")

        y_text = y1 - 10
        for line in reversed(label_lines):
            cv2.putText(overlay, line, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            y_text -= 20
    else:
        cv2.putText(overlay, "Machine bbox: Not configured", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # Distance info
    distance = config.get("machine_distance_ft", "?")
    cv2.putText(overlay, f"Machine distance: {distance} ft from home plate",
                (10, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    return overlay


def main():
    if len(sys.argv) < 3:
        print("Usage: python calibrate_cage.py <video_path> <output_image> [--verify] [--frame N]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2]
    verify_mode = "--verify" in sys.argv

    # Parse optional frame number
    frame_num = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--frame" and i + 1 < len(sys.argv):
            frame_num = int(sys.argv[i + 1])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"=== Cage Calibration Helper ===")
    print(f"Video: {video_path}")
    print(f"Frame: {frame_num}")
    print(f"Mode: {'VERIFY' if verify_mode else 'REFERENCE'}")

    frame = extract_reference_frame(video_path, frame_num)
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    if verify_mode:
        # Load config and overlay calibration
        config_path = "/app/config/cage_config.json"
        machines_path = "/app/config/machines.json"

        if not os.path.exists(config_path):
            print(f"ERROR: Config not found at {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            config = json.load(f)
        with open(machines_path) as f:
            machines_db = json.load(f)

        result = draw_calibration_overlay(frame, config, machines_db)
        print(f"\nCalibration overlay:")

        # Report calibration status
        cal = config.get("calibration", {})
        bbox = cal.get("machine_bbox_px")
        if bbox:
            print(f"  Machine bbox: {bbox}")
            machine_type = config.get("machine_type")
            machine_info = machines_db.get("machines", {}).get(machine_type, {})
            dims = machine_info.get("dimensions_inches", {})
            if dims:
                ppi = abs(bbox[3] - bbox[1]) / dims["height"]
                print(f"  Pixels per inch: {ppi:.2f}")
                print(f"  At {config['machine_distance_ft']}ft, "
                      f"1 foot = {ppi * 12:.0f} pixels")
        else:
            print("  Machine bbox: NOT SET")

        roi = config.get("roi_polygon", [])
        if roi:
            print(f"  ROI polygon: {len(roi)} points")
        else:
            print("  ROI polygon: NOT SET (using full frame)")
    else:
        result = draw_grid(frame, spacing=100)

    cv2.imwrite(output_path, result)
    print(f"\nSaved: {output_path}")
    print(f"Open this image to identify machine bbox and cage boundaries.")


if __name__ == "__main__":
    main()
