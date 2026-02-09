"""
Trace the machine-exit detector frame by frame to see what it's finding
and compare with ground truth.
"""
import sys
import os
import json
import cv2
import numpy as np

sys.path.insert(0, "/app")
from common.machine_exit_detector import MachineExitDetector, MachineExitConfig

GT = {
    97:  (988, 450),
    100: (993, 445),
    105: (1005, 443),
    110: (1019, 445),
    115: (1039, 451),
    120: (1068, 463),
}

def main():
    video_path = sys.argv[1]
    config_path = "/app/config/cage_config.json"

    with open(config_path) as f:
        config = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    machine_bbox = config["calibration"]["machine_bbox_px"]
    me_config_data = config.get("machine_exit_detector", {})
    me_config = MachineExitConfig(
        machine_bbox=machine_bbox,
        search_extend_right=me_config_data.get("search_extend_right", 250),
        search_extend_left=me_config_data.get("search_extend_left", 20),
        search_extend_vertical=me_config_data.get("search_extend_vertical", 40),
        min_diff_intensity=me_config_data.get("min_diff_intensity", 80.0),
        bg_warmup_frames=me_config_data.get("bg_warmup_frames", 30),
        bg_alpha=me_config_data.get("bg_alpha", 0.02),
        min_blob_area=me_config_data.get("min_blob_area", 4),
        max_blob_area=me_config_data.get("max_blob_area", 400),
        max_aspect_ratio=me_config_data.get("max_aspect_ratio", 3.0),
    )

    detector = MachineExitDetector(me_config, height, width)
    print(f"Search region: {detector.search_region}")
    print(f"Tracing frames 0-200...\n")

    frame_idx = 0
    detection_count = 0

    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.update(frame)

        gt_str = ""
        if frame_idx in GT:
            gx, gy = GT[frame_idx]
            gt_str = f"  GT=({gx},{gy})"

        if result is not None:
            cx, cy, radius, conf, diff_val = result
            detection_count += 1

            # Check distance to ground truth if available
            err_str = ""
            if frame_idx in GT:
                gx, gy = GT[frame_idx]
                err = np.sqrt((cx - gx)**2 + (cy - gy)**2)
                err_str = f" err={err:.1f}px"

            print(f"  F{frame_idx:3d}: DETECTED at ({cx:.0f},{cy:.0f}) "
                  f"r={radius:.1f} conf={conf:.2f} diff={diff_val:.0f}{err_str}{gt_str}")
        elif frame_idx in GT or (80 <= frame_idx <= 140):
            print(f"  F{frame_idx:3d}: no detection{gt_str}")

        frame_idx += 1

    print(f"\nTotal detections: {detection_count}")
    cap.release()


if __name__ == "__main__":
    main()
