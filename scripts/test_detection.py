"""
Test script: Extract frames from a video and run YOLO detection.
Outputs which frames have a ball detected and where.

Usage (inside Docker):
    python /app/scripts/test_detection.py /videos/TrainingClips00351348.mov
"""
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"\n=== Ball Detection Test ===")
    print(f"Video: {video_path}")

    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.1f}s")

    # Load YOLOv8 model (pretrained on COCO - has 'sports ball' class #32)
    print(f"\nLoading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # nano model, fastest for testing

    # COCO class 32 = 'sports ball'
    SPORTS_BALL_CLASS = 32

    print(f"Scanning all {total_frames} frames for ball detections...\n")

    ball_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference (verbose=False to reduce noise)
        results = model(frame, verbose=False, conf=0.15)

        # Check for sports ball detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]

                # Log sports ball detections
                if cls_id == SPORTS_BALL_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    ball_detections.append({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "conf": conf,
                    })

                    print(f"  Frame {frame_idx:4d} ({frame_idx/fps:.2f}s) | "
                          f"Ball at ({cx:.0f}, {cy:.0f}) "
                          f"size {w:.0f}x{h:.0f} "
                          f"conf={conf:.2f}")

        frame_idx += 1

        # Progress every 60 frames (~1 second)
        if frame_idx % 60 == 0:
            print(f"  ... scanned {frame_idx}/{total_frames} frames")

    cap.release()

    # Summary
    print(f"\n=== Results ===")
    print(f"Total frames scanned: {frame_idx}")
    print(f"Ball detections: {len(ball_detections)}")

    if ball_detections:
        frames_with_ball = sorted(set(d["frame"] for d in ball_detections))
        print(f"Frames with ball: {frames_with_ball}")

        # Show trajectory if we got multiple detections
        if len(ball_detections) >= 2:
            print(f"\n=== Trajectory ===")
            for d in ball_detections:
                print(f"  t={d['time_s']:.3f}s  pos=({d['cx']:.0f}, {d['cy']:.0f})  conf={d['conf']:.2f}")

            # Rough speed estimate from pixel displacement
            first = ball_detections[0]
            last = ball_detections[-1]
            dx = last["cx"] - first["cx"]
            dy = last["cy"] - first["cy"]
            pixel_dist = np.sqrt(dx**2 + dy**2)
            time_diff = last["time_s"] - first["time_s"]

            if time_diff > 0:
                pixels_per_sec = pixel_dist / time_diff
                print(f"\n=== Rough Speed Estimate ===")
                print(f"  Pixel displacement: {pixel_dist:.0f} px over {time_diff:.3f}s")
                print(f"  Speed: {pixels_per_sec:.0f} pixels/sec")
                print(f"  (Need camera calibration to convert to mph)")
    else:
        print("No ball detected. May need to adjust confidence threshold or try a different model.")

    # Also report any other objects detected frequently (for context)
    print(f"\n=== Other Objects Detected (sampling every 30 frames) ===")
    cap = cv2.VideoCapture(video_path)
    object_counts = {}
    sample_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if sample_idx % 30 == 0:
            results = model(frame, verbose=False, conf=0.25)
            for result in results:
                for box in result.boxes:
                    cls_name = model.names[int(box.cls[0])]
                    object_counts[cls_name] = object_counts.get(cls_name, 0) + 1
        sample_idx += 1
    cap.release()

    for obj, count in sorted(object_counts.items(), key=lambda x: -x[1]):
        print(f"  {obj}: {count} detections")


if __name__ == "__main__":
    main()
