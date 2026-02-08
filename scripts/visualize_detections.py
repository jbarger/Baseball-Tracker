"""
Visualize YOLO detections: draws bounding boxes on every frame and writes
an annotated MP4 you can scrub through to see exactly what the model sees.

Usage (inside Docker):
    python /app/scripts/visualize_detections.py /videos/TrainingClips00351348.mov /output/annotated.mp4

Color key:
    GREEN  = sports ball (COCO class 32)
    BLUE   = baseball bat (COCO class 34)
    YELLOW = person (COCO class 0)
    WHITE  = everything else
"""
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- colour palette by object class ----------
COLORS = {
    32: (0, 255, 0),     # sports ball  -> GREEN
    34: (255, 128, 0),   # baseball bat -> ORANGE
    0:  (0, 255, 255),   # person       -> YELLOW
}
DEFAULT_COLOR = (255, 255, 255)  # everything else -> WHITE

BALL_CLASS = 32


def draw_detections(frame, boxes, model_names, frame_idx, fps):
    """Draw bounding boxes, labels and frame info HUD onto a frame."""
    overlay = frame.copy()
    ball_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_name = model_names[cls_id]
        color = COLORS.get(cls_id, DEFAULT_COLOR)

        if cls_id == BALL_CLASS:
            ball_count += 1

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Centre dot for balls
        if cls_id == BALL_CLASS:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)  # red dot

    # ---------- HUD (top-left) ----------
    time_s = frame_idx / fps
    hud_lines = [
        f"Frame {frame_idx}  |  {time_s:.2f}s",
        f"Balls detected: {ball_count}",
    ]
    y_offset = 30
    for line in hud_lines:
        cv2.putText(overlay, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 28

    return overlay


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_detections.py <input_video> <output_video>")
        print("Example: python visualize_detections.py /videos/clip.mov /output/annotated.mp4")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"=== Detection Visualizer ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f}fps")
    print(f"Frames: {total_frames} ({duration:.1f}s)")
    print(f"Confidence threshold: {conf_threshold}")

    # Output video writer (H.264 in MP4 container)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {output_path}")
        sys.exit(1)

    # Load YOLO model
    print(f"\nLoading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"Processing frames...\n")
    frame_idx = 0
    total_ball_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, verbose=False, conf=conf_threshold)

        # Collect all boxes from this frame
        all_boxes = []
        has_ball = False
        for result in results:
            for box in result.boxes:
                all_boxes.append(box)
                if int(box.cls[0]) == BALL_CLASS:
                    has_ball = True

        if has_ball:
            total_ball_frames += 1

        # Draw detections on frame
        annotated = draw_detections(frame, all_boxes, model.names, frame_idx, fps)

        writer.write(annotated)
        frame_idx += 1

        # Progress
        if frame_idx % 60 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} frames ({pct:.0f}%)"
                  f"  |  ball visible in {total_ball_frames} frames so far")

    cap.release()
    writer.release()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n=== Done ===")
    print(f"Wrote {frame_idx} annotated frames to {output_path}")
    print(f"Output size: {file_size_mb:.1f} MB")
    print(f"Ball visible in {total_ball_frames}/{frame_idx} frames")
    print(f"\nOpen the output video in any player to scrub through detections.")


if __name__ == "__main__":
    main()
