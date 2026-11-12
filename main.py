"""Real-time fire and smoke detection using a fine-tuned YOLO11 model and OpenCV.

Opens the default webcam, runs inference on each frame, draws bounding boxes,
and displays the annotated result. Press 'q' to exit.

Usage:
    python main.py                  # use default camera (index 0) and conf 0.25
    python main.py --camera 1       # use camera index 1
    python main.py --conf 0.4       # higher confidence to reduce false positives
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Resolve weights next to this script so the process works from any cwd.
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

# Default confidence threshold for detections. Lower values increase sensitivity
# but may raise false positives. Typical range: 0.2-0.5.
# For fire/smoke, 0.25-0.4 often works well depending on lighting and camera quality.
# Adjust based on false positive rate observed during testing.
DEFAULT_CONF_THRESHOLD: float = 0.25

# Window title shown in the OpenCV display.
WINDOW_TITLE = "Fire and Smoke Detection"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time fire and smoke detection")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold for detections (default: {DEFAULT_CONF_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_index = args.camera
    conf_threshold = args.conf

    if not (0.0 < conf_threshold <= 1.0):
        print(f"Error: Confidence threshold must be between 0 and 1 (got {conf_threshold})")
        sys.exit(1)

    if not MODEL_PATH.is_file():
        print(f"Error: model weights not found at {MODEL_PATH}")
        print("Please ensure 'best.pt' is in the same directory as this script.")
        sys.exit(1)

    # Load the trained fire & smoke detection model (classes typically include fire and smoke).
    # Note: the first inference call may be slower due to model warm-up.
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"Error: Failed to load YOLO model from {MODEL_PATH}: {e}")
        print("Check that the weights file is valid and ultralytics is installed correctly.")
        sys.exit(1)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            f"Error: Could not open webcam (index {camera_index}). "
            "Please check that a camera is connected, not in use by another app, and accessible."
        )
        print("Tip: try a different index with --camera N if you have multiple devices.")
        sys.exit(1)

    # Optional: reduce resolution for smoother real-time performance on lower-end hardware.
    # Uncomment the lines below if you experience lag (also lowers CPU/GPU load).
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 15)  # lower FPS can help on weaker machines

    print("Starting real-time fire and smoke detection...")
    print(f"Using camera index: {camera_index}")
    print(f"Window title: {WINDOW_TITLE}")
    print(f"Confidence threshold: {conf_threshold} (adjust with --conf to tune sensitivity)")
    print("Press 'q' to quit the detection window.")
    print("Note: the first frame may take longer while the model warms up.")
    print("Tip: ensure good lighting and clear view of potential fire/smoke sources for best results.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame from webcam. Exiting.")
                break

            # Run inference with configurable confidence
            results = model(frame, conf=conf_threshold)

            # Draw bounding boxes and labels
            annotated_frame = results[0].plot()

            # Display the result
            cv2.imshow(WINDOW_TITLE, annotated_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the device, including when inference raises.
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped. Webcam and windows released successfully.")


if __name__ == "__main__":
    main()
