"""Real-time fire and smoke detection using a fine-tuned YOLO11 model and OpenCV.

Opens the default webcam, runs inference on each frame, draws bounding boxes,
and displays the annotated result. Press 'q' to exit.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Resolve weights next to this script so the process works from any cwd.
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

# Change this if your camera is not the first device (0).
CAMERA_INDEX = 0

# Confidence threshold for detections. Lower values increase sensitivity
# but may raise false positives. Typical range: 0.2-0.5.
CONF_THRESHOLD = 0.25

if not MODEL_PATH.is_file():
    print(f"Error: model weights not found at {MODEL_PATH}")
    sys.exit(1)

# Load the trained fire & smoke detection model (classes typically include fire and smoke).
model = YOLO(str(MODEL_PATH))

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open webcam (index {CAMERA_INDEX}). Please check that a camera is connected and accessible.")
    sys.exit(1)

print("Starting real-time fire and smoke detection...")
print("Press 'q' to quit the detection window.")
print(f"Using confidence threshold: {CONF_THRESHOLD}")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame from webcam. Exiting.")
            break

        # Run inference with configurable confidence
        results = model(frame, conf=CONF_THRESHOLD)

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the result
        cv2.imshow("Fire and Smoke Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Always release the device, including when inference raises.
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped. Resources released.")
