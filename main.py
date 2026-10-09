import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Resolve weights next to this script so the process works from any cwd.
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

if not MODEL_PATH.is_file():
    print(f"Error: model weights not found at {MODEL_PATH}")
    sys.exit(1)

# Load the trained fire & smoke detection model
model = YOLO(str(MODEL_PATH))

# Open default webcam (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Please check that a camera is connected and accessible.")
    sys.exit(1)

print("Starting real-time fire and smoke detection...")
print("Press 'q' to quit the detection window.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame from webcam. Exiting.")
            break

        # Run inference
        results = model(frame)

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
