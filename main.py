import cv2
from ultralytics import YOLO

# Load the trained fire & smoke detection model
model = YOLO("best.pt")

# Open default webcam (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting real-time fire and smoke detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
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

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
