# Fire Detection

Real-time fire and smoke detection using YOLO11 and OpenCV.

## Features

- Fire detection
- Smoke detection
- Real-time webcam detection
- Bounding boxes with confidence scores

## Tech Stack

- Python
- YOLO11 (Ultralytics)
- OpenCV
- PyTorch

## Model

The project uses a YOLO11 model fine-tuned on a fire and smoke dataset.

- `yolo11n.pt` — pretrained YOLO11 model
- `best.pt` — trained fire and smoke detection model

## Setup & Run

1. Install dependencies:

```bash
uv sync
```

2. Run the detection script:

```bash
python main.py
```

Press `q` to quit the webcam window.

## Notes

- Requires a working webcam.
- Make sure `best.pt` is present in the project root.
