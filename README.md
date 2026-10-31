# Fire Detection

Real-time fire and smoke detection using YOLO11 and OpenCV.

## Features

- Fire detection
- Smoke detection
- Real-time webcam detection
- Bounding boxes with confidence scores
- High accuracy on common fire/smoke scenarios

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
- Make sure `best.pt` is present in the project root (the script will exit with a clear error if it is missing).
- For best results, ensure good lighting conditions.
- Model works best with clear visibility of potential fire/smoke sources.
- Camera resources are always released, even if inference fails.
- You can adjust the detection confidence threshold by modifying the `CONF_THRESHOLD` constant in `main.py` (default 0.25). Lower values increase sensitivity but may raise false positives. A value around 0.4–0.5 often balances accuracy for fire/smoke scenes.
- If your webcam is not the first device, change `CAMERA_INDEX` in `main.py` (default is 0).

A Jupyter notebook (`app.ipynb`) is also included for interactive experimentation.

---
*Last updated: 2026-10-31*
