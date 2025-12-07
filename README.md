# Local Vision Test

A real-time vision detection system using YOLOv8 for fast and accurate object detection.

## Features

- Real-time camera object detection with YOLOv8
- Web-based interface with Flask
- Multiple detection modes (all objects, people, faces, etc.)
- Adjustable processing rates (1-60 FPS)
- Hardware-optimized models (desktop vs Raspberry Pi)
- Compatible with Raspberry Pi and other edge devices

## Requirements

- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Flask

## Installation

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the vision detection server:
```bash
cd vision_experiment
python vision_test_realtime.py
```

Then open your browser to: http://localhost:8000

## Detection Models

- **YOLOv8** (Default): Most capable, 80+ object classes from COCO dataset
- **OpenCV**: Lightweight fallback for basic detection

## Detection Modes

- **All Detection**: Detect all objects
- **Face Features**: Faces and people only
- **People**: People only
- **General Objects**: All objects except people
- **None**: No detection overlay

## Camera Permissions

On macOS, grant camera access to Terminal/Python in System Settings > Privacy & Security > Camera.

## License

MIT