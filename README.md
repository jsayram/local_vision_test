# Local Vision Test

A real-time vision analysis tool using Moondream and Ollama for local AI-powered image understanding.

## Features

- Real-time camera vision analysis
- Single image file analysis
- Fast processing with background threading
- Compatible with Raspberry Pi and other edge devices

## Requirements

- Python 3.8+
- Ollama with Moondream model
- OpenCV
- Requests

## Installation

1. Install Ollama: https://ollama.com
2. Pull Moondream model: `ollama pull moondream`
3. Install Python dependencies: `pip install opencv-python requests`

## Usage

Run the vision test:
```bash
python vision_test_realtime.py
```

Choose your input source:
1. Live camera for real-time analysis
2. Image file for single analysis

## Camera Permissions

On macOS, grant camera access to Terminal/Python in System Settings > Privacy & Security > Camera.

## License

MIT