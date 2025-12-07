#!/bin/bash
# Download Vosk model for offline speech recognition

cd "$(dirname "$0")/vision_experiment"

echo "Creating models directory..."
mkdir -p models

cd models

echo "Downloading Vosk English model (50MB)..."
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

echo "Extracting model..."
unzip vosk-model-small-en-us-0.15.zip

echo "Renaming to vosk-en..."
mv vosk-model-small-en-us-0.15 vosk-en

echo "Cleaning up..."
rm vosk-model-small-en-us-0.15.zip

echo ""
echo "✓ Vosk model installed successfully!"
echo "  Location: vision_experiment/models/vosk-en"
echo ""
