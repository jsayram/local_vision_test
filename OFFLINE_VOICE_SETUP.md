# Offline Voice Setup Instructions

## Complete 100% Offline Voice System

This system now uses **Vosk** for completely offline speech recognition and **pyttsx3** for text-to-speech. No internet required for voice features!

## Installation Steps

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install offline speech recognition and TTS
pip install vosk pyttsx3 pyaudio
```

### 2. Download Vosk Model

```bash
# Make download script executable
chmod +x download_vosk_model.sh

# Run download script (downloads ~50MB model)
./download_vosk_model.sh
```

**OR manually:**
```bash
cd vision_experiment/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-en
rm vosk-model-small-en-us-0.15.zip
```

### 3. Grant Microphone Permissions

**macOS:**
1. System Preferences → Security & Privacy → Privacy → Microphone
2. Grant permission to Terminal (or your Python executable)
3. You may need to restart the application after granting permission

## Usage

### Start the Server
```bash
./start_server.sh
```

### Using Voice Features

1. **Open browser:** http://localhost:8000
2. **Click "Start Speaking" button** in the header
3. **Speak your message** - you'll see live transcription in:
   - Portrait overlay (green banner at top)
   - Camera panel transcription log
4. **Final transcription** automatically sent to chat
5. **Portrait responds** with streaming text and speaks via TTS

### Stop the Server
```bash
./stop_server.sh
```
or press `Ctrl+C` in the terminal

## Features

### 100% Offline Components

✅ **Speech Recognition** - Vosk (no internet needed)
✅ **Text-to-Speech** - pyttsx3 (macOS system voices)
✅ **AI Processing** - Moondream (local)
✅ **Camera Detection** - YOLO (local)
✅ **All UI** - Runs locally

### Voice Workflow

```
User clicks "Start Speaking"
        ↓
PyAudio captures microphone
        ↓
Vosk processes audio (offline)
        ↓
Real-time transcription via WebSocket
        ↓
Display in portrait overlay + log
        ↓
Final text sent to Moondream (local)
        ↓
Response streams word-by-word
        ↓
pyttsx3 speaks response
```

## Transcription Display

### Portrait Overlay
- **Green banner** at top of portrait
- Shows **live interim** transcription (updates as you speak)
- Shows **final** transcription for 3 seconds
- **"You:"** label prefix

### Camera Panel Log
- Scrollable **history** of final transcriptions
- **Timestamps** for each entry
- **Confidence scores** (percentage)
- Keeps last **10 transcriptions**

## Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
# Install PortAudio first
brew install portaudio

# Then install PyAudio
pip install pyaudio
```

**If still failing:**
```bash
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" pip install pyaudio
```

### Vosk Not Recognizing Speech

1. **Check model installation:**
   ```bash
   ls vision_experiment/models/vosk-en
   # Should show: am/, conf/, graph/, ivector/
   ```

2. **Check microphone permissions** - System Preferences → Security → Microphone

3. **Check console output:**
   ```
   [Vosk] Loading model from: ...
   [Vosk] ✓ Model loaded successfully
   [Vosk] Opening audio stream...
   [Vosk] ✓ Started listening
   ```

### No TTS Output

1. **Check pyttsx3 installation:**
   ```bash
   pip list | grep pyttsx3
   ```

2. **Check console:**
   ```
   [TTS HH:MM:SS] 🔊 Speaking: "text"
   [TTS HH:MM:SS] ✓ Queued for speech
   ```

3. **Check system volume** - macOS system voices use system audio

### Speech Recognition Stops

- **Expected behavior:** Recognition stops after each phrase
- **Click "Start Speaking"** again to continue
- This prevents continuous accidental recording

## Files Removed

The following files are **no longer used** and can be deleted:

- `vision_experiment/core/wake_word_listener.py` - Replaced by offline_speech_recognizer.py
- Wake word functionality removed (was dependent on Porcupine/paid service)
- Browser Web Speech API code removed from voice.js

## What Changed

### Replaced
- ❌ Porcupine wake word → ✅ Push-to-talk button
- ❌ Browser Web Speech API → ✅ Vosk offline recognition

### Added
- ✅ `offline_speech_recognizer.py` - Complete offline speech system
- ✅ Live transcription overlays (2 locations)
- ✅ Comprehensive logging
- ✅ Status indicators

### Improved
- ✅ Stream overlay visibility (white text, darker background)
- ✅ Scrollable camera info panels
- ✅ Enhanced TTS integration
- ✅ Better error handling

## Console Output Examples

### Successful Voice Recognition
```
[Speech 14:23:45.123] ✓ FINAL: "hello how are you"
[Chat 14:23:45.124] 🎤 VOICE: "hello how are you"
[Chat 14:23:45.124] Person: unknown
```

### TTS Speaking
```
[TTS 14:23:46.234] 🔊 Speaking: "I'm doing well, thank you!"
[TTS 14:23:46.235] ✓ Queued for speech
```

### Vosk Status
```
[Vosk] Loading model from: vision_experiment/models/vosk-en
[Vosk] ✓ Model loaded successfully
[Vosk] Opening audio stream (rate=16000, chunk=4000)
[Vosk] ✓ Started listening
[Vosk] Listen loop started
[Vosk] ... interim: "hello"
[Vosk] ... interim: "hello how"
[Vosk] ... interim: "hello how are"
[Vosk] ✓ FINAL: "hello how are you"
```

## Performance

- **Vosk latency:** ~100-300ms
- **pyttsx3 latency:** Immediate (uses system voices)
- **Model size:** 50MB (small English model)
- **Memory usage:** ~200MB additional
- **CPU usage:** Low (real-time capable)

## Advanced

### Use Larger Vosk Model (Better Accuracy)

```bash
cd vision_experiment/models
# Download large model (1.8GB)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 vosk-en-large

# Update app.py line ~135
# model_path="models/vosk-en-large"
```

### Custom TTS Voice

In `app.py`, modify voice settings:
```python
voice_settings = VoiceSettings()
voice_settings.tts_rate = 175  # Words per minute (default: 150)
voice_settings.tts_volume = 1.0  # Volume 0-1 (default: 0.9)
```

---

**Everything now works 100% offline!** 🎉
