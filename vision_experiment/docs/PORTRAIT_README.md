# Magical Living Portrait - Architecture Documentation

## 🎨 Overview

A clean, event-driven "living portrait" system that watches through a camera, detects people, and responds with personality using vision-language AI.

### Architecture Philosophy
- **Fast Loop**: Camera + YOLO detection (30-60 FPS)
- **Slow Loop**: Moondream vision-language calls (event-driven, ~15-45s intervals)  
- **Animation Loop**: Sprite-based rendering with state machine
- **Memory**: JSON-based local storage for persistence

---

## 📁 Project Structure

```
vision_experiment/
├── config.py                 # All configuration and constants
├── models.py                 # Data structures (Detection, PersonState, Event, etc.)
├── storage.py                # JSON storage (people, interactions, settings)
├── moondream_client.py       # Moondream API integration
├── detector.py               # YOLO integration + event detection logic
├── animator.py               # Sprite loading and rendering system
├── portrait.py               # Main application (ENTRY POINT)
│
├── yolo_detector.py          # YOLO detector wrapper (existing, reused)
├── opencv_detector.py        # OpenCV fallback (existing, kept)
├── detection_manager.py      # Detection manager (existing, simplified)
│
├── sprites/                  # PNG sprite images (add your own!)
│   ├── idle.png
│   ├── happy.png
│   ├── curious.png
│   ├── thoughtful.png
│   ├── talking_open.png
│   └── talking_closed.png
│
├── memory/                   # JSON storage (auto-created)
│   ├── people.json
│   ├── interactions.json
│   └── settings.json
│
└── templates/                # Web UI (optional, for Flask server)
    └── index.html
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install required packages (already done)
pip install -r requirements.txt
```

### 2. Add Sprite Images

Create or add PNG images to `vision_experiment/sprites/`:

Required sprites:
- `idle.png` - Default/neutral expression
- `happy.png` - Happy/excited expression  
- `curious.png` - Curious/questioning expression
- `thoughtful.png` - Thoughtful/contemplative expression
- `talking_open.png` - Talking with mouth open
- `talking_closed.png` - Talking with mouth closed

**Sprite Requirements:**
- PNG format with transparent background (RGBA)
- Recommended size: 400x400 to 600x600 pixels
- Should show a face/character portrait

**Fallback**: If sprites are missing, the system uses colored rectangles as placeholders.

### 3. Configure Moondream API (Optional)

Set environment variables for Moondream:

```bash
export MOONDREAM_API_URL="http://localhost:11434/api/generate"  # Ollama default
export MOONDREAM_MODEL="moondream"
export DEBUG_MODE="True"  # Use stub responses for testing
```

**For testing without Moondream:**
- System will automatically use stub responses if no API URL is configured
- Stub provides canned responses based on event type

### 4. Run the Portrait

```bash
cd vision_experiment
python3 portrait.py
```

**First Run:**
1. Camera selection prompt (if multiple cameras available)
2. System initialization
3. Portrait window opens
4. Start moving in front of the camera!

---

## 🎮 Controls

While the portrait window is active:

| Key | Action |
|-----|--------|
| **Q** or **ESC** | Quit application |
| **P** | Pause/Resume detection |
| **D** | Toggle debug logging |
| **R** | Reset animation state |

---

## 🧠 How It Works

### Event Detection

The system triggers Moondream calls based on these events:

1. **NEW_PERSON** - Someone appears in frame
   - No one detected → Person detected
   - Triggers immediately

2. **POSE_CHANGED** - Significant movement
   - Bounding box changes by >30% (configurable)
   - Respects minimum interval (15-30s)

3. **PERIODIC_UPDATE** - Regular check-in
   - After 45s of presence with no other events
   - Keeps conversation flowing

4. **PERSON_LEFT** - Someone leaves
   - Currently doesn't trigger Moondream (can be enabled)

### Threading Model

```
Main Thread (Animation Loop)
  ├─ Render portrait based on animation_state
  ├─ Display via cv2.imshow()
  └─ Handle keyboard input
  
Vision Thread (Background)
  ├─ Capture camera frames
  ├─ Run YOLO detection (every N frames)
  ├─ Detect events (NEW_PERSON, POSE_CHANGED, etc.)
  └─ Queue Moondream jobs

Moondream Worker Thread (Background)
  ├─ Process jobs from queue
  ├─ Call Moondream API with face crop + context
  ├─ Update animation_state with response
  └─ Save interaction to JSON storage
```

### State Management

**PersonState** (thread-safe with lock)
- Current person's bounding box
- Timing information (first_seen, last_seen, last_moondream_call)
- Identity (person_id, name if recognized)

**AnimationState** (thread-safe with lock)
- Current mood (idle, happy, curious, thoughtful)
- Speaking status (talking animation)
- Current subtitle text
- Timing for animations

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Device Modes
- Auto-detects: M1 Max, Raspberry Pi 3/4/5, generic desktop
- Adjusts: resolution, model size, intervals automatically

### Detection Tuning
```python
POSE_CHANGE_THRESHOLD = 0.7      # Lower = more sensitive to movement
MOONDREAM_MIN_INTERVAL = 15.0    # Min seconds between calls
PERIODIC_UPDATE_INTERVAL = 45.0  # Max seconds before check-in
MIN_BBOX_SIZE = 50               # Min detection size (pixels)
```

### Animation Timing
```python
TALKING_MOUTH_TOGGLE_INTERVAL = 0.15  # Mouth flap speed
SPEAKING_DURATION = 5.0               # How long to show talking
SUBTITLE_DURATION = 8.0               # How long to show subtitles
```

---

## 💾 JSON Storage

### people.json
```json
[
  {
    "person_id": 1,
    "name": "Jose",
    "notes": "Owner of the house",
    "first_seen": "2025-12-07T10:30:00",
    "last_seen": "2025-12-07T14:22:15"
  }
]
```

### interactions.json
```json
[
  {
    "person_id": 1,
    "timestamp": "2025-12-07T14:22:15",
    "mood": "happy",
    "text": "Good to see you again, Jose!",
    "event_type": "NEW_PERSON"
  }
]
```

### settings.json
```json
{
  "moondream_interval_seconds": 20,
  "pose_change_threshold": 0.3,
  "system_prompt": "You are a magical living portrait..."
}
```

---

## 🔧 Moondream API Integration

### Current Implementation

The system is ready for Moondream integration:

**File**: `moondream_client.py`

**Function**: `call_moondream_on_face(face_image, context)`

**Stub vs Real**:
- Stub responses provided for testing
- Real API calls ready - just need endpoint configuration

### To Use Real Moondream

1. **Ollama (Recommended)**:
```bash
# Install Ollama
brew install ollama  # or visit ollama.com

# Pull moondream model
ollama pull moondream

# Run Ollama server
ollama serve  # Default: http://localhost:11434

# Configure
export MOONDREAM_API_URL="http://localhost:11434/api/generate"
export MOONDREAM_MODEL="moondream"
```

2. **Custom API**:
- Update `MOONDREAM_API_URL` in config
- Modify `parse_moondream_response()` in `moondream_client.py` to match your API format

---

## 🐛 Debugging

### Enable Debug Mode

```bash
export DEBUG_MODE="True"
python3 portrait.py
```

**Debug Features:**
- Detailed console logging
- Detection decision logs
- Moondream call details
- On-screen debug overlay

### Common Issues

**Camera not working:**
```bash
# List available cameras
python3 -c "import detector; print(detector.list_available_cameras())"

# Grant camera permissions (macOS)
# System Settings → Privacy & Security → Camera → Terminal
```

**Sprites not loading:**
- Check `sprites/` folder exists
- Verify PNG files are present
- System will use fallback colored rectangles if missing

**Moondream not responding:**
- Check `MOONDREAM_API_URL` is set
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- System falls back to stub responses on failure

---

## 🎯 Next Steps

### Immediate Enhancements

1. **Add Sprite Images**
   - Create or download portrait sprites
   - Add to `sprites/` folder
   - Recommended: Use consistent art style

2. **Connect Moondream**
   - Install and run Ollama
   - Test with real vision-language responses
   - Tune system prompt in `settings.json`

3. **Personalize Memory**
   - Add yourself to `people.json`
   - Train the portrait to recognize you
   - Customize greetings and responses

### Future Enhancements

- Face recognition for auto-identification
- Web interface for remote viewing (Flask already set up)
- Voice synthesis (TTS) for spoken responses
- Multi-person tracking
- Gesture recognition for interactive commands
- Cloud storage sync for memory
- Mobile app control

---

## 📊 Performance

**M1 Max MacBook Pro:**
- Vision Loop: ~30 FPS (with YOLO on every frame)
- Animation Loop: ~33 FPS (cv2.waitKey(30))
- Moondream Calls: ~2-5s each (event-driven, not continuous)

**Raspberry Pi 3:**
- Vision Loop: ~10 FPS (YOLO every 3rd frame)
- Animation Loop: ~30 FPS
- Moondream Calls: ~5-10s each
- Recommended: Use Moondream stub or remote API

---

## 🤝 Contributing

This is a modular, well-documented system. To extend:

1. **New Detection Events**: Edit `detector.py` → `detect_event_from_person_state()`
2. **New Moods/Sprites**: Add to `config.SPRITE_FILES` and `config.FALLBACK_COLORS`
3. **Custom Storage**: Extend `storage.py` with new JSON files
4. **Different AI Model**: Create new client in style of `moondream_client.py`

---

## 📝 License

MIT License - Feel free to use, modify, and distribute!

---

## 🎨 Credits

- **YOLO**: Ultralytics YOLOv8
- **OpenCV**: Computer vision and rendering
- **Moondream**: Vision-language model
- **Flask**: Web server (optional component)

Built with ❤️ for the joy of creating interactive AI art.
