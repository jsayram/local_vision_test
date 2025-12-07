# 🎨 MAGICAL LIVING PORTRAIT - Quick Start

## What You Just Got

A complete refactored **event-driven living portrait system** with:

✅ **Clean modular architecture** (8 Python modules)
✅ **YOLO integration** for fast person detection  
✅ **Moondream integration** ready (with stub fallback)
✅ **Sprite-based animation** with state machine
✅ **JSON memory storage** (people, interactions, settings)
✅ **Thread-safe** concurrent processing
✅ **Works on M1 Mac AND Raspberry Pi 3**

---

## 🚀 Run It NOW (3 Commands)

```bash
cd vision_experiment

# Test the system (no camera required)
python3 test_system.py

# Run the living portrait!
python3 portrait.py
```

**That's it!** The system will:
1. Auto-detect your camera
2. Show a portrait window
3. Detect when you appear
4. Respond with personality (stub mode)
5. Remember interactions in JSON files

---

## 📦 What Was Created

### Core Modules (NEW!)

| File | Purpose |
|------|---------|
| `config.py` | All constants, device detection, configuration |
| `models.py` | Data structures (Detection, Event, PersonState, etc.) |
| `storage.py` | JSON storage helpers (people, interactions, settings) |
| `moondream_client.py` | Moondream API wrapper with stub fallback |
| `detector.py` | YOLO integration + event detection logic |
| `animator.py` | Sprite loading and rendering system |
| **`portrait.py`** | **MAIN APPLICATION** - run this! |
| `test_system.py` | Test script to verify everything works |

### Reused from Existing
- `yolo_detector.py` ✓ (integrated)
- `opencv_detector.py` ✓ (kept as fallback)
- `detection_manager.py` ✓ (simplified)

### New Folders
- `sprites/` - Add your PNG images here (optional, has fallback)
- `memory/` - JSON storage auto-created on first run

---

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION LOOP (Thread)                     │
│  Camera → YOLO → Event Detection → Queue Moondream Jobs    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│               MOONDREAM WORKER (Thread)                     │
│  Process Queue → Call AI → Update State → Save Memory      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              ANIMATION LOOP (Main Thread)                   │
│  Update State → Render Portrait → Display → Handle Keys    │
└─────────────────────────────────────────────────────────────┘
```

### Events That Trigger Moondream

1. **NEW_PERSON** - Someone appears (instant)
2. **POSE_CHANGED** - Significant movement (>30% bbox change)
3. **PERIODIC_UPDATE** - Check-in after 45s of presence

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **P** | Pause/Resume |
| **D** | Debug mode |
| **R** | Reset animation |

---

## 🎨 Next Steps

### 1. Add Sprites (Optional but Cool!)

```bash
cd vision_experiment/sprites
# Add 6 PNG files:
# - idle.png, happy.png, curious.png, thoughtful.png
# - talking_open.png, talking_closed.png
```

See `sprites/README.md` for details.

**Without sprites**: System uses colored rectangles (works fine!)

### 2. Connect Real Moondream (Optional)

```bash
# Install Ollama
brew install ollama

# Pull moondream
ollama pull moondream

# Run server
ollama serve

# Configure
export MOONDREAM_API_URL="http://localhost:11434/api/generate"
export MOONDREAM_MODEL="moondream"

# Run portrait
python3 portrait.py
```

**Without Moondream**: System uses stub responses (still fun!)

### 3. Customize Configuration

Edit `config.py` to tune:
- Detection sensitivity
- Moondream call intervals  
- Animation timing
- Camera resolution
- Device-specific settings

### 4. Explore Memory

Check `memory/` folder after running:
- `people.json` - Known people
- `interactions.json` - Conversation history
- `settings.json` - System configuration

---

## 🔧 Configuration Highlights

### Auto Device Detection

System automatically detects:
- **M1 Max / Desktop**: High resolution, YOLOv8s model, 15s Moondream interval
- **Raspberry Pi 3**: Low resolution, YOLOv8n model, 30s Moondream interval
- **Raspberry Pi 4/5**: Medium settings

### Tuning Event Detection

```python
# config.py
POSE_CHANGE_THRESHOLD = 0.7      # Lower = more sensitive
MOONDREAM_MIN_INTERVAL = 15.0    # Min seconds between AI calls
PERIODIC_UPDATE_INTERVAL = 45.0  # Max seconds before check-in
```

### Debug Mode

```bash
export DEBUG_MODE="True"
python3 portrait.py
```

Shows detailed logging of all detection decisions!

---

## 📊 File Structure Summary

```
vision_experiment/
├── portrait.py              ← RUN THIS! Main application
├── config.py                ← Configuration
├── models.py                ← Data structures
├── storage.py               ← JSON helpers
├── moondream_client.py      ← AI integration
├── detector.py              ← YOLO + events
├── animator.py              ← Rendering
├── test_system.py           ← Test script
│
├── yolo_detector.py         ← (Existing, reused)
├── opencv_detector.py       ← (Existing, kept)
├── detection_manager.py     ← (Existing, simplified)
│
├── sprites/
│   ├── README.md            ← Sprite creation guide
│   └── (add your PNGs here)
│
├── memory/
│   ├── people.json          ← Auto-created
│   ├── interactions.json    ← Auto-created
│   └── settings.json        ← Auto-created
│
└── PORTRAIT_README.md       ← Full documentation
```

---

## 🐛 Troubleshooting

**Camera not opening?**
```bash
# macOS: Grant camera permission
# System Settings → Privacy & Security → Camera → Terminal

# Test camera
python3 -c "import detector; print(detector.list_available_cameras())"
```

**Sprites not loading?**
- They're optional! System works with fallback rectangles
- Add PNGs to `sprites/` folder when ready

**YOLO not working?**
- Check: `pip list | grep ultralytics`
- Should already be installed from requirements.txt

**Want faster testing?**
```bash
export DEBUG_MODE="True"  # Uses stub Moondream responses
python3 portrait.py
```

---

## 💡 Key Features

✅ **No blocking** - Moondream runs in background thread
✅ **Event-driven** - Only calls AI when needed (not every frame!)
✅ **Memory** - Remembers people and conversations
✅ **Portable** - Same code for Mac and Raspberry Pi
✅ **Debuggable** - Clear logging and state visibility
✅ **Extensible** - Clean modules, easy to modify
✅ **Fallback-first** - Works without sprites or Moondream

---

## 🎓 Learn More

- **Full documentation**: `PORTRAIT_README.md`
- **Sprite guide**: `sprites/README.md`
- **Code comments**: Every module is well-documented
- **Test before camera**: `python3 test_system.py`

---

## 🚀 You're Ready!

```bash
python3 portrait.py
```

Stand in front of the camera and watch the magic! 🎨✨

The portrait will:
1. Detect you (YOLO)
2. Recognize it's a NEW_PERSON event
3. Call Moondream (or stub) with your face
4. Say something based on the response
5. Show subtitles and animate
6. Remember the interaction in JSON

**Enjoy your living portrait!** 🖼️
