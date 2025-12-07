# Living Portrait - Quick Start Guide

## 🚀 Starting the Server

### Option 1: Using the start script (Recommended)
```bash
./start_server.sh
```
This will start the server and handle Ctrl+C gracefully.

### Option 2: Manual start
```bash
source .venv/bin/activate
cd vision_experiment
python app.py
```

## 🛑 Stopping the Server

### Option 1: Using the stop script (Safest)
```bash
./stop_server.sh
```

### Option 2: Using Ctrl+C (When started with start_server.sh)
Just press `Ctrl+C` in the terminal where the server is running.

### Option 3: Force kill (If server is stuck)
```bash
lsof -ti :8000 | xargs kill -9
```

## 🌐 Accessing the Interface

Open your browser to: **http://localhost:8000**

## 📋 Features Available

### ✅ Currently Working
- **Chat Interface**: Type messages and get AI responses
- **Streaming Responses**: Word-by-word display like ChatGPT
- **Portrait Animation**: Real-time mood changes and speaking animation
- **Camera Feed**: Live person detection with YOLO
- **Event Detection**: Automatic responses to new people, movement, periodic updates

### ⚠️ Requires Additional Setup
To enable full features, install optional dependencies:

```bash
# Face Recognition
pip install cmake dlib face_recognition

# Wake Word Detection  
pip install pvporcupine

# Voice I/O
pip install pyttsx3 pyaudio
```

## 🎯 Usage

### Chat with the Portrait
1. Type a message in the chat input box (left panel)
2. Press Enter or click Send
3. Watch the response stream word-by-word in the portrait overlay
4. See the conversation history in the chat panel

### Special Commands
- **"forget that"** - Deletes the last exchange
- **"clear chat"** - Clears the entire conversation
- **"start over"** - Same as clear chat

### Voice Interaction (When Libraries Installed)
1. Click the "Wake Word" button in the header
2. Say "hey portrait" (or use the manual trigger)
3. Speak your message when the mic icon appears
4. Portrait responds with voice

### Face Recognition (When Libraries Installed)
- System automatically detects and recognizes faces
- If confidence > 70%: Auto-identifies
- If confidence < 70%: Shows confirmation dialog
- Click "Yes" or "No" to confirm/reject (30s timeout)

## 🔧 Troubleshooting

### Server won't stop
```bash
# Use the stop script
./stop_server.sh

# Or force kill
lsof -ti :8000 | xargs kill -9
```

### Camera not working
- Check camera permissions in System Preferences
- Make sure no other app is using the camera
- Try restarting the server

### Port 8000 already in use
```bash
# Kill whatever is using port 8000
lsof -ti :8000 | xargs kill -9

# Then start the server again
./start_server.sh
```

### Dependencies missing
```bash
# Reinstall requirements
source .venv/bin/activate
pip install -r requirements.txt
```

## 📁 Project Structure

```
local_vision_test/
├── start_server.sh          # Start the server
├── stop_server.sh           # Stop the server gracefully
├── requirements.txt         # Python dependencies
├── vision_experiment/
│   ├── app.py              # Main Flask-SocketIO server
│   ├── models/
│   │   └── models.py       # Data structures
│   ├── core/
│   │   ├── face_recognition_manager.py
│   │   ├── wake_word_listener.py
│   │   ├── voice_manager.py
│   │   ├── stream_manager.py
│   │   ├── chat_manager.py
│   │   └── ...
│   ├── templates/
│   │   └── index.html      # 3-column web interface
│   └── static/
│       ├── css/
│       │   └── style.css   # UI styling
│       └── js/
│           ├── socket.js   # WebSocket manager
│           ├── chat.js     # Chat interface
│           ├── voice.js    # Voice controls
│           └── streaming.js # Streaming display
```

## 🎨 UI Layout

```
┌─────────────────────────────────────────────────┐
│              Header (Voice Controls)             │
├───────────┬──────────────────┬──────────────────┤
│   Chat    │     Portrait     │      Camera      │
│ Interface │   with Overlay   │       Feed       │
│           │                  │                  │
│ Messages  │   Streaming      │   Detection      │
│ History   │   Response       │      Info        │
│           │                  │                  │
│ Input Box │                  │                  │
└───────────┴──────────────────┴──────────────────┘
```

## 💡 Tips

1. **Chat is always available** - Even without voice libraries installed
2. **Streaming works immediately** - Responses appear word-by-word
3. **Camera detection is automatic** - System responds to movement
4. **Messages are saved** - Per-person conversation history
5. **50 message limit** - Older messages auto-archive by date

## 🐛 Known Issues

1. **Ctrl+C doesn't work when running in background** - Use `./stop_server.sh` instead
2. **Voice libraries can be tricky to install** - Optional, system works without them
3. **Face recognition needs dlib** - Requires build tools (cmake)

## 📞 Getting Help

Check the console output for detailed logs:
- `[Chat]` - Chat manager events
- `[Stream Manager]` - Streaming events
- `[Moondream Worker]` - AI response generation
- `[Animation Loop]` - Portrait animation
- `[SocketIO]` - WebSocket connections
- `[Cleanup]` - Shutdown events

---

**Enjoy your Living Portrait! 🎨✨**
