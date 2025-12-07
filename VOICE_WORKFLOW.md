# Living Portrait Voice & Animation Workflow

## Overview
This document explains how the voice conversation and mouth animation system works in the Living Portrait application.

## Complete Workflow (Voice Conversation)

### 1. User Speaks
```
User clicks "Start Speaking" button
  ↓
Frontend emits 'start_speech_recognition' via SocketIO
  ↓
Server starts OfflineSpeechRecognizer (Vosk)
  ↓
PyAudio captures microphone audio (16kHz, mono, 8000 chunk size)
```

### 2. Speech Recognition
```
Vosk processes audio in real-time
  ↓
Interim results → on_speech_partial() callback
  ↓
Frontend displays partial transcription (green overlay + log)
  ↓
Final result (after pause) → on_speech_final() callback
  ↓
Transcription sent to chat system
```

### 3. AI Processing
```
Chat receives voice message
  ↓
MoondreamPrompt built with:
  - User's transcribed message
  - Recent conversation history (last 5 messages)
  - Person context (name, ID)
  - Event hints ("respond to what they said")
  ↓
Moondream worker processes prompt
  ↓
Generates response text + mood
```

### 4. Response Streaming
```
StreamManager receives response
  ↓
Text streamed word-by-word at 5 words/second
  ↓
Each word emitted via 'stream_chunk' event
  ↓
Frontend displays in real-time
  ↓
Sentences buffered for TTS (wait for punctuation)
```

### 5. Text-to-Speech (TTS)
```
Complete sentence sent to VoiceManager
  ↓
Queued in tts_queue
  ↓
TTS worker thread picks up text
  ↓
on_speak_start() callback:
  - Emits 'tts_speaking' {speaking: true}
  - Sets animation_state.speaking = True
  - Frontend adds 'speaking' class to canvas
  ↓
pyttsx3 speaks the text (185 macOS voices available)
  ↓
on_speak_end() callback:
  - Emits 'tts_speaking' {speaking: false}
  - Sets animation_state.speaking = False
  - Frontend removes 'speaking' class
```

### 6. Mouth Animation
```
Animation loop runs at ~30 FPS
  ↓
Checks animation_state.speaking
  ↓
If speaking = True:
  - Toggle mouth_open every 0.15s
  - Render talking sprite (mouth open/closed)
  ↓
If speaking = False:
  - Render idle sprite (closed mouth)
  ↓
Canvas updated with current sprite
```

## Data Flow Diagram

```
┌──────────────┐
│   User       │ Speaks into microphone
│  (Browser)   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Frontend (JavaScript)                               │
│  - voice.js: Handles "Start Speaking" button         │
│  - socket.js: Emits 'start_speech_recognition'       │
│  - Displays transcription overlay                    │
└──────┬───────────────────────────────────────────────┘
       │ SocketIO
       ▼
┌──────────────────────────────────────────────────────┐
│  Backend (Python - app.py)                           │
│  - Receives SocketIO event                           │
│  - Creates OfflineSpeechRecognizer (Vosk)            │
│  - Captures audio via PyAudio                        │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Vosk Speech Recognition                             │
│  - Model: vosk-model-small-en-us-0.15 (50MB)         │
│  - Processes audio chunks                            │
│  - Detects speech patterns                           │
└──────┬───────────────────────────────────────────────┘
       │
       ├──> on_speech_partial("hello...") → Frontend
       │
       └──> on_speech_final("hello there") → ChatManager
              │
              ▼
       ┌──────────────────────────────────────────────┐
       │  ChatManager                                 │
       │  - Stores message with is_voice=True         │
       │  - Formats conversation history              │
       └──────┬───────────────────────────────────────┘
              │
              ▼
       ┌──────────────────────────────────────────────┐
       │  MoondreamWorker                             │
       │  - Builds prompt with context                │
       │  - Calls Moondream VL model                  │
       │  - Generates response + mood                 │
       └──────┬───────────────────────────────────────┘
              │
              ▼
       ┌──────────────────────────────────────────────┐
       │  StreamManager                               │
       │  - Streams response word-by-word             │
       │  - Buffers sentences for TTS                 │
       └──────┬───────────────────────────────────────┘
              │
              ├──> 'stream_chunk' → Frontend (real-time display)
              │
              └──> Complete sentence → VoiceManager
                     │
                     ▼
              ┌──────────────────────────────────────────┐
              │  VoiceManager (TTS)                      │
              │  - Queue: tts_queue                      │
              │  - Worker thread processes speech        │
              │  - on_speak_start() → SocketIO emit      │
              │  - pyttsx3.say() → macOS TTS             │
              │  - on_speak_end() → SocketIO emit        │
              └──────┬───────────────────────────────────┘
                     │
                     ├──> 'tts_speaking' {true} → Frontend
                     │       │
                     │       ▼
                     │    ┌────────────────────────────┐
                     │    │  Canvas Animation          │
                     │    │  - Sets speaking=true      │
                     │    │  - Toggles mouth sprite    │
                     │    │  - 0.15s intervals         │
                     │    └────────────────────────────┘
                     │
                     └──> 'tts_speaking' {false} → Frontend
                             │
                             ▼
                          ┌────────────────────────────┐
                          │  Canvas Animation          │
                          │  - Sets speaking=false     │
                          │  - Returns to idle sprite  │
                          └────────────────────────────┘
```

## Key Components

### Audio Capture (PyAudio + Vosk)
- **Sample Rate**: 16000 Hz (16kHz)
- **Channels**: 1 (mono)
- **Chunk Size**: 8000 samples (~0.5 seconds)
- **Device**: MacBook Pro Microphone (device 0)
- **Audio Level Monitoring**: NumPy calculates `mean(abs(audio_data))`
  - Normal speaking: 500-2000
  - Quiet room: 50-200
  - Silent (permissions issue): 0

### Timeout Mechanism
- **Silent Duration Tracking**: Resets when audio level > 100
- **Auto-Stop**: After 10 seconds of complete silence (level < 100)
- **Warning Messages**: Show at 2s, 4.5s, 7s intervals
- **Permission Instructions**: Displayed in console + browser alert

### Conversation Context
The AI uses the last 5 messages to maintain conversation flow:
```python
Recent conversation:
  User: "what are you doing"
  You: "Greetings, visitor."
  User: "hello from the frame would you know from within the frame"
  You: "What are you looking for?"
  User: "why can i see the chat as scroll"

User says: "what are we doing"
IMPORTANT: Respond DIRECTLY to what they said.
DO NOT just say 'Hello from within the frame' - actually engage!
```

### Response Variety
Prompts now include:
- "VARY your responses - don't repeat the same phrases!"
- Event-specific hints for natural conversation
- User message highlighting: "User says: ..."
- Direct engagement instruction

## Fixed Issues

### ✅ Chat Scrolling
**Problem**: Chat messages not scrollable  
**Solution**: Added `overflow-y: auto`, `max-height: calc(100vh - 200px)`

### ✅ Mouth Animation Not Syncing
**Problem**: Portrait mouth not moving while speaking  
**Solution**: 
- Added TTS callbacks (`on_speak_start`, `on_speak_end`)
- Emit `tts_speaking` events via SocketIO
- Update `animation_state.speaking` attribute
- Frontend adds/removes 'speaking' class to canvas

### ✅ Repetitive Responses
**Problem**: AI kept saying "Hello from within the frame"  
**Solution**:
- Enhanced prompts with "VARY your responses"
- Added "Respond DIRECTLY to what they said"
- Improved event hints for VOICE_MESSAGE
- Better conversation history formatting

### ✅ Microphone Permissions
**Problem**: Audio level always 0  
**Solution**:
- Added audio level monitoring with NumPy
- Auto-stop after 10s of silence
- Clear diagnostic messages
- macOS permissions instructions

## Configuration

### Voice Settings (models/models.py)
```python
@dataclass
class VoiceSettings:
    tts_voice: Optional[str] = None  # Use default macOS voice
    tts_rate: int = 150  # Words per minute
    tts_volume: float = 0.9  # 0.0 to 1.0
    punctuation_buffer: bool = True  # Wait for sentence endings
```

### Stream Settings (app.py)
```python
stream_manager = StreamManager(words_per_second=5.0)
```

### Animation Timing (models/models.py)
```python
mouth_toggle_interval = 0.15  # Toggle mouth every 0.15 seconds
speaking_duration = len(text.split()) / 3  # ~3 words per second
```

## Testing Checklist

- [ ] Microphone permissions granted (System Preferences → Microphone)
- [ ] Click "Start Speaking" button
- [ ] Audio level shows > 500 when speaking
- [ ] Interim transcription appears (green overlay)
- [ ] Final transcription sent to chat
- [ ] Portrait responds with varied answers
- [ ] TTS speaks the response
- [ ] Mouth animates while speaking
- [ ] Chat messages are scrollable
- [ ] Auto-stop works after 10s of silence

## Troubleshooting

### No Audio Detected (level: 0)
1. System Preferences → Security & Privacy → Privacy → Microphone
2. Enable Terminal/Python
3. Refresh browser page
4. Click "Start Speaking" again

### Repetitive Responses
- Check recent conversation history in console
- Verify prompts include "VARY your responses"
- Ensure user message is being passed correctly

### Mouth Not Animating
- Check browser console for `tts_speaking` events
- Verify `speaking` class is added to canvas
- Check animation_state.speaking in backend logs

### Chat Not Scrolling
- Inspect `.chat-messages` CSS
- Should have `overflow-y: auto` and `max-height`
- Try scrolling manually to test

## Files Modified

1. **vision_experiment/core/voice_manager.py**: Added TTS callbacks
2. **vision_experiment/app.py**: Wired callbacks to SocketIO + animation state
3. **vision_experiment/static/js/socket.js**: Handle `tts_speaking` events
4. **vision_experiment/static/css/style.css**: Fixed chat scrolling
5. **vision_experiment/models/models.py**: Improved prompts for variety
6. **vision_experiment/core/offline_speech_recognizer.py**: Audio diagnostics
