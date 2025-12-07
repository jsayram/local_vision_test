#!/usr/bin/env python3
"""
LIVING PORTRAIT - Integrated Flask Web Application
Combines the original vision detection system with the new Living Portrait features
Side-by-side display: Camera Feed + Portrait Animation
Now with voice interaction, face recognition, and chat!
"""

import cv2
import requests
import base64
import time
import threading
import os
import platform
import signal
import atexit
import numpy as np
import uuid
import re
from difflib import SequenceMatcher
from queue import Queue, Empty
from collections import defaultdict, deque
from typing import Optional, List
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Import detection manager (original system)
from detectors.detection_manager import DetectionManager

# Import portrait system modules
from core import config
from models.models import (
    Event, EventType, PersonState, AnimationState, MoondreamContext,
    VoiceSettings, ChatMessage, FaceRecognitionResult
)
from core import storage
from core import moondream_client
from core import detector
from core import animator
from detectors.yolo_detector import YOLODetector

# Import new feature modules
from core.face_recognition_manager import FaceRecognitionManager
from core.voice_manager import create_voice_manager
from core.stream_manager import StreamManager
from core.chat_manager import ChatManager, CommandParser
from core.offline_speech_recognizer import create_offline_speech_recognizer
from core.llm_client import create_llm_client

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))
app.config['SECRET_KEY'] = 'living-portrait-secret-key-2025'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================================
# GLOBAL STATE - Original System
# ============================================================================
current_description = ""

# Camera and processing control
cap = None
processing = False
last_capture = 0
interval = 1.0
paused = False

# Detection control
detection_model = "yolov8"
detection_mode = "all_detection"
processing_fps = "1 FPS"

# Global stats
current_processing_time = 0.0
current_frame_size = (0, 0)
last_detection_time_global = 0
cached_detections_global = []

# Server control
server_running = True

# Initialize detection manager
detection_manager = DetectionManager()

# ============================================================================
# GLOBAL STATE - Portrait System
# ============================================================================

# YOLO detector for portrait
yolo = None
person_state = PersonState()
person_state_lock = threading.Lock()

# Animation state
animation_state = AnimationState()
animation_state_lock = threading.Lock()

# Moondream job queue
moondream_queue = Queue(maxsize=10)

# Latest portrait frame for streaming
latest_portrait_frame = None
portrait_frame_lock = threading.Lock()

# Portrait subtitle (what the portrait is saying)
portrait_subtitle = "Waiting for someone to appear..."
portrait_subtitle_lock = threading.Lock()

# Recent TTS guard (prevents echo from being interpreted as user speech)
tts_guard_lock = threading.Lock()
tts_guard_active = False
tts_guard_window = 12.0  # seconds to keep guard active after speech ends
recent_tts_phrases = deque(maxlen=5)  # store recent TTS phrases with timestamps

# ============================================================================
# NEW FEATURE MANAGERS
# ============================================================================

# Face recognition
face_recognition_manager = FaceRecognitionManager()

# Voice settings
voice_settings = VoiceSettings()
voice_manager = create_voice_manager(voice_settings, use_stub=False)  # Use real TTS

# LLM for conversation (Llama 3.2 3B for desktop, 1B for Raspberry Pi)
llm_client = create_llm_client(model_name="llama3.2:latest")  # Use the installed version
print(f"[LLM] Client initialized: available={llm_client.available}")

# Set up voice manager callbacks for speaking events
def on_tts_start(text):
    """Called when TTS starts speaking"""
    global tts_guard_active
    socketio.emit('tts_speaking', {'speaking': True, 'text': text[:50]})
    # Update animation state with speaking duration estimate
    # Estimate ~50ms per character for TTS speaking rate
    estimated_duration = len(text) * 0.05 + 0.5  # Add 0.5s buffer
    now = time.time()
    with animation_state_lock:
        animation_state.speaking = True
        animation_state.speaking_until = now + estimated_duration
        animation_state.last_mouth_toggle = now  # Reset mouth toggle timer
    with tts_guard_lock:
        tts_guard_active = True
        recent_tts_phrases.append((text.strip(), now))
    print(f"[TTS] Started speaking ({len(text)} chars, ~{estimated_duration:.1f}s)")
    # Echo cancellation: pause speech recognition while TTS is playing
    if offline_speech:
        offline_speech.set_tts_playing(True)

def on_tts_end():
    """Called when TTS stops speaking"""
    global tts_guard_active
    socketio.emit('tts_speaking', {'speaking': False})
    # Update animation state
    with animation_state_lock:
        animation_state.speaking = False
        animation_state.mouth_open = False  # Close mouth when done
    with tts_guard_lock:
        tts_guard_active = False
        if recent_tts_phrases:
            phrase, _ = recent_tts_phrases.pop()
            recent_tts_phrases.append((phrase, time.time()))
    print(f"[TTS] Stopped speaking")
    # Echo cancellation: resume speech recognition when TTS stops
    if offline_speech:
        threading.Timer(0.75, lambda: offline_speech.set_tts_playing(False)).start()

if voice_manager:
    voice_manager.on_speak_start = on_tts_start
    voice_manager.on_speak_end = on_tts_end

# Offline speech recognition
offline_speech = None  # Initialize on demand

# Chat manager
chat_manager = ChatManager()

# Stream manager for word-by-word responses
stream_manager = StreamManager(words_per_second=5.0)

# Current person being interacted with
current_person_id = None
pending_face_confirmation = None  # Stores FaceRecognitionResult awaiting confirmation

def _normalize_text_for_compare(text: str) -> str:
    """Normalize text for fuzzy comparison (lowercase, alphanumeric only)."""
    return re.sub(r'[^a-z0-9 ]+', '', text.lower()).strip()


def _should_ignore_transcription(text: str) -> bool:
    """Return True if the transcription matches the portrait's recent TTS output."""
    if not text:
        return False
    normalized_transcript = _normalize_text_for_compare(text)
    if not normalized_transcript:
        return False
    with tts_guard_lock:
        guard_active = tts_guard_active
        recent_phrases = list(recent_tts_phrases)
    if not recent_phrases:
        return False
    now = time.time()
    for phrase_text, phrase_time in recent_phrases:
        if not phrase_text:
            continue
        if not guard_active and (now - phrase_time) > tts_guard_window:
            continue
        normalized_tts = _normalize_text_for_compare(phrase_text)
        if not normalized_tts:
            continue
        if normalized_transcript in normalized_tts or normalized_tts in normalized_transcript:
            return True
        tokens_transcript = set(normalized_transcript.split())
        tokens_tts = set(normalized_tts.split())
        if tokens_transcript and tokens_tts:
            overlap_ratio = len(tokens_transcript & tokens_tts) / max(1, len(tokens_transcript))
            if overlap_ratio >= 0.7:
                return True
        similarity = SequenceMatcher(None, normalized_transcript, normalized_tts).ratio()
        if similarity >= 0.65:
            return True
    return False

# Speech recognition callbacks
def on_speech_partial(text, result):
    """Callback for interim speech results"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    if _should_ignore_transcription(text):
        return
    # Emit to frontend for live display
    socketio.emit('speech_partial', {
        'text': text,
        'timestamp': timestamp
    })

def on_speech_final(text, result):
    """Callback for final speech results"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    if _should_ignore_transcription(text):
        print(f"[Speech {timestamp}] Ignored (matched portrait speech): \"{text}\"")
        return
    print(f"[Speech {timestamp}] ✓ FINAL: \"{text}\"")
    
    # Get confidence if available
    confidence = result.get('result', [{}])[0].get('conf', 1.0) if result.get('result') else 1.0
    
    # Emit to frontend
    socketio.emit('speech_final', {
        'text': text,
        'timestamp': timestamp,
        'confidence': confidence
    })
    
    # Auto-send to chat (simulating voice message)
    socketio.emit('auto_chat_message', {
        'text': text,
        'is_voice': True
    })

def on_speech_error(error_message):
    """Callback for speech recognition errors"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[Speech {timestamp}] ❌ ERROR: {error_message}")
    
    # Emit error to frontend
    socketio.emit('speech_status', {
        'listening': False,
        'status': error_message,
        'error': True,
        'requires_refresh': True
    })

# ============================================================================
# CAMERA CLEANUP
# ============================================================================

def cleanup_all():
    """Comprehensive cleanup of all system components"""
    global cap, server_running, offline_speech
    
    print("\n[Cleanup] Shutting down Living Portrait System...")
    server_running = False
    
    # Stop speech recognition
    if offline_speech:
        try:
            offline_speech.stop()
            print("[Cleanup] Speech recognition stopped")
        except Exception as e:
            print(f"[Cleanup] Error stopping speech: {e}")
    
    # Stop voice manager
    if voice_manager:
        try:
            voice_manager.shutdown()
            print("[Cleanup] Voice manager stopped")
        except Exception as e:
            print(f"[Cleanup] Error stopping voice: {e}")
    
    # Stop stream manager
    if stream_manager:
        try:
            stream_manager.stop()
            print("[Cleanup] Stream manager stopped")
        except Exception as e:
            print(f"[Cleanup] Error stopping stream: {e}")
    
    # Release camera
    if cap is not None:
        try:
            if cap.isOpened():
                cap.release()
                print("[Cleanup] Camera released")
        except Exception as e:
            print(f"[Cleanup] Error releasing camera: {e}")
        cap = None
    
    print("[Cleanup] Shutdown complete")

def cleanup_camera():
    """Legacy cleanup function for atexit"""
    global cap
    if cap is not None and cap.isOpened():
        cap.release()

atexit.register(cleanup_all)

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure clean shutdown"""
    global server_running
    print(f"\n[Signal] Received signal {signum}, shutting down...")
    server_running = False
    cleanup_all()
    os._exit(0)  # Force exit immediately

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# HELPER FUNCTIONS - Original System
# ============================================================================

def extract_keywords_enhanced(description):
    """Enhanced keyword extraction from AI description"""
    base_keywords = [
        'man', 'woman', 'person', 'human', 'people', 'child', 'adult', 'boy', 'girl',
        'couch', 'sofa', 'chair', 'table', 'desk', 'lamp', 'light', 'vase', 'flowers', 'plant', 'tree',
        'window', 'wall', 'door', 'ceiling', 'floor', 'carpet', 'rug', 'curtains', 'blinds',
        'shirt', 'sweater', 'jacket', 'pants', 'jeans', 'dress', 'skirt', 'hat', 'cap', 'glasses', 'watch',
        'headphones', 'earbuds', 'bed', 'pillow', 'blanket', 'mirror', 'picture', 'frame',
        'book', 'phone', 'cellphone', 'computer', 'laptop', 'screen', 'monitor', 'keyboard', 'mouse',
        'bottle', 'glass', 'cup', 'plate', 'bowl', 'food', 'fruit', 'apple', 'banana', 'vegetable',
        'bag', 'backpack', 'purse', 'shoes', 'sneakers', 'boots', 'hand', 'arm', 'leg', 'foot',
        'potted plant', 'television', 'tv', 'remote', 'clock', 'fan', 'heater', 'air conditioner',
        'car', 'vehicle', 'bicycle', 'motorcycle', 'truck', 'bus', 'train',
        'dog', 'cat', 'animal', 'bird', 'fish', 'horse', 'cow'
    ]
    
    found_keywords = []
    description_lower = description.lower()
    for keyword in base_keywords:
        if keyword in description_lower:
            found_keywords.append(keyword.title())
    
    return list(set(found_keywords))

def correlate_ai_detection(ai_description, detections):
    """Strict correlation: only return objects where both AI and detection agree"""
    if not ai_description or not detections:
        return []
    
    correlated_results = []
    ai_keywords = extract_keywords_enhanced(ai_description)
    
    synonyms = {
        'person': ['man', 'woman', 'human', 'people', 'boy', 'girl', 'child', 'adult'],
        'car': ['vehicle', 'auto', 'automobile'],
        'chair': ['seat', 'stool'],
        'table': ['desk', 'counter'],
        'couch': ['sofa', 'loveseat'],
        'phone': ['cellphone', 'mobile', 'smartphone'],
        'computer': ['laptop', 'pc'],
        'bottle': ['container', 'flask'],
        'cup': ['glass', 'mug'],
        'book': ['novel', 'magazine'],
        'dog': ['puppy', 'canine'],
        'cat': ['kitten', 'feline'],
        'tv': ['television', 'monitor'],
        'potted plant': ['plant', 'flower pot'],
    }
    
    synonym_map = {}
    for canonical, variants in synonyms.items():
        synonym_map[canonical] = canonical
        for variant in variants:
            synonym_map[variant.lower()] = canonical
    
    for det in detections:
        obj_name = det['label'].lower()
        det_conf = det['confidence']
        
        canonical_obj = synonym_map.get(obj_name, obj_name)
        
        ai_agrees = False
        best_match = None
        
        for keyword in ai_keywords:
            keyword_lower = keyword.lower()
            canonical_keyword = synonym_map.get(keyword_lower, keyword_lower)
            
            if canonical_obj == canonical_keyword:
                ai_agrees = True
                best_match = keyword
                break
        
        if ai_agrees:
            display_name = canonical_obj.title()
            combined_conf = max(det_conf, 0.95)
            correlated_results.append((display_name, combined_conf, "AI + Detection Confirmed"))
    
    correlated_results.sort(key=lambda x: x[1], reverse=True)
    return correlated_results

def parse_fps_string(fps_string):
    """Parse FPS string and return interval in seconds"""
    fps_string = fps_string.lower().strip()
    
    if fps_string.endswith(' fps'):
        try:
            fps_value = float(fps_string.replace(' fps', ''))
            return 1.0 / fps_value if fps_value > 0 else 1.0
        except ValueError:
            pass
    
    if fps_string.startswith('1 fp ') and fps_string.endswith(' seconds'):
        try:
            seconds = float(fps_string.replace('1 fp ', '').replace(' seconds', ''))
            return seconds if seconds > 0 else 1.0
        except ValueError:
            pass
    
    try:
        fps_value = float(fps_string)
        return 1.0 / fps_value if fps_value > 0 else 1.0
    except ValueError:
        pass
    
    return 1.0

def process_frame(frame, timestamp, detections=None):
    """Original AI description processing"""
    global current_description, current_processing_time, current_frame_size
    start_time = time.time()
    
    frame = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    base_prompt = "Describe what you see in this image briefly."
    
    if detections:
        detection_summary = ", ".join([f"{det['label']} ({det['confidence']:.2f})" for det in detections[:5]])
        base_prompt += f" I can see these objects: {detection_summary}. Please focus on describing these and any other objects you notice."
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": base_prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "No response")
            processing_time = time.time() - start_time
            current_description = description
            current_processing_time = processing_time
            height, width = frame.shape[:2]
            current_frame_size = (width, height)
            print(f"AI Description: {description}")
    except Exception as e:
        print(f"Error in AI processing: {str(e)}")

# ============================================================================
# PORTRAIT SYSTEM - MOONDREAM WORKER THREAD
# ============================================================================

def moondream_worker():
    """Background worker that processes Moondream vision-language calls"""
    global animation_state, portrait_subtitle, current_person_id
    print("[Moondream Worker] Started")
    
    def on_stream_chunk(chunk):
        """Callback for streaming chunks"""
        socketio.emit('stream_chunk', {
            'text': chunk.text,
            'is_complete': chunk.is_complete,
            'metadata': chunk.metadata
        })
    
    # Start stream manager with callback
    stream_manager.on_chunk = on_stream_chunk
    stream_manager.start()
    
    while server_running:
        try:
            # Expecting tuple: (event, context, face_img)
            job = moondream_queue.get(timeout=1.0)
            
            if isinstance(job, dict):
                # Old format compatibility
                event_type = job["event_type"]
                face_img = job["face_img"]
                person_id = job.get("person_id", "unknown")
                
                # Convert to new format
                context = MoondreamContext(
                    event_type=event_type.name,
                    person_id=person_id if person_id != "unknown" else None
                )
            else:
                # New format: (event, context, face_img)
                event, context, face_img = job
                event_type = event.event_type
                person_id = context.person_id or "unknown"
            
            print(f"[Moondream Worker] Processing {event_type.name} event for person {person_id}")
            
            # If this is a chat/voice message, ensure it's in chat history
            if event_type in [EventType.CHAT_MESSAGE, EventType.VOICE_MESSAGE] and context.user_message:
                # User message should already be added by handle_chat_message, but ensure it's there
                pass
            
            # Show typing indicator
            socketio.emit('typing_indicator', {'is_typing': True})
            
            # STEP 1: Use Moondream for VISION ONLY (what do I see?)
            print(f"[Moondream Worker] Step 1: Getting vision description...")
            result = moondream_client.call_moondream(face_img, context, use_stub=True)
            vision_description = result.text  # This is just image description
            
            # STEP 2: Use LLM for CONVERSATION (how do I respond?)
            print(f"[Moondream Worker] Step 2: Generating conversational response...")
            print(f"[Moondream Worker] Vision saw: {vision_description[:100]}...")
            
            conversation_response = llm_client.generate_response(
                vision_description=vision_description,
                user_message=context.user_message,
                conversation_history=context.recent_interactions if hasattr(context, 'recent_interactions') else [],
                person_name=context.name if hasattr(context, 'name') else None,
                event_type=event_type.name
            )
            
            # Send debug info
            socketio.emit('debug_info', {
                'event_type': event_type.name,
                'person_id': person_id,
                'user_message': context.user_message,
                'vision_description': vision_description,
                'llm_response': conversation_response,
                'recent_interactions_count': len(context.recent_interactions) if hasattr(context, 'recent_interactions') else 0
            })
            
            print(f"[Moondream Worker] Final response: {conversation_response[:100]}...")
            
            # Use the LLM response instead of Moondream's generic caption
            result.text = conversation_response
            
            # Hide typing indicator
            socketio.emit('typing_indicator', {'is_typing': False})
            
            # Stream the response word-by-word
            stream_id = str(uuid.uuid4())
            stream_manager.stream_text(result.text, stream_id, result.mood, {
                'person_id': person_id,
                'event_type': event_type.name
            })
            
            # Update animation state (mood and subtitle only - speaking controlled by TTS callbacks)
            with animation_state_lock:
                animation_state.mood = result.mood
                animation_state.subtitle = result.text
            
            # Update portrait subtitle
            with portrait_subtitle_lock:
                portrait_subtitle = result.text
            
            # Add portrait message to chat manager (for history)
            if person_id != "unknown":
                chat_manager.add_portrait_message(
                    person_id if isinstance(person_id, int) else None,
                    result.text,
                    result.mood,
                    is_voice=False
                )
            
            # Send chat message to frontend BEFORE TTS starts
            # This way the text appears as the portrait begins speaking
            message_id = str(uuid.uuid4())
            socketio.emit('chat_message', {
                'speaker': 'portrait',
                'text': result.text,
                'mood': result.mood,
                'is_voice': False,
                'timestamp': datetime.now().isoformat(),
                'message_id': message_id
            })
            
            # SPEAK using TTS - this triggers on_tts_start/on_tts_end callbacks
            # which control animation_state.speaking and echo cancellation
            if voice_manager and voice_manager.is_available():
                print(f"[Moondream Worker] 🔊 Speaking: {result.text[:50]}...")
                voice_manager.speak(result.text, use_buffer=False)
            else:
                # No TTS available - manually set speaking duration
                with animation_state_lock:
                    animation_state.speaking = True
                    animation_state.speaking_until = time.time() + len(result.text) * 0.05  # ~50ms per char
            
            # Emit stream complete
            socketio.emit('stream_complete', {
                'stream_id': stream_id,
                'full_text': result.text,
                'mood': result.mood
            })
            
            # Save interaction with proper format
            interaction = {
                "person_id": person_id,
                "text": result.text,
                "mood": result.mood,
                "timestamp": datetime.now().isoformat()
            }
            storage.append_interaction(interaction)
            
            # Console log for debugging
            print(f"\n{'='*60}")
            print(f"[Moondream Worker] Response Generated:")
            print(f"  Event Type: {event_type.name}")
            print(f"  Person ID: {person_id}")
            print(f"  Portrait Says: {result.text}")
            print(f"  Portrait Mood: {result.mood}")
            print(f"  Stream ID: {stream_id}")
            if event_type.name == "PERIODIC_UPDATE":
                print(f"  ⏰ Next periodic update in ~{config.PERIODIC_UPDATE_INTERVAL}s")
            print(f"{'='*60}\n")
            
            moondream_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"[Moondream Worker] Error: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('typing_indicator', {'is_typing': False})
            if not moondream_queue.empty():
                moondream_queue.task_done()

# ============================================================================
# PORTRAIT SYSTEM - ANIMATION LOOP THREAD
# ============================================================================

def animation_loop():
    """Background thread that renders the portrait"""
    global animation_state, latest_portrait_frame
    print("[Animation Loop] Started")
    
    last_logged_mood = None
    last_logged_speaking = None
    
    while server_running:
        try:
            now = time.time()
            
            with animation_state_lock:
                # Update speaking animation (toggles mouth_open for talking)
                animation_state.update_speaking(now, 0.15)  # Toggle mouth every 0.15s
                
                # Update subtitle visibility  
                animation_state.update_subtitle(now)
                
                current_state = AnimationState()
                current_state.mood = animation_state.mood
                current_state.speaking = animation_state.speaking
                current_state.subtitle = animation_state.subtitle
                current_state.mouth_open = animation_state.mouth_open
                current_state.speaking_until = animation_state.speaking_until
                current_state.last_mouth_toggle = animation_state.last_mouth_toggle
                
                # Log state changes
                if (current_state.mood != last_logged_mood or 
                    current_state.speaking != last_logged_speaking):
                    print(f"[Animation Loop] State Update: mood={current_state.mood}, "
                          f"speaking={current_state.speaking}, mouth_open={current_state.mouth_open}, "
                          f"subtitle={current_state.subtitle[:50] if current_state.subtitle else 'None'}...")
                    last_logged_mood = current_state.mood
                    last_logged_speaking = current_state.speaking
            
            portrait_frame = animator.render_portrait(current_state)
            
            # Validate portrait frame
            if portrait_frame is None or portrait_frame.size == 0:
                print("[Animation Loop] Warning: Invalid portrait frame generated")
                time.sleep(0.1)
                continue
            
            with portrait_frame_lock:
                latest_portrait_frame = portrait_frame.copy()
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"[Animation Loop] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Original camera feed with detection boxes"""
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/animation_feed')
def animation_feed():
    """New portrait animation feed"""
    return Response(generate_portrait_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_subtitle')
def get_subtitle():
    """Get current portrait subtitle"""
    global portrait_subtitle
    with portrait_subtitle_lock:
        return jsonify({'subtitle': portrait_subtitle})

@app.route('/get_status')
def get_status():
    """Get system status including platform info"""
    return jsonify({
        'platform': config.DEVICE_MODE,
        'paused': paused
    })

@app.route('/toggle_pause')
def toggle_pause():
    global paused
    paused = not paused
    status = "PAUSED" if paused else "RESUMED"
    print(f"Processing {status}")
    return jsonify({'paused': paused})

@app.route('/stop_server')
def stop_server():
    """Stop the server gracefully"""
    def shutdown():
        time.sleep(0.5)
        cleanup_all()
        os._exit(0)
    
    threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({'status': 'Server shutting down...'})

# ============================================================================
# SOCKETIO EVENT HANDLERS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"[SocketIO] Client connected")
    emit('connected', {'status': 'Connected to Living Portrait'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"[SocketIO] Client disconnected")

@socketio.on('send_chat_message')
def handle_chat_message(data):
    """
    Handle incoming chat message (text or voice)
    data: {text: str, person_id: int|None, is_voice: bool}
    """
    global current_person_id, moondream_queue
    
    text = data.get('text', '').strip()
    is_voice = data.get('is_voice', False)
    person_id = data.get('person_id') or current_person_id
    
    if not text:
        return
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[Chat {timestamp}] {'🎤 VOICE' if is_voice else '⌨️  TEXT'}: \"{text}\"")
    print(f"[Chat {timestamp}] Person: {person_id or 'unknown'}")
    
    # Check for special commands
    command = CommandParser.parse_command(text)
    if command:
        handle_command(command, person_id)
        return
    
    # Add user message to chat history
    chat_manager.add_user_message(person_id, text, is_voice)
    
    # Send to client for display
    emit('chat_message', {
        'speaker': 'user',
        'text': text,
        'is_voice': is_voice,
        'timestamp': datetime.now().isoformat(),
        'message_id': str(uuid.uuid4())
    })
    
    # Queue Moondream response with user message
    with person_state_lock:
        current_state = PersonState(
            person_id=person_id,
            name=storage.get_person_name(person_id) if person_id else None
        )
    
    context = MoondreamContext(
        person_id=person_id,
        name=current_state.name,
        recent_interactions=chat_manager.get_recent_messages_formatted(person_id, count=5),
        event_type=EventType.VOICE_MESSAGE.value if is_voice else EventType.CHAT_MESSAGE.value,
        user_message=text
    )
    
    event = Event(
        event_type=EventType.VOICE_MESSAGE if is_voice else EventType.CHAT_MESSAGE,
        timestamp=time.time(),
        person_state=current_state,
        description=f"User {'said' if is_voice else 'typed'}: {text}"
    )
    
    try:
        moondream_queue.put((event, context, None), block=False)
    except:
        print("[Chat] Moondream queue full, skipping")

def handle_command(command, person_id):
    """Handle special chat commands"""
    cmd_type = command['command']
    
    if cmd_type == 'forget_last':
        deleted = chat_manager.delete_last_exchange(person_id)
        socketio.emit('forget_command', {
            'deleted_count': deleted,
            'message': f"Forgot last exchange ({deleted} messages)"
        })
        
        # Portrait acknowledges
        if voice_manager:
            voice_manager.speak("Okay, I've forgotten that.", use_buffer=False)
    
    elif cmd_type == 'clear_conversation':
        chat_manager.clear_conversation(person_id)
        socketio.emit('chat_cleared', {'message': 'Conversation cleared'})
        
        if voice_manager:
            voice_manager.speak("Starting fresh!", use_buffer=False)

@socketio.on('start_speech_recognition')
def handle_start_speech(data=None):
    """Start offline speech recognition"""
    global offline_speech
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[Speech {timestamp}] Starting offline speech recognition...")
    
    if offline_speech is None:
        offline_speech = create_offline_speech_recognizer(
            model_path="models/vosk-en",
            on_partial=on_speech_partial,
            on_final=on_speech_final,
            on_error=on_speech_error,
            use_stub=False
        )
    
    if offline_speech.start():
        print(f"[Speech {timestamp}] ✓ Listening")
        emit('speech_status', {'listening': True, 'status': 'Listening...'})
    else:
        print(f"[Speech {timestamp}] ✗ Failed to start")
        emit('speech_status', {
            'listening': False, 
            'status': 'Failed - Check Vosk model installation',
            'error': True
        })

@socketio.on('get_debug_info')
def handle_get_debug_info(data=None):
    """Send comprehensive debug information to client"""
    try:
        global animation_state, offline_speech, voice_manager, stream_manager, chat_manager
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'animation_state': {
                'mood': animation_state.mood,
                'speaking': animation_state.speaking,
                'subtitle': animation_state.subtitle,
                'mouth_open': animation_state.mouth_open,
                'last_line': animation_state.last_line[:100] if animation_state.last_line else None
            },
            'voice_status': {
                'available': voice_manager.is_available() if voice_manager else False,
                'is_speaking': voice_manager.is_speaking if voice_manager else False,
                'queue_size': voice_manager.tts_queue.qsize() if voice_manager else 0
            },
            'speech_recognition': {
                'active': offline_speech.is_listening if offline_speech else False,
                'available': offline_speech is not None
            },
            'stream_manager': {
                'active': stream_manager.is_active() if hasattr(stream_manager, 'is_active') else False
            }
        }
        
        socketio.emit('debug_info', debug_info)
        
    except Exception as e:
        print(f"[Debug] Error getting debug info: {e}")

@socketio.on('set_vision_detail')
def handle_set_vision_detail(data):
    """Set vision analysis detail level"""
    from core.vision_prompts import set_detail_level, get_vision_config
    
    level = data.get('level', 'standard')
    set_detail_level(level)
    
    config = get_vision_config()
    emit('vision_config_updated', {
        'detail_level': config.detail_level.value,
        'features': config.get_feature_list()
    })
    print(f"[Vision] Detail level changed to: {level}")

@socketio.on('get_vision_config')
def handle_get_vision_config(data=None):
    """Get current vision config"""
    from core.vision_prompts import get_vision_config
    
    config = get_vision_config()
    emit('vision_config', {
        'detail_level': config.detail_level.value,
        'features': config.get_feature_list(),
        'settings': {
            'facial_expressions': config.include_facial_expressions,
            'body_language': config.include_body_language,
            'room_details': config.include_room_details,
            'objects': config.include_objects,
            'clothing': config.include_clothing,
            'emotion': config.include_emotion,
        }
    })

@socketio.on('stop_speech_recognition')
def handle_stop_speech(data=None):
    """Stop offline speech recognition"""
    global offline_speech
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    if offline_speech:
        offline_speech.stop()
        print(f"[Speech {timestamp}] Stopped")
    
    emit('speech_status', {'listening': False, 'status': 'Idle'})

@socketio.on('tts_speak')
def handle_tts_speak(data):
    """Handle TTS request from client"""
    text = data.get('text', '').strip()
    if not text:
        return
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[TTS {timestamp}] 🔊 Speaking: \"{text}\"")
    
    if voice_manager and voice_manager.is_available():
        socketio.emit('tts_started', {'text': text, 'timestamp': timestamp})
        voice_manager.speak(text, use_buffer=False)
        print(f"[TTS {timestamp}] ✓ Queued for speech")
    else:
        print(f"[TTS {timestamp}] ✗ Voice manager unavailable (install pyttsx3)")
        socketio.emit('tts_error', {'error': 'TTS not available'})

@socketio.on('face_confirmation')
def handle_face_confirmation(data):
    """
    Handle user response to face recognition confirmation
    data: {confirmed: bool|None}  (None = timeout)
    """
    global pending_face_confirmation, current_person_id
    
    confirmed = data.get('confirmed')
    
    if pending_face_confirmation is None:
        print("[Face] No pending confirmation")
        return
    
    face_result = pending_face_confirmation
    pending_face_confirmation = None
    
    if confirmed is True:
        # User confirmed - register face
        print(f"[Face] Confirmed: {face_result.name}")
        current_person_id = face_result.person_id
        
        # Update person state
        with person_state_lock:
            person_state.person_id = face_result.person_id
            person_state.name = face_result.name
            person_state.face_recognition_confidence = face_result.confidence
        
        socketio.emit('person_identified', {
            'person_id': face_result.person_id,
            'name': face_result.name
        })
        
        if voice_manager:
            voice_manager.speak(f"Hello {face_result.name}! Good to see you.", use_buffer=False)
    
    elif confirmed is False:
        # User rejected - this is NOT the person
        print(f"[Face] Rejected match for {face_result.name}")
        if voice_manager:
            voice_manager.speak("Sorry about that. Who are you?", use_buffer=False)
    
    else:
        # Timeout - no response
        print("[Face] Confirmation timeout")
        if voice_manager:
            voice_manager.speak("Okay then, don't tell me. I'll keep it to yourself, I'll still talk.", use_buffer=False)

@socketio.on('tts_speak')
def handle_tts_speak(data):
    """
    Speak text via TTS
    data: {text: str}
    """
    text = data.get('text', '').strip()
    if text and voice_manager:
        voice_manager.speak(text, use_buffer=False)
        socketio.emit('tts_started', {})
    return jsonify({'success': True, 'message': 'Server stopped'})

@app.route('/get_ai_description')
def get_ai_description():
    """Get original AI description (for reference)"""
    global current_description
    return jsonify({'description': current_description or 'Waiting for AI analysis...'})

@app.route('/get_terminal_data')
def get_terminal_data():
    """Get detection and correlation data (for reference)"""
    if not hasattr(get_terminal_data, 'detection_objects'):
        get_terminal_data.detection_objects = []
    if not hasattr(get_terminal_data, 'ai_objects'):
        get_terminal_data.ai_objects = []
    if not hasattr(get_terminal_data, 'correlated_objects'):
        get_terminal_data.correlated_objects = []
    if not hasattr(get_terminal_data, 'detection_count'):
        get_terminal_data.detection_count = 0
    if not hasattr(get_terminal_data, 'ai_count'):
        get_terminal_data.ai_count = 0
    if not hasattr(get_terminal_data, 'combined_count'):
        get_terminal_data.combined_count = 0
    if not hasattr(get_terminal_data, 'avg_confidence'):
        get_terminal_data.avg_confidence = '--'

    return jsonify({
        'detection_objects': get_terminal_data.detection_objects,
        'ai_objects': get_terminal_data.ai_objects,
        'correlated_objects': get_terminal_data.correlated_objects,
        'detection_count': get_terminal_data.detection_count,
        'ai_count': get_terminal_data.ai_count,
        'combined_count': get_terminal_data.combined_count,
        'avg_confidence': get_terminal_data.avg_confidence
    })

# ============================================================================
# VIDEO GENERATION
# ============================================================================

def generate_camera_frames():
    """Generate camera feed frames with detection boxes"""
    global cap, processing, last_capture, interval, server_running, processing_fps, paused
    global last_detection_time_global, cached_detections_global, yolo, person_state
    last_frame = None

    try:
        while server_running:
            if not server_running:
                break

            if cap is None or not cap.isOpened():
                time.sleep(0.5)
                continue

            if paused and last_frame is not None:
                ret, buffer = cv2.imencode('.jpg', last_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Camera] Failed to read frame, attempting recovery...")
                # Try to reinitialize camera
                try:
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        print("[Camera] Recovery successful")
                except Exception as e:
                    print(f"[Camera] Recovery failed: {e}")
                time.sleep(0.5)
                continue

            if not server_running:
                break

            interval = parse_fps_string(processing_fps)

            current_time = time.time()
            should_process = (current_time - last_detection_time_global) >= interval

            clean_frame = frame.copy()

            if should_process:
                # Run detection for camera feed
                raw_detections = detection_manager.detect_objects(frame, detection_model, detection_mode)
                
                # Filter and deduplicate detections
                filtered_detections = []
                seen_labels = {}
                
                for det in raw_detections:
                    label = det.get('label', '').lower()
                    conf = det.get('confidence', 0)
                    
                    # Skip false positives (unlikely objects with low confidence)
                    if label in ['toilet', 'bed', 'sink'] and conf < 0.7:
                        continue
                    
                    # For person detections, only keep the highest confidence one
                    if label == 'person':
                        if label not in seen_labels or conf > seen_labels[label]['confidence']:
                            # Remove previous person if exists
                            filtered_detections = [d for d in filtered_detections if d.get('label') != 'person']
                            filtered_detections.append(det)
                            seen_labels[label] = det
                    else:
                        # For other objects, keep all unique ones
                        if label not in seen_labels:
                            filtered_detections.append(det)
                            seen_labels[label] = det
                
                cached_detections_global = filtered_detections
                last_detection_time_global = current_time
                
                # Emit detection update to frontend
                people_count = sum(1 for d in filtered_detections if d.get('label') == 'person')
                object_count = len(filtered_detections)
                socketio.emit('detection_update', {
                    'people_count': people_count,
                    'object_count': object_count,
                    'detections': [f"{d['label']} ({d['confidence']:.0%})" for d in filtered_detections]
                })
                
                # Also run YOLO for portrait system (only detects people)
                try:
                    portrait_detections = detector.run_yolo_on_frame(clean_frame, yolo)
                    
                    # Find best person detection for event system
                    current_person = detector.find_best_person_detection(portrait_detections)
                    
                    # Detect events for portrait
                    with person_state_lock:
                        event = detector.detect_event_from_person_state(
                            person_state, current_person, current_time
                        )
                        
                        # Update person state if we got a new state from event
                        if event is not None:
                            person_state = event.person_state
                    
                    # If event detected, queue Moondream job
                    if event is not None:
                        print(f"[Vision Loop] Event detected: {event.event_type.name}")
                        
                        if current_person:
                            face_img = detector.crop_person_for_moondream(clean_frame, current_person)
                            person_id = event.person_state.person_id if event.person_state.person_id else "unknown"
                            
                            job = {
                                "event_type": event.event_type,
                                "face_img": face_img,
                                "person_id": person_id
                            }
                            
                            try:
                                moondream_queue.put_nowait(job)
                                print(f"[Vision Loop] Queued Moondream job for {event.event_type.name}")
                            except:
                                print(f"[Vision Loop] Moondream queue full, dropping event")
                except Exception as e:
                    print(f"[Vision Loop] Error in portrait detection: {e}")
                    import traceback
                    traceback.print_exc()
            
            detections = cached_detections_global

            detection_objects = [f"{det['label'].title()} ({det['confidence']:.2f})" for det in detections]

            ai_objects = []
            if current_description:
                ai_keywords = extract_keywords_enhanced(current_description)
                ai_objects = list(set([keyword.title() for keyword in ai_keywords]))

            correlated_results = correlate_ai_detection(current_description, detections) if (detections and current_description) else []

            if correlated_results:
                confirmed_labels = set([name.lower() for name, _, _ in correlated_results])
                detections_to_draw = [d for d in detections if d.get('label','').lower() in confirmed_labels]
                if detections_to_draw:
                    detection_manager.draw_detections(frame, detections_to_draw)

            get_terminal_data.detection_objects = detection_objects
            get_terminal_data.ai_objects = ai_objects
            get_terminal_data.correlated_objects = [f"{obj} ({conf:.2f}) [{source}]" for obj, conf, source in correlated_results]

            if correlated_results:
                avg_conf = sum(c[1] for c in correlated_results) / len(correlated_results)
                get_terminal_data.avg_confidence = f"{avg_conf:.0%}"
                get_terminal_data.detection_count = len(detection_objects)
                get_terminal_data.ai_count = len(ai_objects)
                get_terminal_data.combined_count = len(correlated_results)
            else:
                get_terminal_data.avg_confidence = "--"
                get_terminal_data.detection_count = len(detection_objects)
                get_terminal_data.ai_count = len(ai_objects)
                get_terminal_data.combined_count = 0

            last_frame = frame.copy()

            # Validate frame before encoding
            if frame is None or frame.size == 0:
                print("[Camera] Invalid frame, skipping")
                time.sleep(0.1)
                continue
            
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret or buffer is None:
                    print("[Camera] Failed to encode frame")
                    time.sleep(0.1)
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"[Camera] Error encoding frame: {e}")
                time.sleep(0.1)
                continue

            if not server_running:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if should_process and not processing and not paused:
                processing = True
                last_capture = current_time
                frame_to_process = clean_frame.copy()
                detections_for_ai = detections.copy() if detections else []
                def process_and_reset():
                    process_frame(frame_to_process, current_time, detections_for_ai)
                    global processing
                    processing = False
                thread = threading.Thread(target=process_and_reset)
                thread.daemon = True
                thread.start()

            if not server_running:
                break

            time.sleep(0.033)

    except GeneratorExit:
        pass
    except Exception as e:
        import traceback
        print(f"Error in generate_camera_frames: {e}")
        traceback.print_exc()

def generate_portrait_frames():
    """Generate portrait animation frames"""
    global latest_portrait_frame, server_running
    
    try:
        while server_running:
            if latest_portrait_frame is not None:
                try:
                    with portrait_frame_lock:
                        frame = latest_portrait_frame.copy()
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        time.sleep(0.033)
                        continue
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret or buffer is None:
                        time.sleep(0.033)
                        continue
                        
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"[Portrait] Error encoding frame: {e}")
                    time.sleep(0.033)
                    continue
            
            time.sleep(0.033)
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"Error in generate_portrait_frames: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    global cap, yolo, server_running
    
    # Initialize camera
    print("Initializing camera...")
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Using camera index {i} at {width}x{height} resolution")
            break
    else:
        print("Could not open any camera")
        return
    
    # Camera warmup
    print("Warming up camera...")
    for attempt in range(30):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            if test_frame.mean() > 5:
                print(f"Camera ready after {attempt+1} frames")
                break
        time.sleep(0.1)
    else:
        print("Camera failed to produce valid frames")
        cap.release()
        return
    
    # Initialize YOLO for portrait system
    print("Initializing YOLO detector...")
    yolo = YOLODetector()
    
    # Start background threads
    print("Starting Moondream worker thread...")
    moondream_thread = threading.Thread(target=moondream_worker, daemon=True)
    moondream_thread.start()
    
    print("Starting animation loop thread...")
    animation_thread = threading.Thread(target=animation_loop, daemon=True)
    animation_thread.start()
    
    print("\n" + "="*60)
    print("Living Portrait System Started!")
    print(f"Platform: {config.DEVICE_MODE}")
    print("Web server at http://localhost:8000")
    print("\nEvent Detection Settings:")
    print(f"  • NEW_PERSON: Triggers when someone appears")
    print(f"  • POSE_CHANGED: IoU < {config.POSE_CHANGE_THRESHOLD} (movement detection)")
    print(f"  • PERIODIC_UPDATE: Every {config.PERIODIC_UPDATE_INTERVAL}s while person present")
    print(f"  • Min interval between calls: {config.MOONDREAM_MIN_INTERVAL}s")
    print("\nNew Features:")
    print(f"  • Face Recognition: {face_recognition_manager.is_available()}")
    print(f"  • Voice (TTS): {voice_manager.is_available()}")
    print(f"  • Chat: Enabled")
    print(f"  • Streaming: Enabled")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Configure Flask logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    # Start Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
