#!/usr/bin/env python3
"""
LIVING PORTRAIT - Integrated Flask Web Application
Combines the original vision detection system with the new Living Portrait features
Side-by-side display: Camera Feed + Portrait Animation
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
from queue import Queue, Empty
from collections import defaultdict, deque
from typing import Optional, List
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, send_from_directory

# Import detection manager (original system)
from detectors.detection_manager import DetectionManager

# Import portrait system modules
from core import config
from models.models import Event, EventType, PersonState, AnimationState, MoondreamContext
from core import storage
from core import moondream_client
from core import detector
from core import animator
from detectors.yolo_detector import YOLODetector

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))

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

# ============================================================================
# CAMERA CLEANUP
# ============================================================================

def cleanup_camera():
    """Ensure camera is properly released on exit"""
    global cap
    if cap is not None:
        if cap.isOpened():
            cap.release()
            print("Camera released during cleanup")
        cap = None

atexit.register(cleanup_camera)

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure clean shutdown"""
    global server_running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    server_running = False
    cleanup_camera()
    exit(0)

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
    global animation_state, portrait_subtitle
    print("[Moondream Worker] Started")
    
    while server_running:
        try:
            job = moondream_queue.get(timeout=1.0)
            
            event_type = job["event_type"]
            face_img = job["face_img"]
            person_id = job.get("person_id", "unknown")
            
            print(f"[Moondream Worker] Processing {event_type.name} event for person {person_id}")
            
            # Load recent interactions if person is known
            recent_interactions = []
            if person_id != "unknown" and person_id in people:
                person_data = people[person_id]
                recent_interactions = person_data.get("interactions", [])[-5:]  # Last 5
            
            context = MoondreamContext(
                event_type=event_type.name,
                person_id=person_id if person_id != "unknown" else None,
                recent_interactions=recent_interactions
            )
            
            # Use the wrapper which handles stub fallback properly
            result = moondream_client.call_moondream(face_img, context, use_stub=True)
            
            # Update animation state
            with animation_state_lock:
                animation_state.mood = result.mood
                animation_state.speaking = True
                animation_state.subtitle = result.text
            
            # Update portrait subtitle
            with portrait_subtitle_lock:
                portrait_subtitle = result.text
            
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
            print(f"  Animation State: mood={animation_state.mood}, speaking={animation_state.speaking}")
            if event_type.name == "PERIODIC_UPDATE":
                print(f"  ⏰ Next periodic update in ~{config.PERIODIC_UPDATE_INTERVAL}s")
            print(f"{'='*60}\n")
            
            moondream_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"[Moondream Worker] Error: {e}")
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
            with animation_state_lock:
                current_state = AnimationState()
                current_state.mood = animation_state.mood
                current_state.speaking = animation_state.speaking
                current_state.subtitle = animation_state.subtitle
                current_state.mouth_open = animation_state.mouth_open
                
                # Log state changes
                if (current_state.mood != last_logged_mood or 
                    current_state.speaking != last_logged_speaking):
                    print(f"[Animation Loop] State Update: mood={current_state.mood}, "
                          f"speaking={current_state.speaking}, "
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
    global server_running
    server_running = False
    time.sleep(0.5)
    cleanup_camera()
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
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Configure Flask logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
