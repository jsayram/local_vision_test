import cv2
import requests
import base64
import time
import threading
import os
import platform
from flask import Flask, Response, render_template_string
from detection_manager import DetectionManager

# Try to import advanced detection libraries (with fallbacks)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available - using OpenCV fallback")

# TensorFlow import is deferred to avoid startup issues
TF_AVAILABLE = False
tf = None
hub = None

app = Flask(__name__)

current_description = ""
PROCESS_FPS = 1  # Configurable processing rate (frames per second), up to 60

# Global stats for GUI overlay
current_processing_time = 0.0
current_frame_size = (0, 0)

# Camera and processing control
cap = None
processing = False
last_capture = 0
interval = 1.0 / PROCESS_FPS
show_overlay = True

# Detection control (separate model and mode)
detection_model = "yolov8"  # Options: "yolov8", "tensorflow", "opencv"
detection_mode = "all_detection"  # Options: "all_detection", "face_features", "people", "general_objects", "none"

# Processing FPS control
processing_fps = "1 FPS"  # Options: "1 FPS", "15 FPS", "1 FP 2 seconds", "1 FP 5 seconds", etc. or custom

def correlate_ai_detection(ai_description, detections):
    """
    Correlate AI description with detection results to create combined confidence scores
    Returns: list of tuples (object_name, combined_confidence, source)
    """
    import re
    from difflib import SequenceMatcher
    
    # Extract keywords from AI description
    keywords = ['man', 'woman', 'person', 'couch', 'chair', 'table', 'lamp', 'vase', 'flowers', 'plant', 
                'window', 'wall', 'shirt', 'sweater', 'headphones', 'bed', 'pillow', 'curtains', 'door', 
                'ceiling', 'floor', 'carpet', 'book', 'phone', 'computer', 'screen', 'keyboard', 'mouse', 
                'bottle', 'glass', 'cup', 'plate', 'food', 'fruit', 'vegetable', 'hat', 'glasses', 'watch', 
                'bag', 'shoes', 'jacket', 'pants', 'dress', 'hair', 'hand', 'arm', 'leg', 'foot', 'potted plant']
    
    # Find AI-identified objects
    ai_objects = []
    description_lower = ai_description.lower()
    for keyword in keywords:
        if keyword.lower() in description_lower:
            ai_objects.append(keyword.title())
    
    # Remove duplicates and clean
    ai_objects = list(set(ai_objects))
    
    # Create correlation results
    correlated_results = []
    
    # Process detected objects
    for det in detections:
        obj_name = det['label'].lower()
        det_conf = det['confidence']
        
        # Find best AI match using fuzzy matching
        best_match = None
        best_ratio = 0
        
        for ai_obj in ai_objects:
            ai_obj_lower = ai_obj.lower()
            ratio = SequenceMatcher(None, obj_name, ai_obj_lower).ratio()
            if ratio > best_ratio and ratio > 0.6:  # 60% similarity threshold
                best_match = ai_obj
                best_ratio = ratio
        
        if best_match:
            # Both AI and detection agree - high confidence
            combined_conf = max(det_conf, 0.85)  # Boost to at least 0.85
            correlated_results.append((det['label'].title(), combined_conf, "AI + Detection"))
            # Remove from AI objects to avoid double counting
            ai_objects.remove(best_match)
        else:
            # Only detection found it - slight penalty
            combined_conf = det_conf * 0.9
            correlated_results.append((det['label'].title(), combined_conf, "Detection Only"))
    
    # Add remaining AI-only objects
    for ai_obj in ai_objects:
        correlated_results.append((ai_obj, 0.6, "AI Only"))
    
    # Sort by confidence (highest first)
    correlated_results.sort(key=lambda x: x[1], reverse=True)
    
    return correlated_results

def parse_fps_string(fps_string):
    """Parse FPS string and return interval in seconds"""
    fps_string = fps_string.lower().strip()
    
    # Handle direct FPS values
    if fps_string.endswith(' fps'):
        try:
            fps_value = float(fps_string.replace(' fps', ''))
            return 1.0 / fps_value if fps_value > 0 else 1.0
        except ValueError:
            pass
    
    # Handle "1 FP X seconds" format
    if fps_string.startswith('1 fp ') and fps_string.endswith(' seconds'):
        try:
            seconds = float(fps_string.replace('1 fp ', '').replace(' seconds', ''))
            return seconds if seconds > 0 else 1.0
        except ValueError:
            pass
    
    # Handle "1 frame per X seconds" format
    if fps_string.startswith('1 frame per ') and fps_string.endswith(' seconds'):
        try:
            seconds = float(fps_string.replace('1 frame per ', '').replace(' seconds', ''))
            return seconds if seconds > 0 else 1.0
        except ValueError:
            pass
    
    # Try to parse as direct number (FPS)
    try:
        fps_value = float(fps_string)
        return 1.0 / fps_value if fps_value > 0 else 1.0
    except ValueError:
        pass
    
    # Default fallback
    return 1.0

# Server control
server_running = True

# Initialize detection manager
detection_manager = DetectionManager()

def process_frame(frame, timestamp):
    global current_description, current_processing_time, current_frame_size
    start_time = time.time()
    
    # Reduce resolution for better performance while maintaining usability
    frame = cv2.resize(frame, (640, 480))
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send to Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "Describe what you see in this image briefly.",
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
            print(f"[\033[91mMODEL: {detection_model.upper()}\033[0m] Frame: \033[92m{width}x{height}\033[0m | Processing: \033[92m{processing_time:.2f}s\033[0m | Description: {description}")
            print("Press 'q' in camera window to exit")
        else:
            print(f"Error: \033[92m{response.status_code}\033[0m - {response.text}")
            
    except Exception as e:
        print(f"\033[92mError: {str(e)}\033[0m")

def main():
    # print("Choose data source:")
    # print("1. Live camera (real-time)")
    # print("2. Image file (single analysis)")
    # choice = input("Enter choice (1 or 2): ").strip()
    
    # if choice == "1":
    #     # Live camera mode
    #     run_camera_mode()
    # elif choice == "2":
    #     # Image file mode
    #     image_path = input("Enter image file path: ").strip()
    #     run_image_mode(image_path)
    # else:
    #     print("Invalid choice")
    #     return
    
    # For now, directly run camera mode
    run_camera_mode()

def run_camera_mode():
    global cap, processing, last_capture, interval, show_overlay
    # Check if Ollama is running and moondream is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if not any("moondream" in name for name in model_names):
                print("Moondream model not found. Please run: \033[92mollama pull moondream\033[0m")
                return
        else:
            print("Ollama not running. Please start Ollama first.")
            return
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return
    
    # Open camera
    cap = None
    for i in range(3):  # Try camera indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Set camera properties for Raspberry Pi camera resolution (Camera Module v2: 3280x2464)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2464)
            cap.set(cv2.CAP_PROP_FPS, 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Using camera index {i} at \033[92m{width}x{height}\033[0m resolution")
            break
    else:
        print("Could not open any camera")
        return
    
    # Try to read more frames to stabilize - camera needs warmup time
    print("Initializing camera (waiting for warmup)...")
    for attempt in range(30):  # More warmup attempts
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            # Check if frame is not all black
            if test_frame.mean() > 5:  # Not completely black
                print(f"Camera ready after {attempt+1} frames")
                break
        time.sleep(0.1)
    else:
        print("Camera failed to produce valid frames. Check camera permissions:")
        print("  System Settings > Privacy & Security > Camera")
        print("  Ensure Terminal or your IDE has camera access.")
        cap.release()
        return
    
    print("Vision Test Started!")
    print(f"Real-time capture at \033[92m{PROCESS_FPS} FPS\033[0m, processing in background.")
    print("Starting web server at http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    # Configure Flask logging to reduce HTTP request noise
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)  # Only show warnings and errors, not INFO level requests
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

def generate_frames():
    global cap, processing, last_capture, interval, show_overlay, server_running, processing_fps
    while server_running:
        if not server_running:  # Check again before reading frame
            break
            
        if cap is None or not cap.isOpened():
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        if not server_running:  # Check again before processing
            break
            
        # Update interval based on current FPS setting
        interval = parse_fps_string(processing_fps)
        if not server_running:  # Check again before reading frame
            break
            
        if cap is None or not cap.isOpened():
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        if not server_running:  # Check again before processing
            break
            
        # Keep a clean copy for AI processing (before overlays)
        clean_frame = frame.copy()
        
        # Apply detection using DetectionManager
        detections = detection_manager.detect_objects(frame, detection_model, detection_mode)
        detected_objects = [f"{det['label']} ({det['confidence']:.2f})" for det in detections]

        # Draw detections on frame
        detection_manager.draw_detections(frame, detections)

        # Create terminal-like window at bottom
        frame_height, frame_width = frame.shape[:2]
        terminal_height = 120
        cv2.rectangle(frame, (0, frame_height - terminal_height), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_height - terminal_height), (frame_width, frame_height), (255, 255, 255), 2)

        # Display three-column view: Detection | AI | Detection+AI
        y_pos = frame_height - terminal_height + 25

        # Get detection objects
        detection_objects = [f"{det['label'].title()} ({det['confidence']:.2f})" for det in detections]

        # Get AI objects
        ai_objects = []
        if show_overlay and current_description:
            keywords = ['man', 'woman', 'person', 'couch', 'chair', 'table', 'lamp', 'vase', 'flowers', 'plant',
                       'window', 'wall', 'shirt', 'sweater', 'headphones', 'bed', 'pillow', 'curtains', 'door',
                       'ceiling', 'floor', 'carpet', 'book', 'phone', 'computer', 'screen', 'keyboard', 'mouse',
                       'bottle', 'glass', 'cup', 'plate', 'food', 'fruit', 'vegetable', 'hat', 'glasses', 'watch',
                       'bag', 'shoes', 'jacket', 'pants', 'dress', 'hair', 'hand', 'arm', 'leg', 'foot', 'potted plant']
            description_lower = current_description.lower()
            ai_objects = [word.title() for word in keywords if word.lower() in description_lower]
            ai_objects = list(set(ai_objects))  # Remove duplicates

        # Get correlated results
        correlated_results = correlate_ai_detection(current_description, detections) if (detections or ai_objects) else []

        # Column headers
        col_width = frame_width // 3
        cv2.putText(frame, 'Detection', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'AI', (col_width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, 'Detection+AI', (2 * col_width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw column separators
        cv2.line(frame, (col_width, frame_height - terminal_height), (col_width, frame_height), (255, 255, 255), 1)
        cv2.line(frame, (2 * col_width, frame_height - terminal_height), (2 * col_width, frame_height), (255, 255, 255), 1)

        y_pos += 20

        # Display items in each column (max 6 per column)
        max_items = 6
        for i in range(max_items):
            # Detection column
            if i < len(detection_objects):
                cv2.putText(frame, f'- {detection_objects[i]}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # AI column
            if i < len(ai_objects):
                cv2.putText(frame, f'- {ai_objects[i]}', (col_width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Detection+AI column
            if i < len(correlated_results):
                obj_name, conf, source = correlated_results[i]
                # Color code by source
                if source == "AI + Detection":
                    color = (0, 255, 0)  # Green for both
                elif source == "Detection Only":
                    color = (255, 255, 0)  # Yellow for detection only
                else:  # AI Only
                    color = (255, 165, 0)  # Orange for AI only

                cv2.putText(frame, f'- {obj_name} ({conf:.2f})', (2 * col_width + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            y_pos += 15
        
        # Store data for terminal endpoint
        get_terminal_data.detection_objects = detection_objects
        get_terminal_data.ai_objects = ai_objects
        get_terminal_data.correlated_objects = [f"{obj} ({conf:.2f}) [{source}]" for obj, conf, source in correlated_results]
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        if not server_running:  # Final check before yielding frame
            break
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Check if we should process this frame
        current_time = time.time()
        if current_time - last_capture > interval and not processing:
            processing = True
            last_capture = current_time
            # Start processing thread - use clean_frame without overlays
            frame_to_process = clean_frame.copy()
            def process_and_reset():
                process_frame(frame_to_process, current_time)
                global processing
                processing = False
            thread = threading.Thread(target=process_and_reset)
            thread.daemon = True
            thread.start()
        
        if not server_running:  # Check before sleep
            break
            
        time.sleep(0.1)  # Small delay to prevent overwhelming the stream
    
    # Cleanup when server stops
    if cap is not None and cap.isOpened():
        cap.release()
        print("Camera released due to server stop")

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Vision Analysis</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f0f0f0; 
                height: 100vh;
                box-sizing: border-box;
            }
            .container { 
                max-width: 100%; 
                margin: 0 auto; 
                height: calc(100vh - 40px);
                display: flex;
                flex-direction: column;
            }
            .header {
                flex-shrink: 0;
                margin-bottom: 20px;
            }
            .main-content {
                flex: 1;
                display: flex;
                gap: 20px;
                min-height: 0;
            }
            .controls { 
                background: white; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 15px;
                flex-shrink: 0;
            }
            .control-group { 
                margin-bottom: 12px; 
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .control-group label {
                min-width: 180px;
                font-weight: bold;
                font-size: 14px;
            }
            select { 
                padding: 6px; 
                border: 1px solid #ccc; 
                border-radius: 4px; 
                flex: 1;
                max-width: 200px;
            }
            button { 
                padding: 8px 16px; 
                margin: 2px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                font-size: 14px;
            }
            .btn-primary { background: #007bff; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-success { background: #28a745; color: white; }
            .video-section {
                flex: 1;
                display: flex;
                flex-direction: column;
                min-width: 0;
            }
            .video-container { 
                background: black; 
                flex: 1;
                position: relative;
                border-radius: 8px;
                overflow: hidden;
                min-height: 400px;
            }
            img { 
                width: 100%; 
                height: 100%; 
                object-fit: contain;
                display: block;
            }
            .stats-section {
                width: 350px;
                flex-shrink: 0;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            .status-banner { 
                background: #dc3545; 
                color: white; 
                padding: 12px; 
                border-radius: 8px; 
                font-weight: bold; 
                text-align: center; 
                font-size: 16px;
                flex-shrink: 0;
            }
            .terminal { 
                background: black; 
                color: #00ff00; 
                font-family: 'Courier New', monospace; 
                padding: 12px; 
                border-radius: 8px; 
                font-size: 12px; 
                white-space: pre-wrap; 
                flex: 1;
                overflow-y: auto;
                min-height: 200px;
                max-height: none;
            }
            .performance-stats {
                background: white;
                padding: 12px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                flex-shrink: 0;
            }
            .performance-stats div {
                margin-bottom: 4px;
            }
            @media (max-width: 1200px) {
                .main-content {
                    flex-direction: column;
                }
                .stats-section {
                    width: 100%;
                    order: -1;
                }
                .video-section {
                    order: 1;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0; text-align: center;">🤖 AI Vision Analysis System</h1>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label for="detectionModel">🔧 Detection Model (AI Library):</label>
                    <select id="detectionModel" onchange="changeDetectionModel()">
                        <option value="yolov8">YOLOv8 AI (Most Capable)</option>
                        <option value="tensorflow">TensorFlow AI</option>
                        <option value="opencv">OpenCV (Lightweight)</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="detectionMode">🎯 Detection Mode (What to Detect):</label>
                    <select id="detectionMode" onchange="changeDetectionMode()">
                        <option value="all_detection">All Detection</option>
                        <option value="face_features">Face Features Only</option>
                        <option value="people">People Detection Only</option>
                        <option value="general_objects">General Objects Only</option>
                        <option value="none">No Detection</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="processingFps">⚡ Processing Rate:</label>
                    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                        <select id="processingFpsSelect" onchange="changeProcessingFps()" style="min-width: 140px;">
                            <optgroup label="Fast Processing">
                                <option value="15 FPS">15 FPS (fast)</option>
                                <option value="1 FPS">1 FPS (normal)</option>
                            </optgroup>
                            <optgroup label="Slow Intervals">
                                <option value="0.5 FPS">Every 2 seconds</option>
                                <option value="0.2 FPS">Every 5 seconds</option>
                                <option value="0.1 FPS">Every 10 seconds</option>
                                <option value="0.033 FPS">Every 30 seconds</option>
                                <option value="0.017 FPS">Every 60 seconds</option>
                            </optgroup>
                        </select>
                        <div style="display: flex; gap: 5px; align-items: center;">
                            <span style="font-size: 12px; color: #666;">or</span>
                            <input type="text" id="processingFpsInput" placeholder="e.g., 0.5 FPS, 2 FP 3s"
                                   style="padding: 6px; border: 1px solid #ccc; border-radius: 4px; width: 140px; font-size: 12px;"
                                   onkeypress="handleFpsInputKeyPress(event)"
                                   title="Examples: '0.5 FPS', '2 FP 3 seconds', '1 frame per 10 seconds'">
                            <button onclick="applyCustomFps()" style="padding: 6px 10px; font-size: 11px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Apply</button>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #666; margin-top: 4px;">
                        Current: <span id="currentFpsDisplay">1 FPS</span>
                    </div>
                </div>
                
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn-primary" onclick="toggleOverlay()">Toggle AI Overlay</button>
                    <button class="btn-danger" onclick="stopServer()">Stop Server</button>
                    <button class="btn-success" onclick="refreshPage()">Refresh</button>
                </div>
            </div>
            
            <div class="main-content">
                <div class="stats-section">
                    <div class="status-banner" id="statusBanner">
                        🔴 Active Model: YOLOV8 (Most Capable)
                    </div>
                    
                    <div class="performance-stats" id="performanceStats">
                        <div><strong>Performance Stats:</strong></div>
                        <div>Processing Time: --</div>
                        <div>Frame Size: --</div>
                        <div>Status: Initializing...</div>
                    </div>
                    
                    <div class="terminal" id="terminal">
Detected Objects:
- Loading...

AI Identified:
- Initializing...
                    </div>
                </div>
                
                <div class="video-section">
                    <div class="video-container">
                        <img src="/video_feed" alt="Live Video Feed">
                    </div>
                </div>
            </div>
        </div>

        <script>
            function changeDetectionModel() {
                const model = document.getElementById('detectionModel').value;
                fetch('/change_detection_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: model })
                }).then(() => updateStatus());
            }
            
            function changeDetectionMode() {
                const mode = document.getElementById('detectionMode').value;
                fetch('/change_detection_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode })
                });
            }
            
            function changeProcessingFps() {
                const fps = document.getElementById('processingFpsSelect').value;
                document.getElementById('processingFpsInput').value = ''; // Clear custom input
                applyFpsChange(fps);
            }
            
            function handleFpsInputKeyPress(event) {
                if (event.key === 'Enter') {
                    applyCustomFps();
                }
            }
            
            function applyCustomFps() {
                const customFps = document.getElementById('processingFpsInput').value.trim();
                if (customFps) {
                    applyFpsChange(customFps);
                }
            }
            
            function applyFpsChange(fps) {
                fetch('/change_processing_fps', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fps: fps })
                }).then(() => updateStatus());
            }
            
            function toggleOverlay() {
                fetch('/toggle_overlay');
            }
            
            function stopServer() {
                if (confirm('Are you sure you want to stop the server? This will end the vision analysis.')) {
                    window.location.href = '/stop_server';
                }
            }
            
            function refreshPage() {
                location.reload();
            }
            
            function updateStatus() {
                fetch('/get_status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('statusBanner').textContent = 
                            `🔴 Active Model: ${data.model} (${data.capability})`;
                        
                        // Update performance stats
                        document.getElementById('performanceStats').innerHTML = 
                            `<div><strong>Performance Stats:</strong></div>
                            <div>Processing Time: ${data.processing_time}</div>
                            <div>Frame Size: ${data.frame_size}</div>
                            <div>Processing Rate: ${data.fps}</div>
                            <div>Status: Active</div>`;
                        
                        // Update dropdown selections to match current state
                        document.getElementById('detectionModel').value = data.model.toLowerCase();
                        document.getElementById('detectionMode').value = data.mode;
                        
                        // Handle FPS selection/input
                        const fpsSelect = document.getElementById('processingFpsSelect');
                        const fpsInput = document.getElementById('processingFpsInput');
                        const currentFpsDisplay = document.getElementById('currentFpsDisplay');
                        const currentFps = data.fps;
                        
                        // Update current FPS display
                        currentFpsDisplay.textContent = currentFps;
                        
                        // Check if current FPS is in the dropdown options
                        const options = Array.from(fpsSelect.options).map(opt => opt.value);
                        if (options.includes(currentFps)) {
                            fpsSelect.value = currentFps;
                            fpsInput.value = '';
                        } else {
                            // Custom value - put it in the input field
                            fpsSelect.value = options[0]; // Reset dropdown to first option
                            fpsInput.value = currentFps;
                        }
                    })
                    .catch(() => {
                        document.getElementById('performanceStats').innerHTML = 
                            `<div><strong>Performance Stats:</strong></div>
                            <div>Processing Time: --</div>
                            <div>Frame Size: --</div>
                            <div>Processing Rate: --</div>
                            <div>Status: Error</div>`;
                    });
            }
            
            // Auto-refresh terminal and status every 2 seconds
            setInterval(() => {
                fetch('/get_terminal_data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('terminal').textContent = data.terminal;
                    })
                    .catch(() => {
                        document.getElementById('terminal').textContent = 'Error loading terminal data...';
                    });
                updateStatus();
            }, 2000);
            
            // Initial status update
            updateStatus();
            
            // Make responsive adjustments
            window.addEventListener('resize', function() {
                // Force layout recalculation if needed
                const container = document.querySelector('.container');
                container.style.height = 'calc(100vh - 40px)';
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_overlay')
def toggle_overlay():
    global show_overlay
    show_overlay = not show_overlay
    return {'overlay': show_overlay}

@app.route('/change_detection_mode', methods=['POST'])
def change_detection_mode():
    global detection_mode
    from flask import request
    data = request.get_json()
    new_mode = data.get('mode', 'all_detection')
    if new_mode in ['all_detection', 'face_features', 'people', 'general_objects', 'none']:
        detection_mode = new_mode
        print(f"Detection mode changed to: {detection_mode}")
        return {'mode': detection_mode, 'success': True}
    return {'error': 'Invalid mode'}, 400

@app.route('/change_processing_fps', methods=['POST'])
def change_processing_fps():
    global processing_fps, interval
    from flask import request
    data = request.get_json()
    new_fps = data.get('fps', '1 FPS')
    # Validate by trying to parse
    try:
        test_interval = parse_fps_string(new_fps)
        if test_interval > 0:
            processing_fps = new_fps
            interval = test_interval
            print(f"Processing FPS changed to: {processing_fps} (interval: {interval:.2f}s)")
            return {'fps': processing_fps, 'interval': interval, 'success': True}
    except:
        pass
    return {'error': 'Invalid FPS format'}, 400

@app.route('/change_detection_model', methods=['POST'])
def change_detection_model():
    global detection_model
    from flask import request
    data = request.get_json()
    new_model = data.get('model', 'yolov8')
    if new_model in ['yolov8', 'tensorflow', 'opencv']:
        detection_model = new_model
        print(f"Detection model changed to: {detection_model}")
        return {'model': detection_model, 'success': True}
    return {'error': 'Invalid model'}, 400

@app.route('/get_status')
def get_status():
    global detection_model, detection_mode, current_processing_time, current_frame_size, processing_fps
    
    # Get model capability description
    model_capabilities = {
        'yolov8': 'Most Capable',
        'tensorflow': 'High Accuracy', 
        'opencv': 'Lightweight'
    }
    
    return {
        'model': detection_model.upper(),
        'capability': model_capabilities.get(detection_model, 'Unknown'),
        'mode': detection_mode,
        'processing_time': f"{current_processing_time:.2f}s",
        'frame_size': f"{current_frame_size[0]}x{current_frame_size[1]}",
        'fps': processing_fps
    }

@app.route('/get_terminal_data')
def get_terminal_data():
    # Initialize if not exists
    if not hasattr(get_terminal_data, 'detection_objects'):
        get_terminal_data.detection_objects = []
    if not hasattr(get_terminal_data, 'ai_objects'):
        get_terminal_data.ai_objects = []
    if not hasattr(get_terminal_data, 'correlated_objects'):
        get_terminal_data.correlated_objects = []

    # Create three-column display
    max_items = 8
    lines = ["Detection".ljust(20) + "AI".ljust(15) + "Detection+AI"]

    for i in range(max_items):
        detection = get_terminal_data.detection_objects[i] if i < len(get_terminal_data.detection_objects) else ""
        ai = get_terminal_data.ai_objects[i] if i < len(get_terminal_data.ai_objects) else ""
        correlated = get_terminal_data.correlated_objects[i] if i < len(get_terminal_data.correlated_objects) else ""

        line = f"{detection[:18]:<20}{ai[:13]:<15}{correlated[:25]}"
        lines.append(line)

    terminal_text = "\n".join(lines)
    return {'terminal': terminal_text}

@app.route('/stop_server')
def stop_server():
    global server_running
    server_running = False
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Server Stopped</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f0f0f0; 
                text-align: center;
            }
            .container { 
                max-width: 600px; 
                margin: 100px auto; 
                background: white; 
                padding: 40px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            h1 { 
                color: #f44336; 
            }
            p {
                color: #666;
                font-size: 18px;
                margin: 20px 0;
            }
            .status {
                color: #f44336;
                font-weight: bold;
                font-size: 20px;
            }
            .restart-btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 18px;
                margin-top: 20px;
            }
            .restart-btn:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛑 Server Stopped</h1>
            <p class="status">The vision analysis server has been stopped.</p>
            <p>The camera feed and AI processing have been terminated.</p>
            <button class="restart-btn" onclick="restartServer()">🔄 Restart Vision System</button>
            <p style="margin-top: 20px; font-size: 16px; color: #888;">Or run the script again from the terminal.</p>
        </div>
        
        <script>
            function restartServer() {
                if (confirm('Are you sure you want to restart the vision system?')) {
                    fetch('/restart_server')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                alert('Vision system restarting...');
                                window.location.href = '/';
                            } else {
                                alert('Failed to restart: ' + data.error);
                            }
                        })
                        .catch(error => {
                            alert('Error restarting server: ' + error);
                        });
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/restart_server')
def restart_server():
    global server_running, cap, processing, last_capture, show_overlay, detection_mode, current_description, current_processing_time, current_frame_size
    
    try:
        # Reset all global variables
        server_running = True
        processing = False
        last_capture = 0
        show_overlay = True
        detection_mode = "all_detection"  # Use default detection mode
        current_description = ""
        current_processing_time = 0.0
        current_frame_size = (0, 0)
        
        # Release existing camera if it exists
        if cap is not None:
            if cap.isOpened():
                cap.release()
            cap = None
        
        # Reinitialize camera (similar to run_camera_mode)
        cap = None
        for i in range(3):  # Try camera indices 0, 1, 2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Set camera properties for Raspberry Pi camera resolution (Camera Module v2: 3280x2464)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2464)
                cap.set(cv2.CAP_PROP_FPS, 30)
                break
        
        if cap is None or not cap.isOpened():
            return {'success': False, 'error': 'Could not open camera'}, 500
        
        # Camera warmup
        for attempt in range(10):  # Shorter warmup for restart
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.mean() > 5:
                break
            time.sleep(0.1)
        
        print("Vision system restarted successfully!")
        return {'success': True, 'message': 'Vision system restarted'}
        
    except Exception as e:
        print(f"Error restarting vision system: {e}")
        return {'success': False, 'error': str(e)}, 500

def run_image_mode(image_path):
    # Check if Ollama is running and moondream is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if not any("moondream" in name for name in model_names):
                print("\033[92mMoondream model not found. Please run: ollama pull moondream\033[0m")
                return
        else:
            print("\033[92mOllama not running. Please start Ollama first.\033[0m")
            return
    except Exception as e:
        print(f"\033[92mCannot connect to Ollama: {e}\033[0m")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"\033[92mCould not load image: {image_path}\033[0m")
        return
    
    print(f"\033[92mProcessing image: {image_path}\033[0m")
    process_frame(frame, time.time())
    print("\033[92mImage analysis complete.\033[0m")

if __name__ == "__main__":
    main()