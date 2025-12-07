import cv2
import requests
import base64
import time
import threading
import os
import platform
import signal
import atexit
from flask import Flask, Response, render_template, send_from_directory
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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))

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
paused = False  # Pause/Resume control

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

def cleanup_camera():
    """Ensure camera is properly released on exit"""
    global cap
    if cap is not None:
        if cap.isOpened():
            cap.release()
            print("Camera released during cleanup")
        cap = None

# Register cleanup handlers
atexit.register(cleanup_camera)

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure clean shutdown"""
    global server_running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    server_running = False
    cleanup_camera()
    exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    global cap, processing, last_capture, interval, show_overlay, server_running, processing_fps, paused
    last_frame = None  # Store last frame for pause mode

    try:
        while server_running:
            if not server_running:  # Check again before reading frame
                break

            if cap is None or not cap.isOpened():
                break

            # If paused, keep yielding the last frame
            if paused and last_frame is not None:
                ret, buffer = cv2.imencode('.jpg', last_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

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

            # Only apply detection and overlays if show_overlay is True
            if show_overlay:
                # Apply detection using DetectionManager
                detections = detection_manager.detect_objects(frame, detection_model, detection_mode)
                detected_objects = [f"{det['label']} ({det['confidence']:.2f})" for det in detections]

                # Draw detections on frame
                detection_manager.draw_detections(frame, detections)
            else:
                # No overlay - just use clean frame, no detections displayed
                detections = []
                detected_objects = []

            frame_height, frame_width = frame.shape[:2]

            # Get detection objects with confidence (always compute for web UI)
            detection_objects = [f"{det['label'].title()} ({det['confidence']:.2f})" for det in detections]

            # Get AI objects
            ai_objects = []
            if current_description:
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

            # Store data for web UI endpoint (always available, even when overlay is off)
            get_terminal_data.detection_objects = detection_objects
            get_terminal_data.ai_objects = ai_objects
            get_terminal_data.correlated_objects = [f"{obj} ({conf:.2f}) [{source}]" for obj, conf, source in correlated_results]

            # Calculate stats for web UI
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

            # Store the frame for pause mode
            last_frame = frame.copy()

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            if not server_running:  # Final check before yielding frame
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Check if we should process this frame (skip if paused)
            current_time = time.time()
            if current_time - last_capture > interval and not processing and not paused:
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

    finally:
        # Cleanup when server stops or generator is interrupted
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released due to server stop or interruption")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_overlay')
def toggle_overlay():
    global show_overlay
    show_overlay = not show_overlay
    return {'overlay': show_overlay}

@app.route('/toggle_pause')
def toggle_pause():
    global paused
    paused = not paused
    status = "PAUSED" if paused else "RESUMED"
    print(f"Processing {status}")
    return {'paused': paused}

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
    global detection_model, detection_mode, current_processing_time, current_frame_size, processing_fps, paused
    
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
        'fps': processing_fps,
        'paused': paused
    }

@app.route('/get_ai_description')
def get_ai_description():
    global current_description
    return {'description': current_description or 'Initializing AI vision analysis...'}

@app.route('/get_terminal_data')
def get_terminal_data():
    # Initialize if not exists
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

    # Return structured data for copyable display
    return {
        'detection_objects': get_terminal_data.detection_objects,
        'ai_objects': get_terminal_data.ai_objects,
        'correlated_objects': get_terminal_data.correlated_objects,
        'detection_count': get_terminal_data.detection_count,
        'ai_count': get_terminal_data.ai_count,
        'combined_count': get_terminal_data.combined_count,
        'avg_confidence': get_terminal_data.avg_confidence
    }

@app.route('/get_system_stats')
def get_system_stats():
    import psutil
    
    # CPU usage
    cpu_usage = f"{psutil.cpu_percent(interval=0.1):.1f}%"
    
    # RAM usage
    ram_usage = f"{psutil.virtual_memory().percent:.1f}%"
    
    # GPU usage (simplified - use CPU as placeholder on systems without GPU monitoring)
    gpu_usage = '--'
    try:
        # Try to get GPU info if available (works on some systems)
        import subprocess
        import platform
        if platform.system() == 'Darwin':  # macOS
            # On M1/M2 Macs, GPU is integrated - show as active when processing
            gpu_usage = 'Active' if processing else 'Idle'
        else:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                gpu_usage = f"{result.stdout.strip()}%"
    except:
        gpu_usage = 'N/A'
    
    return {
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'gpu_usage': gpu_usage
    }

@app.route('/stop_server')
def stop_server():
    global server_running
    server_running = False

    # Give some time for cleanup
    time.sleep(0.5)

    # Ensure camera is released
    cleanup_camera()

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