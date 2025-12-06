import cv2
import requests
import base64
import time
import threading
from flask import Flask, Response, render_template_string

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
            print(f"[{time.strftime('%H:%M:%S')}] Frame: \033[92m{width}x{height}\033[0m | Processing: \033[92m{processing_time:.2f}s\033[0m | Description: {description}")
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
            # Set camera properties for better compatibility
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

def generate_frames():
    global cap, processing, last_capture, interval, show_overlay
    while True:
        if cap is None or not cap.isOpened():
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep a clean copy for AI processing (before overlays)
        clean_frame = frame.copy()
        
        # Overlay description on frame for web display
        if show_overlay and current_description:
            frame_height = frame.shape[0]
            # Start from bottom
            y = frame_height - 100  # Start near bottom
            
            # Description in green bold
            lines = current_description.split('.')
            for line in lines:
                line = line.strip()
                if line:
                    cv2.putText(frame, line + '.', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # Green, thicker
                    y += 40
                    if y > frame_height - 10:
                        break
            
            # Stats below in black
            y += 20
            if current_frame_size[0] > 0:
                cv2.putText(frame, f"Resolution: {current_frame_size[0]}x{current_frame_size[1]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y += 30
            if current_processing_time > 0:
                cv2.putText(frame, f"Processing: {current_processing_time:.2f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y += 30
            cv2.putText(frame, f"FPS: {PROCESS_FPS}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
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
        
        time.sleep(0.1)  # Small delay to prevent overwhelming the stream

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Vision Analysis</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f0f0f0; 
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            h1 { 
                color: #333; 
                text-align: center; 
            }
            .video-container { 
                text-align: center; 
                margin: 20px 0; 
            }
            img { 
                max-width: 100%; 
                border: 2px solid #333; 
                border-radius: 5px; 
            }
            .controls { 
                text-align: center; 
                margin: 20px 0; 
            }
            button { 
                background: #4CAF50; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                margin: 0 10px; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px; 
            }
            button:hover { 
                background: #45a049; 
            }
            .status { 
                text-align: center; 
                margin: 20px 0; 
                font-size: 18px; 
                color: #666; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time Vision Analysis</h1>
            <div class="video-container">
                <img src="/video_feed" alt="Camera Feed">
            </div>
            <div class="status">
                <p>AI-powered object and scene recognition with live descriptions</p>
                <p>Green text shows AI descriptions, black text shows performance stats</p>
            </div>
            <div class="controls">
                <button onclick="toggleOverlay()">Toggle Overlay</button>
                <button onclick="location.reload()">Refresh</button>
            </div>
        </div>
        
        <script>
            function toggleOverlay() {
                fetch('/toggle_overlay')
                    .then(response => response.json())
                    .then(data => {
                        alert('Overlay ' + (data.overlay ? 'enabled' : 'disabled'));
                    });
            }
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