import cv2
import requests
import base64
import time
import threading
import os
import platform
from flask import Flask, Response, render_template_string

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

# Detection mode control
detection_mode = "face_features"  # Options: "face_features", "objects", "both", "none"

# Server control
server_running = True

# Load detection classifiers with error checking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Car cascade may not be available in all OpenCV installations
car_cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
if os.path.exists(car_cascade_path):
    car_cascade = cv2.CascadeClassifier(car_cascade_path)
    car_cascade_loaded = car_cascade is not None and not car_cascade.empty()
else:
    car_cascade = None
    car_cascade_loaded = False

print(f"Cascade loading status:")
print(f"Face cascade loaded: {face_cascade is not None and not face_cascade.empty()}")
print(f"Eye cascade loaded: {eye_cascade is not None and not eye_cascade.empty()}")
print(f"Profile face cascade loaded: {profile_face_cascade is not None and not profile_face_cascade.empty()}")
print(f"Full body cascade loaded: {fullbody_cascade is not None and not fullbody_cascade.empty()}")
print(f"Upper body cascade loaded: {upperbody_cascade is not None and not upperbody_cascade.empty()}")
print(f"Smile cascade loaded: {smile_cascade is not None and not smile_cascade.empty()}")
print(f"Eye glasses cascade loaded: {eye_glasses_cascade is not None and not eye_glasses_cascade.empty()}")
print(f"Car cascade loaded: {car_cascade_loaded}")

# Hardware detection for Raspberry Pi optimization
def detect_hardware():
    """Detect if running on Raspberry Pi and hardware capabilities"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                # Get Pi model
                if 'Raspberry Pi 5' in cpuinfo:
                    return {'platform': 'rpi5', 'cpu_cores': 4, 'ram_gb': 8, 'gpu': True}
                elif 'Raspberry Pi 4' in cpuinfo:
                    return {'platform': 'rpi4', 'cpu_cores': 4, 'ram_gb': 4, 'gpu': False}
                elif 'Raspberry Pi 3' in cpuinfo:
                    return {'platform': 'rpi3', 'cpu_cores': 4, 'ram_gb': 1, 'gpu': False}
                else:
                    return {'platform': 'rpi_other', 'cpu_cores': 1, 'ram_gb': 0.5, 'gpu': False}
    except:
        pass
    return {'platform': 'desktop', 'cpu_cores': 8, 'ram_gb': 16, 'gpu': True}

hardware_info = detect_hardware()
print(f"Hardware detected: {hardware_info}")

# Initialize advanced detectors with Raspberry Pi optimizations
class YOLODetector:
    def __init__(self):
        self.available = False
        self.model = None
        if YOLO_AVAILABLE:
            try:
                # Use nano model for Raspberry Pi, small for desktop
                model_size = 'yolov8n.pt' if hardware_info['platform'].startswith('rpi') else 'yolov8s.pt'
                self.model = YOLO(model_size)
                self.available = True
                print(f"YOLOv8 {model_size} loaded successfully")
            except Exception as e:
                print(f"YOLOv8 failed to load: {e}")

    def detect(self, frame):
        if not self.available or self.model is None:
            return []
        try:
            results = self.model(frame, conf=0.5, verbose=False)
            detections = []
            if len(results) > 0:
                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class': int(cls),
                        'label': results[0].names[int(cls)]
                    })
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

class TensorFlowDetector:
    def __init__(self):
        self.available = False
        self.detector = None
        global TF_AVAILABLE, tf, hub
        if not TF_AVAILABLE:
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                TF_AVAILABLE = True
            except ImportError:
                TF_AVAILABLE = False
                print("TensorFlow import failed")
                return
        
        if TF_AVAILABLE:
            try:
                # Use efficient model for Raspberry Pi
                if hardware_info['platform'].startswith('rpi'):
                    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
                else:
                    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

                self.detector = hub.load(model_url)
                self.available = True
                print("TensorFlow Object Detection loaded successfully")
            except Exception as e:
                print(f"TensorFlow OD failed to load: {e}")
                self.available = False

    def detect(self, frame):
        if not self.available or self.detector is None:
            return []
        try:
            # Convert to tensor
            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run detection
            detections = self.detector(input_tensor)

            results = []
            detection_boxes = detections['detection_boxes'][0].numpy()
            detection_classes = detections['detection_classes'][0].numpy().astype(int)
            detection_scores = detections['detection_scores'][0].numpy()

            # COCO class labels (simplified)
            coco_labels = {
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
                50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
                60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
                65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
                74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
                79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
                85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                90: 'toothbrush'
            }

            for i, score in enumerate(detection_scores):
                if score > 0.5:  # Confidence threshold
                    box = detection_boxes[i]
                    h, w = frame.shape[:2]
                    y1, x1, y2, x2 = box
                    results.append({
                        'bbox': (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                        'confidence': float(score),
                        'class': detection_classes[i],
                        'label': coco_labels.get(detection_classes[i], f'class_{detection_classes[i]}')
                    })
            return results
        except Exception as e:
            print(f"TensorFlow detection error: {e}")
            return []

# Initialize detectors
yolo_detector = YOLODetector()
tf_detector = TensorFlowDetector()

print(f"Advanced detectors status:")
print(f"YOLOv8 available: {yolo_detector.available}")
print(f"TensorFlow OD available: {tf_detector.available}")

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
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

def generate_frames():
    global cap, processing, last_capture, interval, show_overlay, server_running
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
            
        # Keep a clean copy for AI processing (before overlays)
        clean_frame = frame.copy()
        
        # Detect objects based on selected mode
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_objects = []
        
        if detection_mode in ['face_features', 'both', 'all']:
            # Detect frontal faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                detected_objects.append('Face')
            
            # Detect profile faces
            profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in profile_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)  # Orange
                cv2.putText(frame, 'Profile', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                detected_objects.append('Profile Face')
            
            # Detect eyes (within faces for better accuracy)
            all_faces = list(faces) + list(profile_faces)
            for (x, y, w, h) in all_faces:
                if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0] and w > 20 and h > 20:
                    try:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                        if roi_gray.size > 0:
                            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                                cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                detected_objects.append('Eye')
                            
                            # Detect smiles (within faces)
                            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
                            for (sx, sy, sw, sh) in smiles:
                                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 2)  # Yellow
                                cv2.putText(roi_color, 'Smile', (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                detected_objects.append('Smile')
                    except Exception as e:
                        # Skip this face if detection fails
                        continue
            
            # Detect eyes with glasses
            try:
                eyes_glasses = eye_glasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                for (x, y, w, h) in eyes_glasses:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 255), 2)  # Light blue
                    cv2.putText(frame, 'Eye+Glasses', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                    detected_objects.append('Eye with Glasses')
            except Exception as e:
                # Skip eye glasses detection if it fails
                pass
        
        if detection_mode in ['people', 'both', 'all']:
            # Detect full bodies
            bodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120))
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 128), 2)  # Purple
                cv2.putText(frame, 'Body', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
                detected_objects.append('Full Body')
            
            # Detect upper bodies
            upper_bodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 80))
            for (x, y, w, h) in upper_bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Cyan
                cv2.putText(frame, 'Upper Body', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                detected_objects.append('Upper Body')
        
        if detection_mode in ['objects', 'all']:
            # Detect cars (only if cascade is available)
            if car_cascade_loaded and car_cascade is not None:
                try:
                    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80))
                    for (x, y, w, h) in cars:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)  # Magenta
                        cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        detected_objects.append('Car')
                except Exception as e:
                    # Skip car detection if it fails
                    pass
            else:
                # If car cascade not available, show message
                detected_objects.append('Car detection unavailable')
        
        # Advanced detection modes using YOLOv8 and TensorFlow
        if detection_mode == 'yolo':
            if yolo_detector.available:
                detections = yolo_detector.detect(frame)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
                    label = f"{det['label']}:{det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    detected_objects.append(det['label'])
            else:
                detected_objects.append('YOLOv8 not available')
        
        if detection_mode == 'tensorflow':
            if tf_detector.available:
                detections = tf_detector.detect(frame)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange
                    label = f"{det['label']}:{det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    detected_objects.append(det['label'])
            else:
                detected_objects.append('TensorFlow OD not available')
        
        # Create terminal-like window at bottom
        frame_height, frame_width = frame.shape[:2]
        terminal_height = 120
        cv2.rectangle(frame, (0, frame_height - terminal_height), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_height - terminal_height), (frame_width, frame_height), (255, 255, 255), 2)
        
        # Display detected objects in terminal
        y_pos = frame_height - terminal_height + 30
        if detected_objects:
            unique_objects = list(set(detected_objects))  # Remove duplicates
            cv2.putText(frame, 'Detected Objects:', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 25
            for obj in unique_objects:
                cv2.putText(frame, f'- {obj}', (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 20
        else:
            cv2.putText(frame, 'No objects detected', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Overlay AI description if enabled (but simplified)
        if show_overlay and current_description:
            # Parse description for key objects (simple keyword extraction)
            keywords = ['man', 'woman', 'person', 'couch', 'chair', 'table', 'lamp', 'vase', 'flowers', 'plant', 'window', 'wall', 'shirt', 'sweater', 'headphones', 'bed', 'pillow', 'curtains', 'door', 'ceiling', 'floor', 'carpet', 'book', 'phone', 'computer', 'screen', 'keyboard', 'mouse', 'bottle', 'glass', 'cup', 'plate', 'food', 'fruit', 'vegetable', 'hat', 'glasses', 'watch', 'bag', 'shoes', 'jacket', 'pants', 'dress', 'hair', 'hand', 'arm', 'leg', 'foot']
            found_objects = [word for word in keywords if word.lower() in current_description.lower()]
            if found_objects:
                y_pos += 20
                cv2.putText(frame, 'AI Identified:', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_pos += 25
                for obj in list(set(found_objects))[:5]:  # Limit to 5
                    cv2.putText(frame, f'- {obj.title()}', (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 20
        
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
                <p>AI-powered object and scene recognition with selectable detection modes</p>
                <p>Use the dropdown to choose: Face Features, People, General Objects, or combinations</p>
            </div>
            <div class="controls">
                <div style="margin: 10px 0;">
                    <label for="detectionMode" style="font-weight: bold; margin-right: 10px;">Detection Mode:</label>
                    <select id="detectionMode" onchange="changeDetectionMode(this.value)" style="padding: 8px; border-radius: 5px; border: 1px solid #ccc;">
                        <option value="face_features">Face Features Only</option>
                        <option value="people">People Detection Only</option>
                        <option value="objects">General Objects Only</option>
                        <option value="both">Face Features + People</option>
                        <option value="all">All Detection</option>
                        <option value="yolo">YOLOv8 AI Detection</option>
                        <option value="tensorflow">TensorFlow AI Detection</option>
                        <option value="none">No Detection</option>
                    </select>
                </div>
                <button onclick="toggleOverlay()">Toggle AI Overlay</button>
                <button onclick="location.reload()">Refresh</button>
                <button onclick="stopServer()" style="background: #f44336; color: white;">Stop Server</button>
            </div>
        </div>
        
        <script>
            function toggleOverlay() {
                fetch('/toggle_overlay')
                    .then(response => response.json())
                    .then(data => {
                        alert('AI Overlay ' + (data.overlay ? 'enabled' : 'disabled'));
                    });
            }
            
            function changeDetectionMode(mode) {
                fetch('/change_detection_mode', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ mode: mode })
                })
                .then(response => response.json())
                .then(data => {
                    alert('Detection mode changed to: ' + data.mode);
                });
            }
            
            function stopServer() {
                if (confirm('Are you sure you want to stop the server? This will end the vision analysis and close the video feed.')) {
                    window.location.href = '/stop_server';
                }
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

@app.route('/change_detection_mode', methods=['POST'])
def change_detection_mode():
    global detection_mode
    from flask import request
    data = request.get_json()
    new_mode = data.get('mode', 'face_features')
    if new_mode in ['face_features', 'people', 'both', 'none']:
        detection_mode = new_mode
        return {'mode': detection_mode, 'success': True}
    return {'error': 'Invalid mode'}, 400

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
        detection_mode = "face_features"
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