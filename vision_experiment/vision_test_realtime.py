import cv2
import requests
import base64
import time
import threading

current_description = ""

def process_frame(frame, timestamp):
    global current_description
    start_time = time.time()
    
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
            print(f"[{time.strftime('%H:%M:%S')}] Description: {description}")
            print(f"Processing rate: {processing_time:.2f} seconds")
            print("Press 'q' in camera window to exit")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    print("Choose data source:")
    print("1. Live camera (real-time)")
    print("2. Image file (single analysis)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Live camera mode
        run_camera_mode()
    elif choice == "2":
        # Image file mode
        image_path = input("Enter image file path: ").strip()
        run_image_mode(image_path)
    else:
        print("Invalid choice")
        return

def run_camera_mode():
    # Check if Ollama is running and moondream is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if not any("moondream" in name for name in model_names):
                print("Moondream model not found. Please run: ollama pull moondream")
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
            print(f"Using camera index {i}")
            break
    else:
        print("Could not open any camera")
        return
    
    # Try to read a few frames to stabilize
    print("Initializing camera...")
    for _ in range(10):
        ret, _ = cap.read()
        if ret:
            print("Camera ready")
            break
        time.sleep(0.1)
    else:
        print("Camera failed to stabilize")
        cap.release()
        return
    
    print("Vision Test Started!")
    print("Real-time capture every 1 second, processing in background.")
    print("Press 'q' to quit")
    
    processing = False
    last_capture = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Overlay description on frame
        if current_description:
            lines = current_description.split('.')
            y = 30
            for line in lines:
                line = line.strip()
                if line:
                    cv2.putText(frame, line + '.', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 25
                    if y > frame.shape[0] - 10:  # Prevent going off screen
                        break
        
        cv2.imshow('Camera Feed', frame)
        
        current_time = time.time()
        if current_time - last_capture > 1 and not processing:  # Capture every 1 second if not processing
            print("Capturing frame...")
            processing = True
            last_capture = current_time
            # Start processing thread
            def process_and_reset():
                print("Processing started...")
                process_frame(frame.copy(), current_time)
                global processing
                processing = False
                print("Processing finished.")
            thread = threading.Thread(target=process_and_reset)
            thread.daemon = True
            thread.start()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera turned off. Exited.")

def run_image_mode(image_path):
    # Check if Ollama is running and moondream is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if not any("moondream" in name for name in model_names):
                print("Moondream model not found. Please run: ollama pull moondream")
                return
        else:
            print("Ollama not running. Please start Ollama first.")
            return
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    process_frame(frame, time.time())
    print("Image analysis complete.")

if __name__ == "__main__":
    main()