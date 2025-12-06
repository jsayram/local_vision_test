import cv2
import requests
import base64
import time

def main():
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
    print("The camera feed will show in the window.")
    print("Descriptions will appear in the terminal every 5 seconds.")
    print("Press 'q' to quit")
    
    last_capture = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        cv2.imshow('Camera Feed', frame)
        
        current_time = time.time()
        if current_time - last_capture > 5:  # Capture every 5 seconds
            print("Capturing and describing...")
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to Ollama
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "moondream",
                        "prompt": "Describe what you see in this image in detail.",
                        "images": [img_base64],
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    description = result.get("response", "No response")
                    print(f"[{time.strftime('%H:%M:%S')}] Description: {description}")
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
            
            last_capture = current_time
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()