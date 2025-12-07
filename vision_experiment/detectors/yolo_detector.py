import cv2
import time
import os

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available - using fallback")

class YOLODetector:
    def __init__(self):
        self.detector = None
        self.initialize_detector()

    def initialize_detector(self):
        """Initialize YOLOv8 detector"""
        if not YOLO_AVAILABLE:
            print("✗ YOLOv8 not available")
            return

        try:
            # Use nano model for Raspberry Pi, small for desktop
            hardware_info = self.detect_hardware()
            model_size = 'yolov8n.pt' if hardware_info['platform'].startswith('rpi') else 'yolov8s.pt'
            self.detector = YOLO(model_size)
            print(f"✓ YOLOv8 {model_size} loaded")
        except Exception as e:
            print(f"✗ YOLOv8 failed: {e}")
            self.detector = None

    def detect_hardware(self):
        """Detect hardware capabilities"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo:
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

    def detect(self, frame, mode="all_detection"):
        """
        Detect objects using YOLOv8
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        if not self.detector:
            return []

        try:
            results = self.detector(frame, conf=0.5, verbose=False)
            detections = []

            if len(results) > 0:
                for box in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    label = results[0].names[int(cls)]

                    # Filter based on detection mode
                    if self._should_include_detection(label, mode):
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'label': label,
                            'confidence': conf,
                            'color': (0, 255, 255)  # Yellow
                        })

            # Only log when detections found (reduce spam)
            if detections:
                print(f"YOLO detected {len(detections)} objects: {[d['label'] for d in detections]}")

            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def _should_include_detection(self, label, mode):
        """Check if detection should be included based on mode"""
        if mode == "all_detection":
            return True
        elif mode == "face_features" and label in ['person', 'face']:
            return True
        elif mode == "people" and label == 'person':
            return True
        elif mode == "general_objects" and label not in ['person']:
            return True
        elif mode == "none":
            return False
        return False
        return False