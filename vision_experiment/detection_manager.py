import cv2
import time
import threading
import os
import platform

# Try to import advanced detection libraries (with fallbacks)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# TensorFlow import is deferred to avoid startup issues
TF_AVAILABLE = False
tf = None
hub = None

class DetectionManager:
    def __init__(self):
        self.yolo_detector = None
        self.tf_detector = None
        self.cascades = {}
        self.load_cascades()
        self.initialize_advanced_detectors()

    def load_cascades(self):
        """Load OpenCV cascade classifiers"""
        cascade_files = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'profile_face': 'haarcascade_profileface.xml',
            'fullbody': 'haarcascade_fullbody.xml',
            'upperbody': 'haarcascade_upperbody.xml',
            'smile': 'haarcascade_smile.xml',
            'eye_glasses': 'haarcascade_eye_tree_eyeglasses.xml',
            'car': 'haarcascade_car.xml'
        }

        for name, filename in cascade_files.items():
            cascade_path = cv2.data.haarcascades + filename
            if os.path.exists(cascade_path):
                self.cascades[name] = cv2.CascadeClassifier(cascade_path)
                print(f"✓ {name} cascade loaded")
            else:
                self.cascades[name] = None
                print(f"✗ {name} cascade not found")

    def initialize_advanced_detectors(self):
        """Initialize YOLO and TensorFlow detectors"""
        # YOLOv8 detector
        if YOLO_AVAILABLE:
            try:
                # Use nano model for Raspberry Pi, small for desktop
                hardware_info = self.detect_hardware()
                model_size = 'yolov8n.pt' if hardware_info['platform'].startswith('rpi') else 'yolov8s.pt'
                self.yolo_detector = YOLO(model_size)
                print(f"✓ YOLOv8 {model_size} loaded")
            except Exception as e:
                print(f"✗ YOLOv8 failed: {e}")
                self.yolo_detector = None

        # TensorFlow detector (lazy loading)
        self.tf_detector = None

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

    def initialize_tensorflow_detector(self):
        """Lazy load TensorFlow detector"""
        global TF_AVAILABLE, tf, hub
        if self.tf_detector is not None:
            return self.tf_detector

        if not TF_AVAILABLE:
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                TF_AVAILABLE = True
            except ImportError:
                print("✗ TensorFlow import failed")
                return None

        try:
            hardware_info = self.detect_hardware()
            # Use efficient model for Raspberry Pi
            if hardware_info['platform'].startswith('rpi'):
                model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            else:
                model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

            self.tf_detector = hub.load(model_url)
            print("✓ TensorFlow Object Detection loaded")
            return self.tf_detector
        except Exception as e:
            print(f"✗ TensorFlow OD failed: {e}")
            return None

    def detect_objects(self, frame, model="yolov8", mode="all_detection"):
        """
        Detect objects in frame based on model and mode
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        detections = []
        
        if model == "yolov8" and self.yolo_detector:
            detections = self._detect_yolo(frame, mode)
        elif model == "tensorflow":
            tf_detector = self.initialize_tensorflow_detector()
            if tf_detector:
                detections = self._detect_tensorflow(frame, mode)
        elif model == "opencv":
            detections = self._detect_opencv(frame, mode)
        
        return detections

    def _detect_yolo(self, frame, mode):
        """YOLOv8 detection"""
        try:
            results = self.yolo_detector(frame, conf=0.5, verbose=False)
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

            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def _detect_tensorflow(self, frame, mode):
        """TensorFlow detection"""
        try:
            global tf
            if tf is None:
                import tensorflow as tf

            # Convert to tensor
            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run detection
            detections_output = self.tf_detector(input_tensor)

            results = []
            detection_boxes = detections_output['detection_boxes'][0].numpy()
            detection_classes = detections_output['detection_classes'][0].numpy().astype(int)
            detection_scores = detections_output['detection_scores'][0].numpy()

            # COCO class labels
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

            h, w = frame.shape[:2]
            for i, score in enumerate(detection_scores):
                if score > 0.5:
                    box = detection_boxes[i]
                    y1, x1, y2, x2 = box

                    class_id = detection_classes[i]
                    label = coco_labels.get(class_id, f'class_{class_id}')

                    if self._should_include_detection(label, mode):
                        results.append({
                            'bbox': (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                            'label': label,
                            'confidence': float(score),
                            'color': (255, 165, 0)  # Orange
                        })

            return results
        except Exception as e:
            print(f"TensorFlow detection error: {e}")
            return []

    def _detect_opencv(self, frame, mode):
        """OpenCV cascade detection"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        if mode in ['face_features', 'all_detection']:
            if self.cascades.get('face') is not None:
                faces = self.cascades['face'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Face',
                        'confidence': 0.8,
                        'color': (255, 0, 0)
                    })

            # Profile faces
            if self.cascades.get('profile_face') is not None:
                profile_faces = self.cascades['profile_face'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in profile_faces:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Profile Face',
                        'confidence': 0.7,
                        'color': (255, 165, 0)
                    })

        # People detection
        if mode in ['people', 'all_detection']:
            if self.cascades.get('fullbody') is not None:
                bodies = self.cascades['fullbody'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120))
                for (x, y, w, h) in bodies:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Full Body',
                        'confidence': 0.6,
                        'color': (128, 0, 128)
                    })

            if self.cascades.get('upperbody') is not None:
                upper_bodies = self.cascades['upperbody'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 80))
                for (x, y, w, h) in upper_bodies:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Upper Body',
                        'confidence': 0.5,
                        'color': (0, 255, 255)
                    })

        # General objects
        if mode in ['general_objects', 'all_detection']:
            if self.cascades.get('car') is not None:
                cars = self.cascades['car'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80))
                for (x, y, w, h) in cars:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Car',
                        'confidence': 0.7,
                        'color': (255, 0, 255)
                    })

        return detections

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

    def draw_detections(self, frame, detections):
        """Draw detection bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            confidence = det['confidence']
            color = det.get('color', (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(frame, f"{label}:{confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame