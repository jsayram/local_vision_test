import cv2
import time
import os
import threading

# TensorFlow import is deferred to avoid startup issues
TF_AVAILABLE = None  # None means not checked yet, True/False after check
_tf_module = None
_hub_module = None
_tf_lock = threading.Lock()  # Global lock for TensorFlow operations

class TensorFlowDetector:
    def __init__(self):
        self.detector = None
        self._initialized = False
        self._detect_lock = threading.Lock()  # Instance lock for detection
        # Don't initialize on construction - do it lazily on first detect()
        print("TensorFlowDetector created (lazy initialization)")

    def _check_tensorflow_available(self):
        """Check if TensorFlow is available"""
        global TF_AVAILABLE, _tf_module, _hub_module
        
        # Return cached result if already checked
        if TF_AVAILABLE is not None:
            return TF_AVAILABLE
            
        try:
            import tensorflow
            import tensorflow_hub
            _tf_module = tensorflow
            _hub_module = tensorflow_hub
            TF_AVAILABLE = True
            print(f"✓ TensorFlow {tensorflow.__version__} available")
            return True
        except ImportError as e:
            TF_AVAILABLE = False
            print(f"✗ TensorFlow not installed: {e}")
            return False

    def initialize_detector(self):
        """Lazy load TensorFlow detector"""
        global _hub_module
        
        if self._initialized:
            return self.detector
            
        self._initialized = True
        
        if not self._check_tensorflow_available():
            return None

        try:
            hardware_info = self.detect_hardware()
            # Use efficient model for Raspberry Pi
            if hardware_info['platform'].startswith('rpi'):
                model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            else:
                model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

            print(f"Loading TensorFlow model from {model_url}...")
            self.detector = _hub_module.load(model_url)
            print("✓ TensorFlow Object Detection loaded")
            return self.detector
        except Exception as e:
            print(f"✗ TensorFlow OD failed: {e}")
            return None

    def is_available(self):
        """Check if TensorFlow detector is available and can be used"""
        return self._check_tensorflow_available()

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
        Detect objects using TensorFlow
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        # Use lock to prevent concurrent TensorFlow operations
        with self._detect_lock:
            # Lazy initialization on first detect call
            if not self._initialized:
                self.initialize_detector()
                
            if not self.detector:
                return []

            try:
                global _tf_module
                if _tf_module is None:
                    import tensorflow
                    _tf_module = tensorflow

                # Convert to tensor
                input_tensor = _tf_module.convert_to_tensor(frame)
                input_tensor = input_tensor[_tf_module.newaxis, ...]

                # Run detection
                detections_output = self.detector(input_tensor)

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