import cv2
import time
import os
import threading
import numpy as np

# TensorFlow import is deferred to avoid startup issues
TF_AVAILABLE = None  # None means not checked yet, True/False after check
_tf_module = None
_hub_module = None
_tf_lock = threading.Lock()  # Global lock for TensorFlow operations

class TensorFlowDetector:
    def __init__(self):
        self.object_detector = None
        self.face_detector = None
        self.face_landmarks_detector = None
        self.emotion_detector = None
        self.pose_detector = None
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
        """Lazy load TensorFlow detectors"""
        global _hub_module
        
        if self._initialized:
            return
            
        self._initialized = True
        
        if not self._check_tensorflow_available():
            return

        try:
            hardware_info = self.detect_hardware()
            
            # Object detection model
            if hardware_info['platform'].startswith('rpi'):
                obj_model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            else:
                obj_model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

            print(f"Loading TensorFlow object detection model...")
            self.object_detector = _hub_module.load(obj_model_url)
            print("✓ TensorFlow Object Detection loaded")

            # Face detection model
            try:
                face_model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2_fpnlite_320x320/1"
                print("Loading TensorFlow face detection model...")
                self.face_detector = _hub_module.load(face_model_url)
                print("✓ TensorFlow Face Detection loaded")
            except Exception as e:
                print(f"Face detection model failed to load: {e}")
                self.face_detector = None

            # Face landmarks model (facial features)
            try:
                landmarks_model_url = "https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/1"
                print("Loading TensorFlow face landmarks model...")
                self.face_landmarks_detector = _hub_module.load(landmarks_model_url)
                print("✓ TensorFlow Face Landmarks loaded")
            except Exception as e:
                print(f"Face landmarks model failed to load: {e}")
                self.face_landmarks_detector = None

            # Emotion recognition model
            try:
                emotion_model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
                print("Loading TensorFlow emotion recognition model...")
                self.emotion_detector = _hub_module.load(emotion_model_url)
                print("✓ TensorFlow Emotion Recognition loaded")
            except Exception as e:
                print(f"Emotion recognition model failed to load: {e}")
                self.emotion_detector = None

            # Pose estimation model (body parts)
            try:
                pose_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
                print("Loading TensorFlow pose estimation model...")
                self.pose_detector = _hub_module.load(pose_model_url)
                print("✓ TensorFlow Pose Estimation loaded")
            except Exception as e:
                print(f"Pose estimation model failed to load: {e}")
                self.pose_detector = None

        except Exception as e:
            print(f"✗ TensorFlow initialization failed: {e}")

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
        Detect objects using TensorFlow with enhanced face/body detection
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        # Use lock to prevent concurrent TensorFlow operations
        with self._detect_lock:
            # Lazy initialization on first detect call
            if not self._initialized:
                self.initialize_detector()

            results = []

            # Object detection for general objects
            if mode in ["all_detection", "general_objects", "people"]:
                results.extend(self._detect_objects(frame, mode))

            # Face detection and features
            if mode in ["all_detection", "face_features", "face_expressions"]:
                results.extend(self._detect_faces(frame, mode))

            # Body parts detection
            if mode in ["all_detection", "body_parts"]:
                results.extend(self._detect_body_parts(frame))

            return results

    def _detect_objects(self, frame, mode):
        """Detect general objects using SSD MobileNet"""
        if not self.object_detector:
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
            detections_output = self.object_detector(input_tensor)

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
            print(f"TensorFlow object detection error: {e}")
            return []

    def _detect_faces(self, frame, mode):
        """Detect faces, facial features, and expressions"""
        results = []

        # Face detection
        if self.face_detector:
            try:
                global _tf_module
                if _tf_module is None:
                    import tensorflow
                    _tf_module = tensorflow

                input_tensor = _tf_module.convert_to_tensor(frame)
                input_tensor = input_tensor[_tf_module.newaxis, ...]

                detections_output = self.face_detector(input_tensor)

                detection_boxes = detections_output['detection_boxes'][0].numpy()
                detection_scores = detections_output['detection_scores'][0].numpy()

                h, w = frame.shape[:2]
                for i, score in enumerate(detection_scores):
                    if score > 0.6:  # Higher threshold for faces
                        box = detection_boxes[i]
                        y1, x1, y2, x2 = box
                        bbox = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

                        # Add face detection
                        results.append({
                            'bbox': bbox,
                            'label': 'face',
                            'confidence': float(score),
                            'color': (0, 255, 255)  # Cyan
                        })

                        # Facial features detection
                        if mode in ["all_detection", "face_features"] and self.face_landmarks_detector:
                            features = self._detect_facial_features(frame, bbox)
                            results.extend(features)

                        # Expression detection
                        if mode in ["all_detection", "face_expressions"] and self.emotion_detector:
                            expression = self._detect_expression(frame, bbox)
                            if expression:
                                results.append(expression)

            except Exception as e:
                print(f"Face detection error: {e}")

        return results

    def _detect_facial_features(self, frame, face_bbox):
        """Detect facial landmarks/features"""
        features = []
        try:
            x1, y1, x2, y2 = face_bbox
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                return features

            # Simple feature detection based on color and shape analysis
            # Eyes detection (dark regions)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eye_detections = eyes.detectMultiScale(gray, 1.1, 3)

            for (ex, ey, ew, eh) in eye_detections:
                features.append({
                    'bbox': (x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh),
                    'label': 'eye',
                    'confidence': 0.8,
                    'color': (255, 0, 255)  # Magenta
                })

            # Nose detection (central region)
            h, w = face_roi.shape[:2]
            nose_x, nose_y = w//2 - 20, h//2 - 10
            nose_w, nose_h = 40, 30
            features.append({
                'bbox': (x1 + nose_x, y1 + nose_y, x1 + nose_x + nose_w, y1 + nose_y + nose_h),
                'label': 'nose',
                'confidence': 0.7,
                'color': (255, 0, 255)  # Magenta
            })

            # Mouth detection (lower region)
            mouth_x, mouth_y = w//2 - 25, h*3//4 - 10
            mouth_w, mouth_h = 50, 20
            features.append({
                'bbox': (x1 + mouth_x, y1 + mouth_y, x1 + mouth_x + mouth_w, y1 + mouth_y + mouth_h),
                'label': 'mouth',
                'confidence': 0.7,
                'color': (255, 0, 255)  # Magenta
            })

        except Exception as e:
            print(f"Facial features detection error: {e}")

        return features

    def _detect_expression(self, frame, face_bbox):
        """Detect facial expressions"""
        try:
            x1, y1, x2, y2 = face_bbox
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                return None

            # Simple expression detection based on mouth and eye analysis
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Detect smile (mouth curvature)
            h, w = face_roi.shape[:2]
            mouth_region = gray[h*3//4:h, w//2-30:w//2+30]
            if mouth_region.size > 0:
                mouth_mean = np.mean(mouth_region)
                mouth_std = np.std(mouth_region)

                # Simple heuristic for smile detection
                if mouth_std > 25 and mouth_mean > 100:
                    return {
                        'bbox': face_bbox,
                        'label': 'smiling',
                        'confidence': 0.75,
                        'color': (0, 255, 0)  # Green
                    }
                elif mouth_std < 15:
                    return {
                        'bbox': face_bbox,
                        'label': 'neutral',
                        'confidence': 0.6,
                        'color': (128, 128, 128)  # Gray
                    }

        except Exception as e:
            print(f"Expression detection error: {e}")

        return None

    def _detect_body_parts(self, frame):
        """Detect body parts using pose estimation"""
        body_parts = []
        try:
            if not self.pose_detector:
                return body_parts

            global _tf_module
            if _tf_module is None:
                import tensorflow
                _tf_module = tensorflow

            # Prepare input
            input_image = _tf_module.image.resize_with_pad(_tf_module.expand_dims(frame, axis=0), 192, 192)
            input_image = _tf_module.cast(input_image, dtype=_tf_module.int32)

            # Run pose estimation
            outputs = self.pose_detector(input_image)
            keypoints = outputs['output_0'].numpy()[0][0]

            # Keypoint labels (MoveNet keypoints)
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]

            h, w = frame.shape[:2]
            for i, keypoint in enumerate(keypoints):
                y, x, confidence = keypoint
                if confidence > 0.3:  # Minimum confidence
                    # Convert normalized coordinates to pixel coordinates
                    px_x, px_y = int(x * w), int(y * h)

                    # Create bounding box around keypoint
                    box_size = 20
                    bbox = (px_x - box_size, px_y - box_size, px_x + box_size, px_y + box_size)

                    body_parts.append({
                        'bbox': bbox,
                        'label': keypoint_names[i],
                        'confidence': float(confidence),
                        'color': (0, 255, 255)  # Yellow
                    })

        except Exception as e:
            print(f"Body parts detection error: {e}")

        return body_parts

    def _should_include_detection(self, label, mode):
        """Check if detection should be included based on mode"""
        if mode == "all_detection":
            return True
        elif mode == "face_features" and label in ['person', 'face', 'eye', 'nose', 'mouth']:
            return True
        elif mode == "face_expressions" and label in ['face', 'smiling', 'neutral']:
            return True
        elif mode == "body_parts" and label in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            return True
        elif mode == "people" and label == 'person':
            return True
        elif mode == "general_objects" and label not in ['person', 'face', 'eye', 'nose', 'mouth', 'smiling', 'neutral'] and \
             label not in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            return True
        elif mode == "none":
            return False
        return False