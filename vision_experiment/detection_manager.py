import cv2

# Import the separate detector classes
from yolo_detector import YOLODetector
from tensorflow_detector import TensorFlowDetector
from opencv_detector import OpenCVDetector

class DetectionManager:
    def __init__(self):
        self.yolo_detector = None
        self.tf_detector = None
        self.opencv_detector = None
        self.initialize_detectors()

    def initialize_detectors(self):
        """Initialize all detectors"""
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector()

        # Initialize TensorFlow detector (lazy loading)
        self.tf_detector = TensorFlowDetector()

        # Initialize OpenCV detector
        self.opencv_detector = OpenCVDetector()

    def detect_objects(self, frame, model="yolov8", mode="all_detection"):
        """
        Detect objects in frame based on model and mode
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        detections = []

        if model == "yolov8" and self.yolo_detector and self.yolo_detector.detector:
            detections = self.yolo_detector.detect(frame, mode)
        elif model == "tensorflow" and self.tf_detector:
            detections = self.tf_detector.detect(frame, mode)
        elif model == "opencv" and self.opencv_detector:
            detections = self.opencv_detector.detect(frame, mode)

        return detections

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