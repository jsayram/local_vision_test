import cv2
import os

class OpenCVDetector:
    def __init__(self):
        self.cascades = {}
        self.load_cascades()

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

    def detect(self, frame, mode="all_detection"):
        """
        Detect objects using OpenCV cascades
        Returns: list of detections with format [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(f"[DEBUG] OpenCV detection running with mode: {mode}")

        # Face detection
        if mode in ['face_features', 'all_detection']:
            if self.cascades.get('face') is not None:
                faces = self.cascades['face'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                print(f"[DEBUG] Face detection: found {len(faces)} faces")
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
                print(f"[DEBUG] Profile face detection: found {len(profile_faces)} profile faces")
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
                print(f"[DEBUG] Full body detection: found {len(bodies)} bodies")
                for (x, y, w, h) in bodies:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Full Body',
                        'confidence': 0.6,
                        'color': (128, 0, 128)
                    })

            if self.cascades.get('upperbody') is not None:
                upper_bodies = self.cascades['upperbody'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 80))
                print(f"[DEBUG] Upper body detection: found {len(upper_bodies)} upper bodies")
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
                print(f"[DEBUG] Car detection: found {len(cars)} cars")
                for (x, y, w, h) in cars:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'label': 'Car',
                        'confidence': 0.7,
                        'color': (255, 0, 255)
                    })

        print(f"[DEBUG] OpenCV detection completed: {len(detections)} total detections")
        return detections