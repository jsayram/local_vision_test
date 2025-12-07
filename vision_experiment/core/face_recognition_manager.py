"""
Face Recognition Manager
Handles face detection, encoding, matching, and person identification
"""
import os
import pickle
import numpy as np
from typing import Optional, Dict, List, Tuple
import cv2

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition library not installed. Face recognition disabled.")

from models.models import FaceRecognitionResult, Person


class FaceRecognitionManager:
    """Manages face recognition for person identification"""
    
    def __init__(self, encodings_file: str = "memory/face_encodings.pkl",
                 confidence_threshold: float = 0.7,
                 auto_accept_threshold: float = 0.7):
        """
        Initialize face recognition manager
        
        Args:
            encodings_file: Path to pickle file storing face encodings
            confidence_threshold: Minimum confidence for a match (0.0-1.0)
            auto_accept_threshold: Confidence above which no confirmation needed
        """
        self.encodings_file = encodings_file
        self.confidence_threshold = confidence_threshold
        self.auto_accept_threshold = auto_accept_threshold
        
        # Storage: {person_id: [encoding1, encoding2, ...]}
        self.known_encodings: Dict[int, List[np.ndarray]] = {}
        self.person_names: Dict[int, str] = {}  # {person_id: name}
        
        # Load existing encodings
        self._load_encodings()
    
    def is_available(self) -> bool:
        """Check if face recognition is available"""
        return FACE_RECOGNITION_AVAILABLE
    
    def recognize_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> FaceRecognitionResult:
        """
        Attempt to recognize a face in the given frame
        
        Args:
            frame: Full camera frame (BGR format)
            bbox: Bounding box (x1, y1, x2, y2) of the person
            
        Returns:
            FaceRecognitionResult with match info or unknown face
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return FaceRecognitionResult()
        
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Expand bbox slightly for better face detection
            x1, y1, x2, y2 = bbox
            height, width = frame.shape[:2]
            expand = 20
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(width, x2 + expand)
            y2 = min(height, y2 + expand)
            
            # Detect faces in the region
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                # No face detected in frame
                return FaceRecognitionResult()
            
            # Find face closest to bbox center
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            closest_face = self._find_closest_face(face_locations, bbox_center)
            
            if closest_face is None:
                return FaceRecognitionResult()
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, [closest_face])
            
            if not face_encodings:
                return FaceRecognitionResult()
            
            face_encoding = face_encodings[0]
            
            # Convert face_recognition format (top, right, bottom, left) to (x1, y1, x2, y2)
            top, right, bottom, left = closest_face
            face_bbox = (left, top, right, bottom)
            
            # Try to match with known faces
            if self.known_encodings:
                match_result = self._match_face(face_encoding)
                if match_result:
                    person_id, confidence = match_result
                    needs_confirmation = confidence < self.auto_accept_threshold
                    
                    return FaceRecognitionResult(
                        person_id=person_id,
                        name=self.person_names.get(person_id, f"Person {person_id}"),
                        confidence=confidence,
                        needs_confirmation=needs_confirmation,
                        face_encoding=face_encoding.tolist(),
                        bbox=face_bbox
                    )
            
            # Unknown face
            return FaceRecognitionResult(
                person_id=None,
                name=None,
                confidence=0.0,
                needs_confirmation=False,
                face_encoding=face_encoding.tolist(),
                bbox=face_bbox
            )
            
        except Exception as e:
            print(f"[Face Recognition] Error: {e}")
            return FaceRecognitionResult()
    
    def _find_closest_face(self, face_locations: List, target_center: Tuple[int, int]) -> Optional[Tuple]:
        """Find the face location closest to target center point"""
        if not face_locations:
            return None
        
        target_x, target_y = target_center
        closest_face = None
        min_distance = float('inf')
        
        for face_loc in face_locations:
            top, right, bottom, left = face_loc
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            
            distance = np.sqrt((face_center_x - target_x)**2 + (face_center_y - target_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_face = face_loc
        
        return closest_face
    
    def _match_face(self, face_encoding: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Match face encoding against known faces
        
        Returns:
            (person_id, confidence) if match found, None otherwise
        """
        best_match_id = None
        best_confidence = 0.0
        
        for person_id, encodings in self.known_encodings.items():
            # Compare against all encodings for this person
            distances = face_recognition.face_distance(encodings, face_encoding)
            
            if len(distances) > 0:
                # Use minimum distance (best match)
                min_distance = np.min(distances)
                # Convert distance to confidence (0.0-1.0, higher is better)
                confidence = 1.0 - min_distance
                
                if confidence > best_confidence and confidence >= self.confidence_threshold:
                    best_confidence = confidence
                    best_match_id = person_id
        
        if best_match_id is not None:
            return (best_match_id, best_confidence)
        
        return None
    
    def register_face(self, person_id: int, name: str, face_encoding: List[float]) -> bool:
        """
        Register a new face encoding for a person
        
        Args:
            person_id: Person's ID
            name: Person's name
            face_encoding: Face encoding (from FaceRecognitionResult)
            
        Returns:
            True if successful
        """
        try:
            encoding_array = np.array(face_encoding)
            
            if person_id not in self.known_encodings:
                self.known_encodings[person_id] = []
            
            # Add encoding to person's list
            self.known_encodings[person_id].append(encoding_array)
            self.person_names[person_id] = name
            
            # Save to disk
            self._save_encodings()
            
            print(f"[Face Recognition] Registered face for {name} (ID: {person_id})")
            return True
            
        except Exception as e:
            print(f"[Face Recognition] Error registering face: {e}")
            return False
    
    def update_person_name(self, person_id: int, name: str):
        """Update the name for a person"""
        if person_id in self.person_names:
            self.person_names[person_id] = name
            self._save_encodings()
    
    def remove_person(self, person_id: int) -> bool:
        """Remove all face encodings for a person"""
        try:
            if person_id in self.known_encodings:
                del self.known_encodings[person_id]
            if person_id in self.person_names:
                del self.person_names[person_id]
            
            self._save_encodings()
            print(f"[Face Recognition] Removed person ID {person_id}")
            return True
            
        except Exception as e:
            print(f"[Face Recognition] Error removing person: {e}")
            return False
    
    def _load_encodings(self):
        """Load face encodings from disk"""
        if not os.path.exists(self.encodings_file):
            print("[Face Recognition] No existing encodings file found")
            return
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data.get('encodings', {})
                self.person_names = data.get('names', {})
            
            num_people = len(self.known_encodings)
            total_encodings = sum(len(enc) for enc in self.known_encodings.values())
            print(f"[Face Recognition] Loaded {total_encodings} encodings for {num_people} people")
            
        except Exception as e:
            print(f"[Face Recognition] Error loading encodings: {e}")
            self.known_encodings = {}
            self.person_names = {}
    
    def _save_encodings(self):
        """Save face encodings to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            
            data = {
                'encodings': self.known_encodings,
                'names': self.person_names
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"[Face Recognition] Saved encodings to {self.encodings_file}")
            
        except Exception as e:
            print(f"[Face Recognition] Error saving encodings: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about registered faces"""
        return {
            'num_people': len(self.known_encodings),
            'total_encodings': sum(len(enc) for enc in self.known_encodings.values()),
            'people': {pid: self.person_names.get(pid, f"Person {pid}") 
                      for pid in self.known_encodings.keys()}
        }
