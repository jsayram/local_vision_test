"""
Detection Module - Integrates YOLO detector with event detection logic
"""
import time
from typing import List, Optional
import numpy as np

from models.models import Detection, PersonState, Event, EventType
from detectors.yolo_detector import YOLODetector
from core import config

# ============================================================================
# YOLO WRAPPER
# ============================================================================

def run_yolo_on_frame(frame: np.ndarray, detector: YOLODetector) -> List[Detection]:
    """
    Run YOLO detection on a frame and convert to Detection objects
    
    Args:
        frame: OpenCV image (numpy array)
        detector: Initialized YOLODetector instance
    
    Returns:
        List of Detection objects
    """
    # Run YOLO detection
    yolo_results = detector.detect(frame, mode="people")  # Only detect people
    
    # Convert to Detection objects
    detections = []
    for result in yolo_results:
        x1, y1, x2, y2 = result['bbox']
        
        # Get class ID from label (assume person = 0 in COCO)
        class_id = 0 if result['label'].lower() == 'person' else -1
        
        detection = Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_id=class_id,
            label=result['label'],
            confidence=result['confidence']
        )
        
        # Filter out small detections
        if detection.area >= config.MIN_BBOX_SIZE ** 2:
            detections.append(detection)
    
    return detections


def find_best_person_detection(detections: List[Detection]) -> Optional[Detection]:
    """
    Find the best "person" detection from a list
    Uses largest bbox area as the primary criteria
    
    Args:
        detections: List of Detection objects
    
    Returns:
        Best person detection or None if no people detected
    """
    # Filter to only person detections
    person_detections = [d for d in detections if d.label.lower() == 'person']
    
    if not person_detections:
        return None
    
    # Return detection with largest area (closest/most prominent person)
    return max(person_detections, key=lambda d: d.area)


# ============================================================================
# EVENT DETECTION LOGIC
# ============================================================================

def detect_event_from_person_state(
    previous_state: PersonState, 
    current_detection: Optional[Detection],
    now: float
) -> Optional[Event]:
    """
    Detect if an event has occurred based on state changes
    
    Events:
    - NEW_PERSON: No one before, someone detected now
    - PERSON_LEFT: Someone before, no one now
    - POSE_CHANGED: Significant change in bounding box (movement/pose)
    - PERIODIC_UPDATE: Enough time has passed since last Moondream call
    
    Args:
        previous_state: Previous PersonState
        current_detection: Current Detection or None
        now: Current timestamp
    
    Returns:
        Event object if an event occurred, None otherwise
    """
    # NEW_PERSON: Someone appeared
    if not previous_state.is_present() and current_detection is not None:
        new_state = PersonState(
            bbox=current_detection.bbox,
            confidence=current_detection.confidence,
            first_seen=now,
            last_seen=now,
            frame_count=1
        )
        
        event = Event(
            event_type=EventType.NEW_PERSON,
            timestamp=now,
            person_state=new_state,
            description="New person detected"
        )
        
        if config.DEBUG_MODE:
            print(f"🎯 EVENT: NEW_PERSON at {now:.1f}")
        
        return event
    
    # PERSON_LEFT: Someone disappeared
    if previous_state.is_present() and current_detection is None:
        event = Event(
            event_type=EventType.PERSON_LEFT,
            timestamp=now,
            person_state=previous_state,
            description="Person left"
        )
        
        if config.DEBUG_MODE:
            print(f"🎯 EVENT: PERSON_LEFT at {now:.1f}")
        
        # Don't trigger Moondream on person leaving (can enable if desired)
        return None  # Return None to skip Moondream call
    
    # No one present - no event
    if current_detection is None:
        return None
    
    # Person is present - check for POSE_CHANGED or PERIODIC_UPDATE
    # Calculate IoU with previous bbox
    iou = previous_state.compute_iou(current_detection.bbox)
    
    # Check if enough time passed since last Moondream call
    time_since_last_call = now - previous_state.last_moondream_call
    
    # POSE_CHANGED: Significant movement detected
    if iou < config.POSE_CHANGE_THRESHOLD and time_since_last_call >= config.MOONDREAM_MIN_INTERVAL:
        updated_state = PersonState(
            person_id=previous_state.person_id,
            name=previous_state.name,
            bbox=current_detection.bbox,
            confidence=current_detection.confidence,
            first_seen=previous_state.first_seen,
            last_seen=now,
            last_moondream_call=now,  # Will be updated after Moondream call
            frame_count=previous_state.frame_count + 1
        )
        
        event = Event(
            event_type=EventType.POSE_CHANGED,
            timestamp=now,
            person_state=updated_state,
            description=f"Pose changed (IoU: {iou:.2f})"
        )
        
        if config.DEBUG_MODE:
            print(f"🎯 EVENT: POSE_CHANGED at {now:.1f} (IoU: {iou:.2f})")
        
        return event
    
    # PERIODIC_UPDATE: Been a while, check in
    if time_since_last_call >= config.PERIODIC_UPDATE_INTERVAL:
        updated_state = PersonState(
            person_id=previous_state.person_id,
            name=previous_state.name,
            bbox=current_detection.bbox,
            confidence=current_detection.confidence,
            first_seen=previous_state.first_seen,
            last_seen=now,
            last_moondream_call=now,  # Will be updated after Moondream call
            frame_count=previous_state.frame_count + 1
        )
        
        event = Event(
            event_type=EventType.PERIODIC_UPDATE,
            timestamp=now,
            person_state=updated_state,
            description=f"Periodic update (last call: {time_since_last_call:.1f}s ago)"
        )
        
        if config.DEBUG_MODE:
            print(f"🎯 EVENT: PERIODIC_UPDATE at {now:.1f}")
        
        return event
    
    # No event - just update state
    return None


def update_person_state(previous_state: PersonState, current_detection: Optional[Detection], 
                       now: float) -> PersonState:
    """
    Update person state with new detection
    Used when no event triggered but we want to track state
    
    Args:
        previous_state: Previous PersonState
        current_detection: Current Detection or None
        now: Current timestamp
    
    Returns:
        Updated PersonState
    """
    if current_detection is None:
        # Person left - return empty state
        return PersonState()
    
    # Update state
    return PersonState(
        person_id=previous_state.person_id,
        name=previous_state.name,
        bbox=current_detection.bbox,
        confidence=current_detection.confidence,
        first_seen=previous_state.first_seen if previous_state.is_present() else now,
        last_seen=now,
        last_moondream_call=previous_state.last_moondream_call,
        frame_count=previous_state.frame_count + 1 if previous_state.is_present() else 1
    )


# ============================================================================
# FACE/PERSON CROPPING FOR MOONDREAM
# ============================================================================

def crop_person_for_moondream(frame: np.ndarray, detection: Detection, 
                              padding: float = 0.2) -> np.ndarray:
    """
    Crop person from frame with padding for Moondream analysis
    
    Args:
        frame: Full frame image
        detection: Person detection
        padding: Padding around bbox as fraction of bbox size (default 20%)
    
    Returns:
        Cropped image of person
    """
    h, w = frame.shape[:2]
    
    # Add padding
    pad_x = int(detection.width * padding)
    pad_y = int(detection.height * padding)
    
    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(w, detection.x2 + pad_x)
    y2 = min(h, detection.y2 + pad_y)
    
    # Crop
    cropped = frame[y1:y2, x1:x2]
    
    if config.DEBUG_MODE:
        print(f"  Cropped person: {cropped.shape} from {frame.shape}")
    
    return cropped


# ============================================================================
# CAMERA UTILITIES
# ============================================================================

def list_available_cameras(max_test: int = 5) -> List[int]:
    """
    List available camera indices by testing them
    
    Args:
        max_test: Maximum camera index to test
    
    Returns:
        List of available camera indices
    """
    import cv2
    available = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


def initialize_camera(camera_index: int = None) -> Optional['cv2.VideoCapture']:
    """
    Initialize camera with configured settings
    
    Args:
        camera_index: Camera index (uses config.CAMERA_INDEX if None)
    
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    import cv2
    
    if camera_index is None:
        camera_index = config.CAMERA_INDEX
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_index}")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera {camera_index} initialized: {actual_width}x{actual_height}")
    
    return cap
