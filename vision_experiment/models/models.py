"""
Data structures for the Living Portrait system
Uses Python 3.10+ dataclasses with type hints
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ============================================================================
# DETECTION DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """Represents a single object detection from YOLO"""
    x1: int  # Top-left x
    y1: int  # Top-left y
    x2: int  # Bottom-right x
    y2: int  # Bottom-right y
    class_id: int  # YOLO class ID
    label: str  # Human-readable label (e.g., "person")
    confidence: float  # Detection confidence (0.0-1.0)
    
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as tuple"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def center(self) -> tuple[int, int]:
        """Return center point of bbox"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def width(self) -> int:
        """Bounding box width"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Bounding box height"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Bounding box area"""
        return self.width * self.height


# ============================================================================
# PERSON TRACKING STATE
# ============================================================================

@dataclass
class PersonState:
    """Represents the current state of a detected person"""
    person_id: Optional[int] = None  # ID from memory/database (if recognized)
    name: Optional[str] = None  # Name (if known)
    bbox: Optional[tuple[int, int, int, int]] = None  # Current bounding box
    confidence: float = 0.0  # Detection confidence
    first_seen: float = 0.0  # Timestamp when first detected
    last_seen: float = 0.0  # Timestamp when last detected
    last_moondream_call: float = 0.0  # Timestamp of last Moondream call
    frame_count: int = 0  # Number of consecutive frames with this person
    
    def is_present(self) -> bool:
        """Check if person is currently present"""
        return self.bbox is not None
    
    def compute_iou(self, other_bbox: tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union with another bbox
        Returns: 0.0-1.0, higher = more overlap
        """
        if self.bbox is None:
            return 0.0
        
        x1_a, y1_a, x2_a, y2_a = self.bbox
        x1_b, y1_b, x2_b, y2_b = other_bbox
        
        # Intersection rectangle
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        
        # Check if there's any intersection
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Compute areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0


# ============================================================================
# EVENT TYPES
# ============================================================================

class EventType(Enum):
    """Types of events that can trigger Moondream calls"""
    NEW_PERSON = "NEW_PERSON"  # Someone just appeared
    PERSON_LEFT = "PERSON_LEFT"  # Person disappeared
    POSE_CHANGED = "POSE_CHANGED"  # Significant movement/pose change
    PERIODIC_UPDATE = "PERIODIC_UPDATE"  # Time-based check-in
    MANUAL_TRIGGER = "MANUAL_TRIGGER"  # User-requested


@dataclass
class Event:
    """Represents an event that may trigger Moondream"""
    event_type: EventType
    timestamp: float
    person_state: PersonState
    description: str = ""  # Optional description for logging


# ============================================================================
# MOONDREAM INTEGRATION
# ============================================================================

@dataclass
class MoondreamContext:
    """Context information sent to Moondream"""
    person_id: Optional[int] = None
    name: Optional[str] = None
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    event_type: str = "UNKNOWN"
    system_prompt: str = ""
    
    def build_prompt(self, image_description: str = "") -> str:
        """Build the full prompt for Moondream"""
        parts = []
        
        # Add system context
        if self.system_prompt:
            parts.append(self.system_prompt)
        
        # Add person context
        if self.name:
            parts.append(f"This is {self.name}.")
        elif self.person_id:
            parts.append(f"This is person #{self.person_id}.")
        else:
            parts.append("This person is new to me.")
        
        # Add recent interactions
        if self.recent_interactions:
            parts.append("\nRecent conversations:")
            for interaction in self.recent_interactions[-3:]:  # Last 3
                mood = interaction.get('mood', 'neutral')
                text = interaction.get('text', '')
                parts.append(f"  [{mood}] {text}")
        
        # Add event context
        parts.append(f"\nEvent: {self.event_type}")
        
        # Add image description if provided
        if image_description:
            parts.append(f"\nI see: {image_description}")
        
        parts.append("\nRespond in character as a magical living portrait. Be brief, warm, and engaging.")
        
        return "\n".join(parts)


@dataclass
class MoondreamResult:
    """Result from Moondream API call"""
    text: str  # What the portrait says
    mood: str = "idle"  # Mood/emotion: idle, happy, curious, thoughtful, etc.
    raw_response: Optional[Dict[str, Any]] = None  # Full API response
    
    @classmethod
    def parse_from_api(cls, api_response: Dict[str, Any]) -> 'MoondreamResult':
        """
        Parse Moondream API response into MoondreamResult
        
        Expected API response format:
        {
            "response": "Hello! Good to see you again.",
            "mood": "happy"  # optional
        }
        """
        text = api_response.get('response', api_response.get('text', ''))
        mood = api_response.get('mood', 'idle')
        
        # Basic mood extraction from text if not provided
        if mood == 'idle' and text:
            text_lower = text.lower()
            if any(word in text_lower for word in ['?', 'what', 'how', 'why', 'curious']):
                mood = 'curious'
            elif any(word in text_lower for word in ['!', 'great', 'wonderful', 'happy', 'excellent']):
                mood = 'happy'
            elif any(word in text_lower for word in ['hmm', 'think', 'ponder', 'consider']):
                mood = 'thoughtful'
        
        return cls(text=text, mood=mood, raw_response=api_response)


# ============================================================================
# ANIMATION STATE
# ============================================================================

@dataclass
class AnimationState:
    """Current state of the portrait animation"""
    mood: str = "idle"  # Current mood (sprite set to use)
    speaking: bool = False  # Is the portrait currently "speaking"?
    last_line: str = ""  # Last thing the portrait said
    subtitle: str = ""  # Current subtitle to display
    last_mood_change: float = 0.0  # When mood last changed
    last_mouth_toggle: float = 0.0  # For talking animation
    mouth_open: bool = False  # Current mouth state
    speaking_until: float = 0.0  # Timestamp when speaking ends
    subtitle_until: float = 0.0  # Timestamp when subtitle disappears
    
    def update_from_moondream(self, result: MoondreamResult, now: float, 
                             speaking_duration: float, subtitle_duration: float):
        """Update animation state from Moondream result"""
        self.mood = result.mood
        self.last_line = result.text
        self.subtitle = result.text
        self.speaking = True
        self.last_mood_change = now
        self.speaking_until = now + speaking_duration
        self.subtitle_until = now + subtitle_duration
        self.mouth_open = True
    
    def update_speaking(self, now: float, mouth_toggle_interval: float):
        """Update speaking state (called each frame)"""
        # Check if speaking time is over
        if now >= self.speaking_until:
            self.speaking = False
            self.mouth_open = False
        
        # Toggle mouth if speaking
        if self.speaking and now - self.last_mouth_toggle >= mouth_toggle_interval:
            self.mouth_open = not self.mouth_open
            self.last_mouth_toggle = now
    
    def update_subtitle(self, now: float):
        """Update subtitle visibility"""
        if now >= self.subtitle_until:
            self.subtitle = ""


# ============================================================================
# MEMORY/STORAGE DATA STRUCTURES
# ============================================================================

@dataclass
class Person:
    """Represents a known person in memory"""
    person_id: int
    name: str
    notes: str = ""
    first_seen: Optional[str] = None  # ISO datetime string
    last_seen: Optional[str] = None  # ISO datetime string
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'person_id': self.person_id,
            'name': self.name,
            'notes': self.notes,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Person':
        """Create from dictionary (loaded from JSON)"""
        return cls(
            person_id=data['person_id'],
            name=data['name'],
            notes=data.get('notes', ''),
            first_seen=data.get('first_seen'),
            last_seen=data.get('last_seen')
        )


@dataclass
class Interaction:
    """Represents a single interaction record"""
    person_id: Optional[int]
    timestamp: str  # ISO datetime string
    mood: str
    text: str
    event_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'person_id': self.person_id,
            'timestamp': self.timestamp,
            'mood': self.mood,
            'text': self.text,
            'event_type': self.event_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """Create from dictionary (loaded from JSON)"""
        return cls(
            person_id=data.get('person_id'),
            timestamp=data['timestamp'],
            mood=data['mood'],
            text=data['text'],
            event_type=data['event_type']
        )
    
    @classmethod
    def create_now(cls, person_id: Optional[int], mood: str, text: str, 
                   event_type: str) -> 'Interaction':
        """Create a new interaction with current timestamp"""
        return cls(
            person_id=person_id,
            timestamp=datetime.now().isoformat(),
            mood=mood,
            text=text,
            event_type=event_type
        )


# ============================================================================
# CAMERA SELECTION
# ============================================================================

@dataclass
class CameraInfo:
    """Information about an available camera"""
    index: int
    name: str
    is_available: bool = True
