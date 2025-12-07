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
    face_recognition_confidence: float = 0.0  # Face recognition confidence (0.0-1.0)
    pending_face_confirmation: bool = False  # Awaiting user confirmation for face match
    
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
    CHAT_MESSAGE = "CHAT_MESSAGE"  # User sent a text chat message
    VOICE_MESSAGE = "VOICE_MESSAGE"  # User sent a voice message
    FACE_CONFIRMED = "FACE_CONFIRMED"  # Face recognition confirmed by user
    FACE_DENIED = "FACE_DENIED"  # Face recognition rejected by user


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
    user_message: Optional[str] = None  # Text from chat or voice input
    
    def build_prompt(self, image_description: str = "") -> str:
        """Build the full prompt for Moondream"""
        parts = []
        
        # Add system context
        if self.system_prompt:
            parts.append(self.system_prompt)
        else:
            parts.append("You are a magical living portrait that can see and converse with people.")
        
        # Add person context
        if self.name:
            parts.append(f"You are speaking with {self.name}, whom you recognize.")
        elif self.person_id:
            parts.append(f"You are speaking with person #{self.person_id}, but you don't know their name yet.")
        else:
            parts.append("This is someone new. You should introduce yourself and ask their name warmly.")
        
        # Add recent interactions for context
        if self.recent_interactions:
            parts.append("\nRecent conversation:")
            for interaction in self.recent_interactions[-5:]:  # Last 5 exchanges
                role = interaction.get('role', 'portrait')
                text = interaction.get('text', '')
                if role == 'user':
                    parts.append(f"  User: {text}")
                else:
                    parts.append(f"  You: {text}")
        
        # Add event context
        event_hints = {
            "NEW_PERSON": "A new person just appeared. Greet them warmly and ask their name.",
            "CHAT_MESSAGE": "The user sent you a text message. Respond naturally and conversationally.",
            "VOICE_MESSAGE": "The user spoke to you. Give a natural, varied response - don't repeat yourself.",
            "PERIODIC_UPDATE": "Continue the conversation or ask an engaging question.",
            "POSE_CHANGED": "The person moved. Comment briefly if relevant, or continue the conversation."
        }
        hint = event_hints.get(self.event_type, "")
        if hint:
            parts.append(f"\n{hint}")
        
        # Add user's message if provided (chat or voice input)
        if self.user_message:
            parts.append(f"\nUser says: \"{self.user_message}\"")
            parts.append("IMPORTANT: Respond DIRECTLY to what they said. Answer their question or comment on their topic.")
            parts.append("DO NOT just say 'Hello from within the frame' - actually engage with their words!")
        
        # Add image description if provided
        if image_description:
            parts.append(f"\nYou can see: {image_description}")
        
        # Final instructions
        if not self.name and not self.user_message:
            parts.append("\nIntroduce yourself as a living portrait and ask for their name.")
        
        parts.append("\nRespond in 1-2 short sentences. Be warm, engaging, and conversational.")
        parts.append("VARY your responses - don't repeat the same phrases!")
        
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
# FACE RECOGNITION
# ============================================================================

class InteractionMode(Enum):
    """How the user can interact with the portrait"""
    VOICE_ONLY = "VOICE_ONLY"  # Voice interaction enabled
    CHAT_ONLY = "CHAT_ONLY"  # Text chat enabled
    VOICE_AND_CHAT = "VOICE_AND_CHAT"  # Both enabled


@dataclass
class FaceRecognitionResult:
    """Result from face recognition attempt"""
    person_id: Optional[int] = None  # Matched person ID (if any)
    name: Optional[str] = None  # Matched person name (if any)
    confidence: float = 0.0  # Confidence score (0.0-1.0)
    needs_confirmation: bool = False  # True if below auto-accept threshold
    face_encoding: Optional[List[float]] = None  # Face encoding for storage
    bbox: Optional[tuple[int, int, int, int]] = None  # Face location
    
    def is_confident_match(self, threshold: float = 0.7) -> bool:
        """Check if confidence exceeds auto-accept threshold"""
        return self.confidence >= threshold and self.person_id is not None
    
    def is_unknown(self) -> bool:
        """Check if this is an unknown face"""
        return self.person_id is None


# ============================================================================
# CONVERSATION & CHAT
# ============================================================================

@dataclass
class ChatMessage:
    """Single message in a conversation"""
    message_id: str  # Unique ID for this message
    speaker: str  # "user" or "portrait"
    text: str  # Message content
    timestamp: str  # ISO datetime string
    is_voice: bool = False  # True if spoken, False if typed
    mood: Optional[str] = None  # Portrait mood (portrait messages only)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'message_id': self.message_id,
            'speaker': self.speaker,
            'text': self.text,
            'timestamp': self.timestamp,
            'is_voice': self.is_voice,
            'mood': self.mood
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary (loaded from JSON)"""
        return cls(
            message_id=data['message_id'],
            speaker=data['speaker'],
            text=data['text'],
            timestamp=data['timestamp'],
            is_voice=data.get('is_voice', False),
            mood=data.get('mood')
        )
    
    @classmethod
    def create_user_message(cls, text: str, is_voice: bool = False) -> 'ChatMessage':
        """Create a new user message with auto-generated ID and timestamp"""
        import uuid
        return cls(
            message_id=str(uuid.uuid4()),
            speaker='user',
            text=text,
            timestamp=datetime.now().isoformat(),
            is_voice=is_voice
        )
    
    @classmethod
    def create_portrait_message(cls, text: str, mood: str = 'idle', 
                                is_voice: bool = False) -> 'ChatMessage':
        """Create a new portrait message with auto-generated ID and timestamp"""
        import uuid
        return cls(
            message_id=str(uuid.uuid4()),
            speaker='portrait',
            text=text,
            timestamp=datetime.now().isoformat(),
            is_voice=is_voice,
            mood=mood
        )


@dataclass
class PersonConversation:
    """Conversation thread for a specific person"""
    person_id: Optional[int]  # None for unknown person
    messages: List[ChatMessage] = field(default_factory=list)
    max_messages: int = 50  # Maximum messages before archiving
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_message(self, message: ChatMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.last_updated = datetime.now().isoformat()
        
        # Archive old messages if limit exceeded
        if len(self.messages) > self.max_messages:
            self._archive_old_messages()
    
    def _archive_old_messages(self):
        """Move old messages to archive, keeping only recent ones"""
        # Keep most recent max_messages
        archived = self.messages[:-self.max_messages]
        self.messages = self.messages[-self.max_messages:]
        # TODO: Save archived messages to separate file
        return archived
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent N messages"""
        return self.messages[-count:] if self.messages else []
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a specific message (for 'forget that' command)"""
        for i, msg in enumerate(self.messages):
            if msg.message_id == message_id:
                del self.messages[i]
                self.last_updated = datetime.now().isoformat()
                return True
        return False
    
    def delete_last_exchange(self) -> int:
        """Delete the last user message and portrait response (for 'forget that')"""
        deleted_count = 0
        # Remove last portrait message if it exists
        if self.messages and self.messages[-1].speaker == 'portrait':
            self.messages.pop()
            deleted_count += 1
        # Remove last user message if it exists
        if self.messages and self.messages[-1].speaker == 'user':
            self.messages.pop()
            deleted_count += 1
        if deleted_count > 0:
            self.last_updated = datetime.now().isoformat()
        return deleted_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'person_id': self.person_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'max_messages': self.max_messages,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonConversation':
        """Create from dictionary (loaded from JSON)"""
        return cls(
            person_id=data.get('person_id'),
            messages=[ChatMessage.from_dict(msg) for msg in data.get('messages', [])],
            max_messages=data.get('max_messages', 50),
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_updated=data.get('last_updated', datetime.now().isoformat())
        )


# ============================================================================
# VOICE SETTINGS
# ============================================================================

@dataclass
class VoiceSettings:
    """Configuration for voice interaction"""
    wake_word: str = "hey portrait"  # Wake word to activate voice input
    wake_word_threshold: float = 0.5  # Sensitivity (0.0-1.0)
    tts_voice: Optional[str] = None  # TTS voice name (None = default)
    tts_rate: int = 150  # Words per minute
    tts_volume: float = 0.9  # Volume (0.0-1.0)
    confirmation_timeout: int = 30  # Seconds to wait for face confirmation
    chime_on_wake: bool = True  # Play sound when wake word detected
    punctuation_buffer: bool = True  # Wait for punctuation before TTS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'wake_word': self.wake_word,
            'wake_word_threshold': self.wake_word_threshold,
            'tts_voice': self.tts_voice,
            'tts_rate': self.tts_rate,
            'tts_volume': self.tts_volume,
            'confirmation_timeout': self.confirmation_timeout,
            'chime_on_wake': self.chime_on_wake,
            'punctuation_buffer': self.punctuation_buffer
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceSettings':
        """Create from dictionary (loaded from JSON)"""
        return cls(
            wake_word=data.get('wake_word', 'hey portrait'),
            wake_word_threshold=data.get('wake_word_threshold', 0.5),
            tts_voice=data.get('tts_voice'),
            tts_rate=data.get('tts_rate', 150),
            tts_volume=data.get('tts_volume', 0.9),
            confirmation_timeout=data.get('confirmation_timeout', 30),
            chime_on_wake=data.get('chime_on_wake', True),
            punctuation_buffer=data.get('punctuation_buffer', True)
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
