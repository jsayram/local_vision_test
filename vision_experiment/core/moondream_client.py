"""
Moondream API Client
Handles vision-language model calls for the Living Portrait
"""
import requests
import base64
import json
import time
from typing import Optional, Dict, Any
import numpy as np
import cv2

from models.models import MoondreamContext, MoondreamResult
from core import config
from core import storage

# ============================================================================
# IMAGE ENCODING
# ============================================================================

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode OpenCV image (numpy array) to base64 string
    Args:
        image: OpenCV image (BGR format)
    Returns: Base64-encoded JPEG string
    """
    # Encode image as JPEG
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image as JPEG")
    
    # Convert to base64
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


# ============================================================================
# MOONDREAM API CALL
# ============================================================================

def call_moondream_on_face(face_image: np.ndarray, context: MoondreamContext) -> Optional[MoondreamResult]:
    """
    Call Moondream vision-language model with a face/person image
    
    Args:
        face_image: OpenCV image (numpy array) of the person's face/upper body
        context: MoondreamContext with person info and recent interactions
    
    Returns:
        MoondreamResult with mood and text, or None if call fails
    """
    if config.DEBUG_MODE:
        print(f"🌙 Calling Moondream for {context.name or 'unknown person'}...")
    
    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(face_image)
        
        # Build the prompt
        prompt = context.build_prompt()
        
        # Prepare API request
        # This is a STUB - replace with your actual Moondream API format
        # Example assumes Ollama-style API with vision support
        
        if config.MOONDREAM_MODEL == 'moondream':
            # Ollama format
            api_payload = {
                "model": config.MOONDREAM_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        else:
            # Generic format - adapt as needed
            api_payload = {
                "model": config.MOONDREAM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }
                ]
            }
        
        # Add API key if provided
        headers = {
            "Content-Type": "application/json"
        }
        if config.MOONDREAM_API_KEY:
            headers["Authorization"] = f"Bearer {config.MOONDREAM_API_KEY}"
        
        # Make the API call
        start_time = time.time()
        response = requests.post(
            config.MOONDREAM_API_URL,
            json=api_payload,
            headers=headers,
            timeout=config.MOONDREAM_TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"✗ Moondream API error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
        
        # Parse response
        api_response = response.json()
        
        if config.DEBUG_MODE:
            print(f"✓ Moondream responded in {elapsed:.2f}s")
            print(f"  Raw response: {json.dumps(api_response, indent=2)[:500]}")
        
        # Parse into MoondreamResult
        # Adapt this based on your actual API response format
        result = parse_moondream_response(api_response)
        
        if config.DEBUG_MODE:
            print(f"  Mood: {result.mood}")
            print(f"  Text: {result.text}")
        
        return result
        
    except requests.Timeout:
        print(f"✗ Moondream API timeout after {config.MOONDREAM_TIMEOUT}s")
        return None
    except requests.RequestException as e:
        print(f"✗ Moondream API request failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error calling Moondream: {e}")
        return None


def parse_moondream_response(api_response: Dict[str, Any]) -> MoondreamResult:
    """
    Parse Moondream API response into MoondreamResult
    Adapt this function based on your actual Moondream API response format
    
    Args:
        api_response: Raw API response dictionary
    
    Returns:
        MoondreamResult with parsed mood and text
    """
    # Example for Ollama format
    if 'response' in api_response:
        # Ollama format: {"model": "moondream", "response": "text here", ...}
        text = api_response.get('response', '')
        mood = extract_mood_from_text(text)
        return MoondreamResult(text=text, mood=mood, raw_response=api_response)
    
    # Example for OpenAI-style format
    elif 'choices' in api_response:
        # OpenAI format: {"choices": [{"message": {"content": "text"}}]}
        try:
            text = api_response['choices'][0]['message']['content']
            mood = extract_mood_from_text(text)
            return MoondreamResult(text=text, mood=mood, raw_response=api_response)
        except (KeyError, IndexError):
            pass
    
    # Generic fallback
    text = api_response.get('text', api_response.get('content', str(api_response)))
    mood = extract_mood_from_text(text)
    return MoondreamResult(text=text, mood=mood, raw_response=api_response)


def extract_mood_from_text(text: str) -> str:
    """
    Extract mood from response text using keyword matching
    This is a simple heuristic - you can improve with sentiment analysis
    
    Args:
        text: Response text from Moondream
    
    Returns:
        Mood string: "idle", "happy", "curious", or "thoughtful"
    """
    text_lower = text.lower()
    
    # Check for mood indicators
    happy_words = ['!', 'great', 'wonderful', 'happy', 'excellent', 'amazing', 'fantastic', 
                   'delighted', 'joy', 'pleased', 'glad', 'lovely']
    curious_words = ['?', 'what', 'how', 'why', 'when', 'where', 'curious', 'wonder', 
                    'interesting', 'intriguing', 'hmm']
    thoughtful_words = ['think', 'ponder', 'consider', 'reflect', 'contemplate', 
                       'perhaps', 'maybe', 'seems', 'appears']
    
    # Count indicators
    happy_count = sum(1 for word in happy_words if word in text_lower)
    curious_count = sum(1 for word in curious_words if word in text_lower)
    thoughtful_count = sum(1 for word in thoughtful_words if word in text_lower)
    
    # Determine dominant mood
    if curious_count > 0 and '?' in text:
        return 'curious'
    elif happy_count > thoughtful_count and happy_count > curious_count:
        return 'happy'
    elif thoughtful_count > 0:
        return 'thoughtful'
    else:
        return 'idle'


# ============================================================================
# STUB/FALLBACK FOR TESTING WITHOUT REAL API
# ============================================================================

def call_moondream_stub(face_image: np.ndarray, context: MoondreamContext) -> MoondreamResult:
    """
    Stub implementation for testing without a real Moondream API
    Returns canned responses based on event type
    
    Args:
        face_image: Face image (unused in stub)
        context: Context information
    
    Returns:
        MoondreamResult with fake response
    """
    # Simulate API delay
    time.sleep(0.5)
    
    # Generate response based on event type
    if context.event_type == "NEW_PERSON":
        if context.name:
            responses = [
                (f"Ah, {context.name}! Welcome back to my gallery.", "happy"),
                (f"Hello again, {context.name}. It's been a while.", "curious"),
                (f"Good to see you, {context.name}. What brings you here today?", "curious")
            ]
        else:
            responses = [
                ("A new face! How intriguing. Who might you be?", "curious"),
                ("Hello there! I don't believe we've met.", "happy"),
                ("Welcome to my frame. Care to introduce yourself?", "curious")
            ]
    
    elif context.event_type == "POSE_CHANGED":
        responses = [
            ("What are you looking for?", "curious"),
            ("Something caught your attention?", "curious"),
            ("Ah, a change in posture. Interesting...", "thoughtful")
        ]
    
    elif context.event_type == "PERIODIC_UPDATE":
        responses = [
            ("Still there? Time flies when you're painted.", "thoughtful"),
            ("How are you doing?", "curious"),
            ("Anything on your mind?", "curious")
        ]
    
    else:
        responses = [
            ("Hello from within the frame.", "idle"),
            ("Greetings, visitor.", "idle")
        ]
    
    # Pick a response
    import random
    text, mood = random.choice(responses)
    
    if config.DEBUG_MODE:
        print(f"🌙 [STUB] Moondream response: [{mood}] {text}")
    
    return MoondreamResult(text=text, mood=mood)


# ============================================================================
# MAIN API CALL WRAPPER (SWITCHES BETWEEN REAL AND STUB)
# ============================================================================

def call_moondream(face_image: np.ndarray, context: MoondreamContext, 
                  use_stub: bool = False) -> Optional[MoondreamResult]:
    """
    Main wrapper for Moondream calls
    Switches between real API and stub based on configuration
    
    Args:
        face_image: Face/person image
        context: Moondream context
        use_stub: If True, use stub instead of real API
    
    Returns:
        MoondreamResult or None if failed
    """
    # Check if we should use stub
    # Use stub if:
    # - Explicitly requested
    # - No API URL configured
    # - API URL is localhost and not available
    
    if use_stub or not config.MOONDREAM_API_URL or config.MOONDREAM_API_URL == '':
        return call_moondream_stub(face_image, context)
    
    # Try real API, fall back to stub on failure
    result = call_moondream_on_face(face_image, context)
    
    if result is None:
        print("⚠ Falling back to stub response")
        result = call_moondream_stub(face_image, context)
    
    return result


# ============================================================================
# CONTEXT BUILDER HELPER
# ============================================================================

def build_context_for_person(person_id: Optional[int], event_type: str) -> MoondreamContext:
    """
    Build MoondreamContext for a person
    Loads recent interactions and person info from storage
    
    Args:
        person_id: Person ID (or None if unknown)
        event_type: Type of event triggering this call
    
    Returns:
        MoondreamContext ready for Moondream call
    """
    # Get person info
    person_info = None
    name = None
    if person_id is not None:
        person_info = storage.get_person_by_id(person_id)
        if person_info:
            name = person_info.get('name')
    
    # Get recent interactions
    recent_interactions = storage.get_recent_interactions(
        person_id=person_id, 
        limit=config.MAX_RECENT_INTERACTIONS
    )
    
    # Get system prompt from settings
    system_prompt = storage.get_setting('system_prompt', '')
    
    # Build context
    context = MoondreamContext(
        person_id=person_id,
        name=name,
        recent_interactions=recent_interactions,
        event_type=event_type,
        system_prompt=system_prompt
    )
    
    return context
