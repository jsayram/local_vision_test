"""
JSON-based local storage for the Living Portrait system
Handles people, interactions, and settings persistence
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.models import Person, Interaction
from core import config

# ============================================================================
# INITIALIZATION
# ============================================================================

def ensure_memory_dir():
    """Create memory directory if it doesn't exist"""
    Path(config.MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    if config.DEBUG_MODE:
        print(f"✓ Memory directory: {config.MEMORY_DIR}")


def initialize_storage():
    """
    Initialize all JSON storage files with defaults if they don't exist
    Called on application startup
    """
    ensure_memory_dir()
    
    # Initialize people.json
    if not os.path.exists(config.PEOPLE_JSON):
        save_people([])
        print(f"✓ Created {config.PEOPLE_JSON}")
    
    # Initialize interactions.json
    if not os.path.exists(config.INTERACTIONS_JSON):
        save_interactions([])
        print(f"✓ Created {config.INTERACTIONS_JSON}")
    
    # Initialize settings.json with defaults
    if not os.path.exists(config.SETTINGS_JSON):
        default_settings = {
            'moondream_interval_seconds': config.MOONDREAM_MIN_INTERVAL,
            'pose_change_threshold': config.POSE_CHANGE_THRESHOLD,
            'periodic_update_interval': config.PERIODIC_UPDATE_INTERVAL,
            'debug_mode': config.DEBUG_MODE,
            'system_prompt': (
                "You are a magical living portrait hanging on the wall. "
                "You observe the world through your frame and speak with wisdom, "
                "warmth, and a touch of mystery. Keep responses brief (1-2 sentences). "
                "Be friendly and remember past conversations."
            )
        }
        save_settings(default_settings)
        print(f"✓ Created {config.SETTINGS_JSON}")


# ============================================================================
# PEOPLE STORAGE
# ============================================================================

def load_people() -> List[Dict[str, Any]]:
    """
    Load all people from people.json
    Returns: List of person dictionaries
    """
    try:
        with open(config.PEOPLE_JSON, 'r') as f:
            data = json.load(f)
            if config.DEBUG_MODE:
                print(f"✓ Loaded {len(data)} people from storage")
            return data
    except FileNotFoundError:
        if config.DEBUG_MODE:
            print(f"⚠ People file not found, returning empty list")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Error loading people.json: {e}")
        return []


def save_people(people: List[Dict[str, Any]]) -> bool:
    """
    Save people list to people.json
    Args:
        people: List of person dictionaries
    Returns: True if successful, False otherwise
    """
    try:
        ensure_memory_dir()
        with open(config.PEOPLE_JSON, 'w') as f:
            json.dump(people, f, indent=2)
        if config.DEBUG_MODE:
            print(f"✓ Saved {len(people)} people to storage")
        return True
    except Exception as e:
        print(f"✗ Error saving people.json: {e}")
        return False


def get_person_by_id(person_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific person by ID
    Args:
        person_id: Person ID to lookup
    Returns: Person dictionary or None if not found
    """
    people = load_people()
    for person in people:
        if person.get('person_id') == person_id:
            return person
    return None


def get_person_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific person by name (case-insensitive)
    Args:
        name: Person name to lookup
    Returns: Person dictionary or None if not found
    """
    people = load_people()
    name_lower = name.lower()
    for person in people:
        if person.get('name', '').lower() == name_lower:
            return person
    return None


def add_person(name: str, notes: str = "") -> Dict[str, Any]:
    """
    Add a new person to storage
    Args:
        name: Person's name
        notes: Optional notes about the person
    Returns: The created person dictionary
    """
    people = load_people()
    
    # Generate new ID (max existing ID + 1)
    max_id = max([p.get('person_id', 0) for p in people], default=0)
    new_id = max_id + 1
    
    # Create new person
    new_person = {
        'person_id': new_id,
        'name': name,
        'notes': notes,
        'first_seen': datetime.now().isoformat(),
        'last_seen': datetime.now().isoformat()
    }
    
    people.append(new_person)
    save_people(people)
    
    if config.DEBUG_MODE:
        print(f"✓ Added new person: {name} (ID: {new_id})")
    
    return new_person


def update_person_last_seen(person_id: int):
    """
    Update the last_seen timestamp for a person
    Args:
        person_id: Person ID to update
    """
    people = load_people()
    for person in people:
        if person.get('person_id') == person_id:
            person['last_seen'] = datetime.now().isoformat()
            save_people(people)
            if config.DEBUG_MODE:
                print(f"✓ Updated last_seen for person ID {person_id}")
            return
    
    if config.DEBUG_MODE:
        print(f"⚠ Person ID {person_id} not found for last_seen update")


# ============================================================================
# INTERACTIONS STORAGE
# ============================================================================

def load_interactions() -> List[Dict[str, Any]]:
    """
    Load all interactions from interactions.json
    Returns: List of interaction dictionaries
    """
    try:
        with open(config.INTERACTIONS_JSON, 'r') as f:
            data = json.load(f)
            if config.DEBUG_MODE:
                print(f"✓ Loaded {len(data)} interactions from storage")
            return data
    except FileNotFoundError:
        if config.DEBUG_MODE:
            print(f"⚠ Interactions file not found, returning empty list")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Error loading interactions.json: {e}")
        return []


def save_interactions(interactions: List[Dict[str, Any]]) -> bool:
    """
    Save interactions list to interactions.json
    Args:
        interactions: List of interaction dictionaries
    Returns: True if successful, False otherwise
    """
    try:
        ensure_memory_dir()
        with open(config.INTERACTIONS_JSON, 'w') as f:
            json.dump(interactions, f, indent=2)
        if config.DEBUG_MODE:
            print(f"✓ Saved {len(interactions)} interactions to storage")
        return True
    except Exception as e:
        print(f"✗ Error saving interactions.json: {e}")
        return False


def append_interaction(interaction: Dict[str, Any]) -> bool:
    """
    Append a new interaction to interactions.json
    Args:
        interaction: Interaction dictionary to append
    Returns: True if successful, False otherwise
    """
    interactions = load_interactions()
    interactions.append(interaction)
    return save_interactions(interactions)


def get_recent_interactions(person_id: Optional[int] = None, 
                           limit: int = None) -> List[Dict[str, Any]]:
    """
    Get recent interactions, optionally filtered by person_id
    Args:
        person_id: If provided, only return interactions for this person
        limit: Maximum number of interactions to return (most recent first)
    Returns: List of interaction dictionaries
    """
    interactions = load_interactions()
    
    # Filter by person_id if provided
    if person_id is not None:
        interactions = [i for i in interactions if i.get('person_id') == person_id]
    
    # Sort by timestamp (most recent first)
    interactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Apply limit
    if limit is not None:
        interactions = interactions[:limit]
    
    return interactions


def create_interaction_record(person_id: Optional[int], mood: str, text: str, 
                              event_type: str) -> Dict[str, Any]:
    """
    Create a new interaction record and save it
    Args:
        person_id: Person ID (or None if unknown)
        mood: Mood/emotion of the interaction
        text: What the portrait said
        event_type: Type of event that triggered this
    Returns: The created interaction dictionary
    """
    interaction = Interaction.create_now(person_id, mood, text, event_type)
    interaction_dict = interaction.to_dict()
    
    if append_interaction(interaction_dict):
        if config.DEBUG_MODE:
            print(f"✓ Saved interaction: [{mood}] {text[:50]}...")
        return interaction_dict
    else:
        print(f"✗ Failed to save interaction")
        return interaction_dict


# ============================================================================
# SETTINGS STORAGE
# ============================================================================

def load_settings() -> Dict[str, Any]:
    """
    Load settings from settings.json
    Returns: Settings dictionary
    """
    try:
        with open(config.SETTINGS_JSON, 'r') as f:
            data = json.load(f)
            if config.DEBUG_MODE:
                print(f"✓ Loaded settings from storage")
            return data
    except FileNotFoundError:
        if config.DEBUG_MODE:
            print(f"⚠ Settings file not found, returning defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"✗ Error loading settings.json: {e}")
        return {}


def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Save settings to settings.json
    Args:
        settings: Settings dictionary
    Returns: True if successful, False otherwise
    """
    try:
        ensure_memory_dir()
        with open(config.SETTINGS_JSON, 'w') as f:
            json.dump(settings, f, indent=2)
        if config.DEBUG_MODE:
            print(f"✓ Saved settings to storage")
        return True
    except Exception as e:
        print(f"✗ Error saving settings.json: {e}")
        return False


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific setting value
    Args:
        key: Setting key to retrieve
        default: Default value if key not found
    Returns: Setting value or default
    """
    settings = load_settings()
    return settings.get(key, default)


def update_setting(key: str, value: Any) -> bool:
    """
    Update a specific setting
    Args:
        key: Setting key to update
        value: New value
    Returns: True if successful, False otherwise
    """
    settings = load_settings()
    settings[key] = value
    return save_settings(settings)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_storage_stats() -> Dict[str, Any]:
    """
    Get statistics about stored data
    Returns: Dictionary with counts and sizes
    """
    people = load_people()
    interactions = load_interactions()
    settings = load_settings()
    
    return {
        'people_count': len(people),
        'interactions_count': len(interactions),
        'settings_count': len(settings),
        'memory_dir': config.MEMORY_DIR,
        'total_size_bytes': sum([
            os.path.getsize(config.PEOPLE_JSON) if os.path.exists(config.PEOPLE_JSON) else 0,
            os.path.getsize(config.INTERACTIONS_JSON) if os.path.exists(config.INTERACTIONS_JSON) else 0,
            os.path.getsize(config.SETTINGS_JSON) if os.path.exists(config.SETTINGS_JSON) else 0
        ])
    }


def clear_all_data():
    """
    WARNING: Clear all stored data (use with caution!)
    Reinitializes with empty/default data
    """
    if config.DEBUG_MODE:
        print("⚠ Clearing all storage data...")
    
    save_people([])
    save_interactions([])
    initialize_storage()  # Recreate with defaults
    
    print("✓ All storage data cleared and reinitialized")
