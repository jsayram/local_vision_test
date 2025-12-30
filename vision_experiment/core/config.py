"""
Configuration and Constants for Living Portrait System
Supports both M1 Max (macOS) and Raspberry Pi 3 deployment
"""
import os
import platform

# ============================================================================
# DEVICE DETECTION AND MODE
# ============================================================================

def detect_device_mode():
    """Auto-detect if running on macOS or Raspberry Pi"""
    system = platform.system()
    
    # Check for Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                if 'Raspberry Pi 3' in cpuinfo:
                    return 'pi3'
                elif 'Raspberry Pi 4' in cpuinfo:
                    return 'pi4'
                elif 'Raspberry Pi 5' in cpuinfo:
                    return 'pi5'
                else:
                    return 'pi_other'
    except:
        pass
    
    # Check for macOS
    if system == 'Darwin':
        return 'mac'
    
    # Default to desktop Linux
    return 'desktop'

DEVICE_MODE = detect_device_mode()

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

if DEVICE_MODE in ['pi3', 'pi_other']:
    # Raspberry Pi 3 - lower resolution
    CAMERA_INDEX = 0
    FRAME_WIDTH = 416
    FRAME_HEIGHT = 416
    DETECTION_SKIP_FRAMES = 3  # Run detection every 3rd frame
elif DEVICE_MODE in ['pi4', 'pi5']:
    # Raspberry Pi 4/5 - medium resolution
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    DETECTION_SKIP_FRAMES = 2
else:
    # M1 Max or desktop - higher resolution
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    DETECTION_SKIP_FRAMES = 1  # Run detection every frame

# ============================================================================
# YOLO MODEL CONFIGURATION
# ============================================================================

if DEVICE_MODE in ['pi3', 'pi_other']:
    # Raspberry Pi 3 - lightest model
    YOLO_MODEL = 'yolov8n.pt'  # Nano model
    YOLO_CONFIDENCE = 0.5
elif DEVICE_MODE in ['pi4', 'pi5']:
    # Raspberry Pi 4/5 - small model
    YOLO_MODEL = 'yolov8s.pt'  # Small model
    YOLO_CONFIDENCE = 0.5
else:
    # M1 Max or desktop - medium model
    YOLO_MODEL = 'yolov8s.pt'  # Small/Medium model
    YOLO_CONFIDENCE = 0.5

# Class ID for "person" in COCO dataset
PERSON_CLASS_ID = 0
PERSON_CLASS_LABEL = 'person'

# ============================================================================
# EVENT DETECTION THRESHOLDS
# ============================================================================

# Minimum time between Moondream calls (seconds)
if DEVICE_MODE in ['pi3', 'pi_other']:
    MOONDREAM_MIN_INTERVAL = 30.0  # Longer interval on Pi 3
else:
    MOONDREAM_MIN_INTERVAL = 15.0  # Faster on M1 Max

# Pose change threshold (0.0-1.0, higher = more sensitive)
# Based on IoU (Intersection over Union) - lower IoU = bigger change
POSE_CHANGE_THRESHOLD = 0.7  # If IoU < 0.7, consider it a pose change

# Minimum bounding box size to consider (avoid tiny false detections)
MIN_BBOX_SIZE = 50  # pixels

# Maximum time to wait before periodic update (seconds)
PERIODIC_UPDATE_INTERVAL = 45.0

# ============================================================================
# ANIMATION CONFIGURATION
# ============================================================================

# Sprites folder - in the main vision_experiment directory
SPRITES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sprites')

# Sprite filenames (expected in SPRITES_DIR)
SPRITE_FILES = {
    'idle': 'idle.png',
    'happy': 'happy.png',
    'curious': 'curious.png',
    'thoughtful': 'thoughtful.png',
    'talking_open': 'talking_open.png',
    'talking_closed': 'talking_closed.png'
}

# Animation timing
TALKING_MOUTH_TOGGLE_INTERVAL = 0.15  # seconds between mouth open/closed
SPEAKING_DURATION = 5.0  # seconds to show talking animation
SUBTITLE_DURATION = 8.0  # seconds to show subtitle

# Canvas size for portrait display
PORTRAIT_WIDTH = 640
PORTRAIT_HEIGHT = 480

# Fallback colors (if sprites missing)
FALLBACK_COLORS = {
    'idle': (100, 100, 100),      # Gray
    'happy': (0, 200, 0),          # Green
    'curious': (200, 200, 0),      # Yellow
    'thoughtful': (150, 0, 150),   # Purple
    'talking_open': (255, 255, 255),  # White
    'talking_closed': (200, 200, 200) # Light gray
}

# ============================================================================
# JSON STORAGE PATHS
# ============================================================================

MEMORY_DIR = os.path.join(os.path.dirname(__file__), 'memory')
PEOPLE_JSON = os.path.join(MEMORY_DIR, 'people.json')
INTERACTIONS_JSON = os.path.join(MEMORY_DIR, 'interactions.json')
SETTINGS_JSON = os.path.join(MEMORY_DIR, 'settings.json')
ARCHIVED_CONVERSATIONS_JSON = os.path.join(MEMORY_DIR, 'archived_conversations.json')

# Number of recent interactions to include in Moondream context
MAX_RECENT_INTERACTIONS = 3

# ============================================================================
# MOONDREAM API CONFIGURATION
# ============================================================================

# API endpoint (placeholder - update with real endpoint)
MOONDREAM_API_URL = os.environ.get('MOONDREAM_API_URL', 'http://localhost:11434/api/generate')
MOONDREAM_API_KEY = os.environ.get('MOONDREAM_API_KEY', '')
MOONDREAM_MODEL = os.environ.get('MOONDREAM_MODEL', 'moondream')

# Request timeout (seconds)
MOONDREAM_TIMEOUT = 30.0

# ============================================================================
# DISPLAY AND UI CONFIGURATION
# ============================================================================

# Window names
MAIN_WINDOW_NAME = "Magical Living Portrait"

# Debug mode (show detection overlays, logging, etc.)
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')

# Show camera feed alongside portrait (useful for debugging)
SHOW_CAMERA_FEED = DEBUG_MODE

# UI colors (BGR format for OpenCV)
COLOR_TEXT = (255, 255, 255)  # White
COLOR_SUBTITLE_BG = (0, 0, 0)  # Black
COLOR_DEBUG_INFO = (0, 255, 255)  # Yellow

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# ============================================================================
# THREADING CONFIGURATION
# ============================================================================

# Queue size for Moondream jobs
MOONDREAM_QUEUE_SIZE = 5

# Thread names
MOONDREAM_WORKER_THREAD_NAME = "MoondreamWorker"
VISION_LOOP_THREAD_NAME = "VisionLoop"
ANIMATION_LOOP_THREAD_NAME = "AnimationLoop"

# ============================================================================
# WEB SERVER CONFIGURATION (FLASK)
# ============================================================================

FLASK_HOST = '0.0.0.0'
FLASK_PORT = 8000
FLASK_DEBUG = False

# Template and static folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(SCRIPT_DIR, 'templates')
STATIC_FOLDER = os.path.join(SCRIPT_DIR, 'static')

# ============================================================================
# SYSTEM MESSAGES
# ============================================================================

STARTUP_MESSAGE = f"""
╔══════════════════════════════════════════════════════════════╗
║         MAGICAL LIVING PORTRAIT - Starting Up                ║
║──────────────────────────────────────────────────────────────║
║  Device Mode: {DEVICE_MODE:<48} ║
║  YOLO Model:  {YOLO_MODEL:<48} ║
║  Resolution:  {FRAME_WIDTH}x{FRAME_HEIGHT:<44} ║
║  Debug Mode:  {str(DEBUG_MODE):<48} ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_config():
    """Print current configuration"""
    print(STARTUP_MESSAGE)
