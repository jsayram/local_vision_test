#!/usr/bin/env python3
"""
MAGICAL LIVING PORTRAIT
A living portrait that watches, remembers, and speaks using:
- OpenCV for camera and rendering
- YOLO for fast person detection
- Moondream for vision-language reasoning
- Sprite-based animation system
- JSON local memory storage

Architecture:
- Vision Loop: Camera → YOLO → Event Detection → Queue Moondream jobs
- Moondream Worker: Background thread processing vision-language calls
- Animation Loop: Render portrait based on state (separate from vision)
- Flask Web Server: Serve web UI and video stream

Author: Jose Ramirez
Date: December 2025
"""

import cv2
import time
import threading
from queue import Queue, Empty
from typing import Optional
import signal
import sys

# Import all our modules
from core import config
from models.models import PersonState, AnimationState, EventType
from detectors.yolo_detector import YOLODetector
from core import detector
from core import moondream_client
from core import animator
from core import storage

# ============================================================================
# GLOBAL STATE (Thread-safe with locks where needed)
# ============================================================================

# Camera and detection
camera = None
yolo = None
person_state = PersonState()
person_state_lock = threading.Lock()

# Animation state
animation_state = AnimationState()
animation_state_lock = threading.Lock()

# Moondream job queue
moondream_queue = Queue(maxsize=config.MOONDREAM_QUEUE_SIZE)

# Control flags
running = True
paused = False
debug_log_enabled = config.DEBUG_MODE

# Current camera index
current_camera_index = config.CAMERA_INDEX

# Frame counters
frame_count = 0

# ============================================================================
# MOONDREAM WORKER THREAD
# ============================================================================

def moondream_worker():
    """
    Background worker thread that processes Moondream jobs from queue
    Runs continuously until program exits
    """
    global animation_state, person_state
    
    print(f"🌙 Moondream worker thread started")
    
    while running:
        try:
            # Block waiting for job (with timeout to check running flag)
            job = moondream_queue.get(timeout=1.0)
            
            if job is None:  # Poison pill to stop thread
                break
            
            # Unpack job
            face_image, context, event_type = job
            
            if config.DEBUG_MODE:
                print(f"🌙 Processing Moondream job: {event_type}")
            
            # Call Moondream (this is the slow part)
            # Use stub if DEBUG_MODE for faster testing
            result = moondream_client.call_moondream(
                face_image, 
                context, 
                use_stub=(config.DEBUG_MODE and not config.MOONDREAM_API_URL)
            )
            
            if result is None:
                print("✗ Moondream call failed")
                continue
            
            # Update animation state (thread-safe)
            with animation_state_lock:
                now = time.time()
                animation_state.update_from_moondream(
                    result, 
                    now,
                    config.SPEAKING_DURATION,
                    config.SUBTITLE_DURATION
                )
            
            # Save interaction to memory
            with person_state_lock:
                person_id = person_state.person_id
            
            storage.create_interaction_record(
                person_id=person_id,
                mood=result.mood,
                text=result.text,
                event_type=event_type
            )
            
            # Update person last_seen if known
            if person_id is not None:
                storage.update_person_last_seen(person_id)
            
            print(f"✓ Portrait says [{result.mood}]: {result.text}")
            
        except Empty:
            # Timeout - just continue
            continue
        except Exception as e:
            print(f"✗ Error in Moondream worker: {e}")
            import traceback
            traceback.print_exc()
    
    print("🌙 Moondream worker thread stopped")


# ============================================================================
# VISION DETECTION LOOP
# ============================================================================

def vision_loop():
    """
    Main vision loop:
    - Read camera frames
    - Run YOLO detection (every Nth frame)
    - Detect events (NEW_PERSON, POSE_CHANGED, PERIODIC_UPDATE)
    - Queue Moondream jobs when events occur
    """
    global camera, yolo, person_state, frame_count
    
    print("👁  Vision loop started")
    
    # Initialize YOLO detector
    yolo = YOLODetector()
    
    # Initialize camera
    camera = detector.initialize_camera(current_camera_index)
    if camera is None:
        print("✗ Failed to initialize camera - vision loop cannot start")
        return
    
    frame_skip_counter = 0
    
    while running:
        if paused:
            time.sleep(0.1)
            continue
        
        # Read frame
        ret, frame = camera.read()
        if not ret:
            print("✗ Failed to read camera frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        now = time.time()
        
        # Run detection every Nth frame (performance optimization)
        if frame_skip_counter % config.DETECTION_SKIP_FRAMES == 0:
            # Run YOLO
            detections = detector.run_yolo_on_frame(frame, yolo)
            
            # Find best person detection
            best_person = detector.find_best_person_detection(detections)
            
            # Detect events
            with person_state_lock:
                event = detector.detect_event_from_person_state(
                    person_state,
                    best_person,
                    now
                )
                
                # Update person state
                if event is not None:
                    # Event occurred - state is in event.person_state
                    person_state = event.person_state
                    person_state.last_moondream_call = now  # Mark that we're calling
                else:
                    # No event - just update state
                    person_state = detector.update_person_state(
                        person_state,
                        best_person,
                        now
                    )
            
            # If event occurred, queue Moondream job
            if event is not None and best_person is not None:
                # Crop face/person for Moondream
                face_crop = detector.crop_person_for_moondream(frame, best_person)
                
                # Build context
                with person_state_lock:
                    pid = person_state.person_id
                
                context = moondream_client.build_context_for_person(
                    person_id=pid,
                    event_type=event.event_type.value
                )
                
                # Queue job (non-blocking - drop if queue full)
                try:
                    moondream_queue.put_nowait((face_crop, context, event.event_type.value))
                    if debug_log_enabled:
                        print(f"  ➡ Queued Moondream job for {event.event_type.value}")
                except:
                    if debug_log_enabled:
                        print(f"  ⚠ Moondream queue full - skipping")
        
        frame_skip_counter += 1
        
        # Small sleep to prevent CPU overload
        time.sleep(0.01)
    
    # Cleanup
    if camera is not None:
        camera.release()
    
    print("👁  Vision loop stopped")


# ============================================================================
# ANIMATION/RENDER LOOP
# ============================================================================

def animation_loop():
    """
    Animation loop:
    - Update animation state (mouth movement, timers)
    - Render portrait
    - Display via cv2.imshow
    - Handle keyboard input
    """
    global animation_state, running, paused, debug_log_enabled
    
    print("🎨 Animation loop started")
    
    # Create window
    cv2.namedWindow(config.MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    while running:
        now = time.time()
        
        # Render portrait (thread-safe)
        with animation_state_lock:
            canvas = animator.update_and_render(animation_state, now)
        
        # Show window
        cv2.imshow(config.MAIN_WINDOW_NAME, canvas)
        
        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF  # 30ms = ~33 FPS
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("User requested quit")
            running = False
        elif key == ord('p'):  # 'p' = pause/resume
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"  {status}")
        elif key == ord('d'):  # 'd' = toggle debug
            debug_log_enabled = not debug_log_enabled
            print(f"  Debug logging: {debug_log_enabled}")
        elif key == ord('r'):  # 'r' = reset animation
            with animation_state_lock:
                animation_state = AnimationState()
            print("  Animation state reset")
    
    cv2.destroyAllWindows()
    print("🎨 Animation loop stopped")


# ============================================================================
# CAMERA SELECTION
# ============================================================================

def select_camera():
    """Interactive camera selection at startup"""
    global current_camera_index
    
    print("\n" + "="*60)
    print("CAMERA SELECTION")
    print("="*60)
    
    available = detector.list_available_cameras(max_test=5)
    
    if not available:
        print("✗ No cameras found!")
        print(f"  Using default index {config.CAMERA_INDEX}")
        return
    
    print(f"Available cameras: {available}")
    
    # Auto-select if only one
    if len(available) == 1:
        current_camera_index = available[0]
        print(f"✓ Auto-selected camera {current_camera_index}")
        return
    
    # Let user choose
    while True:
        try:
            choice = input(f"Select camera index {available}: ")
            choice = int(choice)
            if choice in available:
                current_camera_index = choice
                print(f"✓ Selected camera {current_camera_index}")
                break
            else:
                print(f"  Invalid choice - must be one of {available}")
        except ValueError:
            print("  Invalid input - enter a number")
        except KeyboardInterrupt:
            print("\n  Using default")
            break


# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

def startup():
    """Initialize system on startup"""
    # Print config
    config.print_config()
    
    # Initialize storage
    print("\nInitializing storage...")
    storage.initialize_storage()
    
    # Load settings and update config if needed
    settings = storage.load_settings()
    
    # Camera selection
    if not config.DEBUG_MODE:  # Skip in debug mode for faster startup
        select_camera()
    
    print("\n" + "="*60)
    print("SYSTEM READY")
    print("="*60)
    print("Controls:")
    print("  Q or ESC - Quit")
    print("  P - Pause/Resume detection")
    print("  D - Toggle debug logging")
    print("  R - Reset animation state")
    print("="*60 + "\n")


def shutdown(signum=None, frame=None):
    """Clean shutdown"""
    global running
    
    print("\n" + "="*60)
    print("SHUTTING DOWN...")
    print("="*60)
    
    running = False
    
    # Stop Moondream worker
    try:
        moondream_queue.put(None, timeout=1.0)  # Poison pill
    except:
        pass
    
    # Give threads time to finish
    time.sleep(0.5)
    
    # Release camera
    global camera
    if camera is not None:
        camera.release()
    
    cv2.destroyAllWindows()
    
    # Print stats
    stats = storage.get_storage_stats()
    print(f"\nSession stats:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  People in memory: {stats['people_count']}")
    print(f"  Interactions recorded: {stats['interactions_count']}")
    print(f"  Memory size: {stats['total_size_bytes']} bytes")
    
    print("\n✓ Shutdown complete\n")
    sys.exit(0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Setup signal handlers for clean shutdown
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Startup
    startup()
    
    # Start background threads
    moondream_thread = threading.Thread(
        target=moondream_worker,
        name=config.MOONDREAM_WORKER_THREAD_NAME,
        daemon=True
    )
    moondream_thread.start()
    
    vision_thread = threading.Thread(
        target=vision_loop,
        name=config.VISION_LOOP_THREAD_NAME,
        daemon=True
    )
    vision_thread.start()
    
    # Run animation loop in main thread (needs to handle cv2.imshow)
    try:
        animation_loop()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
