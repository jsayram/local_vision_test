"""
Wake Word Listener
Listens for "hey portrait" using Porcupine wake word detection
"""
import threading
import time
from typing import Optional, Callable
import struct

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    print("Warning: pvporcupine not installed. Wake word detection disabled.")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: pyaudio not installed. Audio input disabled.")


class WakeWordListener:
    """Listens for wake word to activate voice input"""
    
    def __init__(self, 
                 wake_word: str = "hey portrait",
                 sensitivity: float = 0.5,
                 on_wake_word: Optional[Callable] = None):
        """
        Initialize wake word listener
        
        Args:
            wake_word: Wake word to listen for (limited by Porcupine models)
            sensitivity: Detection sensitivity 0.0-1.0 (higher = more sensitive)
            on_wake_word: Callback function when wake word detected
        """
        self.wake_word = wake_word
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.on_wake_word = on_wake_word
        
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self.audio_stream = None
        self.pa = None
        
        self.is_listening = False
        self.listen_thread: Optional[threading.Thread] = None
        
        self._enabled = False
    
    def is_available(self) -> bool:
        """Check if wake word detection is available"""
        return PORCUPINE_AVAILABLE and PYAUDIO_AVAILABLE
    
    def start(self) -> bool:
        """Start listening for wake word"""
        if not self.is_available():
            print("[Wake Word] Not available - missing dependencies")
            return False
        
        if self.is_listening:
            print("[Wake Word] Already listening")
            return True
        
        try:
            # Initialize Porcupine
            # Note: Using built-in keywords. For custom "hey portrait", 
            # you'd need to train a custom model or use a similar keyword
            # Available keywords: alexa, americano, blueberry, bumblebee, computer,
            # grapefruit, grasshopper, hey google, hey siri, jarvis, ok google, 
            # picovoice, porcupine, terminator
            
            # Map our wake word to available Porcupine keyword
            keyword = self._map_wake_word(self.wake_word)
            
            self.porcupine = pvporcupine.create(
                keywords=[keyword],
                sensitivities=[self.sensitivity]
            )
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            # Start listening thread
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
            
            print(f"[Wake Word] Started listening for '{keyword}' (mapped from '{self.wake_word}')")
            print(f"[Wake Word] Sensitivity: {self.sensitivity}")
            return True
            
        except Exception as e:
            print(f"[Wake Word] Error starting: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """Stop listening for wake word"""
        print("[Wake Word] Stopping...")
        self.is_listening = False
        
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        
        self._cleanup()
        print("[Wake Word] Stopped")
    
    def _map_wake_word(self, wake_word: str) -> str:
        """
        Map custom wake word to available Porcupine keyword
        
        For production, you'd train a custom model for "hey portrait"
        For now, map to closest available keyword
        """
        wake_word_lower = wake_word.lower()
        
        # Direct matches
        keyword_map = {
            'hey portrait': 'porcupine',  # Closest available
            'hey google': 'hey google',
            'ok google': 'ok google',
            'hey siri': 'hey siri',
            'alexa': 'alexa',
            'computer': 'computer',
            'jarvis': 'jarvis',
        }
        
        # Check for direct match
        if wake_word_lower in keyword_map:
            return keyword_map[wake_word_lower]
        
        # Default to 'porcupine' for portrait-related phrases
        if 'portrait' in wake_word_lower or 'picture' in wake_word_lower:
            return 'porcupine'
        
        # Default fallback
        return 'porcupine'
    
    def _listen_loop(self):
        """Main listening loop (runs in thread)"""
        print("[Wake Word] Listen loop started")
        
        try:
            while self.is_listening:
                # Read audio frame
                pcm = self.audio_stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Check for wake word
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print(f"[Wake Word] Detected! (index: {keyword_index})")
                    self._handle_wake_word()
                
        except Exception as e:
            if self.is_listening:  # Only log if unexpected
                print(f"[Wake Word] Listen loop error: {e}")
        
        finally:
            print("[Wake Word] Listen loop ended")
    
    def _handle_wake_word(self):
        """Handle wake word detection"""
        if self.on_wake_word:
            try:
                self.on_wake_word()
            except Exception as e:
                print(f"[Wake Word] Error in callback: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None
        
        if self.porcupine:
            try:
                self.porcupine.delete()
            except:
                pass
            self.porcupine = None
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.stop()


# ============================================================================
# STUB IMPLEMENTATION FOR TESTING
# ============================================================================

class StubWakeWordListener:
    """Stub implementation when Porcupine not available"""
    
    def __init__(self, wake_word: str = "hey portrait", 
                 sensitivity: float = 0.5,
                 on_wake_word: Optional[Callable] = None):
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.on_wake_word = on_wake_word
        self.is_listening = False
        print(f"[Wake Word Stub] Initialized (wake word: '{wake_word}')")
    
    def is_available(self) -> bool:
        return False
    
    def start(self) -> bool:
        self.is_listening = True
        print("[Wake Word Stub] Pretending to listen...")
        return True
    
    def stop(self):
        self.is_listening = False
        print("[Wake Word Stub] Stopped")
    
    def trigger_manually(self):
        """Manual trigger for testing"""
        print("[Wake Word Stub] Manual trigger!")
        if self.on_wake_word:
            self.on_wake_word()


def create_wake_word_listener(wake_word: str = "hey portrait",
                               sensitivity: float = 0.5,
                               on_wake_word: Optional[Callable] = None,
                               use_stub: bool = False):
    """
    Factory function to create appropriate wake word listener
    
    Args:
        wake_word: Wake word to listen for
        sensitivity: Detection sensitivity
        on_wake_word: Callback when wake word detected
        use_stub: Force stub implementation for testing
        
    Returns:
        WakeWordListener or StubWakeWordListener
    """
    if use_stub or not (PORCUPINE_AVAILABLE and PYAUDIO_AVAILABLE):
        return StubWakeWordListener(wake_word, sensitivity, on_wake_word)
    else:
        return WakeWordListener(wake_word, sensitivity, on_wake_word)
