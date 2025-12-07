"""
Voice Manager
Handles text-to-speech and speech-to-text with punctuation buffering
"""
import threading
import time
import queue
import re
from typing import Optional, Callable, List

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Text-to-speech disabled.")

from models.models import VoiceSettings


class VoiceManager:
    """Manages voice input/output for the portrait"""
    
    def __init__(self, settings: Optional[VoiceSettings] = None):
        """
        Initialize voice manager
        
        Args:
            settings: Voice configuration settings
        """
        self.settings = settings or VoiceSettings()
        
        # TTS engine
        self.tts_engine = None
        self.tts_queue = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None
        self.tts_running = False
        
        # Speech buffer for punctuation-based chunking
        self.speech_buffer = ""
        self.buffer_lock = threading.Lock()
        
        # Current speech state
        self.is_speaking = False
        self.current_speaker = None  # Thread-safe flag
        
        # Event callbacks
        self.on_speak_start = None
        self.on_speak_end = None
        
        self._initialize_tts()
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return TTS_AVAILABLE
    
    def _initialize_tts(self):
        """Initialize TTS engine"""
        if not TTS_AVAILABLE:
            print("[Voice] TTS not available")
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            if self.settings.tts_voice:
                self.tts_engine.setProperty('voice', self.settings.tts_voice)
            
            self.tts_engine.setProperty('rate', self.settings.tts_rate)
            self.tts_engine.setProperty('volume', self.settings.tts_volume)
            
            # Get available voices for info
            voices = self.tts_engine.getProperty('voices')
            print(f"[Voice] TTS initialized with {len(voices)} voices available")
            print(f"[Voice] Rate: {self.settings.tts_rate} WPM, Volume: {self.settings.tts_volume}")
            
            # Start TTS worker thread
            self.tts_running = True
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
        except Exception as e:
            print(f"[Voice] Error initializing TTS: {e}")
            self.tts_engine = None
    
    def speak(self, text: str, use_buffer: bool = True):
        """
        Speak text using TTS
        
        Args:
            text: Text to speak
            use_buffer: If True, buffer until punctuation before speaking
        """
        if not self.tts_engine:
            print(f"[Voice] TTS unavailable - would say: {text}")
            return
        
        if use_buffer and self.settings.punctuation_buffer:
            self._add_to_buffer(text)
        else:
            # Speak immediately
            self.tts_queue.put(text)
    
    def _add_to_buffer(self, text: str):
        """
        Add text to speech buffer, speak when punctuation found
        
        Buffers text until sentence-ending punctuation (. ! ?)
        Then speaks the complete sentence
        """
        with self.buffer_lock:
            self.speech_buffer += text
            
            # Check for sentence-ending punctuation
            sentences = self._extract_complete_sentences(self.speech_buffer)
            
            if sentences:
                # Speak complete sentences
                for sentence in sentences:
                    self.tts_queue.put(sentence.strip())
                
                # Keep remaining incomplete text in buffer
                # Find where last complete sentence ended
                last_punct_idx = max(
                    self.speech_buffer.rfind('.'),
                    self.speech_buffer.rfind('!'),
                    self.speech_buffer.rfind('?')
                )
                
                if last_punct_idx >= 0:
                    self.speech_buffer = self.speech_buffer[last_punct_idx + 1:].lstrip()
                else:
                    self.speech_buffer = ""
    
    def _extract_complete_sentences(self, text: str) -> List[str]:
        """
        Extract complete sentences from text
        
        Returns list of complete sentences (with punctuation)
        """
        # Split on sentence-ending punctuation, keeping the punctuation
        pattern = r'([^.!?]*[.!?])'
        sentences = re.findall(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def flush_buffer(self):
        """Force speak any buffered text immediately"""
        with self.buffer_lock:
            if self.speech_buffer.strip():
                self.tts_queue.put(self.speech_buffer.strip())
                self.speech_buffer = ""
    
    def clear_buffer(self):
        """Clear speech buffer without speaking"""
        with self.buffer_lock:
            self.speech_buffer = ""
    
    def stop_speaking(self):
        """Stop current speech immediately"""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
            except Exception as e:
                print(f"[Voice] Error stopping speech: {e}")
    
    def clear_queue(self):
        """Clear all pending speech"""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        self.clear_buffer()
    
    def _tts_worker(self):
        """TTS worker thread - processes speech queue"""
        print("[Voice] TTS worker started")
        
        while self.tts_running:
            try:
                # Get next text to speak (with timeout)
                text = self.tts_queue.get(timeout=0.5)
                
                if text:
                    self.is_speaking = True
                    print(f"[Voice] Speaking: {text[:50]}...")
                    
                    # Notify that speaking started (for animation and echo cancellation)
                    if self.on_speak_start:
                        self.on_speak_start(text)
                    
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"[Voice] TTS error: {e}")
                    
                    self.is_speaking = False
                    
                    # Notify that speaking stopped
                    if self.on_speak_end:
                        self.on_speak_end()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Voice] TTS worker error: {e}")
        
        print("[Voice] TTS worker stopped")
    
    def shutdown(self):
        """Shutdown voice manager"""
        print("[Voice] Shutting down...")
        
        self.tts_running = False
        
        if self.tts_thread:
            self.tts_thread.join(timeout=2.0)
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        print("[Voice] Shutdown complete")
    
    def update_settings(self, settings: VoiceSettings):
        """Update voice settings"""
        self.settings = settings
        
        if self.tts_engine:
            try:
                if settings.tts_voice:
                    self.tts_engine.setProperty('voice', settings.tts_voice)
                self.tts_engine.setProperty('rate', settings.tts_rate)
                self.tts_engine.setProperty('volume', settings.tts_volume)
                print(f"[Voice] Settings updated")
            except Exception as e:
                print(f"[Voice] Error updating settings: {e}")
    
    def get_available_voices(self) -> List[dict]:
        """Get list of available TTS voices"""
        if not self.tts_engine:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            return [
                {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': getattr(voice, 'gender', 'unknown')
                }
                for voice in voices
            ]
        except Exception as e:
            print(f"[Voice] Error getting voices: {e}")
            return []
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.shutdown()


# ============================================================================
# STUB IMPLEMENTATION FOR TESTING
# ============================================================================

class StubVoiceManager:
    """Stub implementation when TTS not available"""
    
    def __init__(self, settings: Optional[VoiceSettings] = None):
        self.settings = settings or VoiceSettings()
        self.is_speaking = False
        self.speech_buffer = ""
        print("[Voice Stub] Initialized")
    
    def is_available(self) -> bool:
        return False
    
    def speak(self, text: str, use_buffer: bool = True):
        print(f"[Voice Stub] Would speak: {text}")
    
    def flush_buffer(self):
        print("[Voice Stub] Would flush buffer")
    
    def clear_buffer(self):
        self.speech_buffer = ""
    
    def stop_speaking(self):
        self.is_speaking = False
    
    def clear_queue(self):
        pass
    
    def shutdown(self):
        print("[Voice Stub] Shutdown")
    
    def update_settings(self, settings: VoiceSettings):
        self.settings = settings
    
    def get_available_voices(self) -> List[dict]:
        return []


def create_voice_manager(settings: Optional[VoiceSettings] = None,
                         use_stub: bool = False):
    """
    Factory function to create appropriate voice manager
    
    Args:
        settings: Voice configuration
        use_stub: Force stub implementation for testing
        
    Returns:
        VoiceManager or StubVoiceManager
    """
    if use_stub or not TTS_AVAILABLE:
        return StubVoiceManager(settings)
    else:
        return VoiceManager(settings)
