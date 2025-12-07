"""
Offline Speech Recognition using Vosk
100% offline, no internet required
"""
import threading
import json
import time
from typing import Optional, Callable

try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Warning: vosk or pyaudio not installed. Offline speech recognition disabled.")


class OfflineSpeechRecognizer:
    """Offline speech recognition using Vosk"""
    
    def __init__(self, 
                 model_path: str = "models/vosk-en",
                 on_partial: Optional[Callable] = None,
                 on_final: Optional[Callable] = None,
                 on_error: Optional[Callable] = None,
                 input_device_index: Optional[int] = None):
        """
        Initialize offline speech recognizer
        
        Args:
            model_path: Path to Vosk model directory
            on_partial: Callback for interim results (partial transcription)
            on_final: Callback for final results
            on_error: Callback for errors (e.g., no audio detected)
            input_device_index: Specific microphone device index (None = default)
        """
        self.model_path = model_path
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.input_device_index = input_device_index
        
        self.model = None
        self.recognizer = None
        self.pa = None
        self.stream = None
        
        self.is_listening = False
        self.listen_thread: Optional[threading.Thread] = None
        
        # Echo cancellation: pause recognition when TTS is playing
        self._is_tts_playing = False
        self._tts_lock = threading.Lock()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 8000
    
    def set_tts_playing(self, is_playing: bool):
        """Set whether TTS is currently playing (for echo cancellation)"""
        with self._tts_lock:
            self._is_tts_playing = is_playing
            if is_playing:
                print("[Vosk] 🔇 TTS started - ignoring microphone input (echo cancellation)")
            else:
                print("[Vosk] 🎤 TTS ended - resuming microphone input")
    
    def is_tts_playing(self) -> bool:
        """Check if TTS is currently playing"""
        with self._tts_lock:
            return self._is_tts_playing
        
    def is_available(self) -> bool:
        """Check if Vosk is available"""
        return VOSK_AVAILABLE
    
    def initialize(self) -> bool:
        """Initialize Vosk model and audio"""
        if not VOSK_AVAILABLE:
            print("[Vosk] Not available - missing dependencies")
            return False
        
        try:
            import os
            model_full_path = os.path.join(os.path.dirname(__file__), '..', self.model_path)
            
            print(f"[Vosk] Loading model from: {model_full_path}")
            self.model = Model(model_full_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # Get word-level timestamps
            self.recognizer.SetMaxAlternatives(0)  # Disable alternatives for simpler output
            # Note: Vosk automatically handles silence detection
            
            print("[Vosk] ✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[Vosk] ✗ Error loading model: {e}")
            print(f"[Vosk] Make sure model is downloaded to {self.model_path}")
            return False
    
    def start(self) -> bool:
        """Start listening to microphone"""
        if not self.model:
            if not self.initialize():
                return False
        
        if self.is_listening:
            print("[Vosk] Already listening")
            return True
        
        try:
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # List available input devices
            print(f"[Vosk] Looking for microphone...")
            device_index = self.input_device_index
            
            if device_index is None:
                # Find default input device
                try:
                    default_device = self.pa.get_default_input_device_info()
                    device_index = default_device['index']
                    print(f"[Vosk] Using default mic: {default_device['name']}")
                except Exception as e:
                    print(f"[Vosk] Warning: Could not get default device: {e}")
                    # Try device 0
                    device_index = 0
            
            print(f"[Vosk] Opening audio stream (device={device_index}, rate={self.sample_rate}, chunk={self.chunk_size})")
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            print(f"[Vosk] ✓ Audio stream opened successfully")
            
            # Start listening thread
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
            
            print("[Vosk] ✓ Started listening")
            return True
            
        except Exception as e:
            print(f"[Vosk] ✗ Error starting: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """Stop listening"""
        print("[Vosk] Stopping...")
        self.is_listening = False
        
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        
        self._cleanup()
        print("[Vosk] Stopped")
    
    def _listen_loop(self):
        """Main listening loop (runs in thread)"""
        print("[Vosk] Listen loop started")
        
        frame_count = 0
        last_log_time = time.time()
        silent_start_time = time.time()
        total_silent_time = 0
        silence_timeout = 10.0  # Stop after 10 seconds of complete silence
        
        try:
            import numpy as np
            
            while self.is_listening:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                frame_count += 1
                
                # Calculate audio level for debugging
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_level = np.abs(audio_data).mean()
                
                # Track silence duration
                if audio_level < 100:
                    total_silent_time = time.time() - silent_start_time
                    
                    # Auto-stop if microphone is completely silent for too long
                    if total_silent_time > silence_timeout:
                        print(f"\n[Vosk] ❌ MICROPHONE ERROR: No audio detected for {silence_timeout:.0f} seconds")
                        print(f"[Vosk] This indicates microphone permissions are blocked.")
                        print(f"[Vosk] ")
                        print(f"[Vosk] SOLUTION:")
                        print(f"[Vosk] 1. Open System Preferences → Security & Privacy → Privacy → Microphone")
                        print(f"[Vosk] 2. Grant permission to Terminal/Python")
                        print(f"[Vosk] 3. Refresh the webpage")
                        print(f"[Vosk] 4. Click 'Start Speaking' again")
                        print(f"[Vosk] ")
                        print(f"[Vosk] Auto-stopping speech recognition...\n")
                        
                        # Notify frontend about the error
                        if hasattr(self, 'on_error'):
                            self.on_error("Microphone permissions required. Please grant access and refresh.")
                        
                        self.is_listening = False
                        break
                else:
                    # Reset silence timer when audio is detected
                    silent_start_time = time.time()
                    total_silent_time = 0
                
                # Log audio capture every 2 seconds
                if time.time() - last_log_time > 2.0:
                    print(f"[Vosk] Processing audio... ({frame_count} frames, level: {audio_level:.0f})")
                    if audio_level < 100:
                        print(f"[Vosk] ⚠️  Audio level very low! ({total_silent_time:.1f}s silent, will auto-stop at {silence_timeout:.0f}s)")
                        print(f"[Vosk] → Check: System Preferences → Security & Privacy → Privacy → Microphone")
                    last_log_time = time.time()
                    frame_count = 0
                
                if self.recognizer.AcceptWaveform(data):
                    # Final result (sentence complete)
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    # Skip if TTS is playing (echo cancellation)
                    if self.is_tts_playing():
                        if text:
                            print(f"[Vosk] 🔇 Ignored (TTS playing): \"{text}\"")
                        continue
                    
                    if text:
                        print(f"[Vosk] ✓ FINAL: \"{text}\"")
                        if self.on_final:
                            self.on_final(text, result)
                else:
                    # Skip partial results if TTS is playing
                    if self.is_tts_playing():
                        continue
                        
                    # Partial result (interim transcription)
                    result = json.loads(self.recognizer.PartialResult())
                    partial = result.get('partial', '').strip()
                    
                    if partial:
                        # Only log every ~500ms to avoid spam
                        if not hasattr(self, '_last_partial') or partial != self._last_partial:
                            print(f"[Vosk] ... interim: \"{partial}\"")
                            self._last_partial = partial
                            if self.on_partial:
                                self.on_partial(partial, result)
                            print(f"[Vosk] ... interim: \"{partial}\"")
                            self._last_partial = partial
                            
                        if self.on_partial:
                            self.on_partial(partial, result)
        
        except Exception as e:
            if self.is_listening:  # Only log if unexpected
                print(f"[Vosk] Listen loop error: {e}")
        
        finally:
            print("[Vosk] Listen loop ended")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None
        
        # Note: Don't delete model, we'll reuse it
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.stop()


class StubOfflineSpeechRecognizer:
    """Stub for testing when Vosk not available"""
    
    def __init__(self, model_path=None, on_partial=None, on_final=None, on_error=None):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.is_listening = False
        self._is_tts_playing = False
        print("[Vosk Stub] Initialized (offline speech recognition disabled)")
    
    def set_tts_playing(self, is_playing: bool):
        """Stub for TTS playing state"""
        self._is_tts_playing = is_playing
    
    def is_tts_playing(self) -> bool:
        """Check if TTS is playing"""
        return self._is_tts_playing
    
    def is_available(self):
        return False
    
    def initialize(self):
        return False
    
    def start(self):
        print("[Vosk Stub] Would start listening...")
        return False
    
    def stop(self):
        print("[Vosk Stub] Would stop listening")


def create_offline_speech_recognizer(model_path="models/vosk-en",
                                     on_partial=None,
                                     on_final=None,
                                     on_error=None,
                                     use_stub=False):
    """Factory function for speech recognizer"""
    if use_stub or not VOSK_AVAILABLE:
        return StubOfflineSpeechRecognizer(model_path, on_partial, on_final, on_error)
    else:
        return OfflineSpeechRecognizer(model_path, on_partial, on_final, on_error)
