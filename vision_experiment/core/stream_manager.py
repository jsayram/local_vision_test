"""
Stream Manager
Handles word-by-word streaming of Moondream responses via WebSocket
"""
import time
import queue
import threading
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """Single chunk of streamed content"""
    text: str  # Word or text fragment
    is_complete: bool = False  # True if this is the final chunk
    metadata: Optional[Dict[str, Any]] = None  # Additional info (mood, etc.)


class StreamManager:
    """Manages streaming responses to clients"""
    
    def __init__(self, 
                 on_chunk: Optional[Callable[[StreamChunk], None]] = None,
                 words_per_second: float = 5.0):
        """
        Initialize stream manager
        
        Args:
            on_chunk: Callback for each streamed chunk
            words_per_second: Streaming speed (higher = faster)
        """
        self.on_chunk = on_chunk
        self.words_per_second = words_per_second
        self.chunk_delay = 1.0 / words_per_second
        
        # Stream queue and worker
        self.stream_queue = queue.Queue()
        self.stream_thread: Optional[threading.Thread] = None
        self.is_streaming = False
        
        # Current stream state
        self.current_stream_id: Optional[str] = None
        self.is_active = False
    
    def start(self):
        """Start the stream manager"""
        if self.is_active:
            return
        
        self.is_active = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        print("[Stream Manager] Started")
    
    def stop(self):
        """Stop the stream manager"""
        self.is_active = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        print("[Stream Manager] Stopped")
    
    def stream_text(self, text: str, stream_id: str, mood: str = "idle", 
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Stream text word-by-word
        
        Args:
            text: Full text to stream
            stream_id: Unique ID for this stream
            mood: Portrait mood
            metadata: Additional metadata to include
        """
        if not self.is_active:
            print("[Stream Manager] Not active - starting...")
            self.start()
        
        # Cancel any existing stream
        self.cancel_stream()
        
        self.current_stream_id = stream_id
        self.is_streaming = True
        
        # Queue the text for streaming
        stream_data = {
            'text': text,
            'stream_id': stream_id,
            'mood': mood,
            'metadata': metadata or {}
        }
        
        self.stream_queue.put(stream_data)
        print(f"[Stream Manager] Queued stream: {stream_id}")
    
    def cancel_stream(self):
        """Cancel current stream"""
        if self.is_streaming:
            self.is_streaming = False
            # Clear queue
            while not self.stream_queue.empty():
                try:
                    self.stream_queue.get_nowait()
                except queue.Empty:
                    break
            print("[Stream Manager] Stream cancelled")
    
    def _stream_worker(self):
        """Worker thread that processes stream queue"""
        print("[Stream Manager] Worker started")
        
        while self.is_active:
            try:
                # Get next text to stream
                stream_data = self.stream_queue.get(timeout=0.5)
                
                text = stream_data['text']
                stream_id = stream_data['stream_id']
                mood = stream_data['mood']
                metadata = stream_data['metadata']
                
                print(f"[Stream Manager] Streaming: {text[:50]}...")
                
                # Split into words
                words = text.split()
                
                for i, word in enumerate(words):
                    if not self.is_streaming or self.current_stream_id != stream_id:
                        print("[Stream Manager] Stream interrupted")
                        break
                    
                    is_last = (i == len(words) - 1)
                    
                    # Create chunk
                    chunk = StreamChunk(
                        text=word + (" " if not is_last else ""),
                        is_complete=is_last,
                        metadata={
                            'mood': mood,
                            'stream_id': stream_id,
                            'word_index': i,
                            'total_words': len(words),
                            **metadata
                        }
                    )
                    
                    # Send chunk via callback
                    if self.on_chunk:
                        try:
                            self.on_chunk(chunk)
                        except Exception as e:
                            print(f"[Stream Manager] Callback error: {e}")
                    
                    # Delay before next word (unless it's the last one)
                    if not is_last:
                        time.sleep(self.chunk_delay)
                
                # Stream complete
                self.is_streaming = False
                print(f"[Stream Manager] Stream complete: {stream_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Stream Manager] Worker error: {e}")
                self.is_streaming = False
        
        print("[Stream Manager] Worker stopped")
    
    def is_currently_streaming(self) -> bool:
        """Check if currently streaming"""
        return self.is_streaming
    
    def set_speed(self, words_per_second: float):
        """Update streaming speed"""
        self.words_per_second = max(1.0, min(20.0, words_per_second))
        self.chunk_delay = 1.0 / self.words_per_second
        print(f"[Stream Manager] Speed set to {self.words_per_second} WPS")
    
    def __del__(self):
        """Ensure cleanup"""
        self.stop()


# ============================================================================
# SENTENCE BUFFER FOR TTS
# ============================================================================

class SentenceBuffer:
    """
    Buffers streamed words and sends complete sentences to TTS
    Implements punctuation-based chunking for voice output
    """
    
    def __init__(self, on_sentence: Optional[Callable[[str], None]] = None):
        """
        Args:
            on_sentence: Callback when complete sentence is ready
        """
        self.on_sentence = on_sentence
        self.buffer = ""
    
    def add_word(self, word: str):
        """Add word to buffer, trigger callback on complete sentence"""
        self.buffer += word
        
        # Check if buffer contains sentence-ending punctuation
        if any(p in word for p in ['.', '!', '?']):
            # Extract complete sentence(s)
            sentences = self._extract_sentences(self.buffer)
            
            for sentence in sentences:
                if self.on_sentence:
                    self.on_sentence(sentence)
            
            # Keep any remaining text
            last_punct = max(
                self.buffer.rfind('.'),
                self.buffer.rfind('!'),
                self.buffer.rfind('?')
            )
            
            if last_punct >= 0:
                self.buffer = self.buffer[last_punct + 1:].lstrip()
            else:
                self.buffer = ""
    
    def _extract_sentences(self, text: str) -> list[str]:
        """Extract complete sentences from buffer"""
        import re
        pattern = r'([^.!?]*[.!?])'
        sentences = re.findall(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def flush(self):
        """Send any remaining buffered text"""
        if self.buffer.strip() and self.on_sentence:
            self.on_sentence(self.buffer.strip())
            self.buffer = ""
    
    def clear(self):
        """Clear buffer without sending"""
        self.buffer = ""


# ============================================================================
# TYPING INDICATOR
# ============================================================================

class TypingIndicator:
    """Manages typing indicator state for UI"""
    
    def __init__(self):
        self.is_typing = False
        self.typing_start: Optional[float] = None
    
    def start_typing(self):
        """Start showing typing indicator"""
        if not self.is_typing:
            self.is_typing = True
            self.typing_start = time.time()
    
    def stop_typing(self):
        """Stop showing typing indicator"""
        self.is_typing = False
        self.typing_start = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current typing indicator state"""
        return {
            'is_typing': self.is_typing,
            'duration': time.time() - self.typing_start if self.typing_start else 0
        }
