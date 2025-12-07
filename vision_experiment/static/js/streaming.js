/**
 * Streaming Response Handler
 * Displays word-by-word streaming from Moondream
 */

class StreamingDisplay {
    constructor(socketManager, voiceInterface) {
        this.socket = socketManager;
        this.voice = voiceInterface;
        this.streamContainer = document.getElementById('stream-output');
        this.currentStreamId = null;
        this.streamBuffer = '';
        this.sentenceBuffer = '';
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.socket.on('stream_chunk', (data) => this.handleStreamChunk(data));
        this.socket.on('stream_complete', (data) => this.handleStreamComplete(data));
    }
    
    handleStreamChunk(data) {
        // data: {text, is_complete, metadata: {mood, stream_id, word_index, total_words}}
        
        if (data.metadata.stream_id !== this.currentStreamId) {
            // New stream, clear display
            this.startNewStream(data.metadata.stream_id);
        }
        
        // Add word to buffer and display
        this.streamBuffer += data.text;
        this.updateDisplay(this.streamBuffer, data.metadata.mood);
        
        // Buffer for TTS (wait for sentence endings)
        this.sentenceBuffer += data.text;
        
        // Check for sentence-ending punctuation
        if (this.hasSentenceEnding(data.text)) {
            // Send complete sentence to TTS
            const sentence = this.sentenceBuffer.trim();
            if (sentence) {
                console.log(`[Streaming] 🔊 Sending to TTS: "${sentence}"`);
                this.socket.send('tts_speak', {text: sentence});
            }
            this.sentenceBuffer = '';
        }
        
        // If this is the last chunk, flush any remaining text
        if (data.is_complete && this.sentenceBuffer.trim()) {
            console.log(`[Streaming] 🔊 Sending final to TTS: "${this.sentenceBuffer.trim()}"`);
            this.socket.send('tts_speak', {text: this.sentenceBuffer.trim()});
            this.sentenceBuffer = '';
        }
    }
    
    handleStreamComplete(data) {
        // data: {stream_id, full_text, mood}
        console.log('[Streaming] Stream complete:', data.stream_id);
        
        // Final update to display
        this.updateDisplay(data.full_text, data.mood);
        
        // Add to chat history
        if (window.chatInterface) {
            window.chatInterface.addMessage('portrait', data.full_text, false, data.mood);
        }
        
        // Clear stream buffer
        setTimeout(() => {
            this.clearStream();
        }, 5000);  // Keep visible for 5 seconds
    }
    
    startNewStream(streamId) {
        this.currentStreamId = streamId;
        this.streamBuffer = '';
        this.sentenceBuffer = '';
        
        if (this.streamContainer) {
            this.streamContainer.classList.add('streaming');
        }
        
        console.log('[Streaming] Started stream:', streamId);
    }
    
    updateDisplay(text, mood = 'idle') {
        if (!this.streamContainer) return;
        
        // Update text content
        const textElement = this.streamContainer.querySelector('.stream-text');
        if (textElement) {
            textElement.textContent = text;
        }
        
        // Update mood indicator
        const moodElement = this.streamContainer.querySelector('.stream-mood');
        if (moodElement) {
            moodElement.textContent = mood;
            moodElement.className = `stream-mood mood-${mood}`;
        }
    }
    
    clearStream() {
        this.currentStreamId = null;
        this.streamBuffer = '';
        
        if (this.streamContainer) {
            this.streamContainer.classList.remove('streaming');
            
            const textElement = this.streamContainer.querySelector('.stream-text');
            if (textElement) {
                textElement.textContent = '';
            }
        }
    }
    
    hasSentenceEnding(text) {
        return /[.!?]\s*$/.test(text);
    }
}
