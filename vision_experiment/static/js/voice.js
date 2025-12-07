/**
 * Voice Interaction
 * Handles wake word detection, speech recognition, and TTS coordination
 */

class VoiceInterface {
    constructor(socketManager) {
        this.socket = socketManager;
        this.isListening = false;
        this.isWakeWordActive = false;
        this.recognition = null;
        
        // UI elements
        this.wakeWordButton = document.getElementById('wake-word-toggle');
        this.micIcon = document.getElementById('mic-icon');
        this.voiceStatus = document.getElementById('voice-status');
        this.confirmationModal = document.getElementById('confirmation-modal');
        
        this.initializeSpeechRecognition();
        this.initializeEventListeners();
    }
    
    initializeSpeechRecognition() {
        // Check if browser supports Web Speech API
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('[Voice] Speech recognition not supported');
            this.showStatus('Voice input not supported in this browser', 'warning');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            console.log('[Voice] Recognized:', transcript);
            this.handleSpeechResult(transcript);
        };
        
        this.recognition.onerror = (event) => {
            console.error('[Voice] Recognition error:', event.error);
            this.stopListening();
        };
        
        this.recognition.onend = () => {
            console.log('[Voice] Recognition ended');
            this.stopListening();
        };
        
        console.log('[Voice] Speech recognition initialized');
    }
    
    initializeEventListeners() {
        // Wake word toggle
        this.wakeWordButton?.addEventListener('click', () => this.toggleWakeWord());
        
        // Socket events
        this.socket.on('wake_word_detected', () => this.handleWakeWord());
        this.socket.on('face_confirmation_request', (data) => this.showConfirmationModal(data));
        this.socket.on('tts_started', () => this.handleTTSStarted());
        this.socket.on('tts_finished', () => this.handleTTSFinished());
        
        // Confirmation modal buttons
        document.getElementById('confirm-yes')?.addEventListener('click', () => {
            this.sendConfirmation(true);
        });
        
        document.getElementById('confirm-no')?.addEventListener('click', () => {
            this.sendConfirmation(false);
        });
    }
    
    toggleWakeWord() {
        if (this.isWakeWordActive) {
            this.stopWakeWord();
        } else {
            this.startWakeWord();
        }
    }
    
    startWakeWord() {
        this.socket.emit('start_wake_word');
        this.isWakeWordActive = true;
        
        this.wakeWordButton?.classList.add('active');
        this.showStatus('Listening for "Hey Portrait"...', 'listening');
        
        console.log('[Voice] Wake word listening started');
    }
    
    stopWakeWord() {
        this.socket.emit('stop_wake_word');
        this.isWakeWordActive = false;
        
        this.wakeWordButton?.classList.remove('active');
        this.showStatus('Wake word disabled', 'info');
        
        console.log('[Voice] Wake word listening stopped');
    }
    
    handleWakeWord() {
        console.log('[Voice] Wake word detected!');
        
        // Play chime
        this.playChime();
        
        // Show mic icon
        this.showMicIcon();
        
        // Start speech recognition
        this.startListening();
    }
    
    startListening() {
        if (!this.recognition) {
            console.warn('[Voice] Speech recognition not available');
            return;
        }
        
        if (this.isListening) {
            console.log('[Voice] Already listening');
            return;
        }
        
        try {
            this.recognition.start();
            this.isListening = true;
            this.showStatus('Listening...', 'listening');
            console.log('[Voice] Started listening');
        } catch (error) {
            console.error('[Voice] Error starting recognition:', error);
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
        
        this.isListening = false;
        this.hideMicIcon();
        this.showStatus('', '');
    }
    
    handleSpeechResult(transcript) {
        console.log('[Voice] Speech result:', transcript);
        
        // Send to server
        this.socket.emit('send_chat_message', {
            text: transcript,
            person_id: null,  // Will be determined by server
            is_voice: true
        });
        
        // Add to chat interface (if available)
        if (window.chatInterface) {
            window.chatInterface.addMessage('user', transcript, true);
        }
        
        this.stopListening();
    }
    
    playChime() {
        // Simple beep sound using Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        } catch (error) {
            console.error('[Voice] Error playing chime:', error);
        }
    }
    
    showMicIcon() {
        if (this.micIcon) {
            this.micIcon.style.display = 'block';
            this.micIcon.classList.add('pulse');
        }
    }
    
    hideMicIcon() {
        if (this.micIcon) {
            this.micIcon.style.display = 'none';
            this.micIcon.classList.remove('pulse');
        }
    }
    
    showConfirmationModal(data) {
        // data: {person_id, name, confidence}
        if (!this.confirmationModal) return;
        
        const message = `Is this ${data.name}? (${Math.round(data.confidence * 100)}% match)`;
        document.getElementById('confirmation-message').textContent = message;
        
        this.confirmationModal.style.display = 'flex';
        
        // Auto-timeout after 30 seconds
        this.confirmationTimeout = setTimeout(() => {
            this.hideConfirmationModal();
            this.sendConfirmation(null);  // Timeout
        }, 30000);
        
        console.log('[Voice] Showing confirmation modal:', data);
    }
    
    hideConfirmationModal() {
        if (this.confirmationModal) {
            this.confirmationModal.style.display = 'none';
        }
        
        if (this.confirmationTimeout) {
            clearTimeout(this.confirmationTimeout);
            this.confirmationTimeout = null;
        }
    }
    
    sendConfirmation(confirmed) {
        // confirmed: true, false, or null (timeout)
        this.socket.emit('face_confirmation', {confirmed});
        this.hideConfirmationModal();
        
        console.log('[Voice] Sent confirmation:', confirmed);
    }
    
    handleTTSStarted() {
        // Portrait started speaking
        console.log('[Voice] Portrait speaking...');
        this.showStatus('Portrait speaking...', 'speaking');
    }
    
    handleTTSFinished() {
        // Portrait finished speaking
        console.log('[Voice] Portrait finished speaking');
        this.showStatus('', '');
    }
    
    showStatus(message, type = 'info') {
        if (this.voiceStatus) {
            this.voiceStatus.textContent = message;
            this.voiceStatus.className = `voice-status ${type}`;
        }
    }
}
