/**
 * Voice Interaction - Offline Speech Recognition
 * Uses server-side Vosk for completely offline speech-to-text
 */

class VoiceInterface {
    constructor(socketManager) {
        this.socket = socketManager;
        this.isListening = false;
        
        // UI elements
        this.speakButton = document.getElementById('start-speaking-button');
        this.micIcon = document.getElementById('mic-icon');
        this.voiceStatus = document.getElementById('voice-status');
        this.micStatus = document.getElementById('mic-status');
        this.voiceStatusText = document.getElementById('voice-status-text');
        this.confirmationModal = document.getElementById('confirmation-modal');
        
        // Transcription elements
        this.transcriptionLog = document.getElementById('transcription-log');
        this.transcriptionOverlay = document.getElementById('transcription-overlay-portrait');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Push-to-talk button
        this.speakButton?.addEventListener('click', () => this.toggleListening());
        
        // Server events
        this.socket.on('speech_partial', (data) => this.handlePartial(data));
        this.socket.on('speech_final', (data) => this.handleFinal(data));
        this.socket.on('speech_status', (data) => this.handleStatus(data));
        this.socket.on('tts_started', (data) => this.handleTTSStarted(data));
        this.socket.on('tts_error', (data) => this.handleTTSError(data));
        this.socket.on('auto_chat_message', (data) => this.handleAutoChat(data));
        this.socket.on('face_confirmation_request', (data) => this.showConfirmationModal(data));
        
        // Confirmation modal buttons
        document.getElementById('confirm-yes')?.addEventListener('click', () => {
            this.sendConfirmation(true);
        });
        
        document.getElementById('confirm-no')?.addEventListener('click', () => {
            this.sendConfirmation(false);
        });
    }
    
    toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }
    
    startListening() {
        console.log('[Voice] 🎤 Starting offline speech recognition...');
        this.socket.send('start_speech_recognition', {});
        this.isListening = true;
        
        // Update UI
        this.speakButton?.classList.add('active', 'listening');
        if (this.speakButton) {
            this.speakButton.querySelector('span:last-child').textContent = 'Listening...';
        }
        this.showMicIcon();
        this.updateStatus('Starting...', 'active');
    }
    
    stopListening() {
        console.log('[Voice] 🛑 Stopping speech recognition...');
        this.socket.send('stop_speech_recognition', {});
        this.isListening = false;
        
        // Update UI
        this.speakButton?.classList.remove('active', 'listening');
        if (this.speakButton) {
            this.speakButton.querySelector('span:last-child').textContent = 'Start Speaking';
        }
        this.hideMicIcon();
        this.updateStatus('Idle', 'idle');
    }
    
    handlePartial(data) {
        // Live interim transcription from Vosk
        console.log(`[Voice ${data.timestamp}] ... interim: "${data.text}"`);
        this.showTranscription(data.text, false, data.timestamp);
    }
    
    handleFinal(data) {
        // Final transcription from Vosk
        console.log(`[Voice ${data.timestamp}] ✓ FINAL (${(data.confidence * 100).toFixed(0)}%): "${data.text}"`);
        this.showTranscription(data.text, true, data.timestamp, data.confidence);
    }
    
    handleStatus(data) {
        this.updateStatus(data.status, data.error ? 'error' : 'info');
        
        // If error requires refresh, show persistent message
        if (data.error && data.requires_refresh) {
            this.isListening = false;
            this.speakButton?.classList.remove('active', 'listening');
            if (this.speakButton) {
                this.speakButton.querySelector('span:last-child').textContent = 'Start Speaking';
            }
            this.hideMicIcon();
            
            // Show alert with refresh instruction
            alert('Microphone Error: No audio detected.\n\n' +
                  'Please grant microphone permissions:\n' +
                  '1. System Preferences → Security & Privacy → Privacy → Microphone\n' +
                  '2. Enable Terminal/Python\n' +
                  '3. Refresh this page (⌘+R or F5)\n' +
                  '4. Try again');
            return;
        }
        
        if (data.listening) {
            if (this.micStatus) {
                this.micStatus.textContent = 'Active';
                this.micStatus.style.color = '#4CAF50';
            }
        } else {
            if (this.micStatus) {
                this.micStatus.textContent = 'Ready';
                this.micStatus.style.color = '#999';
            }
            this.isListening = false;
            this.speakButton?.classList.remove('active', 'listening');
            if (this.speakButton) {
                this.speakButton.querySelector('span:last-child').textContent = 'Start Speaking';
            }
        }
    }
    
    handleAutoChat(data) {
        // Automatically send recognized speech to chat
        if (window.chatInterface) {
            window.chatInterface.addMessage('user', data.text, true);
        }
        
        // Also send to server
        this.socket.send('send_chat_message', {
            text: data.text,
            person_id: null,
            is_voice: true
        });
    }
    
    showTranscription(text, isFinal, timestamp, confidence = null) {
        // Update portrait overlay
        if (this.transcriptionOverlay) {
            const textElement = this.transcriptionOverlay.querySelector('.transcription-text');
            if (textElement) {
                textElement.textContent = text;
                textElement.className = isFinal ? 'transcription-final' : 'transcription-interim';
            }
            
            this.transcriptionOverlay.classList.add('active');
            
            if (isFinal) {
                // Hide after 3 seconds
                setTimeout(() => {
                    this.transcriptionOverlay.classList.remove('active');
                }, 3000);
            }
        }
        
        // Update camera panel log (final only)
        if (isFinal && this.transcriptionLog) {
            // Remove placeholder
            const placeholder = this.transcriptionLog.querySelector('.transcription-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
            
            // Add new entry at top
            const entry = document.createElement('div');
            entry.className = 'transcription-item';
            entry.innerHTML = `
                <span class="transcription-timestamp">${timestamp}</span>
                <span class="transcription-text-final">"${text}"</span>
                ${confidence ? `<span class="transcription-confidence">${(confidence * 100).toFixed(0)}%</span>` : ''}
            `;
            
            this.transcriptionLog.insertBefore(entry, this.transcriptionLog.firstChild);
            
            // Keep only last 10
            const items = this.transcriptionLog.querySelectorAll('.transcription-item');
            if (items.length > 10) {
                items[items.length - 1].remove();
            }
        }
    }
    
    updateStatus(message, type = 'info') {
        if (this.voiceStatusText) {
            this.voiceStatusText.textContent = message;
            this.voiceStatusText.className = `info-value status-${type}`;
        }
    }
    
    handleTTSStarted(data) {
        console.log(`[TTS ${data.timestamp}] 🔊 Speaking: "${data.text}"`);
        this.updateStatus('Portrait speaking...', 'speaking');
    }
    
    handleTTSError(data) {
        console.error('[TTS] Error:', data.error);
        this.updateStatus('TTS unavailable', 'error');
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
        this.socket.send('face_confirmation', {confirmed});
        this.hideConfirmationModal();
        
        console.log('[Voice] Sent confirmation:', confirmed);
    }
}
