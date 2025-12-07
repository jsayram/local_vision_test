/**
 * Chat Interface
 * Handles text chat with the portrait
 */

class ChatInterface {
    constructor(socketManager) {
        this.socket = socketManager;
        this.currentPersonId = null;
        this.messages = [];
        
        this.chatInput = document.getElementById('chat-input');
        this.chatMessages = document.getElementById('chat-messages');
        this.sendButton = document.getElementById('send-button');
        this.chatStatus = document.getElementById('chat-status');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Send button
        this.sendButton?.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send
        this.chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Socket events
        this.socket.on('chat_message', (data) => this.handleChatMessage(data));
        this.socket.on('typing_indicator', (data) => this.handleTypingIndicator(data));
        this.socket.on('person_identified', (data) => this.handlePersonIdentified(data));
    }
    
    sendMessage() {
        const text = this.chatInput?.value.trim();
        if (!text) return;
        
        // Add user message to UI immediately
        this.addMessage('user', text, false);
        
        // Clear input
        this.chatInput.value = '';
        
        // Send to server
        this.socket.emit('send_chat_message', {
            text: text,
            person_id: this.currentPersonId,
            is_voice: false
        });
        
        console.log('[Chat] Sent message:', text);
    }
    
    handleChatMessage(data) {
        // data: {speaker, text, mood, is_voice, timestamp, message_id}
        this.addMessage(data.speaker, data.text, data.is_voice, data.mood, data.timestamp);
    }
    
    addMessage(speaker, text, isVoice, mood = null, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${speaker}-message`;
        
        // Add voice indicator if applicable
        const voiceClass = isVoice ? ' voice-message' : '';
        messageDiv.className += voiceClass;
        
        // Timestamp
        const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        
        // Build message HTML
        let html = `
            <div class="message-header">
                <span class="message-speaker">${speaker === 'user' ? 'You' : 'Portrait'}</span>
                <span class="message-time">${time}</span>
                ${isVoice ? '<span class="voice-indicator">🎤</span>' : ''}
            </div>
            <div class="message-text">${this.escapeHtml(text)}</div>
        `;
        
        // Add mood indicator for portrait messages
        if (speaker === 'portrait' && mood) {
            html += `<div class="message-mood">${mood}</div>`;
        }
        
        messageDiv.innerHTML = html;
        
        // Add to messages container
        this.chatMessages?.appendChild(messageDiv);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Store in memory
        this.messages.push({speaker, text, isVoice, mood, timestamp: timestamp || new Date().toISOString()});
    }
    
    handleTypingIndicator(data) {
        // data: {is_typing: bool}
        const indicator = document.getElementById('typing-indicator');
        
        if (data.is_typing) {
            if (!indicator) {
                const div = document.createElement('div');
                div.id = 'typing-indicator';
                div.className = 'typing-indicator';
                div.innerHTML = '<span class="typing-dots">Portrait is typing<span>.</span><span>.</span><span>.</span></span>';
                this.chatMessages?.appendChild(div);
                this.scrollToBottom();
            }
        } else {
            indicator?.remove();
        }
    }
    
    handlePersonIdentified(data) {
        // data: {person_id, name}
        this.currentPersonId = data.person_id;
        
        // Show notification
        this.showStatus(`Connected to ${data.name || 'Unknown Person'}`);
        
        console.log('[Chat] Person identified:', data);
    }
    
    clearMessages() {
        this.chatMessages.innerHTML = '';
        this.messages = [];
    }
    
    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }
    
    showStatus(message, type = 'info') {
        if (this.chatStatus) {
            this.chatStatus.textContent = message;
            this.chatStatus.className = `chat-status ${type}`;
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                this.chatStatus.textContent = '';
            }, 3000);
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Handle "forget that" command response
    handleForgetCommand(data) {
        // data: {deleted_count, message}
        if (data.deleted_count > 0) {
            // Remove last N messages from UI
            const messageElements = this.chatMessages?.querySelectorAll('.chat-message');
            if (messageElements) {
                for (let i = 0; i < data.deleted_count; i++) {
                    const lastMsg = messageElements[messageElements.length - 1 - i];
                    lastMsg?.remove();
                }
            }
            
            this.showStatus(data.message || 'Messages deleted', 'success');
        }
    }
}
