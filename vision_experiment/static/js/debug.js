/**
 * Debug Panel for Living Portrait
 * Shows system state for diagnosing issues
 */

class DebugPanel {
    constructor(socketManager) {
        this.socket = socketManager;
        this.visible = true;  // Start visible by default for debugging
        this.eventLog = [];
        this.maxEvents = 10;
        
        this.createPanel();
        this.initializeEventListeners();
        this.startUpdating();
    }
    
    createPanel() {
        // Create debug panel HTML
        const panel = document.createElement('div');
        panel.id = 'debug-panel';
        panel.className = 'debug-panel';
        panel.style.display = 'block';  // Show by default
        
        panel.innerHTML = `
            <div class="debug-header">
                <h3>🐛 SYSTEM DEBUG</h3>
                <button id="debug-close">✕</button>
            </div>
            <div class="debug-content">
                <div class="debug-section">
                    <h4>🎤 VOICE STATUS</h4>
                    <pre id="debug-voice">status: idle
listening: false
audio_level: 0
last_transcription: null</pre>
                </div>
                
                <div class="debug-section">
                    <h4>🎭 ANIMATION STATE</h4>
                    <pre id="debug-animation">mood: idle
speaking: false
subtitle: ""
mouth_open: false</pre>
                </div>
                
                <div class="debug-section">
                    <h4>📝 LAST PROMPT → MOONDREAM</h4>
                    <pre id="debug-prompt" style="max-height: 150px; overflow-y: auto;">Waiting for prompt...</pre>
                </div>
                
                <div class="debug-section">
                    <h4>💬 MOONDREAM RESPONSE</h4>
                    <pre id="debug-response">text: null
mood: null
event_type: null</pre>
                </div>
                
                <div class="debug-section">
                    <h4>📡 STREAM STATE</h4>
                    <pre id="debug-stream">active: false
stream_id: null
buffer: ""
words_sent: 0</pre>
                </div>
                
                <div class="debug-section">
                    <h4>📋 EVENT LOG (Last ${this.maxEvents})</h4>
                    <pre id="debug-events" style="max-height: 200px; overflow-y: auto;">No events yet...</pre>
                </div>
                
                <div class="debug-section">
                    <h4>🔧 DIAGNOSIS</h4>
                    <pre id="debug-diagnosis" style="color: #ffa500;">Analyzing...</pre>
                </div>
                
                <div class="debug-section">
                    <h4>👁️ VISION CONFIG</h4>
                    <div style="margin-bottom: 8px;">
                        <label>Detail Level: </label>
                        <select id="vision-detail-level" style="background: #333; color: white; padding: 4px 8px; border: 1px solid #555; border-radius: 4px;">
                            <option value="basic">Basic (RPi)</option>
                            <option value="standard" selected>Standard</option>
                            <option value="detailed">Detailed</option>
                            <option value="full">Full</option>
                        </select>
                    </div>
                    <pre id="debug-vision">level: standard
features: loading...</pre>
                </div>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // Create toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'debug-toggle';
        toggleBtn.className = 'debug-toggle-btn';
        toggleBtn.innerHTML = '🐛 Debug';
        toggleBtn.style.cssText = `
            position: fixed;
            top: 15px;
            right: 15px;
            z-index: 10000;
            padding: 10px 20px;
            background: #ff6b35;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(toggleBtn);
    }
    
    initializeEventListeners() {
        // Toggle button
        document.getElementById('debug-toggle')?.addEventListener('click', () => {
            this.toggle();
        });
        
        // Close button
        document.getElementById('debug-close')?.addEventListener('click', () => {
            this.hide();
        });
        
        // Socket events
        this.socket.on('speech_partial', (data) => {
            this.logEvent('SPEECH_PARTIAL', data.text);
            this.updateVoiceStatus(data);
        });
        
        this.socket.on('speech_final', (data) => {
            this.logEvent('SPEECH_FINAL', data.text);
            this.updateVoiceStatus(data);
        });
        
        this.socket.on('stream_chunk', (data) => {
            this.logEvent('STREAM_CHUNK', data.text);
            this.updateStreamStatus(data);
        });
        
        this.socket.on('stream_complete', (data) => {
            this.logEvent('STREAM_COMPLETE', data.full_text);
            this.updateMoondreamResponse(data);
        });
        
        this.socket.on('tts_speaking', (data) => {
            this.logEvent('TTS_SPEAKING', data.speaking ? 'started' : 'stopped');
            this.updateAnimationState(data);
        });
        
        // Listen for debug info from server
        this.socket.on('debug_info', (data) => {
            this.updateFromServer(data);
        });
        
        // Vision config events
        this.socket.on('vision_config', (data) => {
            this.updateVisionConfig(data);
        });
        
        this.socket.on('vision_config_updated', (data) => {
            this.updateVisionConfig(data);
            this.logEvent('VISION_CONFIG', `Level: ${data.detail_level}`);
        });
        
        // Vision detail level selector
        document.getElementById('vision-detail-level')?.addEventListener('change', (e) => {
            const level = e.target.value;
            this.socket.emit('set_vision_detail', { level: level });
            console.log(`[Debug] Vision detail level changed to: ${level}`);
        });
        
        // Request current vision config on init
        setTimeout(() => {
            this.socket.emit('get_vision_config', {});
        }, 1000);
    }
    
    toggle() {
        this.visible = !this.visible;
        const panel = document.getElementById('debug-panel');
        if (panel) {
            panel.style.display = this.visible ? 'block' : 'none';
        }
    }
    
    show() {
        this.visible = true;
        const panel = document.getElementById('debug-panel');
        if (panel) panel.style.display = 'block';
    }
    
    hide() {
        this.visible = false;
        const panel = document.getElementById('debug-panel');
        if (panel) panel.style.display = 'none';
    }
    
    logEvent(type, data) {
        const timestamp = new Date().toLocaleTimeString();
        this.eventLog.unshift(`[${timestamp}] ${type}: ${JSON.stringify(data).substring(0, 100)}`);
        if (this.eventLog.length > this.maxEvents) {
            this.eventLog = this.eventLog.slice(0, this.maxEvents);
        }
        this.updateEventLog();
    }
    
    updateEventLog() {
        const elem = document.getElementById('debug-events');
        if (elem) {
            elem.textContent = this.eventLog.join('\\n');
        }
    }
    
    updateVoiceStatus(data) {
        const elem = document.getElementById('debug-voice');
        if (elem) {
            elem.textContent = `status: ${data.status || 'active'}
listening: ${!!data.listening}
audio_level: ${data.audio_level || 'unknown'}
last_transcription: "${data.text || 'null'}"
confidence: ${data.confidence || 'N/A'}`;
        }
    }
    
    updateAnimationState(data) {
        const elem = document.getElementById('debug-animation');
        if (elem) {
            elem.textContent = `speaking: ${data.speaking || false}
text_preview: "${(data.text || '').substring(0, 50)}..."`;
        }
    }
    
    updateStreamStatus(data) {
        const elem = document.getElementById('debug-stream');
        if (elem) {
            elem.textContent = `active: true
stream_id: ${data.metadata?.stream_id || 'unknown'}
word_index: ${data.metadata?.word_index || 0}
total_words: ${data.metadata?.total_words || 0}
mood: ${data.metadata?.mood || 'idle'}
latest_chunk: "${data.text}"`;
        }
    }
    
    updateMoondreamResponse(data) {
        const elem = document.getElementById('debug-response');
        if (elem) {
            elem.textContent = `text: "${data.full_text}"
mood: ${data.mood}
stream_id: ${data.stream_id}
length: ${data.full_text?.length || 0} chars`;
        }
    }
    
    updateFromServer(data) {
        if (data.prompt) {
            const elem = document.getElementById('debug-prompt');
            if (elem) {
                elem.textContent = data.prompt;
            }
        }
        
        if (data.animation_state) {
            const elem = document.getElementById('debug-animation');
            if (elem) {
                elem.textContent = JSON.stringify(data.animation_state, null, 2);
            }
        }
        
        this.runDiagnosis();
    }
    
    updateVisionConfig(data) {
        const elem = document.getElementById('debug-vision');
        if (elem) {
            elem.textContent = `level: ${data.detail_level}
features: ${data.features?.join(', ') || 'none'}`;
        }
        
        // Update dropdown to match current level
        const select = document.getElementById('vision-detail-level');
        if (select && data.detail_level) {
            select.value = data.detail_level;
        }
    }
    
    runDiagnosis() {
        const diagElem = document.getElementById('debug-diagnosis');
        if (!diagElem) return;
        
        const issues = [];
        const info = [];
        
        // Check voice status
        const voiceText = document.getElementById('debug-voice')?.textContent || '';
        if (voiceText.includes('audio_level: 0')) {
            issues.push('⚠️ AUDIO LEVEL = 0: Microphone permissions issue');
        } else if (voiceText.includes('audio_level: unknown')) {
            info.push('ℹ️ Audio level not being reported');
        }
        
        // Check animation
        const animText = document.getElementById('debug-animation')?.textContent || '';
        if (animText.includes('speaking: false') && voiceText.includes('listening: true')) {
            issues.push('⚠️ NOT ANIMATING: speaking=false while audio active');
        }
        
        // Check responses
        const responseText = document.getElementById('debug-response')?.textContent || '';
        if (responseText.includes('Hello from within the frame')) {
            issues.push('🔴 REPETITIVE RESPONSES: Moondream not understanding context');
            issues.push('💡 SOLUTION NEEDED: Add Llama 3.2 for conversation (Moondream=vision only)');
        }
        
        // Check prompt
        const promptText = document.getElementById('debug-prompt')?.textContent || '';
        if (promptText.includes('Waiting for prompt')) {
            info.push('ℹ️ No prompts sent to Moondream yet');
        } else if (!promptText.includes('User says:')) {
            issues.push('⚠️ Prompt may not include user message properly');
        }
        
        let diagnosis = '';
        if (issues.length > 0) {
            diagnosis += '🔴 ISSUES FOUND:\\n' + issues.join('\\n') + '\\n\\n';
        }
        if (info.length > 0) {
            diagnosis += 'ℹ️ INFO:\\n' + info.join('\\n');
        }
        if (issues.length === 0 && info.length === 0) {
            diagnosis = '✅ No obvious issues detected';
        }
        
        diagElem.textContent = diagnosis;
    }
    
    startUpdating() {
        // Request debug info from server every 2 seconds
        setInterval(() => {
            if (this.visible) {
                this.socket.send('get_debug_info', {});
                this.runDiagnosis();
            }
        }, 2000);
    }
}

// Initialize when socket is ready
window.addEventListener('load', () => {
    if (window.socketManager) {
        window.debugPanel = new DebugPanel(window.socketManager);
        console.log('[Debug] Panel initialized');
    }
});
