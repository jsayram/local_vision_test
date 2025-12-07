/**
 * WebSocket Manager
 * Handles Socket.IO connection and event routing
 */

class SocketManager extends EventTarget {
    constructor() {
        super();
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        // Initialize Socket.IO connection
        this.socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: this.maxReconnectAttempts
        });
        
        // Connection events
        this.socket.on('connect', () => {
            console.log('[Socket] Connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.emit('connected', {});
        });
        
        this.socket.on('disconnect', () => {
            console.log('[Socket] Disconnected');
            this.connected = false;
            this.emit('disconnected', {});
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('[Socket] Connection error:', error);
            this.reconnectAttempts++;
            this.emit('error', {error, attempts: this.reconnectAttempts});
        });
        
        // Streaming events
        this.socket.on('stream_chunk', (data) => {
            this.emit('stream_chunk', data);
        });
        
        this.socket.on('stream_complete', (data) => {
            this.emit('stream_complete', data);
        });
        
        // Detection updates
        this.socket.on('detection_update', (data) => {
            this.emit('detection_update', data);
        });
        
        // Voice detection updates
        this.socket.on('voice_detection_update', (data) => {
            this.emit('voice_detection_update', data);
        });
        
        console.log('[Socket] Initializing connection...');
    }
    
    // Emit custom event
    emit(eventName, data) {
        const event = new CustomEvent(eventName, {detail: data});
        this.dispatchEvent(event);
    }
    
    // Listen for custom events
    on(eventName, callback) {
        this.addEventListener(eventName, (event) => callback(event.detail));
    }
    
    // Send message to server
    send(eventName, data) {
        if (this.socket && this.connected) {
            this.socket.emit(eventName, data);
        } else {
            console.warn('[Socket] Not connected, cannot send:', eventName);
        }
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}
