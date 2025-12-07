// AI Vision Analysis System - Main JavaScript

function changeDetectionModel() {
    const model = document.getElementById('detectionModel').value;
    fetch('/change_detection_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: model })
    }).then(() => updateStatus());
}

function changeDetectionMode() {
    const mode = document.getElementById('detectionMode').value;
    fetch('/change_detection_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
    });
}

function changeProcessingFps() {
    const fps = document.getElementById('processingFpsSelect').value;
    document.getElementById('processingFpsInput').value = ''; // Clear custom input
    applyFpsChange(fps);
}

function handleFpsInputKeyPress(event) {
    if (event.key === 'Enter') {
        applyCustomFps();
    }
}

function applyCustomFps() {
    const customFps = document.getElementById('processingFpsInput').value.trim();
    if (customFps) {
        applyFpsChange(customFps);
    }
}

function applyFpsChange(fps) {
    fetch('/change_processing_fps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fps: fps })
    }).then(() => updateStatus());
}

let overlayEnabled = true;
function toggleOverlay() {
    fetch('/toggle_overlay')
        .then(response => response.json())
        .then(data => {
            overlayEnabled = data.overlay;
            const btn = document.getElementById('overlayBtn');
            if (overlayEnabled) {
                btn.textContent = '🎨 AI Overlay: ON';
                btn.className = 'overlay-toggle active';
            } else {
                btn.textContent = '🎨 AI Overlay: OFF';
                btn.className = 'overlay-toggle inactive';
            }
        });
}

let isPaused = false;
function togglePause() {
    fetch('/toggle_pause')
        .then(response => response.json())
        .then(data => {
            isPaused = data.paused;
            const btn = document.getElementById('pauseBtn');
            if (isPaused) {
                btn.textContent = '▶️ Resume';
                btn.className = 'btn-resume';
            } else {
                btn.textContent = '⏸️ Pause';
                btn.className = 'btn-warning';
            }
        });
}

function stopServer() {
    if (confirm('Are you sure you want to stop the server? This will end the vision analysis.')) {
        window.location.href = '/stop_server';
    }
}

function refreshPage() {
    location.reload();
}

function updateStatus() {
    fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            const pauseStatus = data.paused ? ' (PAUSED)' : '';
            document.getElementById('statusBanner').textContent = 
                `🔴 Active Model: ${data.model} (${data.capability})${pauseStatus}`;
            
            // Update performance stats
            const status = data.paused ? 'PAUSED' : 'Active';
            document.getElementById('performanceStats').innerHTML = 
                `<div><strong>Performance Stats:</strong></div>
                <div>Processing Time: ${data.processing_time}</div>
                <div>Frame Size: ${data.frame_size}</div>
                <div>Processing Rate: ${data.fps}</div>
                <div>Status: ${status}</div>`;
            
            // Update dropdown selections to match current state
            document.getElementById('detectionModel').value = data.model.toLowerCase();
            document.getElementById('detectionMode').value = data.mode;
            
            // Handle FPS selection/input
            const fpsSelect = document.getElementById('processingFpsSelect');
            const fpsInput = document.getElementById('processingFpsInput');
            const currentFpsDisplay = document.getElementById('currentFpsDisplay');
            const currentFps = data.fps;
            
            // Update current FPS display
            currentFpsDisplay.textContent = currentFps;
            
            // Update countdown display
            const countdownDisplay = document.getElementById('countdownDisplay');
            if (data.countdown !== undefined && data.interval !== undefined) {
                if (data.paused) {
                    countdownDisplay.textContent = '⏸️ Paused';
                    countdownDisplay.className = 'font-mono font-bold text-yellow-600 bg-yellow-100 px-2 py-0.5 rounded';
                } else if (data.countdown <= 0) {
                    countdownDisplay.textContent = '🔄 Processing...';
                    countdownDisplay.className = 'font-mono font-bold text-green-600 bg-green-100 px-2 py-0.5 rounded';
                } else {
                    countdownDisplay.textContent = `⏱️ Next: ${data.countdown}s`;
                    countdownDisplay.className = 'font-mono font-bold text-orange-600 bg-orange-100 px-2 py-0.5 rounded';
                }
            }
            
            // Check if current FPS is in the dropdown options
            const options = Array.from(fpsSelect.options).map(opt => opt.value);
            if (options.includes(currentFps)) {
                fpsSelect.value = currentFps;
                fpsInput.value = '';
            } else {
                // Custom value - put it in the input field
                fpsSelect.value = options[0]; // Reset dropdown to first option
                fpsInput.value = currentFps;
            }
            
            // Sync pause button state
            const pauseBtn = document.getElementById('pauseBtn');
            if (data.paused) {
                pauseBtn.textContent = '▶️ Resume';
                pauseBtn.className = 'btn-resume';
                isPaused = true;
            } else {
                pauseBtn.textContent = '⏸️ Pause';
                pauseBtn.className = 'btn-warning';
                isPaused = false;
            }
        })
        .catch(() => {
            document.getElementById('performanceStats').innerHTML = 
                `<div><strong>Performance Stats:</strong></div>
                <div>Processing Time: --</div>
                <div>Frame Size: --</div>
                <div>Processing Rate: --</div>
                <div>Status: Error</div>`;
        });
}

function copyDetectionData() {
    const detection = document.getElementById('detectionList').textContent;
    const ai = document.getElementById('aiList').textContent;
    const combined = document.getElementById('combinedList').textContent;
    const counts = `Detection: ${document.getElementById('detectionCount').textContent}, AI: ${document.getElementById('aiCount').textContent}, Combined: ${document.getElementById('combinedCount').textContent}, Avg Confidence: ${document.getElementById('avgConfidence').textContent}`;
    
    const fullText = `=== DETECTION RESULTS ===\n${counts}\n\n--- DETECTION ---\n${detection}\n\n--- AI ---\n${ai}\n\n--- DETECTION+AI ---\n${combined}`;
    
    navigator.clipboard.writeText(fullText).then(() => {
        alert('Detection data copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

function updateExtendedStats() {
    // Get system stats
    fetch('/get_system_stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('cpuUsage').textContent = data.cpu_usage || '--%';
            document.getElementById('ramUsage').textContent = data.ram_usage || '--%';
            document.getElementById('gpuUsage').textContent = data.gpu_usage || '--%';
        })
        .catch(() => {
            // Keep default values on error
        });
}

function updateTerminalData() {
    fetch('/get_terminal_data')
        .then(response => response.json())
        .then(data => {
            // Update counts
            document.getElementById('detectionCount').textContent = data.detection_count || 0;
            document.getElementById('aiCount').textContent = data.ai_count || 0;
            document.getElementById('combinedCount').textContent = data.combined_count || 0;
            document.getElementById('avgConfidence').textContent = data.avg_confidence || '--';
            
            // Update lists
            document.getElementById('detectionList').innerHTML = data.detection_objects ? data.detection_objects.map(obj => `<div>${obj}</div>`).join('') : 'No detections';
            document.getElementById('aiList').innerHTML = data.ai_objects ? data.ai_objects.map(obj => `<div>${obj}</div>`).join('') : 'No AI objects';
            document.getElementById('combinedList').innerHTML = data.correlated_objects ? data.correlated_objects.map(obj => `<div>${obj}</div>`).join('') : 'No combined results';
        })
        .catch(() => {
            document.getElementById('detectionList').textContent = 'Error loading...';
        });
}

function updateAIDescription() {
    fetch('/get_ai_description')
        .then(response => response.json())
        .then(data => {
            document.getElementById('aiDescription').textContent = data.description || 'No description available yet...';
        })
        .catch(() => {
            document.getElementById('aiDescription').textContent = 'Error loading AI description...';
        });
}

// Fast countdown updates (every 500ms for smooth countdown display)
setInterval(() => {
    updateStatus();
}, 500);

// Slower data updates (every 2 seconds for terminal/AI/stats)
setInterval(() => {
    updateTerminalData();
    updateAIDescription();
    updateExtendedStats();
}, 2000);

// Initial updates on page load
document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    updateExtendedStats();
    updateTerminalData();
    updateAIDescription();
});
