// Living Portrait - Simplified JavaScript

// State
let isPaused = false;

// Elements
const pauseBtn = document.getElementById('pause-btn');
const refreshBtn = document.getElementById('refresh-btn');
const stopBtn = document.getElementById('stop-btn');
const platformIndicator = document.getElementById('platform-indicator');
const portraitSubtitle = document.getElementById('portrait-subtitle');
const detectionList = document.getElementById('detection-list');
const detectionCount = document.getElementById('detection-count');
const aiDescription = document.getElementById('ai-description');

// Button Handlers
pauseBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/toggle_pause');
        const data = await response.json();
        isPaused = data.paused;
        pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
        pauseBtn.className = isPaused 
            ? 'px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg font-medium transition'
            : 'px-4 py-2 bg-yellow-500 hover:bg-yellow-600 rounded-lg font-medium text-gray-900 transition';
    } catch (error) {
        console.error('Error toggling pause:', error);
    }
});

refreshBtn.addEventListener('click', () => {
    location.reload();
});

stopBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to stop the server?')) {
        try {
            await fetch('/stop_server');
            document.body.innerHTML = '<div class="min-h-screen flex items-center justify-center"><div class="text-center"><h1 class="text-4xl font-bold text-white mb-4">Server Stopped</h1><p class="text-purple-200">The vision system has been terminated.</p></div></div>';
        } catch (error) {
            console.error('Error stopping server:', error);
        }
    }
});

// Update Functions
async function updatePlatform() {
    try {
        const response = await fetch('/get_status');
        const data = await response.json();
        platformIndicator.textContent = `Platform: ${data.platform}`;
    } catch (error) {
        console.error('Error fetching platform:', error);
    }
}

async function updateSubtitle() {
    try {
        const response = await fetch('/get_subtitle');
        const data = await response.json();
        if (data.subtitle) {
            portraitSubtitle.textContent = data.subtitle;
        }
    } catch (error) {
        console.error('Error fetching subtitle:', error);
    }
}

async function updateReferences() {
    try {
        // Update detection results
        const terminalResponse = await fetch('/get_terminal_data');
        const terminalData = await terminalResponse.json();
        
        if (terminalData.detection_objects && terminalData.detection_objects.length > 0) {
            detectionCount.textContent = terminalData.detection_objects.length;
            detectionList.innerHTML = terminalData.detection_objects
                .map(obj => `<li class="text-gray-300">• ${obj}</li>`)
                .join('');
        } else {
            detectionCount.textContent = '0';
            detectionList.innerHTML = '<li class="text-gray-400">No detections yet</li>';
        }
        
        // Update AI description
        const aiResponse = await fetch('/get_ai_description');
        const aiData = await aiResponse.json();
        if (aiData.description) {
            aiDescription.textContent = aiData.description;
        }
    } catch (error) {
        console.error('Error fetching references:', error);
    }
}

// Initialize
updatePlatform();

// Polling intervals
setInterval(updateSubtitle, 500);      // Update subtitle twice per second
setInterval(updateReferences, 1000);   // Update references every second
