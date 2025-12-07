# Raspberry Pi Setup Guide - Living Portrait

## Hardware Requirements

**Minimum:**
- Raspberry Pi 4 (4GB RAM) - will work but slower
- Raspberry Pi 4/5 (8GB RAM) - recommended
- Camera module or USB webcam
- Speakers or headphones for TTS output
- Microphone for voice input

**Storage:**
- At least 16GB SD card
- Models require ~5-6GB space

## Software Setup

### 1. Install Ollama on Raspberry Pi

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Wait a few seconds, then pull models
ollama pull moondream  # ~2GB (vision)
ollama pull llama3.2:1b  # ~1.3GB (conversation - LIGHTER for RPi)

# Alternative: Use 3B for better quality if you have 8GB RAM
# ollama pull llama3.2:3b  # ~2GB (conversation - BETTER QUALITY)
```

### 2. Install Python Dependencies

```bash
cd /path/to/local_vision_test

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv portaudio19-dev

# Install Python packages
pip3 install -r requirements.txt
```

### 3. Configure for Raspberry Pi

Edit `vision_experiment/app.py` line ~125:

```python
# For Raspberry Pi 4GB - use lighter model
llm_client = create_llm_client(model_name="llama3.2:1b")

# For Raspberry Pi 8GB - use better model
# llm_client = create_llm_client(model_name="llama3.2:3b")
```

## Architecture Explanation

### Two-Model System (Optimized for RPi)

```
┌─────────────────────────────────────────────────────────┐
│  Camera Input                                           │
│  640x480 video stream                                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  MODEL 1: Moondream (2B params, ~2GB RAM)              │
│  Purpose: VISION ONLY                                   │
│  Input: Image frame                                     │
│  Output: "A man with dark hair wearing a blue shirt"   │
│  Speed: ~2-3 seconds on RPi 4                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  MODEL 2: Llama 3.2 1B/3B (text-only)                  │
│  Purpose: CONVERSATION ONLY                             │
│  Input:                                                 │
│    - Vision: "A man with dark hair..."                 │
│    - User said: "what are you doing?"                  │
│    - Recent chat history                               │
│  Output: "I'm observing you! Nice shirt by the way."   │
│  Speed: ~1-2 seconds on RPi 4                          │
└─────────────────────────────────────────────────────────┘
```

**Why This Works:**
- Moondream: Small vision model, good for RPi
- Llama 3.2 1B: Very small text model (1.3GB)
- Total RAM usage: ~3-4GB (fits on RPi 4GB)
- Both run 100% offline via Ollama
- Combined: Better than either model alone!

## Performance Expectations

### Raspberry Pi 4 (4GB)
- Model: Llama 3.2 1B
- Response time: 3-5 seconds total
- Quality: Good enough for demo/prototype
- Memory: 3-4GB used

### Raspberry Pi 4/5 (8GB)
- Model: Llama 3.2 3B
- Response time: 4-6 seconds total  
- Quality: Very good, natural conversations
- Memory: 4-5GB used

## Testing

```bash
# 1. Start Ollama (if not running)
ollama serve &

# 2. Test models
ollama run moondream
# Type: /bye to exit

ollama run llama3.2:1b
# Test it responds, then /bye

# 3. Start the Living Portrait
cd /Users/jramirez/Git/local_vision_test
./start_server.sh

# 4. Open browser on same network
http://raspberrypi.local:8000
# Or use IP address: http://192.168.1.XXX:8000
```

## Troubleshooting

### "Ollama not running"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve &
```

### "Model not found"
```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.2:1b
```

### Slow Performance
```bash
# Switch to 1B model in app.py
llm_client = create_llm_client(model_name="llama3.2:1b")

# Or reduce image resolution in vision loop
# Edit detection_manager.py, reduce from 640x480 to 320x240
```

### High Memory Usage
```bash
# Check RAM usage
free -h

# If using 1B and still high:
# 1. Reduce camera resolution
# 2. Increase delay between vision calls
# 3. Consider RPi 5 or desktop for better experience
```

## Model Comparison

| Model | Size | RAM | Speed (RPi4) | Quality |
|-------|------|-----|--------------|---------|
| **Llama 3.2 1B** | 1.3GB | ~2GB | Fast (~1-2s) | Good |
| **Llama 3.2 3B** | 2GB | ~3GB | Moderate (~2-4s) | Very Good |

**Recommendation:**
- RPi 4GB → Use 1B
- RPi 8GB → Use 3B

## Network Access (Optional)

To access from other devices on your network:

```bash
# Find your RPi IP address
hostname -I

# Make sure port 8000 is accessible
sudo ufw allow 8000

# Access from phone/computer
http://YOUR_RPI_IP:8000
```

## Desktop vs Raspberry Pi

**Desktop (your current Mac):**
```python
llm_client = create_llm_client(model_name="llama3.2:3b")  # Better quality
```

**Raspberry Pi:**
```python
llm_client = create_llm_client(model_name="llama3.2:1b")  # Faster, lighter
```

## What Changed From Before

**Before (Broken):**
- Moondream trying to do conversation → repetitive responses
- "Hello from within the frame" over and over

**After (Fixed):**
- Moondream → Vision description only
- Llama 3.2 → Natural conversation
- Result: Varied, contextual responses!

## Example Interaction

```
User: "what are you doing?"

[Moondream Vision]: "A man with dark hair wearing a blue shirt standing near a window"
                    ↓
[Llama 3.2 Conversation]: "I'm observing you from within my frame! I can see you're 
                           wearing a nice blue shirt. What brings you here today?"

Output to user: "I'm observing you from within my frame! I can see you're 
                 wearing a nice blue shirt. What brings you here today?"
```

## Files Modified

1. `vision_experiment/core/llm_client.py` - NEW: Llama conversation handler
2. `vision_experiment/app.py` - Updated worker to use 2-model architecture
3. This guide - Raspberry Pi setup instructions

## Next Steps

1. Install Ollama on your RPi
2. Pull both models (moondream + llama3.2:1b)
3. Update app.py model selection based on your RAM
4. Test and enjoy natural conversations!
