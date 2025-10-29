# 🧪 Testing Guide - GPU Allocation System

## ⚠️ IMPORTANT: You're Testing the Wrong Server!

### Your Console Shows:
```
Websocket URL: wss://whisper2.primumai.eu/asr
```

**This is NOT your local server!** This is an external production server that:
- Does NOT have your GPU allocation code
- May be running an old version
- Could have different issues

---

## ✅ How to Test Locally

### **Step 1: Start Your Local Server**

```bash
# Terminal 1: Start the server
cd c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit

# Option A: Using the CLI command (after installation)
whisperlivekit-server --host localhost --port 8000 --model small --backend simulstreaming

# Option B: Run directly with Python
python -m whisperlivekit.basic_server --host localhost --port 8000 --model small --backend simulstreaming
```

**Expected output:**
```
INFO: ================================================================================
INFO: WhisperLiveKit Server Starting
INFO: ================================================================================
INFO: GPU ALLOCATION STATUS
INFO: ================================================================================
INFO: GPU 0: 0/4 connections | Memory: 0.00/15.75 GB (0.0% used) | Free: 15.75 GB
INFO: GPU 1: 0/4 connections | Memory: 0.00/15.75 GB (0.0% used) | Free: 15.75 GB
INFO: GPU 2: 0/4 connections | Memory: 0.00/15.75 GB (0.0% used) | Free: 15.75 GB
INFO: GPU 3: 0/4 connections | Memory: 0.00/15.75 GB (0.0% used) | Free: 15.75 GB
INFO: ================================================================================
INFO: Application startup complete.
INFO: Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

---

### **Step 2: Open the Web Interface**

Open your browser and go to:
```
http://localhost:8000
```

**Make sure the WebSocket URL shows:**
```
ws://localhost:8000/asr
```

**NOT:**
```
wss://whisper2.primumai.eu/asr  ❌ WRONG!
```

---

### **Step 3: Test Transcription**

1. Click "Start Transcription"
2. Allow microphone access
3. Speak into your microphone
4. Watch the transcription appear in real-time

**In the console, you should see:**
```javascript
Connecting to WebSocket
Connected. Using MediaRecorder (WebM).
// ... transcription results ...
```

---

## 🔍 Troubleshooting

### **Problem 1: No GPU Detected**

**Error:**
```
INFO: GPU ALLOCATION STATUS
INFO: No CUDA GPUs detected! Server will fail without GPU support.
```

**Solution:**
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Expected output:
# CUDA available: True
# GPU count: 4
```

If CUDA is not available:
- Install CUDA drivers
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

---

### **Problem 2: Import Errors**

**Error:**
```python
ImportError: cannot import name 'gpu_manager' from 'whisperlivekit'
```

**Solution:**
```bash
# Make sure the package is installed in development mode
pip install -e .

# Or re-install
pip uninstall whisperlivekit
pip install -e .
```

---

### **Problem 3: Port Already in Use**

**Error:**
```
ERROR: [Errno 10048] error while attempting to bind on address ('localhost', 8000)
```

**Solution:**
```bash
# Use a different port
whisperlivekit-server --host localhost --port 8001 --model small

# Then open: http://localhost:8001
```

---

### **Problem 4: No Transcription Output**

**Check server logs for:**
```
INFO: Allocated GPU 0 to connection abc-123
INFO: Connection abc-123: WebSocket opened on GPU 0
INFO: TranscriptionEngine using GPU 0
INFO: SimulStreamingASR using GPU 0
```

If you see errors like:
```
ERROR: Failed to load Whisper model
ERROR: CUDA out of memory
```

**Solutions:**
1. Use a smaller model: `--model tiny` or `--model base`
2. Reduce `max_connections_per_gpu` in `gpu_manager.py`

---

## 📊 Monitor GPU Usage

### **Terminal 2: Watch GPU stats**

```bash
# Windows PowerShell
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }

# Or just run once
nvidia-smi
```

**Expected output when 2 connections are active:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.X  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| GPU Current Temp   Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util       |
|   0   35C    P0    28W / 70W |   7456MiB / 15360MiB |      45%           |
|   1   32C    P0    26W / 70W |   7321MiB / 15360MiB |      42%           |
|   2   30C    P8     9W / 70W |      0MiB / 15360MiB |       0%           |
|   3   29C    P8     9W / 70W |      0MiB / 15360MiB |       0%           |
+-----------------------------------------------------------------------------+
```

---

### **Terminal 3: Check GPU allocation API**

```bash
# Windows PowerShell
curl http://localhost:8000/gpu-status

# Or use Python
python -c "import requests; import json; print(json.dumps(requests.get('http://localhost:8000/gpu-status').json(), indent=2))"
```

**Expected output:**
```json
{
  "total_gpus": 4,
  "total_connections": 2,
  "max_connections_per_gpu": 4,
  "gpus": [
    {
      "gpu_id": 0,
      "active_connections": 1,
      "connection_ids": ["abc-123"],
      "total_memory_gb": 15.75,
      "allocated_memory_gb": 7.45,
      "free_memory_gb": 8.30,
      "utilization_percent": 47.3
    },
    {
      "gpu_id": 1,
      "active_connections": 1,
      "connection_ids": ["def-456"],
      "total_memory_gb": 15.75,
      "allocated_memory_gb": 7.21,
      "free_memory_gb": 8.54,
      "utilization_percent": 45.8
    },
    {
      "gpu_id": 2,
      "active_connections": 0,
      "allocated_memory_gb": 0.00,
      ...
    }
  ]
}
```

---

## 🧪 Test Multiple Connections

### **Test 1: Single Connection**
1. Open browser tab 1: `http://localhost:8000`
2. Start transcription
3. Check GPU allocation: `curl http://localhost:8000/gpu-status`
4. Expected: 1 connection on GPU 0

### **Test 2: Multiple Connections (Round-Robin)**
1. Open tab 2: `http://localhost:8000`
2. Start transcription
3. Open tab 3: `http://localhost:8000`
4. Start transcription
5. Open tab 4: `http://localhost:8000`
6. Start transcription

**Check allocation:**
```bash
curl http://localhost:8000/gpu-status
```

**Expected:**
- Tab 1 → GPU 0
- Tab 2 → GPU 1
- Tab 3 → GPU 2
- Tab 4 → GPU 3

### **Test 3: Connection Limit**
1. Open 16 tabs, start transcription in each
2. Try opening tab 17 and starting transcription
3. Expected: WebSocket closes with error "No GPU resources available"

---

## 🐛 Debug Mode

### **Enable verbose logging:**

Edit `basic_server.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # Change from INFO to DEBUG
```

**You'll see detailed logs:**
```
DEBUG: Connection abc-123: Processing audio chunk (1024 bytes)
DEBUG: Transcription processor: Processing 0.5s of audio
DEBUG: GPU 0 memory: 7.45 GB / 15.75 GB
DEBUG: Generated 3 tokens: ["Hello", "world", "!"]
DEBUG: Sending transcription to client
```

---

## ✅ Success Indicators

### **Server Logs:**
```
INFO: Allocated GPU 0 to connection abc-123 (Load: 1/4, Memory: 3.21/15.75 GB)
INFO: Connection abc-123: WebSocket opened on GPU 0
INFO: TranscriptionEngine using GPU 0
INFO: SimulStreamingASR using GPU 0
INFO: Sortformer using GPU 0
```

### **Browser Console:**
```javascript
Connecting to WebSocket
Connected. Using MediaRecorder (WebM).
// Transcription results appearing...
Ready to stop received, finalizing display and closing WebSocket.
```

### **Web UI:**
- Transcription text appears in real-time
- Speaker labels show (if diarization enabled)
- No errors in status bar

---

## 🚨 Common Mistakes

### ❌ **Mistake 1: Testing against production server**
```
Websocket URL: wss://whisper2.primumai.eu/asr  ❌
```

✅ **Correct:**
```
Websocket URL: ws://localhost:8000/asr  ✅
```

### ❌ **Mistake 2: Not starting the server**
Opening `http://localhost:8000` without running `whisperlivekit-server` first.

### ❌ **Mistake 3: Wrong model size**
Using `--model large-v3` on a GPU with limited memory.

✅ **Use:** `--model small` or `--model base` for testing.

---

## 📖 Next Steps

After successful local testing:

1. **Deploy to AWS EC2 g4dn.12xlarge**
2. **Configure SSL certificates** for HTTPS/WSS
3. **Set up reverse proxy** (Nginx/Caddy)
4. **Monitor production logs**
5. **Load test** with multiple concurrent connections

---

## 💡 Quick Reference

### **Start server:**
```bash
whisperlivekit-server --host localhost --port 8000 --model small
```

### **Check GPUs:**
```bash
nvidia-smi
```

### **Check allocation:**
```bash
curl http://localhost:8000/gpu-status
```

### **View logs:**
```bash
# In the terminal where server is running
# Or if running as service:
journalctl -u whisperlivekit -f
```

---

## 🎓 Summary

1. ✅ **Always test locally first** (`ws://localhost:8000/asr`)
2. ✅ **Check GPU detection** in startup logs
3. ✅ **Monitor allocation** via `/gpu-status` endpoint
4. ✅ **Watch GPU memory** with `nvidia-smi`
5. ✅ **Verify transcription** appears in real-time

**Only deploy to production after local testing succeeds!**
