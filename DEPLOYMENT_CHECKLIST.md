# 🚀 Deployment Checklist - whisper2.primumai.eu

## ⚡ Quick Deploy Steps

### **Step 1: Update Code on Server**

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@whisper2.primumai.eu

# Navigate to project directory
cd /path/to/WhisperLiveKit

# Pull latest changes
git pull origin main

# Or upload new files via SCP
scp -i your-key.pem whisperlivekit/*.py ec2-user@whisper2.primumai.eu:/path/to/WhisperLiveKit/whisperlivekit/
```

---

### **Step 2: Install/Update Package**

```bash
# On the EC2 server
pip install -e .

# Or reinstall
pip uninstall whisperlivekit
pip install -e .
```

---

### **Step 3: Restart Service**

```bash
# If running as systemd service
sudo systemctl restart whisperlivekit

# If running with screen/tmux
pkill -f whisperlivekit-server
whisperlivekit-server --host 0.0.0.0 --port 8000 --model small --backend simulstreaming

# Or with nohup
nohup whisperlivekit-server --host 0.0.0.0 --port 8000 --model small > /var/log/whisperlivekit.log 2>&1 &
```

---

### **Step 4: Verify Deployment**

```bash
# 1. Check server is running
curl https://whisper2.primumai.eu/health

# 2. Check GPU detection
curl https://whisper2.primumai.eu/gpu-status

# 3. Check logs
tail -f /var/log/whisperlivekit.log
# Or
journalctl -u whisperlivekit -f
```

**Expected in logs:**
```
INFO: ================================================================================
INFO: WhisperLiveKit Server Starting
INFO: ================================================================================
INFO: GPU ALLOCATION STATUS
INFO: GPU 0: 0/4 connections | Memory: 0.00/15.75 GB
INFO: GPU 1: 0/4 connections | Memory: 0.00/15.75 GB
INFO: GPU 2: 0/4 connections | Memory: 0.00/15.75 GB
INFO: GPU 3: 0/4 connections | Memory: 0.00/15.75 GB
```

---

### **Step 5: Test WebSocket Connection**

```bash
# Open browser
https://whisper2.primumai.eu

# Click red button to start transcription
# Should see: "Allocating GPU..." → "Loading models..." → "Ready"
```

**In browser console (F12):**
```javascript
Connecting to WebSocket
Connected. Using MediaRecorder (WebM).
```

---

## 📊 **Monitoring Endpoints (NEW)**

After deployment, these endpoints should work:

### **1. Health Check**
```bash
curl https://whisper2.primumai.eu/health | jq
```

### **2. GPU Status**
```bash
curl https://whisper2.primumai.eu/gpu-status | jq
```

### **3. Performance Metrics**
```bash
curl https://whisper2.primumai.eu/performance | jq
```

### **4. Prometheus Metrics**
```bash
curl https://whisper2.primumai.eu/metrics/prometheus
```

---

## 🔍 **Troubleshooting**

### **Issue: Endpoints return 404**

**Cause:** Old code still running  
**Solution:**
```bash
# Make sure you restarted the service
sudo systemctl status whisperlivekit

# Check which process is running
ps aux | grep whisperlivekit

# Kill old process
pkill -f whisperlivekit
```

---

### **Issue: "No GPUs detected"**

**Check CUDA:**
```bash
nvidia-smi

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

**Expected:**
```
True
4
```

---

### **Issue: Connection still slow**

**Check logs for timing:**
```bash
tail -f /var/log/whisperlivekit.log | grep "WebSocket accepted"
```

**Expected:**
```
Connection abc-123: WebSocket accepted (10ms)
Connection abc-123: GPU 0 allocated (5ms)
Connection abc-123: TranscriptionEngine loaded (2350ms)
```

If `WebSocket accepted` is > 1000ms, old code is still running!

---

## ✅ **Deployment Success Checklist**

- [ ] Server restarted successfully
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] `/gpu-status` shows 4 GPUs
- [ ] Browser can connect to WebSocket in < 1 second
- [ ] Logs show "WebSocket accepted (XXms)"
- [ ] Multiple connections distribute across GPUs
- [ ] GPU memory shown in `/gpu-status`

---

## 🎯 **Quick Test Script**

Save this as `test_deployment.sh`:

```bash
#!/bin/bash

echo "Testing WhisperLiveKit Deployment..."
echo ""

echo "1. Health Check:"
curl -s https://whisper2.primumai.eu/health | jq '.status'
echo ""

echo "2. GPU Count:"
curl -s https://whisper2.primumai.eu/gpu-status | jq '.total_gpus'
echo ""

echo "3. Active Connections:"
curl -s https://whisper2.primumai.eu/gpu-status | jq '.total_connections'
echo ""

echo "4. Performance Stats:"
curl -s https://whisper2.primumai.eu/performance | jq '.averages.setup_time_ms'
echo ""

echo "✅ Deployment test complete!"
```

Run it:
```bash
chmod +x test_deployment.sh
./test_deployment.sh
```

---

## 📝 **Files Modified**

These files need to be on the server:

1. `whisperlivekit/basic_server.py` ✨ Updated
2. `whisperlivekit/gpu_manager.py` ✨ New
3. `whisperlivekit/performance_monitor.py` ✨ New
4. `whisperlivekit/core.py` ✨ Updated
5. `whisperlivekit/simul_whisper/backend.py` ✨ Updated
6. `whisperlivekit/diarization/sortformer_backend.py` ✨ Updated

---

## 🚀 **Ready to Deploy!**

Your monitoring system is complete. After deployment:

1. Test locally first (if possible)
2. Deploy to `whisper2.primumai.eu`
3. Run the test script above
4. Monitor via `/health`, `/gpu-status`, `/performance`

Good luck! 🎉
