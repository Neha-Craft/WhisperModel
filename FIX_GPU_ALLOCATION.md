# 🔧 GPU Allocation Fix - Critical Issue Resolved

## 🚨 **Problem Identified**

You reported: **"One GPU is getting maxed out, processes not going to GPU 2 or 3"**

**Root Cause Found:**  
The `gpu_id` and `device` parameters were **NOT being passed** from [`TranscriptionEngine`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\core.py) to [`SimulStreamingASR`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\simul_whisper\backend.py), causing all models to load on **GPU 0 (default)** instead of the assigned GPU.

---

## ✅ **What Was Fixed**

### **1. Pass GPU Info to ASR Backend** ([`core.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\core.py) line 112-115)

**Before:**
```python
self.asr = SimulStreamingASR(
    **transcription_common_params, **simulstreaming_params
)
```

**After:**
```python
# CRITICAL: Pass GPU device info to SimulStreamingASR
simulstreaming_params['gpu_id'] = self.gpu_id
simulstreaming_params['device'] = self.device

self.asr = SimulStreamingASR(
    **transcription_common_params, **simulstreaming_params
)
```

### **2. Changed Connection Limit** ([`gpu_manager.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\gpu_manager.py) line 53)

**Before:**
```python
self.max_connections_per_gpu = 4
```

**After:**
```python
self.max_connections_per_gpu = 2  # 2 connections per GPU as requested
```

### **3. Enhanced Logging** ([`backend.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\simul_whisper\backend.py) line 273-280)

Added detailed GPU memory logging:
```python
logger.info(f"✅ Loaded Whisper model on {self.device} (GPU {self.gpu_id})")
allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
logger.info(f"📊 GPU {self.gpu_id} memory after model load: {allocated:.2f} GB")
```

---

## 🚀 **Expected Behavior After Fix**

### **With 2 connections per GPU:**

| Connection | GPU Assigned | nvitop/nvidia-smi Memory |
|------------|--------------|--------------------------|
| 1st        | GPU 0        | GPU 0: ~3.5 GB           |
| 2nd        | GPU 1        | GPU 1: ~3.5 GB           |
| 3rd        | GPU 2        | GPU 2: ~3.5 GB           |
| 4th        | GPU 3        | GPU 3: ~3.5 GB           |
| 5th        | GPU 0        | GPU 0: ~7.0 GB (2 connections) |
| 6th        | GPU 1        | GPU 1: ~7.0 GB (2 connections) |
| 7th        | GPU 2        | GPU 2: ~7.0 GB (2 connections) |
| 8th        | GPU 3        | GPU 3: ~7.0 GB (2 connections) |
| 9th        | ❌ REJECTED  | "No GPU resources available" |

---

## 📋 **Deployment Checklist**

### **Step 1: Deploy Updated Code**

```bash
# SSH into server
ssh -i your-key.pem ec2-user@whisper2.primumai.eu

# Navigate to project
cd /path/to/WhisperLiveKit

# Pull changes (if using git)
git pull origin main

# OR upload files manually:
# Upload these 3 critical files:
# - whisperlivekit/core.py (GPU param passing fix)
# - whisperlivekit/gpu_manager.py (2 connections limit)
# - whisperlivekit/simul_whisper/backend.py (enhanced logging)
```

### **Step 2: Reinstall Package**

```bash
pip install -e .
```

### **Step 3: Restart Server**

```bash
# If using systemd
sudo systemctl restart whisperlivekit

# If using screen/tmux
pkill -f whisperlivekit-server
whisperlivekit-server --host 0.0.0.0 --port 8000 --model small --backend simulstreaming
```

### **Step 4: Verify Fix**

```bash
# Run the diagnostic tool
python test_gpu_allocation.py

# Or manually check
curl https://whisper2.primumai.eu/gpu-status | jq
```

---

## 🔍 **How to Verify GPU Allocation is Working**

### **Method 1: Use the Diagnostic Script**

```bash
# One-time check
python test_gpu_allocation.py

# Continuous monitoring
python test_gpu_allocation.py --watch
```

**Expected output:**
```
🔍 GPU Allocation Verification Tool
================================================================================

1️⃣  Checking system health...
   Status: HEALTHY
   Active connections: 4
   Available slots: 4

2️⃣  Checking GPU allocation...
   Total GPUs: 4
   Max connections per GPU: 2
   Total active connections: 4

3️⃣  GPU Distribution Analysis:

   GPU | Connections | Memory Used | Memory Free | Utilization
   ----------------------------------------------------------------------
   ✅ 0  |     1       |   3.45 GB  |  12.30 GB |  21.9%
   ✅ 1  |     1       |   3.42 GB  |  12.33 GB |  21.7%
   ✅ 2  |     1       |   3.48 GB  |  12.27 GB |  22.1%
   ✅ 3  |     1       |   3.41 GB  |  12.34 GB |  21.6%

4️⃣  Verification Results:

   ✅ PASS: Connections are balanced across GPUs
      (Difference: 0 connection)
   ✅ PASS: All active GPUs have memory allocated
      4 GPU(s) with connections and memory
   ✅ PASS: No GPU exceeds limit (2 connections)

================================================================================
✅ ALL CHECKS PASSED - GPU allocation is working correctly!
================================================================================
```

---

### **Method 2: Check Server Logs**

```bash
# Watch logs in real-time
journalctl -u whisperlivekit -f

# Or tail log file
tail -f /var/log/whisperlivekit.log
```

**Look for these log lines (one per connection):**
```
INFO: Allocated GPU 0 to connection abc-123 (Load: 1/2, Memory: 3.21/15.75 GB)
INFO: Connection abc-123: WebSocket opened on GPU 0
INFO: TranscriptionEngine using GPU 0
INFO: SimulStreamingASR using GPU 0
INFO: ✅ Loaded Whisper model on cuda:0 (GPU 0)
INFO: 📊 GPU 0 memory after model load: 3.45 GB

INFO: Allocated GPU 1 to connection def-456 (Load: 1/2, Memory: 3.18/15.75 GB)
INFO: Connection def-456: WebSocket opened on GPU 1
INFO: TranscriptionEngine using GPU 1
INFO: SimulStreamingASR using GPU 1
INFO: ✅ Loaded Whisper model on cuda:1 (GPU 1)
INFO: 📊 GPU 1 memory after model load: 3.42 GB
```

✅ **CORRECT**: Each connection uses a different GPU (0, 1, 2, 3)  
❌ **WRONG**: All connections show "GPU 0"

---

### **Method 3: Check with nvitop/nvidia-smi**

```bash
# SSH to server
ssh -i your-key.pem ec2-user@whisper2.primumai.eu

# Watch GPU usage
nvitop

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

**Expected (4 connections, 1 per GPU):**
```
+-----------------------------------------------------------------------------+
| GPU  Name        | Memory-Usage | GPU-Util | Processes                     |
+------------------+--------------+----------+-------------------------------+
|   0  Tesla T4    |  3500 / 15360 MiB |  25% | python (PID: 12345)           |
|   1  Tesla T4    |  3420 / 15360 MiB |  24% | python (PID: 12346)           |
|   2  Tesla T4    |  3480 / 15360 MiB |  25% | python (PID: 12347)           |
|   3  Tesla T4    |  3410 / 15360 MiB |  24% | python (PID: 12348)           |
+-----------------------------------------------------------------------------+
```

✅ **CORRECT**: All 4 GPUs show ~3.5GB memory usage  
❌ **WRONG BEFORE FIX**: Only GPU 0 shows 14GB, others at 0MB

---

### **Method 4: Check API Endpoint**

```bash
# Check GPU allocation
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | {gpu_id, connections: .active_connections, memory: .allocated_memory_gb}'
```

**Expected output:**
```json
[
  {"gpu_id": 0, "connections": 1, "memory": 3.45},
  {"gpu_id": 1, "connections": 1, "memory": 3.42},
  {"gpu_id": 2, "connections": 1, "memory": 3.48},
  {"gpu_id": 3, "connections": 1, "memory": 3.41}
]
```

✅ **CORRECT**: All GPUs have memory allocated  
❌ **WRONG**: Only GPU 0 has memory > 0

---

## 🎯 **Test Procedure**

### **Test 1: Single Connection**
1. Open browser: `https://whisper2.primumai.eu`
2. Start transcription
3. Check: `python test_gpu_allocation.py`
4. **Expected**: GPU 0 has 1 connection, ~3.5GB memory

### **Test 2: Second Connection**
1. Open new browser tab
2. Start transcription
3. Check: `python test_gpu_allocation.py`
4. **Expected**: GPU 1 has 1 connection, ~3.5GB memory

### **Test 3: Multiple Connections**
1. Open 5 connections
2. Check: `python test_gpu_allocation.py`
3. **Expected**:
   - Connections 1-4: GPU 0-3 (1 connection each)
   - Connection 5: GPU 0 (2 connections on GPU 0)
   - All active GPUs show memory allocation

### **Test 4: Connection Limit**
1. Open 8 connections (2 per GPU = max capacity)
2. Try opening 9th connection
3. **Expected**: WebSocket closes with "No GPU resources available"

---

## 🐛 **Troubleshooting**

### **Issue 1: Still all connections on GPU 0**

**Check:**
```bash
# Verify updated code is running
grep -n "CRITICAL: Pass GPU device info" /path/to/whisperlivekit/core.py

# Should find the line (around line 112)
```

**If not found:**
- Code not deployed
- Old version still running
- Need to restart server

---

### **Issue 2: Logs show GPU X but nvitop shows GPU 0**

**This means the device parameter is not being used.**

**Check:**
```bash
# Look for this in logs
grep "SimulStreamingASR using GPU" /var/log/whisperlivekit.log

# Should see different GPU numbers
```

**If all show GPU 0:**
- The device parameter is not being passed to `model.to(device)`
- Check `backend.py` line 273-280

---

### **Issue 3: Memory not showing in /gpu-status but shows in nvitop**

**API uses PyTorch's memory tracking, which may lag behind hardware.**

**Try:**
```bash
# Force refresh
curl https://whisper2.primumai.eu/gpu-status

# Wait 2-3 seconds, check again
curl https://whisper2.primumai.eu/gpu-status
```

---

## 📊 **Files Modified Summary**

| File | Lines Changed | Purpose |
|------|---------------|---------|
| [`core.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\core.py) | 112-115 | Pass GPU params to ASR |
| [`gpu_manager.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\gpu_manager.py) | 53 | Change limit to 2 connections/GPU |
| [`backend.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\whisperlivekit\simul_whisper\backend.py) | 273-280 | Enhanced GPU logging |

**New Files:**
- [`test_gpu_allocation.py`](file://c:\Users\abhil\OneDrive\Desktop\CursorPilot\WhisperLiveKit\test_gpu_allocation.py) - Diagnostic tool

---

## ✅ **Success Criteria**

Your GPU allocation is **working correctly** if:

- ✅ Connections distributed evenly (±0-1 per GPU)
- ✅ All active GPUs show memory in `nvitop`
- ✅ Logs show different GPU numbers (0, 1, 2, 3)
- ✅ API `/gpu-status` shows memory on all active GPUs
- ✅ 9th connection rejected when 8 active (2×4 GPUs)

---

## 🚀 **Quick Commands**

```bash
# Deploy and test
ssh ec2-user@whisper2.primumai.eu
cd /path/to/WhisperLiveKit
git pull
pip install -e .
sudo systemctl restart whisperlivekit

# Verify
python test_gpu_allocation.py

# Monitor
watch -n 2 'curl -s https://whisper2.primumai.eu/gpu-status | jq ".gpus[] | {gpu_id, connections: .active_connections}"'
```

---

## 🎉 **After This Fix**

You should see in `nvitop`:
- **5 connections** = GPU 0 (2), GPU 1 (2), GPU 2 (1), GPU 3 (0)
- **ALL active GPUs show memory usage**
- **No single GPU maxed out**

The fix ensures models load on the **assigned GPU**, not the default GPU 0!
