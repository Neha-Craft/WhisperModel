# 📊 Monitoring & Scaling Guide - WhisperLiveKit GPU System

## 🎯 Your Questions Answered

### **1. Why is WebSocket connection slow?**
### **2. How to check if it's scaling properly?**
### **3. How to verify GPU usage is correct?**

This guide provides complete answers with monitoring tools and diagnostic steps.

---

## 🚀 **Performance Optimization - WebSocket Connection Speed**

### **Problem: Slow Connection**

When you click the red button on `whisper2.primumai.eu`, the delay comes from **model loading happening BEFORE accepting the WebSocket**.

### **Solution Implemented**

I've optimized the connection flow to accept WebSocket **immediately** and load models in the background:

```
OLD FLOW (SLOW):
Client connects → Load models (2-5s) → Accept WebSocket → Ready
                    ^^^^^^ CLIENT WAITS HERE

NEW FLOW (FAST):
Client connects → Accept WebSocket (10ms) → Load models → Ready
                                ^^^^^^ CLIENT CONNECTED IMMEDIATELY
```

### **Expected Timing (After Update)**

| Step | Time | Status Sent to Client |
|------|------|----------------------|
| WebSocket Accept | 10-50ms | ✅ "Connected" |
| GPU Allocation | 1-5ms | "Allocating GPU..." |
| Model Loading | 1-3s | "Loading models on GPU X..." |
| Ready | - | "Ready to transcribe" |

**Total perceived delay: <100ms** (client knows it's connected)

---

## 📡 **NEW Monitoring Endpoints**

I've added **4 new endpoints** to monitor your system:

### **1. `/gpu-status` - GPU Allocation & Memory**

**What it shows:**
- Which GPUs are active
- Connections per GPU
- GPU memory usage
- Connection IDs

**How to use:**
```bash
curl https://whisper2.primumai.eu/gpu-status
```

**Example response:**
```json
{
  "total_gpus": 4,
  "total_connections": 8,
  "max_connections_per_gpu": 4,
  "gpus": [
    {
      "gpu_id": 0,
      "active_connections": 2,
      "connection_ids": ["abc-123", "def-456"],
      "total_memory_gb": 15.75,
      "allocated_memory_gb": 7.45,
      "free_memory_gb": 8.30,
      "utilization_percent": 47.3
    },
    {
      "gpu_id": 1,
      "active_connections": 2,
      ...
    },
    {
      "gpu_id": 2,
      "active_connections": 2,
      ...
    },
    {
      "gpu_id": 3,
      "active_connections": 2,
      ...
    }
  ]
}
```

**What to look for:**
- ✅ **Balanced load**: All GPUs have similar `active_connections`
- ✅ **Memory headroom**: `free_memory_gb` > 2 GB per GPU
- ⚠️ **High utilization**: If `utilization_percent` > 90%, scale up!

---

### **2. `/performance` - Connection Performance Stats**

**What it shows:**
- Average setup time
- Connection duration
- Audio chunks processed
- Transcriptions generated

**How to use:**
```bash
curl https://whisper2.primumai.eu/performance
```

**Example response:**
```json
{
  "total_completed_connections": 127,
  "active_connections": 8,
  "averages": {
    "setup_time_ms": 2350.5,
    "connection_duration_s": 45.3,
    "audio_chunks_per_connection": 892.1,
    "transcriptions_per_connection": 234.7
  },
  "extremes": {
    "slowest_setup_ms": 4521.2,
    "fastest_setup_ms": 1834.1
  }
}
```

**What to look for:**
- ✅ **Fast setup**: `setup_time_ms` < 3000ms (3 seconds)
- ⚠️ **Slow setup**: > 5000ms means GPU loading issues
- ✅ **Consistent times**: Small difference between slowest/fastest

---

### **3. `/health` - Overall System Health**

**What it shows:**
- System health status (healthy/degraded/critical)
- Capacity utilization
- GPU status summary

**How to use:**
```bash
curl https://whisper2.primumai.eu/health
```

**Example response:**
```json
{
  "status": "healthy",
  "timestamp": 1698765432.123,
  "capacity": {
    "total_slots": 16,
    "used_slots": 8,
    "available_slots": 8,
    "utilization_percent": 50.0
  },
  "gpus": {
    "total": 4,
    "with_connections": 4
  },
  "performance": {
    "total_completed_connections": 127,
    "active_connections": 8,
    ...
  }
}
```

**Health Status:**
- `healthy`: < 70% capacity used
- `degraded`: 70-90% capacity used
- `critical`: > 90% capacity used (scale up!)

---

### **4. `/metrics/prometheus` - Prometheus Metrics**

**What it shows:**
- Prometheus-compatible metrics for Grafana dashboards

**How to use:**
```bash
curl https://whisper2.primumai.eu/metrics/prometheus
```

**Example response:**
```
# HELP whisperlivekit_gpu_total Total number of GPUs
# TYPE whisperlivekit_gpu_total gauge
whisperlivekit_gpu_total 4

# HELP whisperlivekit_connections_active Active WebSocket connections
# TYPE whisperlivekit_connections_active gauge
whisperlivekit_connections_active{gpu="0"} 2
whisperlivekit_connections_active{gpu="1"} 2
whisperlivekit_connections_active{gpu="2"} 2
whisperlivekit_connections_active{gpu="3"} 2

# HELP whisperlivekit_gpu_memory_allocated_gb GPU memory allocated in GB
# TYPE whisperlivekit_gpu_memory_allocated_gb gauge
whisperlivekit_gpu_memory_allocated_gb{gpu="0"} 7.45
...
```

---

## ✅ **How to Verify GPU Scaling is Working**

### **Test 1: Single Connection**

```bash
# Step 1: Check initial state
curl https://whisper2.primumai.eu/gpu-status | jq '.total_connections'
# Expected: 0

# Step 2: Start 1 transcription in browser
# (Open https://whisper2.primumai.eu, click red button)

# Step 3: Check allocation
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | {gpu_id, active_connections}'
# Expected: GPU 0 has 1 connection
```

**Result:**
```json
[
  {"gpu_id": 0, "active_connections": 1},
  {"gpu_id": 1, "active_connections": 0},
  {"gpu_id": 2, "active_connections": 0},
  {"gpu_id": 3, "active_connections": 0}
]
```

✅ **PASS**: Connection assigned to GPU 0

---

### **Test 2: Round-Robin Distribution**

```bash
# Open 8 browser tabs
# Start transcription in each tab

# Check distribution
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | {gpu_id, active_connections}'
```

**Expected result:**
```json
[
  {"gpu_id": 0, "active_connections": 2},
  {"gpu_id": 1, "active_connections": 2},
  {"gpu_id": 2, "active_connections": 2},
  {"gpu_id": 3, "active_connections": 2}
]
```

✅ **PASS**: Connections distributed evenly (round-robin working)

❌ **FAIL**: If all connections on GPU 0 (round-robin broken)

---

### **Test 3: Capacity Limit**

```bash
# Open 16 tabs (4 per GPU = max capacity)
# All should connect successfully

# Try opening tab 17
# Expected: Connection rejected with "No GPU resources available"
```

**Check health:**
```bash
curl https://whisper2.primumai.eu/health | jq '.status'
# Expected: "critical" (100% capacity)
```

✅ **PASS**: 17th connection rejected, health shows "critical"

---

## 🔍 **How to Verify GPU Usage (Hardware Level)**

### **Method 1: nvidia-smi (Direct Server Access)**

If you have SSH access to the EC2 instance:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or get specific metrics
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

**Expected output (8 connections active):**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.X  |
|-------------------------------+----------------------+----------------------+
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| 35C    P0    28W / 70W |   7456MiB / 15360MiB |      45%           |
|   1  Tesla T4            On   | 00000000:00:1B.0 Off |                    0 |
| 32C    P0    26W / 70W |   7321MiB / 15360MiB |      42%           |
|   2  Tesla T4            On   | 00000000:00:1C.0 Off |                    0 |
| 34C    P0    27W / 70W |   7289MiB / 15360MiB |      43%           |
|   3  Tesla T4            On   | 00000000:00:1D.0 Off |                    0 |
| 33C    P0    28W / 70W |   7402MiB / 15360MiB |      46%           |
+-----------------------------------------------------------------------------+
```

**What to look for:**
- ✅ All 4 GPUs show memory usage (~7GB each with 2 connections)
- ✅ GPU utilization 40-60% (actively processing)
- ✅ Power consumption 25-30W per GPU (active)
- ⚠️ GPU 0 at 100%, others at 0% → Round-robin broken!

---

### **Method 2: CloudWatch Metrics (AWS Console)**

1. Open AWS CloudWatch
2. Navigate to EC2 → Metrics → GPU
3. Select your `g4dn.12xlarge` instance
4. View metrics:
   - `GPUUtilization` (should be balanced across 4 GPUs)
   - `GPUMemoryUsed` (should show ~7GB per active GPU)
   - `GPUTemperature` (should be 30-40°C)

---

### **Method 3: Application Metrics (Our API)**

```bash
# Check memory usage per GPU
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | {gpu_id, allocated_memory_gb, free_memory_gb}'
```

**Expected (2 connections per GPU):**
```json
[
  {"gpu_id": 0, "allocated_memory_gb": 7.45, "free_memory_gb": 8.30},
  {"gpu_id": 1, "allocated_memory_gb": 7.21, "free_memory_gb": 8.54},
  {"gpu_id": 2, "allocated_memory_gb": 7.33, "free_memory_gb": 8.42},
  {"gpu_id": 3, "allocated_memory_gb": 7.18, "free_memory_gb": 8.57}
]
```

✅ **PASS**: All GPUs have memory allocated (~7GB)  
❌ **FAIL**: Only GPU 0 has memory, others at 0GB

---

## 📈 **Scaling Verification Checklist**

Run these tests to confirm scaling works:

### ✅ **1. Connection Distribution**
```bash
# Start 8 connections, verify each GPU has 2
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[].active_connections'
# Expected: [2, 2, 2, 2]
```

### ✅ **2. Memory Distribution**
```bash
# Check memory is allocated on all GPUs
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | select(.allocated_memory_gb > 0) | .gpu_id'
# Expected: [0, 1, 2, 3]
```

### ✅ **3. Performance Consistency**
```bash
# Setup time should be similar for all connections
curl https://whisper2.primumai.eu/performance | jq '.extremes'
# Expected: slowest_setup_ms ~2x fastest_setup_ms (not 10x)
```

### ✅ **4. Health Status**
```bash
# With 8/16 slots used (50%), should be healthy
curl https://whisper2.primumai.eu/health | jq '.status'
# Expected: "healthy"
```

### ✅ **5. Capacity Enforcement**
```bash
# Try 17th connection (should fail)
# Check logs for: "No GPU resources available"
```

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: All connections go to GPU 0**

**Symptoms:**
```bash
curl /gpu-status
# GPU 0: 8 connections
# GPU 1-3: 0 connections
```

**Cause:** GPU manager not working  
**Solution:**
```bash
# Check logs
journalctl -u whisperlivekit -n 100 | grep "Allocated GPU"

# Should see:
# Allocated GPU 0 to connection...
# Allocated GPU 1 to connection...
# Allocated GPU 2 to connection...
# Allocated GPU 3 to connection...
```

---

### **Issue 2: Slow connection (5-10 seconds)**

**Symptoms:** Client waits 5-10s before seeing "Connected"

**Cause:** WebSocket not accepted immediately  
**Check:** Server logs should show:
```
Connection abc-123: WebSocket accepted (10ms)
Connection abc-123: GPU 0 allocated (5ms)
Connection abc-123: TranscriptionEngine loaded (2350ms)
```

If `WebSocket accepted` is > 100ms, update your code!

---

### **Issue 3: GPU memory not released**

**Symptoms:**
```bash
# After disconnecting 4 connections
curl /gpu-status
# GPU 0 still shows 7GB allocated
```

**Cause:** Cleanup not working  
**Solution:** Check logs for:
```
Released GPU 0 from connection abc-123
```

---

## 📊 **Grafana Dashboard (Optional)**

### **Setup Prometheus Scraping**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'whisperlivekit'
    static_configs:
      - targets: ['whisper2.primumai.eu:443']
    metrics_path: '/metrics/prometheus'
    scheme: 'https'
```

### **Grafana Panels to Create**

1. **GPU Connections (Time Series)**
   - Query: `whisperlivekit_connections_active`
   - Split by: `gpu`

2. **GPU Memory (Stacked Area)**
   - Query: `whisperlivekit_gpu_memory_allocated_gb`

3. **Connection Setup Time (Histogram)**
   - Query: `whisperlivekit_setup_time_ms_avg`

4. **System Capacity (Gauge)**
   - Query: `(sum(whisperlivekit_connections_active) / (whisperlivekit_gpu_total * 4)) * 100`

---

## 🎓 **Quick Monitoring Commands**

### **Check if system is scaling:**
```bash
curl https://whisper2.primumai.eu/health | jq '{status, capacity}'
```

### **Check GPU distribution:**
```bash
curl https://whisper2.primumai.eu/gpu-status | jq '.gpus[] | {gpu_id, connections: .active_connections}'
```

### **Check performance:**
```bash
curl https://whisper2.primumai.eu/performance | jq '.averages.setup_time_ms'
```

### **Monitor in real-time:**
```bash
watch -n 2 'curl -s https://whisper2.primumai.eu/health | jq .'
```

---

## 🎯 **Success Criteria**

Your system is **properly scaled** if:

✅ Connections distributed evenly (±1 connection per GPU)  
✅ All GPUs show memory allocation when connections active  
✅ Setup time < 3 seconds average  
✅ Health status = "healthy" at < 70% capacity  
✅ 17th connection rejected when at max capacity  
✅ GPU memory released after disconnect  

---

## 📞 **Debugging Workflow**

If something isn't working:

1. **Check health:** `curl /health`
2. **Check GPU allocation:** `curl /gpu-status`
3. **Check performance:** `curl /performance`
4. **Check server logs:** `journalctl -u whisperlivekit -f`
5. **Check GPU hardware:** `nvidia-smi`

---

## 🚀 **You're All Set!**

Your WhisperLiveKit server now has:
- ✅ Fast WebSocket connection (< 100ms)
- ✅ 4 monitoring endpoints
- ✅ Performance tracking
- ✅ Health checks
- ✅ Prometheus metrics

**Test it now on `whisper2.primumai.eu`!** 🎉
