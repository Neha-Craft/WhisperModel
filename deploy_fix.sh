#!/bin/bash
# Deployment script for WhisperLiveKit with multi-GPU fix

echo "================================================"
echo "WhisperLiveKit Multi-GPU Fix Deployment"
echo "================================================"

# Stop existing server
echo "Stopping existing server..."
pkill -f whisperlivekit-server || echo "No running server found"
sleep 2

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Reinstall package in development mode
echo "Reinstalling package..."
pip install -e . --force-reinstall --no-deps

# Start server with optimal settings:
# - medium.en: English-only model (no Hindi/Chinese/Urdu detection)
# - --lan en: Force English language
# - --preload-model-count 2: Preload 2 models per GPU (2x4=8 total models for instant connections)
echo "Starting server with multi-GPU fix..."
nohup whisperlivekit-server \
  --model medium.en \
  --lan en \
  --preload-model-count 2 \
  --host 0.0.0.0 \
  --port 8000 > whisper.log 2>&1 &

echo "Server started. PID: $!"
echo "Waiting 10 seconds for initialization..."
sleep 10

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Deployment complete!"
echo "Check logs: tail -f whisper.log"
echo "Monitor GPUs: watch -n 1 nvidia-smi"
echo "Verify fix: python verify_gpu_fix.py"