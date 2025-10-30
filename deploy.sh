#!/bin/bash
# Deployment script for WhisperLiveKit with GPU load balancing fix

echo "================================================"
echo "WhisperLiveKit Deployment Script"
echo "================================================"

# Stop existing server
echo "Stopping existing server..."
pkill -f whisperlivekit-server || echo "No running server found"
sleep 2

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Pull latest changes
echo "Pulling latest changes from git..."
git pull origin master

# Reinstall package in development mode
echo "Reinstalling package..."
pip install -e . --force-reinstall --no-deps

# Start server
echo "Starting server..."
nohup whisperlivekit-server --model medium --host 0.0.0.0 --port 8000 > whisper.log 2>&1 &

echo "Server started. PID: $!"
echo "Waiting 5 seconds for initialization..."
sleep 5

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Deployment complete!"
echo "Check logs: tail -f whisper.log"
echo "Monitor GPUs: watch -n 1 nvidia-smi"
