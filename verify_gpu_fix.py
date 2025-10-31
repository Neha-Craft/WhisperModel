#!/usr/bin/env python3
"""
Script to verify that the multi-GPU fix is working correctly.
This script checks the GPU allocation and model distribution.
"""

import requests
import time
import json

def check_gpu_status():
    """Check the GPU status via the API."""
    try:
        response = requests.get('http://localhost:8000/gpu-status', timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get GPU status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")
        return None

def main():
    print("WhisperLiveKit Multi-GPU Fix Verification")
    print("=" * 50)
    
    # Check initial GPU status
    print("Checking initial GPU status...")
    gpu_status = check_gpu_status()
    
    if gpu_status:
        print(f"Total GPUs: {gpu_status['total_gpus']}")
        print(f"Total connections: {gpu_status['total_connections']}")
        print("\nGPU Details:")
        for gpu in gpu_status['gpus']:
            print(f"  GPU {gpu['gpu_id']}: {gpu['active_connections']} connections, "
                  f"{gpu['allocated_memory_gb']} GB allocated, "
                  f"{gpu['utilization_percent']}% utilization")
        
        # Check if multiple GPUs are being used
        active_gpus = sum(1 for gpu in gpu_status['gpus'] if gpu['active_connections'] > 0)
        total_connections = gpu_status['total_connections']
        
        print(f"\nActive GPUs: {active_gpus}")
        print(f"Total connections: {total_connections}")
        
        if active_gpus > 1 and total_connections > 1:
            print("✅ SUCCESS: Multiple GPUs are being utilized!")
            print("✅ Multi-GPU allocation fix is working correctly.")
        elif total_connections > 1 and active_gpus == 1:
            print("⚠️  WARNING: Multiple connections but only one GPU is active.")
            print("⚠️  The fix may not be working correctly.")
        else:
            print("ℹ️  INFO: No active connections or single connection.")
            print("ℹ️  Start multiple connections to verify the fix.")
    else:
        print("❌ Failed to get GPU status. Is the server running?")

if __name__ == "__main__":
    main()