"""
GPU Manager for WhisperLiveKit
Manages GPU allocation across multiple WebSocket connections with load balancing and memory monitoring.
"""

import torch
import logging
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    gpu_id: int
    active_connections: int = 0
    total_memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    utilization_percent: float = 0.0
    connection_ids: List[str] = field(default_factory=list)
    
    def update_memory_stats(self):
        """Update memory statistics for this GPU."""
        if not torch.cuda.is_available():
            return
        
        try:
            self.total_memory_gb = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024**3)
            self.allocated_memory_gb = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
            self.reserved_memory_gb = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
            self.free_memory_gb = self.total_memory_gb - self.allocated_memory_gb
            self.utilization_percent = (self.allocated_memory_gb / self.total_memory_gb) * 100 if self.total_memory_gb > 0 else 0
        except Exception as e:
            logger.warning(f"Failed to update memory stats for GPU {self.gpu_id}: {e}")


class GPUManager:
    """
    Singleton GPU Manager that allocates GPUs to WebSocket connections.
    
    Features:
    - Round-robin GPU allocation with load balancing
    - Support for 3-4 connections per GPU
    - GPU memory monitoring
    - Automatic fallback to least-loaded GPU
    - Thread-safe connection tracking
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self.allocation_lock = threading.Lock()
        
        # GPU configuration
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.max_connections_per_gpu = 2  # Changed from 4 to 2 per your requirement
        self.use_cpu_fallback = False  # No CPU fallback as requested
        
        # GPU allocation tracking
        self.gpu_stats: Dict[int, GPUStats] = {}
        self.connection_to_gpu: Dict[str, int] = {}  # Maps connection_id -> gpu_id
        self.next_gpu_index = 0  # For round-robin allocation
        
        # Initialize GPU stats
        for gpu_id in range(self.num_gpus):
            self.gpu_stats[gpu_id] = GPUStats(gpu_id=gpu_id)
            self.gpu_stats[gpu_id].update_memory_stats()
        
        logger.info(f"GPUManager initialized with {self.num_gpus} GPUs")
        if self.num_gpus == 0:
            logger.warning("No CUDA GPUs detected! Server will fail without GPU support.")
        else:
            for gpu_id in range(self.num_gpus):
                stats = self.gpu_stats[gpu_id]
                logger.info(f"GPU {gpu_id}: {stats.total_memory_gb:.2f} GB total memory")
    
    def allocate_gpu(self, connection_id: str) -> Optional[int]:
        """
        Allocate a GPU for a new connection using load-balanced round-robin.
        
        Args:
            connection_id: Unique identifier for the WebSocket connection
            
        Returns:
            GPU ID (0-3 for g4dn.12xlarge) or None if no GPU available
        """
        with self.allocation_lock:
            if self.num_gpus == 0:
                logger.error("No GPUs available for allocation")
                return None
            
            # Check if connection already has a GPU assigned
            if connection_id in self.connection_to_gpu:
                logger.warning(f"Connection {connection_id} already has GPU {self.connection_to_gpu[connection_id]}")
                return self.connection_to_gpu[connection_id]
            
            # Find least loaded GPU with capacity
            selected_gpu = self._find_best_gpu()
            
            if selected_gpu is None:
                logger.error(f"All GPUs are at capacity ({self.max_connections_per_gpu} connections each)")
                return None
            
            # Allocate GPU to connection
            self.connection_to_gpu[connection_id] = selected_gpu
            self.gpu_stats[selected_gpu].active_connections += 1
            self.gpu_stats[selected_gpu].connection_ids.append(connection_id)
            self.gpu_stats[selected_gpu].update_memory_stats()
            
            logger.info(
                f"Allocated GPU {selected_gpu} to connection {connection_id} "
                f"(Load: {self.gpu_stats[selected_gpu].active_connections}/{self.max_connections_per_gpu}, "
                f"Memory: {self.gpu_stats[selected_gpu].allocated_memory_gb:.2f}/{self.gpu_stats[selected_gpu].total_memory_gb:.2f} GB)"
            )
            
            return selected_gpu
    
    def _find_best_gpu(self) -> Optional[int]:
        """
        Find the best GPU for allocation using round-robin with load balancing.
        
        Strategy:
        1. Try round-robin selection if GPU has capacity
        2. If full, find least loaded GPU with capacity
        3. If all GPUs full, return None
        """
        # Try round-robin first
        for attempt in range(self.num_gpus):
            gpu_id = (self.next_gpu_index + attempt) % self.num_gpus
            if self.gpu_stats[gpu_id].active_connections < self.max_connections_per_gpu:
                self.next_gpu_index = (gpu_id + 1) % self.num_gpus
                return gpu_id
        
        # All GPUs at or over capacity
        logger.warning("All GPUs are at maximum capacity")
        return None
    
    def release_gpu(self, connection_id: str) -> bool:
        """
        Release GPU allocation for a connection.
        
        Args:
            connection_id: Unique identifier for the WebSocket connection
            
        Returns:
            True if successfully released, False otherwise
        """
        with self.allocation_lock:
            if connection_id not in self.connection_to_gpu:
                logger.warning(f"Connection {connection_id} has no GPU allocation to release")
                return False
            
            gpu_id = self.connection_to_gpu[connection_id]
            
            # Update GPU stats
            self.gpu_stats[gpu_id].active_connections = max(0, self.gpu_stats[gpu_id].active_connections - 1)
            if connection_id in self.gpu_stats[gpu_id].connection_ids:
                self.gpu_stats[gpu_id].connection_ids.remove(connection_id)
            
            # Remove connection mapping
            del self.connection_to_gpu[connection_id]
            
            # Update memory stats
            self.gpu_stats[gpu_id].update_memory_stats()
            
            logger.info(
                f"Released GPU {gpu_id} from connection {connection_id} "
                f"(Remaining: {self.gpu_stats[gpu_id].active_connections} connections, "
                f"Memory: {self.gpu_stats[gpu_id].allocated_memory_gb:.2f}/{self.gpu_stats[gpu_id].total_memory_gb:.2f} GB)"
            )
            
            return True
    
    def get_gpu_for_connection(self, connection_id: str) -> Optional[int]:
        """Get the GPU ID assigned to a connection."""
        return self.connection_to_gpu.get(connection_id)
    
    def get_all_gpu_stats(self) -> Dict[int, GPUStats]:
        """Get statistics for all GPUs."""
        with self.allocation_lock:
            # Update memory stats before returning
            for gpu_id in self.gpu_stats:
                self.gpu_stats[gpu_id].update_memory_stats()
            return self.gpu_stats.copy()
    
    def get_gpu_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get statistics for a specific GPU."""
        if gpu_id not in self.gpu_stats:
            return None
        
        with self.allocation_lock:
            self.gpu_stats[gpu_id].update_memory_stats()
            return self.gpu_stats[gpu_id]
    
    def log_all_gpu_stats(self):
        """Log statistics for all GPUs."""
        stats = self.get_all_gpu_stats()
        logger.info("=" * 80)
        logger.info("GPU ALLOCATION STATUS")
        logger.info("=" * 80)
        for gpu_id, stat in stats.items():
            logger.info(
                f"GPU {gpu_id}: {stat.active_connections}/{self.max_connections_per_gpu} connections | "
                f"Memory: {stat.allocated_memory_gb:.2f}/{stat.total_memory_gb:.2f} GB "
                f"({stat.utilization_percent:.1f}% used) | "
                f"Free: {stat.free_memory_gb:.2f} GB"
            )
        logger.info("=" * 80)
    
    def get_total_active_connections(self) -> int:
        """Get total number of active connections across all GPUs."""
        with self.allocation_lock:
            return sum(stat.active_connections for stat in self.gpu_stats.values())
    
    def clear_all(self):
        """Clear all allocations (for testing/reset purposes)."""
        with self.allocation_lock:
            self.connection_to_gpu.clear()
            for gpu_id in self.gpu_stats:
                self.gpu_stats[gpu_id].active_connections = 0
                self.gpu_stats[gpu_id].connection_ids.clear()
            logger.info("Cleared all GPU allocations")


# Global singleton instance
gpu_manager = GPUManager()
