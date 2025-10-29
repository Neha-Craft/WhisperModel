"""
Performance monitoring utilities for WhisperLiveKit.
Tracks connection times, GPU usage, and system metrics.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for a single WebSocket connection."""
    connection_id: str
    gpu_id: int
    start_time: float
    accept_time: float = 0.0
    gpu_alloc_time: float = 0.0
    engine_load_time: float = 0.0
    processor_create_time: float = 0.0
    first_audio_time: float = 0.0
    first_transcription_time: float = 0.0
    end_time: float = 0.0
    total_audio_chunks: int = 0
    total_transcriptions: int = 0
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing breakdown in milliseconds."""
        if self.start_time == 0:
            return {}
        
        return {
            "websocket_accept_ms": (self.accept_time - self.start_time) * 1000 if self.accept_time else 0,
            "gpu_allocation_ms": (self.gpu_alloc_time - self.accept_time) * 1000 if self.gpu_alloc_time else 0,
            "engine_loading_ms": (self.engine_load_time - self.gpu_alloc_time) * 1000 if self.engine_load_time else 0,
            "processor_creation_ms": (self.processor_create_time - self.engine_load_time) * 1000 if self.processor_create_time else 0,
            "total_setup_ms": (self.processor_create_time - self.start_time) * 1000 if self.processor_create_time else 0,
            "time_to_first_audio_ms": (self.first_audio_time - self.start_time) * 1000 if self.first_audio_time else 0,
            "time_to_first_transcription_ms": (self.first_transcription_time - self.start_time) * 1000 if self.first_transcription_time else 0,
            "total_duration_s": (self.end_time - self.start_time) if self.end_time else (time.time() - self.start_time),
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        timings = self.get_timing_summary()
        return {
            "connection_id": self.connection_id,
            "gpu_id": self.gpu_id,
            "timings": timings,
            "audio_chunks_processed": self.total_audio_chunks,
            "transcriptions_generated": self.total_transcriptions,
        }


class PerformanceMonitor:
    """
    Global performance monitor for tracking connection metrics.
    Thread-safe singleton for monitoring all connections.
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
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.metrics_lock = threading.Lock()
        self.active_connections: Dict[str, ConnectionMetrics] = {}
        self.completed_connections: List[ConnectionMetrics] = []
        self.max_history = 100  # Keep last 100 completed connections
        
        logger.info("PerformanceMonitor initialized")
    
    def start_connection(self, connection_id: str, gpu_id: int) -> ConnectionMetrics:
        """Start tracking a new connection."""
        with self.metrics_lock:
            metrics = ConnectionMetrics(
                connection_id=connection_id,
                gpu_id=gpu_id,
                start_time=time.time()
            )
            self.active_connections[connection_id] = metrics
            logger.debug(f"Started tracking connection {connection_id} on GPU {gpu_id}")
            return metrics
    
    def record_accept(self, connection_id: str):
        """Record WebSocket accept time."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].accept_time = time.time()
    
    def record_gpu_allocation(self, connection_id: str):
        """Record GPU allocation time."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].gpu_alloc_time = time.time()
    
    def record_engine_loaded(self, connection_id: str):
        """Record TranscriptionEngine load time."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].engine_load_time = time.time()
    
    def record_processor_created(self, connection_id: str):
        """Record AudioProcessor creation time."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].processor_create_time = time.time()
    
    def record_first_audio(self, connection_id: str):
        """Record time when first audio chunk received."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                metrics = self.active_connections[connection_id]
                if metrics.first_audio_time == 0:
                    metrics.first_audio_time = time.time()
    
    def record_first_transcription(self, connection_id: str):
        """Record time when first transcription sent."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                metrics = self.active_connections[connection_id]
                if metrics.first_transcription_time == 0:
                    metrics.first_transcription_time = time.time()
    
    def record_audio_chunk(self, connection_id: str):
        """Increment audio chunk counter."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].total_audio_chunks += 1
    
    def record_transcription(self, connection_id: str):
        """Increment transcription counter."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                self.active_connections[connection_id].total_transcriptions += 1
    
    def end_connection(self, connection_id: str):
        """Mark connection as complete and move to history."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                metrics = self.active_connections[connection_id]
                metrics.end_time = time.time()
                
                # Log summary
                summary = metrics.get_timing_summary()
                logger.info(
                    f"Connection {connection_id} completed: "
                    f"Setup={summary.get('total_setup_ms', 0):.0f}ms, "
                    f"Duration={summary.get('total_duration_s', 0):.1f}s, "
                    f"Audio chunks={metrics.total_audio_chunks}, "
                    f"Transcriptions={metrics.total_transcriptions}"
                )
                
                # Move to history
                self.completed_connections.append(metrics)
                if len(self.completed_connections) > self.max_history:
                    self.completed_connections.pop(0)
                
                del self.active_connections[connection_id]
    
    def get_connection_metrics(self, connection_id: str) -> Dict:
        """Get metrics for a specific connection."""
        with self.metrics_lock:
            if connection_id in self.active_connections:
                return self.active_connections[connection_id].to_dict()
            
            # Check completed connections
            for metrics in reversed(self.completed_connections):
                if metrics.connection_id == connection_id:
                    return metrics.to_dict()
            
            return {}
    
    def get_all_active_metrics(self) -> List[Dict]:
        """Get metrics for all active connections."""
        with self.metrics_lock:
            return [metrics.to_dict() for metrics in self.active_connections.values()]
    
    def get_performance_summary(self) -> Dict:
        """Get aggregate performance statistics."""
        with self.metrics_lock:
            if not self.completed_connections:
                return {
                    "total_completed_connections": 0,
                    "active_connections": len(self.active_connections),
                }
            
            # Calculate averages from completed connections
            avg_setup_ms = sum(
                m.get_timing_summary().get('total_setup_ms', 0) 
                for m in self.completed_connections
            ) / len(self.completed_connections)
            
            avg_duration_s = sum(
                m.get_timing_summary().get('total_duration_s', 0) 
                for m in self.completed_connections
            ) / len(self.completed_connections)
            
            avg_chunks = sum(m.total_audio_chunks for m in self.completed_connections) / len(self.completed_connections)
            avg_transcriptions = sum(m.total_transcriptions for m in self.completed_connections) / len(self.completed_connections)
            
            # Find slowest setup
            slowest = max(self.completed_connections, key=lambda m: m.get_timing_summary().get('total_setup_ms', 0))
            fastest = min(self.completed_connections, key=lambda m: m.get_timing_summary().get('total_setup_ms', 0))
            
            return {
                "total_completed_connections": len(self.completed_connections),
                "active_connections": len(self.active_connections),
                "averages": {
                    "setup_time_ms": round(avg_setup_ms, 1),
                    "connection_duration_s": round(avg_duration_s, 1),
                    "audio_chunks_per_connection": round(avg_chunks, 1),
                    "transcriptions_per_connection": round(avg_transcriptions, 1),
                },
                "extremes": {
                    "slowest_setup_ms": round(slowest.get_timing_summary().get('total_setup_ms', 0), 1),
                    "fastest_setup_ms": round(fastest.get_timing_summary().get('total_setup_ms', 0), 1),
                },
            }
    
    def clear_history(self):
        """Clear completed connection history."""
        with self.metrics_lock:
            self.completed_connections.clear()
            logger.info("Cleared connection history")


# Global singleton instance
performance_monitor = PerformanceMonitor()
