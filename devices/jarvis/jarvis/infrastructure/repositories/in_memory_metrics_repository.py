#!/usr/bin/env python3

"""
In-memory metrics repository for Jarvis smart CV pipeline.

Implements IMetricsRepository using in-memory storage with TTL management.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import time

from ...domain.repositories.frame_repository import IMetricsRepository
from ...domain.entities.frame import FrameId
from ...domain.value_objects import Timestamp

logger = logging.getLogger(__name__)


class MetricEntry:
    """Represents a single metric entry."""
    
    def __init__(self, value: float, timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        self.value = value
        self.timestamp = timestamp
        self.metadata = metadata or {}


class InMemoryMetricsRepository(IMetricsRepository):
    """In-memory implementation of metrics repository."""
    
    def __init__(self, ttl_seconds: float = 3600.0):  # 1 hour default TTL
        self._processing_times: Dict[str, List[MetricEntry]] = {}
        self._detection_counts: Dict[str, List[MetricEntry]] = {}
        self._frame_metrics: Dict[str, List[MetricEntry]] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        
        logger.info(f"[METRICS_REPOSITORY] Initialized with ttl={ttl_seconds}s")
    
    async def record_processing_time(
        self, 
        operation: str, 
        duration_ms: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record processing time for an operation."""
        with self._lock:
            if operation not in self._processing_times:
                self._processing_times[operation] = []
            
            entry = MetricEntry(duration_ms, time.time(), metadata)
            self._processing_times[operation].append(entry)
            
            # Cleanup old entries
            await self._cleanup_old_entries(self._processing_times[operation])
            
            logger.debug(f"[METRICS_REPOSITORY] Recorded processing time: {operation}={duration_ms}ms")
    
    async def record_detection_count(
        self, 
        class_name: str, 
        count: int, 
        timestamp: Optional[Timestamp] = None
    ) -> None:
        """Record detection count for a class."""
        with self._lock:
            if class_name not in self._detection_counts:
                self._detection_counts[class_name] = []
            
            ts = timestamp.value if timestamp else time.time()
            entry = MetricEntry(float(count), ts)
            self._detection_counts[class_name].append(entry)
            
            # Cleanup old entries
            await self._cleanup_old_entries(self._detection_counts[class_name])
            
            logger.debug(f"[METRICS_REPOSITORY] Recorded detection count: {class_name}={count}")
    
    async def record_frame_processed(
        self, 
        frame_id: FrameId, 
        processing_time_ms: float,
        detection_count: int
    ) -> None:
        """Record frame processing metrics."""
        with self._lock:
            frame_id_str = str(frame_id)
            if frame_id_str not in self._frame_metrics:
                self._frame_metrics[frame_id_str] = []
            
            metadata = {
                "frame_id": frame_id_str,
                "detection_count": detection_count
            }
            
            entry = MetricEntry(processing_time_ms, time.time(), metadata)
            self._frame_metrics[frame_id_str].append(entry)
            
            logger.debug(f"[METRICS_REPOSITORY] Recorded frame metrics: {frame_id_str}")
    
    async def get_processing_statistics(
        self, 
        operation: str,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get processing statistics for an operation."""
        with self._lock:
            if operation not in self._processing_times:
                return {
                    "operation": operation,
                    "total_samples": 0,
                    "average_time_ms": 0.0,
                    "min_time_ms": 0.0,
                    "max_time_ms": 0.0,
                    "total_time_ms": 0.0
                }
            
            entries = self._processing_times[operation]
            filtered_entries = self._filter_by_time_range(entries, start_time, end_time)
            
            if not filtered_entries:
                return {
                    "operation": operation,
                    "total_samples": 0,
                    "average_time_ms": 0.0,
                    "min_time_ms": 0.0,
                    "max_time_ms": 0.0,
                    "total_time_ms": 0.0
                }
            
            values = [entry.value for entry in filtered_entries]
            
            return {
                "operation": operation,
                "total_samples": len(values),
                "average_time_ms": sum(values) / len(values),
                "min_time_ms": min(values),
                "max_time_ms": max(values),
                "total_time_ms": sum(values)
            }
    
    async def get_detection_statistics(
        self, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get detection statistics."""
        with self._lock:
            stats = {
                "total_detections": 0,
                "detections_by_class": {},
                "total_samples": 0
            }
            
            for class_name, entries in self._detection_counts.items():
                filtered_entries = self._filter_by_time_range(entries, start_time, end_time)
                
                if filtered_entries:
                    total_count = sum(entry.value for entry in filtered_entries)
                    stats["detections_by_class"][class_name] = {
                        "total_count": int(total_count),
                        "samples": len(filtered_entries),
                        "average_per_sample": total_count / len(filtered_entries)
                    }
                    stats["total_detections"] += int(total_count)
                    stats["total_samples"] += len(filtered_entries)
            
            return stats
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        with self._lock:
            current_time = time.time()
            
            # Calculate processing time statistics
            processing_stats = {}
            for operation, entries in self._processing_times.items():
                recent_entries = [e for e in entries if current_time - e.timestamp < 300]  # Last 5 minutes
                if recent_entries:
                    values = [e.value for e in recent_entries]
                    processing_stats[operation] = {
                        "samples": len(values),
                        "average_ms": sum(values) / len(values),
                        "min_ms": min(values),
                        "max_ms": max(values)
                    }
            
            # Calculate detection statistics
            detection_stats = {}
            for class_name, entries in self._detection_counts.items():
                recent_entries = [e for e in entries if current_time - e.timestamp < 300]  # Last 5 minutes
                if recent_entries:
                    total_count = sum(e.value for e in recent_entries)
                    detection_stats[class_name] = {
                        "total_count": int(total_count),
                        "samples": len(recent_entries)
                    }
            
            return {
                "timestamp": current_time,
                "processing_stats": processing_stats,
                "detection_stats": detection_stats,
                "total_operations_tracked": len(self._processing_times),
                "total_classes_tracked": len(self._detection_counts),
                "total_frames_tracked": len(self._frame_metrics)
            }
    
    async def clear_old_metrics(self, older_than: Timestamp) -> int:
        """Clear metrics older than specified timestamp."""
        with self._lock:
            cutoff_time = older_than.value
            total_cleared = 0
            
            # Clear old processing times
            for operation, entries in self._processing_times.items():
                original_count = len(entries)
                self._processing_times[operation] = [
                    e for e in entries if e.timestamp >= cutoff_time
                ]
                total_cleared += original_count - len(self._processing_times[operation])
            
            # Clear old detection counts
            for class_name, entries in self._detection_counts.items():
                original_count = len(entries)
                self._detection_counts[class_name] = [
                    e for e in entries if e.timestamp >= cutoff_time
                ]
                total_cleared += original_count - len(self._detection_counts[class_name])
            
            # Clear old frame metrics
            for frame_id, entries in self._frame_metrics.items():
                original_count = len(entries)
                self._frame_metrics[frame_id] = [
                    e for e in entries if e.timestamp >= cutoff_time
                ]
                total_cleared += original_count - len(self._frame_metrics[frame_id])
            
            logger.info(f"[METRICS_REPOSITORY] Cleared {total_cleared} old metric entries")
            return total_cleared
    
    def _filter_by_time_range(
        self, 
        entries: List[MetricEntry], 
        start_time: Optional[Timestamp], 
        end_time: Optional[Timestamp]
    ) -> List[MetricEntry]:
        """Filter entries by time range."""
        filtered = entries
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time.value]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time.value]
        
        return filtered
    
    async def _cleanup_old_entries(self, entries: List[MetricEntry]) -> None:
        """Remove old entries based on TTL."""
        current_time = time.time()
        cutoff_time = current_time - self._ttl_seconds
        
        # Remove old entries
        entries[:] = [e for e in entries if e.timestamp >= cutoff_time]
