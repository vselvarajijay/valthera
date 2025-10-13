#!/usr/bin/env python3

"""
Repository interfaces for Jarvis smart CV pipeline.

These interfaces define data access contracts without implementation
details, enabling clean separation between domain logic and persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.frame import Frame, FrameId
from ..entities.detection import Detection
from ..value_objects import Timestamp


class IFrameRepository(ABC):
    """Interface for frame data persistence."""
    
    @abstractmethod
    async def save(self, frame: Frame) -> None:
        """Save a frame."""
        pass
    
    @abstractmethod
    async def get_by_id(self, frame_id: FrameId) -> Optional[Frame]:
        """Get frame by ID."""
        pass
    
    @abstractmethod
    async def get_latest(self) -> Optional[Frame]:
        """Get the most recent frame."""
        pass
    
    @abstractmethod
    async def get_by_timestamp_range(
        self, 
        start_time: Timestamp, 
        end_time: Timestamp
    ) -> List[Frame]:
        """Get frames within timestamp range."""
        pass
    
    @abstractmethod
    async def delete_old_frames(self, older_than: Timestamp) -> int:
        """Delete frames older than specified timestamp. Returns count deleted."""
        pass
    
    @abstractmethod
    async def get_frame_count(self) -> int:
        """Get total number of stored frames."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored frames."""
        pass


class IDetectionRepository(ABC):
    """Interface for detection data persistence."""
    
    @abstractmethod
    async def save_detections(self, frame_id: FrameId, detections: List[Detection]) -> None:
        """Save detections for a frame."""
        pass
    
    @abstractmethod
    async def get_detections_by_frame(self, frame_id: FrameId) -> List[Detection]:
        """Get detections for a specific frame."""
        pass
    
    @abstractmethod
    async def get_detections_by_class(
        self, 
        class_name: str, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> List[Detection]:
        """Get detections by class name within optional time range."""
        pass
    
    @abstractmethod
    async def get_detection_statistics(
        self, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get detection statistics within optional time range."""
        pass
    
    @abstractmethod
    async def delete_detections_by_frame(self, frame_id: FrameId) -> int:
        """Delete detections for a frame. Returns count deleted."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored detections."""
        pass


class IMetricsRepository(ABC):
    """Interface for metrics and statistics persistence."""
    
    @abstractmethod
    async def record_processing_time(
        self, 
        operation: str, 
        duration_ms: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record processing time for an operation."""
        pass
    
    @abstractmethod
    async def record_detection_count(
        self, 
        class_name: str, 
        count: int, 
        timestamp: Optional[Timestamp] = None
    ) -> None:
        """Record detection count for a class."""
        pass
    
    @abstractmethod
    async def record_frame_processed(
        self, 
        frame_id: FrameId, 
        processing_time_ms: float,
        detection_count: int
    ) -> None:
        """Record frame processing metrics."""
        pass
    
    @abstractmethod
    async def get_processing_statistics(
        self, 
        operation: str,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get processing statistics for an operation."""
        pass
    
    @abstractmethod
    async def get_detection_statistics(
        self, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get detection statistics."""
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        pass
    
    @abstractmethod
    async def clear_old_metrics(self, older_than: Timestamp) -> int:
        """Clear metrics older than specified timestamp. Returns count deleted."""
        pass


class ICacheRepository(ABC):
    """Interface for caching operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache. Returns True if key existed."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired entries. Returns count cleaned."""
        pass
