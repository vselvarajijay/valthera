#!/usr/bin/env python3

"""
Tracking service interface for Jarvis smart CV pipeline.

Defines the contract for object tracking operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..entities.detection import Detection


class TrackedObject:
    """Represents a tracked object across multiple frames."""
    
    def __init__(self, track_id: str, detection: Detection):
        self.track_id = track_id
        self.current_detection = detection
        self.detection_history: List[Detection] = [detection]
        self.first_seen_time = detection.timestamp
        self.last_seen_time = detection.timestamp
        self.is_active = True
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        self.current_detection = detection
        self.detection_history.append(detection)
        self.last_seen_time = detection.timestamp
    
    def get_age_seconds(self) -> float:
        """Get age of track in seconds."""
        return self.last_seen_time.age_seconds() - self.first_seen_time.age_seconds()
    
    def get_detection_count(self) -> int:
        """Get number of detections in track."""
        return len(self.detection_history)


class ITrackingService(ABC):
    """Interface for object tracking operations."""
    
    @abstractmethod
    async def track_objects(self, detections: List[Detection]) -> List[TrackedObject]:
        """Track objects across frames."""
        pass
    
    @abstractmethod
    async def update_tracks(self, detections: List[Detection]) -> None:
        """Update existing tracks with new detections."""
        pass
    
    @abstractmethod
    async def get_active_tracks(self) -> List[TrackedObject]:
        """Get currently active tracks."""
        pass
    
    @abstractmethod
    async def clear_tracks(self) -> None:
        """Clear all tracks."""
        pass
    
    @abstractmethod
    async def get_track_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        pass
