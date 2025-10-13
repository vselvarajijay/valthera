#!/usr/bin/env python3

"""
In-memory detection repository for Jarvis smart CV pipeline.

Implements IDetectionRepository using in-memory storage.
"""

import logging
from typing import List, Optional, Dict, Any
import threading

from ...domain.repositories.frame_repository import IDetectionRepository
from ...domain.entities.frame import FrameId
from ...domain.entities.detection import Detection
from ...domain.value_objects import Timestamp

logger = logging.getLogger(__name__)


class InMemoryDetectionRepository(IDetectionRepository):
    """In-memory implementation of detection repository."""
    
    def __init__(self):
        self._detections: Dict[str, List[Detection]] = {}  # frame_id -> detections
        self._detection_stats: Dict[str, int] = {}  # class_name -> count
        self._lock = threading.RLock()
        
        logger.info("[DETECTION_REPOSITORY] Initialized")
    
    async def save_detections(self, frame_id: FrameId, detections: List[Detection]) -> None:
        """Save detections for a frame."""
        with self._lock:
            frame_id_str = str(frame_id)
            self._detections[frame_id_str] = detections.copy()
            
            # Update statistics
            for detection in detections:
                class_name = detection.class_name
                self._detection_stats[class_name] = self._detection_stats.get(class_name, 0) + 1
            
            logger.debug(f"[DETECTION_REPOSITORY] Saved {len(detections)} detections for frame {frame_id_str}")
    
    async def get_detections_by_frame(self, frame_id: FrameId) -> List[Detection]:
        """Get detections for a specific frame."""
        with self._lock:
            frame_id_str = str(frame_id)
            return self._detections.get(frame_id_str, []).copy()
    
    async def get_detections_by_class(
        self, 
        class_name: str, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> List[Detection]:
        """Get detections by class name within optional time range."""
        with self._lock:
            matching_detections = []
            
            for frame_id, detections in self._detections.items():
                for detection in detections:
                    if detection.class_name == class_name:
                        # Check time range if specified
                        if start_time and detection.timestamp.value < start_time.value:
                            continue
                        if end_time and detection.timestamp.value > end_time.value:
                            continue
                        
                        matching_detections.append(detection)
            
            return matching_detections
    
    async def get_detection_statistics(
        self, 
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get detection statistics within optional time range."""
        with self._lock:
            stats = {
                "total_detections": 0,
                "detections_by_class": {},
                "frames_with_detections": 0,
                "average_detections_per_frame": 0.0
            }
            
            total_detections = 0
            frames_with_detections = 0
            
            for frame_id, detections in self._detections.items():
                frame_detections = []
                
                for detection in detections:
                    # Check time range if specified
                    if start_time and detection.timestamp.value < start_time.value:
                        continue
                    if end_time and detection.timestamp.value > end_time.value:
                        continue
                    
                    frame_detections.append(detection)
                    total_detections += 1
                    
                    # Count by class
                    class_name = detection.class_name
                    if class_name not in stats["detections_by_class"]:
                        stats["detections_by_class"][class_name] = 0
                    stats["detections_by_class"][class_name] += 1
                
                if frame_detections:
                    frames_with_detections += 1
            
            stats["total_detections"] = total_detections
            stats["frames_with_detections"] = frames_with_detections
            
            if frames_with_detections > 0:
                stats["average_detections_per_frame"] = total_detections / frames_with_detections
            
            return stats
    
    async def delete_detections_by_frame(self, frame_id: FrameId) -> int:
        """Delete detections for a frame."""
        with self._lock:
            frame_id_str = str(frame_id)
            
            if frame_id_str not in self._detections:
                return 0
            
            detections = self._detections[frame_id_str]
            count = len(detections)
            
            # Update statistics
            for detection in detections:
                class_name = detection.class_name
                if class_name in self._detection_stats:
                    self._detection_stats[class_name] -= 1
                    if self._detection_stats[class_name] <= 0:
                        del self._detection_stats[class_name]
            
            del self._detections[frame_id_str]
            
            logger.debug(f"[DETECTION_REPOSITORY] Deleted {count} detections for frame {frame_id_str}")
            return count
    
    async def clear(self) -> None:
        """Clear all stored detections."""
        with self._lock:
            self._detections.clear()
            self._detection_stats.clear()
            logger.info("[DETECTION_REPOSITORY] Cleared all detections")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        with self._lock:
            total_frames = len(self._detections)
            total_detections = sum(len(detections) for detections in self._detections.values())
            
            return {
                "total_frames": total_frames,
                "total_detections": total_detections,
                "detections_by_class": self._detection_stats.copy(),
                "average_detections_per_frame": total_detections / total_frames if total_frames > 0 else 0
            }
