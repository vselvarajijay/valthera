#!/usr/bin/env python3

"""
In-memory frame repository for Jarvis smart CV pipeline.

Implements IFrameRepository using in-memory storage with TTL management.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading

from ...domain.repositories.frame_repository import IFrameRepository
from ...domain.entities.frame import Frame, FrameId
from ...domain.value_objects import Timestamp
from ...domain.exceptions import FrameNotFoundError

logger = logging.getLogger(__name__)


class InMemoryFrameRepository(IFrameRepository):
    """In-memory implementation of frame repository."""
    
    def __init__(self, max_frames: int = 1000, ttl_seconds: float = 300.0):
        self._frames: Dict[str, Frame] = {}
        self._frame_timestamps: Dict[str, float] = {}
        self._max_frames = max_frames
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        
        logger.info(f"[FRAME_REPOSITORY] Initialized with max_frames={max_frames}, ttl={ttl_seconds}s")
    
    async def save(self, frame: Frame) -> None:
        """Save a frame."""
        with self._lock:
            frame_id = str(frame.id)
            
            # Check if we need to evict old frames
            if len(self._frames) >= self._max_frames:
                await self._evict_oldest()
            
            # Store frame and timestamp
            self._frames[frame_id] = frame
            self._frame_timestamps[frame_id] = frame.timestamp.value
            
            logger.debug(f"[FRAME_REPOSITORY] Saved frame {frame_id}")
    
    async def get_by_id(self, frame_id: FrameId) -> Optional[Frame]:
        """Get frame by ID."""
        with self._lock:
            frame_id_str = str(frame_id)
            
            if frame_id_str not in self._frames:
                return None
            
            # Check if frame has expired
            if await self._is_expired(frame_id_str):
                await self._remove_frame(frame_id_str)
                return None
            
            return self._frames[frame_id_str]
    
    async def get_latest(self) -> Optional[Frame]:
        """Get the most recent frame."""
        with self._lock:
            if not self._frames:
                return None
            
            # Find most recent frame
            latest_frame_id = max(
                self._frame_timestamps.keys(),
                key=lambda fid: self._frame_timestamps[fid]
            )
            
            return await self.get_by_id(FrameId(latest_frame_id))
    
    async def get_by_timestamp_range(
        self, 
        start_time: Timestamp, 
        end_time: Timestamp
    ) -> List[Frame]:
        """Get frames within timestamp range."""
        with self._lock:
            frames = []
            
            for frame_id, timestamp in self._frame_timestamps.items():
                if start_time.value <= timestamp <= end_time.value:
                    frame = await self.get_by_id(FrameId(frame_id))
                    if frame:
                        frames.append(frame)
            
            # Sort by timestamp
            frames.sort(key=lambda f: f.timestamp.value)
            return frames
    
    async def delete_old_frames(self, older_than: Timestamp) -> int:
        """Delete frames older than specified timestamp."""
        with self._lock:
            deleted_count = 0
            frames_to_delete = []
            
            for frame_id, timestamp in self._frame_timestamps.items():
                if timestamp < older_than.value:
                    frames_to_delete.append(frame_id)
            
            for frame_id in frames_to_delete:
                await self._remove_frame(frame_id)
                deleted_count += 1
            
            logger.info(f"[FRAME_REPOSITORY] Deleted {deleted_count} old frames")
            return deleted_count
    
    async def get_frame_count(self) -> int:
        """Get total number of stored frames."""
        with self._lock:
            return len(self._frames)
    
    async def clear(self) -> None:
        """Clear all stored frames."""
        with self._lock:
            self._frames.clear()
            self._frame_timestamps.clear()
            logger.info("[FRAME_REPOSITORY] Cleared all frames")
    
    async def cleanup_expired(self) -> int:
        """Remove expired frames."""
        with self._lock:
            expired_frames = []
            
            for frame_id in self._frame_timestamps.keys():
                if await self._is_expired(frame_id):
                    expired_frames.append(frame_id)
            
            for frame_id in expired_frames:
                await self._remove_frame(frame_id)
            
            if expired_frames:
                logger.info(f"[FRAME_REPOSITORY] Cleaned up {len(expired_frames)} expired frames")
            
            return len(expired_frames)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        with self._lock:
            current_time = datetime.now().timestamp()
            
            # Calculate age statistics
            ages = []
            for timestamp in self._frame_timestamps.values():
                ages.append(current_time - timestamp)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "total_frames": len(self._frames),
                "max_frames": self._max_frames,
                "ttl_seconds": self._ttl_seconds,
                "average_age_seconds": avg_age,
                "oldest_frame_age_seconds": max(ages) if ages else 0,
                "newest_frame_age_seconds": min(ages) if ages else 0
            }
    
    async def _is_expired(self, frame_id: str) -> bool:
        """Check if frame has expired."""
        timestamp = self._frame_timestamps.get(frame_id, 0)
        current_time = datetime.now().timestamp()
        return (current_time - timestamp) > self._ttl_seconds
    
    async def _evict_oldest(self) -> None:
        """Evict the oldest frame."""
        if not self._frame_timestamps:
            return
        
        oldest_frame_id = min(
            self._frame_timestamps.keys(),
            key=lambda fid: self._frame_timestamps[fid]
        )
        
        await self._remove_frame(oldest_frame_id)
        logger.debug(f"[FRAME_REPOSITORY] Evicted oldest frame {oldest_frame_id}")
    
    async def _remove_frame(self, frame_id: str) -> None:
        """Remove frame from storage."""
        if frame_id in self._frames:
            del self._frames[frame_id]
        if frame_id in self._frame_timestamps:
            del self._frame_timestamps[frame_id]
