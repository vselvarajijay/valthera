#!/usr/bin/env python3

"""
Frame cache implementation for Jarvis smart CV pipeline.

Refactored version of the existing cache with cleaner interface
and better separation of concerns.
"""

import logging
import hashlib
import threading
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from ...domain.repositories.frame_repository import ICacheRepository
from ...domain.exceptions import InvalidConfigurationError

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl_seconds: Optional[float]
    access_count: int = 0
    last_access: float = 0.0


class FrameCache(ICacheRepository):
    """Cache implementation for frame data and analysis results."""
    
    def __init__(self, max_size: int = 100, default_ttl_seconds: float = 1.0):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl_seconds = default_ttl_seconds
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"[FRAME_CACHE] Initialized with max_size={max_size}, default_ttl={default_ttl_seconds}s")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if entry.ttl_seconds and (current_time - entry.timestamp) > entry.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"[FRAME_CACHE] Entry expired: {key[:20]}...")
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = current_time
            self._hits += 1
            
            logger.debug(f"[FRAME_CACHE] Cache hit: {key[:20]}...")
            return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self._max_size:
                await self._evict_oldest()
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds or self._default_ttl_seconds,
                access_count=1,
                last_access=time.time()
            )
            
            self._cache[key] = entry
            logger.debug(f"[FRAME_CACHE] Cached: {key[:20]}...")
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"[FRAME_CACHE] Deleted: {key[:20]}...")
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if entry.ttl_seconds and (current_time - entry.timestamp) > entry.ttl_seconds:
                del self._cache[key]
                return False
            
            return True
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("[FRAME_CACHE] Cleared all entries")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "default_ttl_seconds": self._default_ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "evictions": self._evictions,
                "total_requests": total_requests
            }
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.ttl_seconds and (current_time - entry.timestamp) > entry.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"[FRAME_CACHE] Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    async def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        # Find oldest entry (least recently accessed)
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access
        )
        
        del self._cache[oldest_key]
        self._evictions += 1
        
        logger.debug(f"[FRAME_CACHE] Evicted oldest entry: {oldest_key[:20]}...")
    
    def create_frame_key(self, frame_data: bytes, classifiers: List[str], options: Dict[str, Any]) -> str:
        """Create cache key for frame analysis."""
        try:
            # Create frame hash
            if NUMPY_AVAILABLE and len(frame_data) > 0:
                # Sample every 10th byte for performance
                sample_data = frame_data[::10]
                frame_hash = hashlib.md5(sample_data).hexdigest()
            else:
                frame_hash = hashlib.md5(frame_data).hexdigest()
            
            # Create options hash
            sorted_options = sorted(options.items())
            options_str = str(sorted_options)
            
            # Create classifiers string
            classifiers_str = ":".join(sorted(classifiers))
            
            # Combine into key
            key_parts = [frame_hash, classifiers_str, options_str]
            return "|".join(key_parts)
            
        except Exception as e:
            logger.error(f"[FRAME_CACHE] Error creating frame key: {e}")
            return f"error_{int(time.time())}"
    
    def create_detection_key(self, frame_id: str, classifier_type: str, confidence_threshold: float) -> str:
        """Create cache key for detection results."""
        return f"detection:{frame_id}:{classifier_type}:{confidence_threshold}"
    
    def create_depth_key(self, frame_id: str, x: int, y: int) -> str:
        """Create cache key for depth calculations."""
        return f"depth:{frame_id}:{x}:{y}"
