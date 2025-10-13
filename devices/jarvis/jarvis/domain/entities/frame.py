#!/usr/bin/env python3

"""
Frame entity for Jarvis smart CV pipeline.

Represents a captured frame with associated metadata and data.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from ..value_objects import Timestamp, Resolution


@dataclass(frozen=True)
class FrameId:
    """Unique identifier for a frame."""
    
    value: str
    
    @classmethod
    def generate(cls) -> 'FrameId':
        """Generate a new unique frame ID."""
        return cls(str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Frame:
    """Represents a captured frame with RGB and optional depth data."""
    
    id: FrameId
    timestamp: Timestamp
    resolution: Resolution
    
    # Frame data
    color_data: bytes
    depth_data: Optional[bytes] = None
    
    # Camera metadata
    camera_intrinsics: Optional[Dict[str, float]] = None
    device_id: Optional[str] = None
    
    # Processing metadata
    processing_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate frame data after initialization."""
        if not self.color_data:
            raise ValueError("Color data cannot be empty")
        
        if len(self.color_data) != self.expected_color_data_size():
            raise ValueError(
                f"Color data size mismatch. Expected {self.expected_color_data_size()}, "
                f"got {len(self.color_data)}"
            )
    
    def expected_color_data_size(self) -> int:
        """Calculate expected size of color data (RGB format)."""
        return self.resolution.total_pixels * 3  # 3 bytes per pixel (RGB)
    
    def expected_depth_data_size(self) -> int:
        """Calculate expected size of depth data (16-bit format)."""
        return self.resolution.total_pixels * 2  # 2 bytes per pixel (uint16)
    
    def has_depth_data(self) -> bool:
        """Check if frame has depth data."""
        return self.depth_data is not None
    
    def validate_depth_data(self) -> bool:
        """Validate depth data if present."""
        if not self.depth_data:
            return True
        
        return len(self.depth_data) == self.expected_depth_data_size()
    
    def get_age_seconds(self) -> float:
        """Get age of frame in seconds."""
        return self.timestamp.age_seconds()
    
    def is_fresh(self, max_age_seconds: float = 1.0) -> bool:
        """Check if frame is fresh (not too old)."""
        return self.get_age_seconds() <= max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary for serialization."""
        return {
            "id": str(self.id),
            "timestamp": str(self.timestamp),
            "resolution": {
                "width": self.resolution.width,
                "height": self.resolution.height
            },
            "has_depth_data": self.has_depth_data(),
            "device_id": self.device_id,
            "camera_intrinsics": self.camera_intrinsics,
            "processing_metadata": self.processing_metadata
        }
    
    @classmethod
    def create_from_camera_data(
        cls,
        color_data: bytes,
        depth_data: Optional[bytes] = None,
        resolution: Resolution = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        device_id: Optional[str] = None
    ) -> 'Frame':
        """Create frame from camera data."""
        if resolution is None:
            # Try to infer resolution from data size
            # Assuming RGB format (3 bytes per pixel)
            total_pixels = len(color_data) // 3
            # Assume square-ish resolution
            import math
            side = int(math.sqrt(total_pixels))
            resolution = Resolution(width=side, height=side)
        
        return cls(
            id=FrameId.generate(),
            timestamp=Timestamp.now(),
            resolution=resolution,
            color_data=color_data,
            depth_data=depth_data,
            camera_intrinsics=camera_intrinsics,
            device_id=device_id
        )
