#!/usr/bin/env python3

"""
Depth service interface for Jarvis smart CV pipeline.

Defines the contract for depth processing operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.frame import Frame
from ..entities.detection import Detection, Position3D
from ..value_objects import DepthValue


class IDepthService(ABC):
    """Interface for depth processing operations."""
    
    @abstractmethod
    async def get_depth_at_point(self, frame: Frame, x: int, y: int) -> Optional[DepthValue]:
        """Get depth value at specific pixel coordinates."""
        pass
    
    @abstractmethod
    async def calculate_3d_position(
        self, 
        frame: Frame, 
        x: int, 
        y: int, 
        depth_mm: float
    ) -> Optional[Position3D]:
        """Calculate 3D position from pixel coordinates and depth."""
        pass
    
    @abstractmethod
    async def add_depth_to_detections(self, frame: Frame, detections: List[Detection]) -> List[Detection]:
        """Add depth information to detections."""
        pass
    
    @abstractmethod
    async def calculate_center_depth(self, frame: Frame, region_size: float = 0.3) -> Optional[DepthValue]:
        """Calculate average depth in center region of frame."""
        pass
    
    @abstractmethod
    async def is_depth_available(self, frame: Frame) -> bool:
        """Check if frame has valid depth data."""
        pass
