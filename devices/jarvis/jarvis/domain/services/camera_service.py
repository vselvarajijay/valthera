#!/usr/bin/env python3

"""
Domain service interfaces for Jarvis smart CV pipeline.

These interfaces define the contracts for core domain operations
without implementation details. They enable dependency inversion
and clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Dict, Any
from datetime import datetime

from ..entities.frame import Frame, FrameId
from ..entities.detection import Detection, Position3D
from ..entities.camera import CameraConfig, CameraInfo, CameraIntrinsics
from ..value_objects import Resolution, DepthValue


class ICameraService(ABC):
    """Interface for camera operations."""
    
    @abstractmethod
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize camera with given configuration."""
        pass
    
    @abstractmethod
    async def start_streaming(self) -> None:
        """Start camera streaming."""
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> None:
        """Stop camera streaming."""
        pass
    
    @abstractmethod
    async def get_frame(self) -> Optional[Frame]:
        """Get the latest frame from camera."""
        pass

    async def get_last_frame(self) -> Optional[Frame]:
        """Optional: return the most recently captured frame if available."""
        return None
    
    @abstractmethod
    async def stream_frames(self) -> AsyncGenerator[Frame, None]:
        """Stream frames continuously."""
        pass
    
    @abstractmethod
    async def get_status(self) -> CameraInfo:
        """Get current camera status and information."""
        pass
    
    @abstractmethod
    async def get_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get camera intrinsic parameters."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if camera is available."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup camera resources."""
        pass


class IDetectionService(ABC):
    """Interface for object detection operations."""
    
    @abstractmethod
    async def detect_objects(self, frame: Frame, class_names: List[str] = None) -> List[Detection]:
        """Detect objects in frame with optional class filtering."""
        pass
    
    @abstractmethod
    async def detect_persons(self, frame: Frame) -> List[Detection]:
        """Detect persons in frame."""
        pass
    
    @abstractmethod
    async def detect_vehicles(self, frame: Frame) -> List[Detection]:
        """Detect vehicles in frame."""
        pass
    
    @abstractmethod
    async def detect_faces(self, frame: Frame) -> List[Detection]:
        """Detect faces in frame."""
        pass
    
    @abstractmethod
    async def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        pass
    
    @abstractmethod
    async def is_initialized(self) -> bool:
        """Check if detection service is initialized."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        pass


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


class ITrackingService(ABC):
    """Interface for object tracking operations."""
    
    @abstractmethod
    async def track_objects(self, detections: List[Detection]) -> List['TrackedObject']:
        """Track objects across frames."""
        pass
    
    @abstractmethod
    async def update_tracks(self, detections: List[Detection]) -> None:
        """Update existing tracks with new detections."""
        pass
    
    @abstractmethod
    async def get_active_tracks(self) -> List['TrackedObject']:
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


class IEventBus(ABC):
    """Interface for domain event publishing."""
    
    @abstractmethod
    async def publish(self, event: 'DomainEvent') -> None:
        """Publish a domain event."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: type, handler: 'EventHandler') -> None:
        """Subscribe to domain events."""
        pass


class DomainEvent(ABC):
    """Base class for domain events."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.event_id = str(id(self))


class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass


# Forward reference for TrackedObject
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..entities.tracking import TrackedObject
