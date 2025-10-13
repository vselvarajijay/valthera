#!/usr/bin/env python3

"""
Camera entity for Jarvis smart CV pipeline.

Represents camera configuration and status information.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

from ..value_objects import Resolution, Timestamp


class CameraType(Enum):
    """Supported camera types."""
    REALSENSE = "realsense"
    SIMPLE = "simple"
    MOCK = "mock"
    USB = "usb"


class CameraStatus(Enum):
    """Camera status states."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for camera operation."""
    
    # Basic settings
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Camera identification
    device_id: str = "0"
    camera_type: CameraType = CameraType.REALSENSE
    
    # Advanced settings
    auto_exposure: bool = True
    auto_white_balance: bool = True
    exposure_time: Optional[int] = None
    gain: Optional[float] = None
    
    # Depth settings (for RealSense)
    enable_depth: bool = True
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    
    # Processing settings
    enable_color_correction: bool = True
    enable_noise_reduction: bool = True
    
    def __post_init__(self):
        """Validate camera configuration."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        
        if self.exposure_time is not None and self.exposure_time <= 0:
            raise ValueError("Exposure time must be positive")
        
        if self.gain is not None and self.gain < 0:
            raise ValueError("Gain must be non-negative")
    
    @property
    def resolution(self) -> Resolution:
        """Get resolution as Resolution value object."""
        return Resolution(self.width, self.height)
    
    @property
    def depth_resolution(self) -> Resolution:
        """Get depth resolution."""
        if self.depth_width and self.depth_height:
            return Resolution(self.depth_width, self.depth_height)
        return self.resolution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "device_id": self.device_id,
            "camera_type": self.camera_type.value,
            "auto_exposure": self.auto_exposure,
            "auto_white_balance": self.auto_white_balance,
            "exposure_time": self.exposure_time,
            "gain": self.gain,
            "enable_depth": self.enable_depth,
            "depth_width": self.depth_width,
            "depth_height": self.depth_height,
            "enable_color_correction": self.enable_color_correction,
            "enable_noise_reduction": self.enable_noise_reduction
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraConfig':
        """Create configuration from dictionary."""
        return cls(
            width=data.get("width", 640),
            height=data.get("height", 480),
            fps=data.get("fps", 30),
            device_id=data.get("device_id", "0"),
            camera_type=CameraType(data.get("camera_type", "realsense")),
            auto_exposure=data.get("auto_exposure", True),
            auto_white_balance=data.get("auto_white_balance", True),
            exposure_time=data.get("exposure_time"),
            gain=data.get("gain"),
            enable_depth=data.get("enable_depth", True),
            depth_width=data.get("depth_width"),
            depth_height=data.get("depth_height"),
            enable_color_correction=data.get("enable_color_correction", True),
            enable_noise_reduction=data.get("enable_noise_reduction", True)
        )


@dataclass(frozen=True)
class CameraInfo:
    """Information about camera hardware and capabilities."""
    
    # Hardware info
    device_id: str
    camera_type: CameraType
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    
    # Capabilities
    supported_resolutions: list[Resolution] = None
    supported_fps: list[int] = None
    has_depth: bool = False
    has_imu: bool = False
    
    # Current status
    status: CameraStatus = CameraStatus.DISCONNECTED
    last_frame_time: Optional[Timestamp] = None
    frame_count: int = 0
    
    def __post_init__(self):
        """Initialize default values."""
        if self.supported_resolutions is None:
            self.supported_resolutions = [
                Resolution(640, 480),
                Resolution(1280, 720),
                Resolution(1920, 1080)
            ]
        
        if self.supported_fps is None:
            self.supported_fps = [15, 30, 60]
    
    def supports_resolution(self, resolution: Resolution) -> bool:
        """Check if camera supports given resolution."""
        return resolution in self.supported_resolutions
    
    def supports_fps(self, fps: int) -> bool:
        """Check if camera supports given FPS."""
        return fps in self.supported_fps
    
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming."""
        return self.status == CameraStatus.STREAMING
    
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self.status in [CameraStatus.CONNECTED, CameraStatus.STREAMING]
    
    def has_error(self) -> bool:
        """Check if camera has an error."""
        return self.status == CameraStatus.ERROR
    
    def get_frame_rate(self) -> float:
        """Get current frame rate based on last frame time."""
        if not self.last_frame_time:
            return 0.0
        
        age = self.last_frame_time.age_seconds()
        if age > 1.0:  # No frames in last second
            return 0.0
        
        return self.frame_count / max(age, 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert camera info to dictionary."""
        return {
            "device_id": self.device_id,
            "camera_type": self.camera_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "supported_resolutions": [{"width": r.width, "height": r.height} for r in self.supported_resolutions],
            "supported_fps": self.supported_fps,
            "has_depth": self.has_depth,
            "has_imu": self.has_imu,
            "status": self.status.value,
            "last_frame_time": str(self.last_frame_time) if self.last_frame_time else None,
            "frame_count": self.frame_count,
            "current_frame_rate": self.get_frame_rate()
        }


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters for 3D calculations."""
    
    fx: float  # Focal length X
    fy: float  # Focal length Y
    ppx: float  # Principal point X
    ppy: float  # Principal point Y
    width: int
    height: int
    
    def __post_init__(self):
        """Validate intrinsic parameters."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
    
    def pixel_to_3d(self, x: int, y: int, depth_mm: float) -> tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D coordinates."""
        if depth_mm <= 0:
            raise ValueError("Depth must be positive")
        
        # Convert depth from mm to meters
        z = depth_mm / 1000.0
        
        # Calculate 3D coordinates
        x_3d = (x - self.ppx) * z / self.fx
        y_3d = (y - self.ppy) * z / self.fy
        
        return (x_3d, y_3d, z)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert intrinsics to dictionary."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraIntrinsics':
        """Create intrinsics from dictionary."""
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            ppx=data["ppx"],
            ppy=data["ppy"],
            width=data["width"],
            height=data["height"]
        )
