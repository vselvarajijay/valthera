#!/usr/bin/env python3

"""
Configuration management for Jarvis smart CV pipeline.

Centralized configuration using Pydantic for validation and
environment variable support.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class CameraSettings(BaseSettings):
    """Camera configuration settings."""
    
    # Basic camera settings
    width: int = Field(default=640, description="Camera width in pixels")
    height: int = Field(default=480, description="Camera height in pixels")
    fps: int = Field(default=30, description="Camera frames per second")
    device_id: str = Field(default="0", description="Camera device ID")
    
    # Camera type preference
    preferred_type: str = Field(default="realsense", description="Preferred camera type")
    
    # Advanced settings
    auto_exposure: bool = Field(default=True, description="Enable auto exposure")
    auto_white_balance: bool = Field(default=True, description="Enable auto white balance")
    exposure_time: Optional[int] = Field(default=None, description="Manual exposure time")
    gain: Optional[float] = Field(default=None, description="Manual gain")
    
    # Depth settings
    enable_depth: bool = Field(default=True, description="Enable depth sensing")
    depth_width: Optional[int] = Field(default=None, description="Depth image width")
    depth_height: Optional[int] = Field(default=None, description="Depth image height")
    
    @field_validator('width', 'height', 'fps')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    
    @field_validator('preferred_type')
    @classmethod
    def validate_camera_type(cls, v):
        valid_types = ['realsense', 'simple', 'mock', 'usb']
        if v not in valid_types:
            raise ValueError(f"Must be one of {valid_types}")
        return v


class DetectionSettings(BaseSettings):
    """Detection configuration settings."""
    
    # Model settings
    model_path: str = Field(default="yolov8n.pt", description="YOLO model path")
    confidence_threshold: float = Field(default=0.5, description="Detection confidence threshold")
    
    # Detection settings
    max_detections: int = Field(default=10, description="Maximum detections per frame")
    enabled_classes: List[str] = Field(default=["person"], description="Enabled detection classes")
    
    # Processing settings
    enable_tracking: bool = Field(default=True, description="Enable object tracking")
    tracking_threshold: float = Field(default=0.5, description="Tracking confidence threshold")
    
    @field_validator('confidence_threshold', 'tracking_threshold')
    @classmethod
    def validate_confidence_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Must be between 0.0 and 1.0")
        return v
    
    @field_validator('max_detections')
    @classmethod
    def validate_max_detections(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class DepthSettings(BaseSettings):
    """Depth processing configuration settings."""
    
    # Depth processing
    enable_depth_processing: bool = Field(default=True, description="Enable depth processing")
    center_region_size: float = Field(default=0.3, description="Center region size for depth calculation")
    
    # 3D positioning
    enable_3d_positioning: bool = Field(default=True, description="Enable 3D position calculation")
    max_depth_mm: float = Field(default=10000.0, description="Maximum valid depth in mm")
    min_depth_mm: float = Field(default=100.0, description="Minimum valid depth in mm")
    
    # Depth filtering
    enable_depth_filtering: bool = Field(default=True, description="Enable depth filtering")
    depth_smoothing: bool = Field(default=True, description="Enable depth smoothing")
    
    @field_validator('center_region_size')
    @classmethod
    def validate_region_size(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError("Must be between 0.0 and 1.0")
        return v
    
    @field_validator('max_depth_mm', 'min_depth_mm')
    @classmethod
    def validate_depth_range(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class ServerSettings(BaseSettings):
    """Server configuration settings."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_credentials: bool = Field(default=True, description="CORS allow credentials")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="/tmp/jarvis.log", description="Log file path")
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("Must be between 1 and 65535")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Must be one of {valid_levels}")
        return v.upper()


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    # Cache settings
    enable_cache: bool = Field(default=True, description="Enable result caching")
    max_size: int = Field(default=100, description="Maximum cache size")
    ttl_seconds: float = Field(default=1.0, description="Cache TTL in seconds")
    
    # Frame repository settings
    max_frames: int = Field(default=1000, description="Maximum stored frames")
    frame_ttl_seconds: float = Field(default=300.0, description="Frame TTL in seconds")
    
    @field_validator('max_size', 'max_frames')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    
    @field_validator('ttl_seconds', 'frame_ttl_seconds')
    @classmethod
    def validate_ttl(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class MetricsSettings(BaseSettings):
    """Metrics and monitoring configuration settings."""
    
    # Metrics settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_ttl_seconds: float = Field(default=3600.0, description="Metrics TTL in seconds")
    
    # Health check settings
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval: float = Field(default=30.0, description="Health check interval in seconds")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    performance_window_seconds: float = Field(default=300.0, description="Performance monitoring window")
    
    @field_validator('metrics_ttl_seconds', 'health_check_interval', 'performance_window_seconds')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class Settings(BaseSettings):
    """Main configuration settings for Jarvis."""
    
    # Sub-configurations
    camera: CameraSettings = Field(default_factory=CameraSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    depth: DepthSettings = Field(default_factory=DepthSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    
    # Environment settings
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    class Config:
        env_prefix = "JARVIS_"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "camera": self.camera.dict(),
            "detection": self.detection.dict(),
            "depth": self.depth.dict(),
            "server": self.server.dict(),
            "cache": self.cache.dict(),
            "metrics": self.metrics.dict(),
            "environment": self.environment,
            "debug": self.debug
        }
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Settings':
        """Load settings from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Settings file not found: {file_path}")
        
        # Load from file and merge with environment
        return cls(_env_file=file_path)


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.from_env()
    return _settings_instance


def load_settings_from_file(file_path: str) -> Settings:
    """Load settings from file."""
    global _settings_instance
    _settings_instance = Settings.from_file(file_path)
    return _settings_instance


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings_instance
    _settings_instance = Settings.from_env()
    return _settings_instance
