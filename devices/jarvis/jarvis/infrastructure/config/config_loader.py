#!/usr/bin/env python3

"""
Configuration loader for Jarvis smart CV pipeline.

Handles loading configuration from various sources with proper
validation and fallback mechanisms.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with multiple source support."""
    
    def __init__(self):
        self._config_paths = [
            "jarvis.env",
            ".env",
            "config.env",
            "/etc/jarvis/jarvis.env"
        ]
        self._settings: Optional[Settings] = None
    
    def load_config(self, config_path: Optional[str] = None) -> Settings:
        """
        Load configuration from various sources.
        
        Priority order:
        1. Explicit config_path parameter
        2. Environment variables
        3. Config files in order of preference
        4. Default values
        """
        try:
            # Try explicit config path first
            if config_path:
                logger.info(f"[CONFIG_LOADER] Loading config from: {config_path}")
                self._settings = Settings.from_file(config_path)
                return self._settings
            
            # Try to find config file
            config_file = self._find_config_file()
            if config_file:
                logger.info(f"[CONFIG_LOADER] Loading config from: {config_file}")
                self._settings = Settings.from_file(config_file)
                return self._settings
            
            # Fall back to environment variables
            logger.info("[CONFIG_LOADER] Loading config from environment variables")
            self._settings = Settings.from_env()
            return self._settings
            
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Error loading config: {e}")
            logger.info("[CONFIG_LOADER] Falling back to default settings")
            self._settings = Settings()
            return self._settings
    
    def _find_config_file(self) -> Optional[str]:
        """Find the first available config file."""
        for config_path in self._config_paths:
            if os.path.exists(config_path):
                return config_path
        return None
    
    def get_settings(self) -> Settings:
        """Get current settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings
    
    def reload_config(self) -> Settings:
        """Reload configuration."""
        self._settings = None
        return self.load_config()
    
    def validate_config(self, settings: Settings) -> bool:
        """Validate configuration settings."""
        try:
            # Validate camera settings
            if settings.camera.width <= 0 or settings.camera.height <= 0:
                logger.error("[CONFIG_LOADER] Invalid camera dimensions")
                return False
            
            if settings.camera.fps <= 0:
                logger.error("[CONFIG_LOADER] Invalid camera FPS")
                return False
            
            # Validate detection settings
            if not (0.0 <= settings.detection.confidence_threshold <= 1.0):
                logger.error("[CONFIG_LOADER] Invalid confidence threshold")
                return False
            
            if settings.detection.max_detections <= 0:
                logger.error("[CONFIG_LOADER] Invalid max detections")
                return False
            
            # Validate server settings
            if not (1 <= settings.server.port <= 65535):
                logger.error("[CONFIG_LOADER] Invalid server port")
                return False
            
            # Validate cache settings
            if settings.cache.max_size <= 0:
                logger.error("[CONFIG_LOADER] Invalid cache max size")
                return False
            
            if settings.cache.ttl_seconds <= 0:
                logger.error("[CONFIG_LOADER] Invalid cache TTL")
                return False
            
            logger.info("[CONFIG_LOADER] Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Configuration validation failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        settings = self.get_settings()
        
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "camera": {
                "type": settings.camera.preferred_type,
                "resolution": f"{settings.camera.width}x{settings.camera.height}",
                "fps": settings.camera.fps,
                "depth_enabled": settings.camera.enable_depth
            },
            "detection": {
                "model": settings.detection.model_path,
                "confidence_threshold": settings.detection.confidence_threshold,
                "enabled_classes": settings.detection.enabled_classes,
                "tracking_enabled": settings.detection.enable_tracking
            },
            "server": {
                "host": settings.server.host,
                "port": settings.server.port,
                "log_level": settings.server.log_level
            },
            "cache": {
                "enabled": settings.cache.enable_cache,
                "max_size": settings.cache.max_size,
                "ttl_seconds": settings.cache.ttl_seconds
            },
            "metrics": {
                "enabled": settings.metrics.enable_metrics,
                "health_checks_enabled": settings.metrics.enable_health_checks
            }
        }
    
    def create_example_config(self, file_path: str) -> None:
        """Create an example configuration file."""
        example_config = """# Jarvis Configuration Example
# Copy this file to jarvis.env and modify as needed

# Camera Settings
JARVIS_CAMERA__WIDTH=640
JARVIS_CAMERA__HEIGHT=480
JARVIS_CAMERA__FPS=30
JARVIS_CAMERA__DEVICE_ID=0
JARVIS_CAMERA__PREFERRED_TYPE=realsense
JARVIS_CAMERA__ENABLE_DEPTH=true

# Detection Settings
JARVIS_DETECTION__MODEL_PATH=yolov8n.pt
JARVIS_DETECTION__CONFIDENCE_THRESHOLD=0.5
JARVIS_DETECTION__MAX_DETECTIONS=10
JARVIS_DETECTION__ENABLED_CLASSES=["person", "car", "truck"]
JARVIS_DETECTION__ENABLE_TRACKING=true

# Depth Settings
JARVIS_DEPTH__ENABLE_DEPTH_PROCESSING=true
JARVIS_DEPTH__CENTER_REGION_SIZE=0.3
JARVIS_DEPTH__ENABLE_3D_POSITIONING=true
JARVIS_DEPTH__MAX_DEPTH_MM=10000.0

# Server Settings
JARVIS_SERVER__HOST=0.0.0.0
JARVIS_SERVER__PORT=8001
JARVIS_SERVER__LOG_LEVEL=INFO
JARVIS_SERVER__LOG_FILE=/tmp/jarvis.log

# Cache Settings
JARVIS_CACHE__ENABLE_CACHE=true
JARVIS_CACHE__MAX_SIZE=100
JARVIS_CACHE__TTL_SECONDS=1.0
JARVIS_CACHE__MAX_FRAMES=1000

# Metrics Settings
JARVIS_METRICS__ENABLE_METRICS=true
JARVIS_METRICS__ENABLE_HEALTH_CHECKS=true
JARVIS_METRICS__HEALTH_CHECK_INTERVAL=30.0

# Environment
JARVIS_ENVIRONMENT=development
JARVIS_DEBUG=false
"""
        
        try:
            with open(file_path, 'w') as f:
                f.write(example_config)
            logger.info(f"[CONFIG_LOADER] Created example config file: {file_path}")
        except Exception as e:
            logger.error(f"[CONFIG_LOADER] Error creating example config: {e}")


# Global config loader instance
_config_loader_instance: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader_instance
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader()
    return _config_loader_instance


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration using the global config loader."""
    loader = get_config_loader()
    return loader.load_config(config_path)
