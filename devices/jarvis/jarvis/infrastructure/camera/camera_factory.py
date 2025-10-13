#!/usr/bin/env python3

"""
Camera factory for Jarvis smart CV pipeline.

Creates appropriate camera service implementations based on availability
and configuration preferences.
"""

import logging
from typing import Optional

from ...domain.services.camera_service import ICameraService
from ...domain.entities.camera import CameraConfig, CameraType
from ...domain.exceptions import CameraUnavailableError

from .realsense_adapter import RealSenseCameraAdapter
from .simple_camera_adapter import SimpleCameraAdapter

logger = logging.getLogger(__name__)


class CameraFactory:
    """Factory for creating camera service implementations."""
    
    @staticmethod
    async def create_camera_service(
        config: CameraConfig,
        preferred_type: Optional[CameraType] = None
    ) -> ICameraService:
        """
        Create camera service based on configuration and availability.
        
        Args:
            config: Camera configuration
            preferred_type: Preferred camera type (optional)
            
        Returns:
            ICameraService implementation
            
        Raises:
            CameraUnavailableError: If no suitable camera is available
        """
        logger.info(f"[CAMERA_FACTORY] Creating camera service for type: {config.preferred_type}")
        
        # Try preferred type first if specified
        if preferred_type:
            camera_service = await CameraFactory._try_create_camera(preferred_type, config)
            if camera_service:
                logger.info(f"[CAMERA_FACTORY] Created {preferred_type.value} camera")
                return camera_service
        
        # Try configured type
        camera_service = await CameraFactory._try_create_camera(config.preferred_type, config)
        if camera_service:
            logger.info(f"[CAMERA_FACTORY] Created {config.preferred_type} camera")
            return camera_service
        
        # Fallback: try other available camera types
        fallback_types = [
            CameraType.REALSENSE,
            CameraType.SIMPLE,
            CameraType.MOCK
        ]
        
        # Remove already tried types
        fallback_types = [t for t in fallback_types if t != config.preferred_type and t != preferred_type]
        
        for camera_type in fallback_types:
            logger.info(f"[CAMERA_FACTORY] Trying fallback camera type: {camera_type.value}")
            camera_service = await CameraFactory._try_create_camera(camera_type, config)
            if camera_service:
                logger.info(f"[CAMERA_FACTORY] Created fallback {camera_type.value} camera")
                return camera_service
        
        # No camera available
        raise CameraUnavailableError("No suitable camera implementation available")
    
    @staticmethod
    async def _try_create_camera(camera_type: CameraType, config: CameraConfig) -> Optional[ICameraService]:
        """Try to create a camera service of the specified type."""
        try:
            if camera_type == CameraType.REALSENSE:
                camera_service = RealSenseCameraAdapter()
            elif camera_type == CameraType.SIMPLE:
                camera_service = SimpleCameraAdapter()
            elif camera_type == CameraType.MOCK:
                # TODO: Implement MockCameraAdapter
                logger.warning("[CAMERA_FACTORY] Mock camera not yet implemented")
                return None
            else:
                logger.warning(f"[CAMERA_FACTORY] Unknown camera type: {camera_type}")
                return None
            
            # Try to initialize
            success = await camera_service.initialize(config)
            if success:
                return camera_service
            else:
                logger.warning(f"[CAMERA_FACTORY] Failed to initialize {camera_type.value} camera")
                return None
                
        except Exception as e:
            logger.warning(f"[CAMERA_FACTORY] Error creating {camera_type.value} camera: {e}")
            return None
    
    @staticmethod
    async def create_default_camera() -> ICameraService:
        """Create camera service with default configuration."""
        default_config = CameraConfig()
        return await CameraFactory.create_camera_service(default_config)
    
    @staticmethod
    async def get_available_camera_types() -> list[CameraType]:
        """Get list of available camera types."""
        available_types = []
        
        # Test each camera type
        test_config = CameraConfig()
        
        for camera_type in CameraType:
            try:
                camera_service = await CameraFactory._try_create_camera(camera_type, test_config)
                if camera_service:
                    available_types.append(camera_type)
                    await camera_service.cleanup()
            except Exception:
                continue
        
        logger.info(f"[CAMERA_FACTORY] Available camera types: {[t.value for t in available_types]}")
        return available_types
