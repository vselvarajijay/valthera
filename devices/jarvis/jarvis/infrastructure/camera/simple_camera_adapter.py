#!/usr/bin/env python3

"""
Simple camera adapter for Jarvis smart CV pipeline.

Adapts the existing SimpleCamera implementation to the ICameraService interface.
"""

import logging
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from ...domain.services.camera_service import ICameraService
from ...domain.entities.camera import CameraConfig, CameraInfo, CameraIntrinsics, CameraStatus, CameraType
from ...domain.entities.frame import Frame, FrameId
from ...domain.value_objects import Resolution, Timestamp
from ...domain.exceptions import CameraUnavailableError

# Import existing SimpleCamera
from ...simple_camera import SimpleCamera

logger = logging.getLogger(__name__)


class SimpleCameraAdapter(ICameraService):
    """Adapter that wraps SimpleCamera to implement ICameraService."""
    
    def __init__(self):
        self._simple_camera: Optional[SimpleCamera] = None
        self._config: Optional[CameraConfig] = None
        self._is_initialized = False
        self._is_streaming = False
        self._frame_count = 0
        self._last_frame_time: Optional[Timestamp] = None
        self._last_frame: Optional[Frame] = None
        self._last_frame_lock = asyncio.Lock()
    
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize simple camera with given configuration."""
        try:
            logger.info(f"[SIMPLE_CAMERA_ADAPTER] Initializing with config: {config.width}x{config.height} @ {config.fps}fps")
            
            # Create SimpleCamera instance
            self._simple_camera = SimpleCamera(
                width=config.width,
                height=config.height,
                fps=config.fps,
                device_id=config.device_id
            )
            
            # Check if initialization was successful
            if not self._simple_camera.is_initialized():
                logger.error("[SIMPLE_CAMERA_ADAPTER] SimpleCamera failed to initialize")
                self._simple_camera = None
                return False
            
            self._config = config
            self._is_initialized = True
            
            logger.info("[SIMPLE_CAMERA_ADAPTER] Successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Initialization failed: {e}")
            self._simple_camera = None
            self._is_initialized = False
            return False
    
    async def start_streaming(self) -> None:
        """Start camera streaming."""
        if not self._is_initialized or not self._simple_camera:
            raise CameraUnavailableError("Camera not initialized")
        
        if self._is_streaming:
            logger.warning("[SIMPLE_CAMERA_ADAPTER] Already streaming")
            return
        
        try:
            self._simple_camera.start()
            self._is_streaming = True
            logger.info("[SIMPLE_CAMERA_ADAPTER] Started streaming")
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Failed to start streaming: {e}")
            self._is_streaming = False
            raise CameraUnavailableError(f"Failed to start streaming: {e}")
    
    async def stop_streaming(self) -> None:
        """Stop camera streaming."""
        if not self._is_streaming:
            return
        
        try:
            if self._simple_camera:
                self._simple_camera.stop()
            self._is_streaming = False
            logger.info("[SIMPLE_CAMERA_ADAPTER] Stopped streaming")
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Error stopping streaming: {e}")
    
    async def get_frame(self) -> Optional[Frame]:
        """Get the latest frame from camera."""
        if not self._is_initialized or not self._simple_camera:
            return None
        
        try:
            # Get latest frame from SimpleCamera
            frame_data = self._simple_camera.get_latest_frame()
            if not frame_data:
                return None
            
            # Convert to domain Frame
            frame = self._convert_frame_data_to_frame(frame_data)
            
            # Update statistics
            self._frame_count += 1
            self._last_frame_time = frame.timestamp
            async with self._last_frame_lock:
                self._last_frame = frame
            
            return frame

        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Error getting frame: {e}")
            return None

    async def get_last_frame(self) -> Optional[Frame]:
        async with self._last_frame_lock:
            return self._last_frame
    
    async def stream_frames(self) -> AsyncGenerator[Frame, None]:
        """Stream frames continuously."""
        if not self._is_initialized or not self._simple_camera:
            raise CameraUnavailableError("Camera not initialized")
        
        logger.info("[SIMPLE_CAMERA_ADAPTER] Starting frame stream")
        
        try:
            while self._is_streaming:
                frame = await self.get_frame()
                if frame:
                    yield frame
                
                # Control frame rate
                await asyncio.sleep(1.0 / self._config.fps)
                
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Error in frame stream: {e}")
            raise
    
    async def get_status(self) -> CameraInfo:
        """Get current camera status and information."""
        if not self._is_initialized:
            return CameraInfo(
                device_id=self._config.device_id if self._config else "unknown",
                camera_type=CameraType.SIMPLE,
                status=CameraStatus.DISCONNECTED
            )
        
        # Determine status
        if self._is_streaming:
            status = CameraStatus.STREAMING
        elif self._simple_camera and self._simple_camera.is_running:
            status = CameraStatus.CONNECTED
        else:
            status = CameraStatus.DISCONNECTED
        
        return CameraInfo(
            device_id=self._config.device_id if self._config else "unknown",
            camera_type=CameraType.SIMPLE,
            manufacturer="Generic",
            model="USB Camera",
            has_depth=False,
            has_imu=False,
            status=status,
            last_frame_time=self._last_frame_time,
            frame_count=self._frame_count
        )
    
    async def get_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get camera intrinsic parameters."""
        # SimpleCamera doesn't provide intrinsics, return None
        return None
    
    async def is_available(self) -> bool:
        """Check if camera is available."""
        return self._is_initialized and self._simple_camera is not None
    
    async def cleanup(self) -> None:
        """Cleanup camera resources."""
        try:
            await self.stop_streaming()
            
            if self._simple_camera:
                self._simple_camera.cleanup()
                self._simple_camera = None
            
            self._is_initialized = False
            logger.info("[SIMPLE_CAMERA_ADAPTER] Cleaned up")
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA_ADAPTER] Error during cleanup: {e}")
    
    def _convert_frame_data_to_frame(self, frame_data) -> Frame:
        """Convert SimpleCamera frame data to domain Frame."""
        # Convert numpy array to bytes
        color_bytes = frame_data.tobytes()
        
        # Create resolution
        resolution = Resolution(
            width=frame_data.shape[1],
            height=frame_data.shape[0]
        )
        
        # Create frame (no depth data for SimpleCamera)
        frame = Frame.create_from_camera_data(
            color_data=color_bytes,
            depth_data=None,
            resolution=resolution,
            camera_intrinsics=None,
            device_id=self._config.device_id if self._config else None
        )
        
        return frame
    
    @asynccontextmanager
    async def streaming_context(self):
        """Context manager for streaming operations."""
        try:
            await self.start_streaming()
            yield self
        finally:
            await self.stop_streaming()
