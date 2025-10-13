#!/usr/bin/env python3

"""
Capture frame use case for Jarvis smart CV pipeline.

Orchestrates frame capture from camera and storage in repository.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from ...domain.services.camera_service import ICameraService
from ...domain.repositories.frame_repository import IFrameRepository
from ...domain.entities.frame import Frame
from ...domain.exceptions import CameraUnavailableError, FrameNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class CaptureFrameRequest:
    """Request to capture a frame."""
    store_frame: bool = True
    timeout_seconds: float = 5.0


@dataclass
class CaptureFrameResponse:
    """Response from frame capture."""
    frame: Frame
    stored: bool
    capture_time_ms: float


class CaptureFrameUseCase:
    """Use case for capturing frames from camera."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        frame_repository: IFrameRepository
    ):
        self._camera_service = camera_service
        self._frame_repository = frame_repository
    
    async def execute(self, request: CaptureFrameRequest) -> CaptureFrameResponse:
        """Execute frame capture."""
        import time
        start_time = time.time()
        
        try:
            # Check if camera is available
            if not await self._camera_service.is_available():
                raise CameraUnavailableError("Camera service not available")
            
            # Capture frame from camera
            frame = await self._camera_service.get_frame()
            if not frame:
                raise FrameNotFoundError("No frame available from camera")
            
            # Store frame if requested
            stored = False
            if request.store_frame:
                await self._frame_repository.save(frame)
                stored = True
                logger.debug(f"[CAPTURE_FRAME] Stored frame {frame.id}")
            
            capture_time = (time.time() - start_time) * 1000
            
            logger.debug(f"[CAPTURE_FRAME] Captured frame {frame.id} in {capture_time:.1f}ms")
            
            return CaptureFrameResponse(
                frame=frame,
                stored=stored,
                capture_time_ms=capture_time
            )
            
        except Exception as e:
            logger.error(f"[CAPTURE_FRAME] Error capturing frame: {e}")
            raise
    
    async def capture_latest_frame(self) -> Optional[Frame]:
        """Capture the latest available frame."""
        try:
            if not await self._camera_service.is_available():
                return None
            
            return await self._camera_service.get_frame()
            
        except Exception as e:
            logger.error(f"[CAPTURE_FRAME] Error getting latest frame: {e}")
            return None
    
    async def is_camera_ready(self) -> bool:
        """Check if camera is ready for capture."""
        try:
            return await self._camera_service.is_available()
        except Exception as e:
            logger.error(f"[CAPTURE_FRAME] Error checking camera readiness: {e}")
            return False
