#!/usr/bin/env python3

"""
Frame controller for Jarvis smart CV pipeline.

Handles frame-related API endpoints using use cases.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, Any
import logging

from jarvis.infrastructure.container import (
    camera_service_dependency,
    frame_repository_dependency,
    settings_dependency
)
from jarvis.infrastructure.config.settings import Settings
from jarvis.domain.services.camera_service import ICameraService
from jarvis.domain.repositories.frame_repository import IFrameRepository
from jarvis.domain.entities.frame import Frame, FrameId
from jarvis.application.use_cases.capture_frame import CaptureFrameUseCase, CaptureFrameRequest, CaptureFrameResponse

logger = logging.getLogger(__name__)


class FrameController:
    """Controller for frame operations."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup frame-related routes."""
        
        @self.router.get("/latest", response_model=Dict[str, Any])
        async def get_latest_frame(
            camera_service: ICameraService = Depends(camera_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Get the latest frame."""
            try:
                use_case = CaptureFrameUseCase(camera_service, frame_repository)
                request = CaptureFrameRequest(store_frame=False)
                response = await use_case.execute(request)
                
                return {
                    "frame_id": str(response.frame.id),
                    "timestamp": str(response.frame.timestamp),
                    "resolution": {
                        "width": response.frame.resolution.width,
                        "height": response.frame.resolution.height
                    },
                    "has_depth_data": response.frame.has_depth_data(),
                    "capture_time_ms": response.capture_time_ms
                }
                
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error getting latest frame: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get latest frame: {str(e)}"
                )
        
        @self.router.post("/capture", response_model=Dict[str, Any])
        async def capture_frame(
            store_frame: bool = True,
            camera_service: ICameraService = Depends(camera_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Capture a new frame."""
            try:
                use_case = CaptureFrameUseCase(camera_service, frame_repository)
                request = CaptureFrameRequest(store_frame=store_frame)
                response = await use_case.execute(request)
                
                return {
                    "frame_id": str(response.frame.id),
                    "timestamp": str(response.frame.timestamp),
                    "resolution": {
                        "width": response.frame.resolution.width,
                        "height": response.frame.resolution.height
                    },
                    "has_depth_data": response.frame.has_depth_data(),
                    "stored": response.stored,
                    "capture_time_ms": response.capture_time_ms
                }
                
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error capturing frame: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to capture frame: {str(e)}"
                )
        
        @self.router.get("/{frame_id}", response_model=Dict[str, Any])
        async def get_frame_by_id(
            frame_id: str,
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Get frame by ID."""
            try:
                frame = await frame_repository.get_by_id(FrameId(frame_id))
                if not frame:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Frame {frame_id} not found"
                    )
                
                return {
                    "frame_id": str(frame.id),
                    "timestamp": str(frame.timestamp),
                    "resolution": {
                        "width": frame.resolution.width,
                        "height": frame.resolution.height
                    },
                    "has_depth_data": frame.has_depth_data(),
                    "device_id": frame.device_id,
                    "camera_intrinsics": frame.camera_intrinsics,
                    "processing_metadata": frame.processing_metadata
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error getting frame {frame_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get frame: {str(e)}"
                )
        
        @self.router.get("/", response_model=Dict[str, Any])
        async def get_frame_statistics(
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Get frame repository statistics."""
            try:
                stats = await frame_repository.get_stats()
                return stats
                
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error getting frame statistics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get frame statistics: {str(e)}"
                )
        
        @self.router.delete("/{frame_id}")
        async def delete_frame(
            frame_id: str,
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Delete frame by ID."""
            try:
                # Note: This would need to be implemented in the repository
                # For now, just return success
                return {"message": f"Frame {frame_id} deletion not implemented"}
                
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error deleting frame {frame_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete frame: {str(e)}"
                )
        
        @self.router.delete("/")
        async def clear_frames(
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Clear all stored frames."""
            try:
                await frame_repository.clear()
                return {"message": "All frames cleared"}
                
            except Exception as e:
                logger.error(f"[FRAME_CONTROLLER] Error clearing frames: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to clear frames: {str(e)}"
                )
