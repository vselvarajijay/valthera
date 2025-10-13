#!/usr/bin/env python3

"""
Camera controller for Jarvis smart CV pipeline.

Handles camera-related API endpoints using use cases.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from typing import Dict, Any
import logging
import io

from PIL import Image

from ...infrastructure.container import camera_service_dependency, frame_repository_dependency
from ...domain.services.camera_service import ICameraService
from ...domain.repositories.frame_repository import IFrameRepository
from ...application.use_cases.capture_frame import CaptureFrameUseCase, CaptureFrameRequest

logger = logging.getLogger(__name__)


class CameraController:
    """Controller for camera operations."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup camera-related routes."""
        
        @self.router.get("/status", response_model=Dict[str, Any])
        async def get_camera_status(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Get camera status."""
            try:
                camera_info = await camera_service.get_status()
                return camera_info.to_dict()
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error getting camera status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get camera status: {str(e)}"
                )
        
        @self.router.get("/available")
        async def is_camera_available(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Check if camera is available."""
            try:
                available = await camera_service.is_available()
                return {"available": available}
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error checking camera availability: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to check camera availability: {str(e)}"
                )
        
        @self.router.post("/start-streaming")
        async def start_streaming(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Start camera streaming."""
            try:
                await camera_service.start_streaming()
                return {"message": "Camera streaming started"}
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error starting streaming: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start streaming: {str(e)}"
                )
        
        @self.router.post("/stop-streaming")
        async def stop_streaming(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Stop camera streaming."""
            try:
                await camera_service.stop_streaming()
                return {"message": "Camera streaming stopped"}
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error stopping streaming: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to stop streaming: {str(e)}"
                )
        
        @self.router.get("/intrinsics", response_model=Dict[str, Any])
        async def get_camera_intrinsics(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Get camera intrinsic parameters."""
            try:
                intrinsics = await camera_service.get_intrinsics()
                if intrinsics:
                    return intrinsics.to_dict()
                else:
                    return {"message": "Camera intrinsics not available"}
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error getting camera intrinsics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get camera intrinsics: {str(e)}"
                )
        
        @self.router.post("/cleanup")
        async def cleanup_camera(
            camera_service: ICameraService = Depends(camera_service_dependency)
        ):
            """Cleanup camera resources."""
            try:
                await camera_service.cleanup()
                return {"message": "Camera resources cleaned up"}
                
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error cleaning up camera: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to cleanup camera: {str(e)}"
                )

        @self.router.get("/raw")
        async def get_raw_frame(
            camera_id: str = "video2",
            camera_service: ICameraService = Depends(camera_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency)
        ):
            """Return a single JPEG frame for simple live view polling.

            Uses the latest stored frame from the frame repository.
            If none is available, attempts to ensure streaming is started and returns 404.
            """
            try:
                # Use the same capture flow as /frames/latest to obtain a fresh frame
                # Try fresh capture (fast path)
                frame = await camera_service.get_frame()
                if frame is None or getattr(frame, "color_data", None) is None:
                    # Try adapter last-frame buffer
                    try:
                        frame = await camera_service.get_last_frame()
                    except Exception:
                        frame = None
                if frame is None or getattr(frame, "color_data", None) is None:
                    # Try repository latest as final fallback
                    try:
                        frame = await frame_repository.get_latest()
                    except Exception:
                        frame = None

                if frame is None or getattr(frame, "color_data", None) is None:
                    # Ensure streaming is on to populate frames
                    try:
                        if hasattr(camera_service, "start_streaming"):
                            await camera_service.start_streaming()
                    except Exception:
                        pass
                    raise HTTPException(status_code=404, detail="No frame available")

                # Convert raw RGB bytes to PIL Image using frame resolution
                try:
                    size = (frame.resolution.width, frame.resolution.height)
                    img = Image.frombytes('RGB', size, frame.color_data)
                except Exception:
                    raise HTTPException(status_code=500, detail="Failed to convert frame to image")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return Response(content=buf.getvalue(), media_type="image/jpeg")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[CAMERA_CONTROLLER] Error getting raw frame: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get frame: {str(e)}"
                )
