#!/usr/bin/env python3

"""
Pipeline management use cases for Jarvis smart CV pipeline.

Orchestrates system startup, shutdown, and status management.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ...domain.services.camera_service import ICameraService
from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository
from ...domain.exceptions import CameraUnavailableError, ModelLoadError

logger = logging.getLogger(__name__)


@dataclass
class StartPipelineRequest:
    """Request to start pipeline."""
    initialize_camera: bool = True
    initialize_detection: bool = True
    start_streaming: bool = False


@dataclass
class StartPipelineResponse:
    """Response from starting pipeline."""
    success: bool
    camera_initialized: bool
    detection_initialized: bool
    streaming_started: bool
    message: str


class StartPipelineUseCase:
    """Use case for starting the pipeline."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
    
    async def execute(self, request: StartPipelineRequest) -> StartPipelineResponse:
        """Execute pipeline startup."""
        camera_initialized = False
        detection_initialized = False
        streaming_started = False
        errors = []
        
        try:
            # Initialize camera if requested
            if request.initialize_camera:
                try:
                    camera_available = await self._camera_service.is_available()
                    if not camera_available:
                        # Try to initialize camera
                        from ...infrastructure.config.settings import get_settings
                        settings = get_settings()
                        camera_available = await self._camera_service.initialize(settings.camera)
                    
                    camera_initialized = camera_available
                    if camera_initialized:
                        logger.info("[START_PIPELINE] Camera initialized successfully")
                    else:
                        errors.append("Failed to initialize camera")
                        
                except Exception as e:
                    logger.error(f"[START_PIPELINE] Camera initialization error: {e}")
                    errors.append(f"Camera initialization failed: {e}")
            
            # Initialize detection if requested
            if request.initialize_detection:
                try:
                    detection_available = await self._detection_service.is_initialized()
                    if not detection_available:
                        # Try to initialize detection service
                        await self._detection_service.initialize()
                        detection_available = await self._detection_service.is_initialized()
                    
                    detection_initialized = detection_available
                    if detection_initialized:
                        logger.info("[START_PIPELINE] Detection service initialized successfully")
                    else:
                        errors.append("Failed to initialize detection service")
                        
                except Exception as e:
                    logger.error(f"[START_PIPELINE] Detection initialization error: {e}")
                    errors.append(f"Detection initialization failed: {e}")
            
            # Start streaming if requested
            if request.start_streaming and camera_initialized:
                try:
                    await self._camera_service.start_streaming()
                    streaming_started = True
                    logger.info("[START_PIPELINE] Camera streaming started")
                except Exception as e:
                    logger.error(f"[START_PIPELINE] Streaming start error: {e}")
                    errors.append(f"Failed to start streaming: {e}")
            
            # Determine overall success
            success = camera_initialized and detection_initialized
            
            # Create response message
            if success:
                message = "Pipeline started successfully"
                if errors:
                    message += f" with warnings: {'; '.join(errors)}"
            else:
                message = f"Pipeline startup failed: {'; '.join(errors)}"
            
            logger.info(f"[START_PIPELINE] Pipeline startup result: {message}")
            
            return StartPipelineResponse(
                success=success,
                camera_initialized=camera_initialized,
                detection_initialized=detection_initialized,
                streaming_started=streaming_started,
                message=message
            )
            
        except Exception as e:
            logger.error(f"[START_PIPELINE] Unexpected error: {e}")
            return StartPipelineResponse(
                success=False,
                camera_initialized=camera_initialized,
                detection_initialized=detection_initialized,
                streaming_started=streaming_started,
                message=f"Pipeline startup failed: {e}"
            )


@dataclass
class StopPipelineRequest:
    """Request to stop pipeline."""
    stop_streaming: bool = True
    cleanup_resources: bool = True


@dataclass
class StopPipelineResponse:
    """Response from stopping pipeline."""
    success: bool
    streaming_stopped: bool
    resources_cleaned: bool
    message: str


class StopPipelineUseCase:
    """Use case for stopping the pipeline."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
    
    async def execute(self, request: StopPipelineRequest) -> StopPipelineResponse:
        """Execute pipeline shutdown."""
        streaming_stopped = False
        resources_cleaned = False
        errors = []
        
        try:
            # Stop streaming if requested
            if request.stop_streaming:
                try:
                    await self._camera_service.stop_streaming()
                    streaming_stopped = True
                    logger.info("[STOP_PIPELINE] Camera streaming stopped")
                except Exception as e:
                    logger.error(f"[STOP_PIPELINE] Streaming stop error: {e}")
                    errors.append(f"Failed to stop streaming: {e}")
            
            # Cleanup resources if requested
            if request.cleanup_resources:
                try:
                    await self._camera_service.cleanup()
                    await self._detection_service.cleanup()
                    await self._depth_service.cleanup()
                    resources_cleaned = True
                    logger.info("[STOP_PIPELINE] Resources cleaned up")
                except Exception as e:
                    logger.error(f"[STOP_PIPELINE] Resource cleanup error: {e}")
                    errors.append(f"Resource cleanup failed: {e}")
            
            # Determine overall success
            success = streaming_stopped and resources_cleaned
            
            # Create response message
            if success:
                message = "Pipeline stopped successfully"
                if errors:
                    message += f" with warnings: {'; '.join(errors)}"
            else:
                message = f"Pipeline stop failed: {'; '.join(errors)}"
            
            logger.info(f"[STOP_PIPELINE] Pipeline stop result: {message}")
            
            return StopPipelineResponse(
                success=success,
                streaming_stopped=streaming_stopped,
                resources_cleaned=resources_cleaned,
                message=message
            )
            
        except Exception as e:
            logger.error(f"[STOP_PIPELINE] Unexpected error: {e}")
            return StopPipelineResponse(
                success=False,
                streaming_stopped=streaming_stopped,
                resources_cleaned=resources_cleaned,
                message=f"Pipeline stop failed: {e}"
            )


@dataclass
class SystemStatusResponse:
    """Response with system status information."""
    overall_status: str
    camera_status: Dict[str, Any]
    detection_status: Dict[str, Any]
    depth_status: Dict[str, Any]
    system_metrics: Dict[str, Any]


class GetSystemStatusUseCase:
    """Use case for getting system status."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService,
        metrics_repository: IMetricsRepository
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
        self._metrics_repository = metrics_repository
    
    async def execute(self) -> SystemStatusResponse:
        """Execute system status check."""
        try:
            # Get camera status
            camera_status = await self._get_camera_status()
            
            # Get detection status
            detection_status = await self._get_detection_status()
            
            # Get depth status
            depth_status = await self._get_depth_status()
            
            # Get system metrics
            system_metrics = await self._metrics_repository.get_system_metrics()
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                camera_status, detection_status, depth_status
            )
            
            return SystemStatusResponse(
                overall_status=overall_status,
                camera_status=camera_status,
                detection_status=detection_status,
                depth_status=depth_status,
                system_metrics=system_metrics
            )
            
        except Exception as e:
            logger.error(f"[SYSTEM_STATUS] Error getting system status: {e}")
            return SystemStatusResponse(
                overall_status="error",
                camera_status={"error": str(e)},
                detection_status={"error": str(e)},
                depth_status={"error": str(e)},
                system_metrics={"error": str(e)}
            )
    
    async def _get_camera_status(self) -> Dict[str, Any]:
        """Get camera status information."""
        try:
            camera_info = await self._camera_service.get_status()
            return {
                "available": await self._camera_service.is_available(),
                "streaming": camera_info.is_streaming(),
                "connected": camera_info.is_connected(),
                "frame_count": camera_info.frame_count,
                "camera_type": camera_info.camera_type.value,
                "device_id": camera_info.device_id
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_detection_status(self) -> Dict[str, Any]:
        """Get detection service status."""
        try:
            model_info = await self._detection_service.get_model_info()
            return {
                "initialized": await self._detection_service.is_initialized(),
                "supported_classes": await self._detection_service.get_supported_classes(),
                "model_info": model_info
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_depth_status(self) -> Dict[str, Any]:
        """Get depth service status."""
        try:
            return {
                "available": True,  # DepthProcessor is always available
                "initialized": True
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _determine_overall_status(
        self, 
        camera_status: Dict[str, Any], 
        detection_status: Dict[str, Any], 
        depth_status: Dict[str, Any]
    ) -> str:
        """Determine overall system status."""
        if "error" in camera_status or "error" in detection_status or "error" in depth_status:
            return "error"
        
        camera_ok = camera_status.get("available", False)
        detection_ok = detection_status.get("initialized", False)
        depth_ok = depth_status.get("available", False)
        
        if camera_ok and detection_ok and depth_ok:
            return "healthy"
        elif camera_ok or detection_ok:
            return "degraded"
        else:
            return "unhealthy"
