#!/usr/bin/env python3

"""
Pipeline controller for Jarvis smart CV pipeline.

Handles pipeline control API endpoints using use cases.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from jarvis.infrastructure.container import (
    camera_service_dependency,
    detection_service_dependency,
    depth_service_dependency,
    frame_repository_dependency,
    detection_repository_dependency,
    metrics_repository_dependency,
    cache_repository_dependency,
    settings_dependency
)
from jarvis.infrastructure.config.settings import Settings
from jarvis.domain.services.camera_service import ICameraService
from jarvis.domain.services.detection_service import IDetectionService
from jarvis.domain.services.depth_service import IDepthService
from jarvis.domain.repositories.frame_repository import IMetricsRepository, IFrameRepository, IDetectionRepository, ICacheRepository
from jarvis.application.use_cases.start_pipeline import StartPipelineUseCase, StopPipelineUseCase, StartPipelineRequest, StartPipelineResponse, StopPipelineRequest, StopPipelineResponse
from jarvis.application.use_cases.get_system_status import GetSystemStatusUseCase, SystemStatusResponse

logger = logging.getLogger(__name__)


class PipelineController:
    """Controller for pipeline operations."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup pipeline-related routes."""
        
        @self.router.post("/start", response_model=Dict[str, Any])
        async def start_pipeline(
            initialize_camera: bool = True,
            initialize_detection: bool = True,
            start_streaming: bool = False,
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency)
        ):
            """Start the pipeline."""
            try:
                use_case = StartPipelineUseCase(camera_service, detection_service, depth_service)
                
                request = StartPipelineRequest(
                    initialize_camera=initialize_camera,
                    initialize_detection=initialize_detection,
                    start_streaming=start_streaming
                )
                
                response = await use_case.execute(request)
                
                return {
                    "success": response.success,
                    "camera_initialized": response.camera_initialized,
                    "detection_initialized": response.detection_initialized,
                    "streaming_started": response.streaming_started,
                    "message": response.message
                }
                
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error starting pipeline: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start pipeline: {str(e)}"
                )
        
        @self.router.post("/stop", response_model=Dict[str, Any])
        async def stop_pipeline(
            stop_streaming: bool = True,
            cleanup_resources: bool = True,
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency)
        ):
            """Stop the pipeline."""
            try:
                use_case = StopPipelineUseCase(camera_service, detection_service, depth_service)
                
                request = StopPipelineRequest(
                    stop_streaming=stop_streaming,
                    cleanup_resources=cleanup_resources
                )
                
                response = await use_case.execute(request)
                
                return {
                    "success": response.success,
                    "streaming_stopped": response.streaming_stopped,
                    "resources_cleaned": response.resources_cleaned,
                    "message": response.message
                }
                
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error stopping pipeline: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to stop pipeline: {str(e)}"
                )
        
        @self.router.get("/status", response_model=Dict[str, Any])
        async def get_pipeline_status(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency),
            cache_repository: ICacheRepository = Depends(cache_repository_dependency)
        ):
            """Get pipeline status."""
            try:
                use_case = GetSystemStatusUseCase(
                    camera_service,
                    detection_service,
                    depth_service,
                    frame_repository,
                    detection_repository,
                    metrics_repository,
                    cache_repository
                )
                
                response = await use_case.execute()
                
                return {
                    "overall_status": response.overall_status,
                    "timestamp": response.timestamp,
                    "components": response.components,
                    "metrics": response.metrics,
                    "health_score": response.health_score
                }
                
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error getting pipeline status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get pipeline status: {str(e)}"
                )
        
        @self.router.get("/health", response_model=Dict[str, Any])
        async def get_health_summary(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency),
            cache_repository: ICacheRepository = Depends(cache_repository_dependency)
        ):
            """Get health summary."""
            try:
                use_case = GetSystemStatusUseCase(
                    camera_service,
                    detection_service,
                    depth_service,
                    frame_repository,
                    detection_repository,
                    metrics_repository,
                    cache_repository
                )
                
                health_summary = await use_case.get_health_summary()
                return health_summary
                
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error getting health summary: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get health summary: {str(e)}"
                )
        
        @self.router.post("/initialize")
        async def initialize_pipeline(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency)
        ):
            """Initialize pipeline components."""
            try:
                use_case = StartPipelineUseCase(camera_service, detection_service, depth_service)
                
                request = StartPipelineRequest(
                    initialize_camera=True,
                    initialize_detection=True,
                    start_streaming=False
                )
                
                response = await use_case.execute(request)
                
                if response.success:
                    return {"message": "Pipeline initialized successfully"}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=response.message
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error initializing pipeline: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize pipeline: {str(e)}"
                )
        
        @self.router.post("/restart")
        async def restart_pipeline(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency)
        ):
            """Restart the pipeline."""
            try:
                # Stop pipeline
                stop_use_case = StopPipelineUseCase(camera_service, detection_service, depth_service)
                stop_request = StopPipelineRequest(stop_streaming=True, cleanup_resources=True)
                stop_response = await stop_use_case.execute(stop_request)
                
                if not stop_response.success:
                    logger.warning(f"[PIPELINE_CONTROLLER] Pipeline stop had issues: {stop_response.message}")
                
                # Start pipeline
                start_use_case = StartPipelineUseCase(camera_service, detection_service, depth_service)
                start_request = StartPipelineRequest(
                    initialize_camera=True,
                    initialize_detection=True,
                    start_streaming=False
                )
                start_response = await start_use_case.execute(start_request)
                
                return {
                    "success": start_response.success,
                    "stop_message": stop_response.message,
                    "start_message": start_response.message
                }
                
            except Exception as e:
                logger.error(f"[PIPELINE_CONTROLLER] Error restarting pipeline: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to restart pipeline: {str(e)}"
                )
