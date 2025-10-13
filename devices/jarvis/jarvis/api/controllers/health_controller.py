#!/usr/bin/env python3

"""
Health controller for Jarvis smart CV pipeline.

Handles health check and monitoring API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from ...infrastructure.container import (
    camera_service_dependency,
    detection_service_dependency,
    depth_service_dependency,
    frame_repository_dependency,
    detection_repository_dependency,
    metrics_repository_dependency,
    cache_repository_dependency
)
from ...domain.services.camera_service import ICameraService
from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository, ICacheRepository
from ...application.use_cases.get_system_status import GetSystemStatusUseCase

logger = logging.getLogger(__name__)


class HealthController:
    """Controller for health and monitoring operations."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup health-related routes."""
        
        @self.router.get("/", response_model=Dict[str, Any])
        async def health_check(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency),
            cache_repository: ICacheRepository = Depends(cache_repository_dependency)
        ):
            """Basic health check."""
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
                
                # Determine HTTP status code based on health
                if health_summary["status"] == "healthy":
                    status_code = status.HTTP_200_OK
                elif health_summary["status"] == "degraded":
                    status_code = status.HTTP_200_OK  # Still OK but degraded
                else:
                    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                
                return health_summary
                
            except Exception as e:
                logger.error(f"[HEALTH_CONTROLLER] Error in health check: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Health check failed: {str(e)}"
                )
        
        @self.router.get("/detailed", response_model=Dict[str, Any])
        async def detailed_health_check(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency),
            cache_repository: ICacheRepository = Depends(cache_repository_dependency)
        ):
            """Detailed health check with component status."""
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
                
                # Determine HTTP status code based on health
                if response.overall_status == "healthy":
                    status_code = status.HTTP_200_OK
                elif response.overall_status == "degraded":
                    status_code = status.HTTP_200_OK  # Still OK but degraded
                else:
                    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                
                return {
                    "overall_status": response.overall_status,
                    "timestamp": response.timestamp,
                    "components": response.components,
                    "metrics": response.metrics,
                    "health_score": response.health_score
                }
                
            except Exception as e:
                logger.error(f"[HEALTH_CONTROLLER] Error in detailed health check: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Detailed health check failed: {str(e)}"
                )
        
        @self.router.get("/ready")
        async def readiness_check(
            camera_service: ICameraService = Depends(camera_service_dependency),
            detection_service: IDetectionService = Depends(detection_service_dependency)
        ):
            """Readiness check for Kubernetes."""
            try:
                camera_ready = await camera_service.is_available()
                detection_ready = await detection_service.is_initialized()
                
                if camera_ready and detection_ready:
                    return {"status": "ready"}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Service not ready"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[HEALTH_CONTROLLER] Error in readiness check: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Readiness check failed: {str(e)}"
                )
        
        @self.router.get("/live")
        async def liveness_check():
            """Liveness check for Kubernetes."""
            return {"status": "alive"}
        
        @self.router.get("/metrics", response_model=Dict[str, Any])
        async def get_metrics(
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency)
        ):
            """Get system metrics."""
            try:
                metrics = await metrics_repository.get_system_metrics()
                return metrics
                
            except Exception as e:
                logger.error(f"[HEALTH_CONTROLLER] Error getting metrics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get metrics: {str(e)}"
                )
        
        @self.router.get("/stats", response_model=Dict[str, Any])
        async def get_statistics(
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            cache_repository: ICacheRepository = Depends(cache_repository_dependency)
        ):
            """Get system statistics."""
            try:
                frame_stats = await frame_repository.get_stats()
                detection_stats = await detection_repository.get_stats()
                cache_stats = await cache_repository.get_stats()
                
                return {
                    "frames": frame_stats,
                    "detections": detection_stats,
                    "cache": cache_stats
                }
                
            except Exception as e:
                logger.error(f"[HEALTH_CONTROLLER] Error getting statistics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get statistics: {str(e)}"
                )
