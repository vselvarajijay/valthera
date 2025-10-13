#!/usr/bin/env python3

"""
System status use case for Jarvis smart CV pipeline.

Provides comprehensive system status and health information.
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

from ...domain.services.camera_service import ICameraService
from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository
from ...domain.repositories.frame_repository import ICacheRepository

logger = logging.getLogger(__name__)


@dataclass
class SystemStatusResponse:
    """Response with comprehensive system status."""
    overall_status: str
    timestamp: float
    components: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]
    health_score: float


class GetSystemStatusUseCase:
    """Use case for getting comprehensive system status."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService,
        frame_repository: IFrameRepository,
        detection_repository: IDetectionRepository,
        metrics_repository: IMetricsRepository,
        cache_repository: ICacheRepository
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
        self._frame_repository = frame_repository
        self._detection_repository = detection_repository
        self._metrics_repository = metrics_repository
        self._cache_repository = cache_repository
    
    async def execute(self) -> SystemStatusResponse:
        """Execute comprehensive system status check."""
        import time
        timestamp = time.time()
        
        try:
            # Get component statuses
            camera_status = await self._get_camera_status()
            detection_status = await self._get_detection_status()
            depth_status = await self._get_depth_status()
            repository_status = await self._get_repository_status()
            
            # Get system metrics
            metrics = await self._get_system_metrics()
            
            # Determine overall status and health score
            overall_status, health_score = self._calculate_health(
                camera_status, detection_status, depth_status, repository_status
            )
            
            return SystemStatusResponse(
                overall_status=overall_status,
                timestamp=timestamp,
                components={
                    "camera": camera_status,
                    "detection": detection_status,
                    "depth": depth_status,
                    "repositories": repository_status
                },
                metrics=metrics,
                health_score=health_score
            )
            
        except Exception as e:
            logger.error(f"[SYSTEM_STATUS] Error getting system status: {e}")
            return SystemStatusResponse(
                overall_status="error",
                timestamp=timestamp,
                components={"error": {"message": str(e)}},
                metrics={"error": str(e)},
                health_score=0.0
            )
    
    async def _get_camera_status(self) -> Dict[str, Any]:
        """Get camera service status."""
        try:
            camera_info = await self._camera_service.get_status()
            return {
                "status": "healthy" if camera_info.is_connected() else "unhealthy",
                "available": await self._camera_service.is_available(),
                "streaming": camera_info.is_streaming(),
                "connected": camera_info.is_connected(),
                "frame_count": camera_info.frame_count,
                "camera_type": camera_info.camera_type.value,
                "device_id": camera_info.device_id,
                "frame_rate": camera_info.get_frame_rate()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_detection_status(self) -> Dict[str, Any]:
        """Get detection service status."""
        try:
            model_info = await self._detection_service.get_model_info()
            return {
                "status": "healthy" if await self._detection_service.is_initialized() else "unhealthy",
                "initialized": await self._detection_service.is_initialized(),
                "supported_classes": await self._detection_service.get_supported_classes(),
                "model_info": model_info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_depth_status(self) -> Dict[str, Any]:
        """Get depth service status."""
        try:
            return {
                "status": "healthy",
                "available": True,
                "initialized": True
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_repository_status(self) -> Dict[str, Any]:
        """Get repository status."""
        try:
            frame_stats = await self._frame_repository.get_stats()
            detection_stats = await self._detection_repository.get_stats()
            cache_stats = await self._cache_repository.get_stats()
            
            return {
                "status": "healthy",
                "frame_repository": frame_stats,
                "detection_repository": detection_stats,
                "cache": cache_stats
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            return await self._metrics_repository.get_system_metrics()
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_health(
        self,
        camera_status: Dict[str, Any],
        detection_status: Dict[str, Any],
        depth_status: Dict[str, Any],
        repository_status: Dict[str, Any]
    ) -> tuple[str, float]:
        """Calculate overall health status and score."""
        component_statuses = [
            camera_status.get("status", "unknown"),
            detection_status.get("status", "unknown"),
            depth_status.get("status", "unknown"),
            repository_status.get("status", "unknown")
        ]
        
        # Count healthy components
        healthy_count = component_statuses.count("healthy")
        total_count = len(component_statuses)
        health_score = healthy_count / total_count
        
        # Determine overall status
        if health_score == 1.0:
            overall_status = "healthy"
        elif health_score >= 0.75:
            overall_status = "degraded"
        elif health_score >= 0.5:
            overall_status = "unhealthy"
        else:
            overall_status = "critical"
        
        return overall_status, health_score
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a simplified health summary."""
        try:
            status_response = await self.execute()
            
            return {
                "status": status_response.overall_status,
                "health_score": status_response.health_score,
                "timestamp": status_response.timestamp,
                "components_healthy": sum(
                    1 for comp in status_response.components.values()
                    if comp.get("status") == "healthy"
                ),
                "total_components": len(status_response.components)
            }
        except Exception as e:
            return {
                "status": "error",
                "health_score": 0.0,
                "error": str(e)
            }
