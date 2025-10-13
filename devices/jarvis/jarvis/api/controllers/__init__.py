#!/usr/bin/env python3

"""
API controllers for Jarvis smart CV pipeline.

Clean API controllers that delegate to use cases and handle
HTTP concerns only.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict, Any
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
from jarvis.domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository, ICacheRepository

from jarvis.api.controllers.frame_controller import FrameController
from jarvis.api.controllers.detection_controller import DetectionController
from jarvis.api.controllers.pipeline_controller import PipelineController
from jarvis.api.controllers.camera_controller import CameraController
from jarvis.api.controllers.health_controller import HealthController

logger = logging.getLogger(__name__)


def create_api_router() -> APIRouter:
    """Create the main API router with all controllers."""
    router = APIRouter(prefix="/api/v1", tags=["jarvis"])
    
    # Create controllers with dependencies
    frame_controller = FrameController()
    detection_controller = DetectionController()
    pipeline_controller = PipelineController()
    camera_controller = CameraController()
    health_controller = HealthController()
    
    # Register routes
    router.include_router(frame_controller.router, prefix="/frames", tags=["frames"])
    router.include_router(detection_controller.router, prefix="/detections", tags=["detections"])
    router.include_router(pipeline_controller.router, prefix="/pipeline", tags=["pipeline"])
    router.include_router(camera_controller.router, prefix="/camera", tags=["camera"])
    router.include_router(health_controller.router, prefix="/health", tags=["health"])
    
    return router
