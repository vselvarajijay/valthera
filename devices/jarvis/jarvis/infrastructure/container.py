#!/usr/bin/env python3

"""
Dependency injection container for Jarvis smart CV pipeline.

Wires all dependencies using dependency-injector to enable
clean separation of concerns and testability.
"""

import logging
from typing import Optional
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from jarvis.domain.services.camera_service import ICameraService
from jarvis.domain.services.detection_service import IDetectionService
from jarvis.domain.services.depth_service import IDepthService
from jarvis.domain.services.tracking_service import ITrackingService
from jarvis.domain.services.event_bus import IEventBus
from jarvis.domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository, ICacheRepository

# Infrastructure implementations
from jarvis.infrastructure.camera import CameraFactory, RealSenseCameraAdapter, SimpleCameraAdapter
from jarvis.infrastructure.detection import YOLODetector, DetectorRegistry, ModelManager
from jarvis.infrastructure.depth import DepthProcessor, IntrinsicsCalculator
from jarvis.infrastructure.repositories import (
    InMemoryFrameRepository, 
    InMemoryDetectionRepository, 
    InMemoryMetricsRepository, 
    FrameCache
)
from jarvis.infrastructure.config.settings import Settings, get_settings
from jarvis.infrastructure.config.config_loader import ConfigLoader, get_config_loader

logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    settings = providers.Singleton(get_settings)
    config_loader = providers.Singleton(get_config_loader)
    
    # Infrastructure Services
    detector_registry = providers.Singleton(DetectorRegistry)
    model_manager = providers.Singleton(ModelManager)
    intrinsics_calculator = providers.Singleton(IntrinsicsCalculator)
    
    # Camera Service
    camera_service = providers.Singleton(
        CameraFactory.create_camera_service,
        config=settings.provided.camera
    )
    
    # Detection Service
    detection_service = providers.Singleton(
        YOLODetector,
        model_path=settings.provided.detection.model_path,
        confidence_threshold=settings.provided.detection.confidence_threshold
    )
    
    # Depth Service
    depth_service = providers.Singleton(DepthProcessor)
    
    # Tracking Service (placeholder - not implemented yet)
    tracking_service = providers.Singleton(
        providers.Factory(lambda: None)  # TODO: Implement tracking service
    )
    
    # Event Bus (placeholder - not implemented yet)
    event_bus = providers.Singleton(
        providers.Factory(lambda: None)  # TODO: Implement event bus
    )
    
    # Repositories
    frame_repository = providers.Singleton(
        InMemoryFrameRepository,
        max_frames=settings.provided.cache.max_frames,
        ttl_seconds=settings.provided.cache.frame_ttl_seconds
    )
    
    detection_repository = providers.Singleton(InMemoryDetectionRepository)
    
    metrics_repository = providers.Singleton(
        InMemoryMetricsRepository,
        ttl_seconds=settings.provided.metrics.metrics_ttl_seconds
    )
    
    cache_repository = providers.Singleton(
        FrameCache,
        max_size=settings.provided.cache.max_size,
        default_ttl_seconds=settings.provided.cache.ttl_seconds
    )


# Global container instance
_container_instance: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
        
        # Configure container with settings
        try:
            settings = get_settings()
            _container_instance.config.from_dict(settings.to_dict())
            logger.info("[CONTAINER] Container configured successfully")
        except Exception as e:
            logger.error(f"[CONTAINER] Error configuring container: {e}")
    
    return _container_instance


def configure_container(config_dict: dict) -> None:
    """Configure container with custom settings."""
    container = get_container()
    container.config.from_dict(config_dict)
    logger.info("[CONTAINER] Container reconfigured")


def wire_container(packages: list) -> None:
    """Wire container to packages for dependency injection."""
    container = get_container()
    container.wire(packages=packages)
    logger.info(f"[CONTAINER] Wired container to packages: {packages}")


def unwire_container() -> None:
    """Unwire container."""
    container = get_container()
    container.unwire()
    logger.info("[CONTAINER] Unwired container")


# Dependency injection helpers
def get_camera_service() -> ICameraService:
    """Get camera service instance."""
    container = get_container()
    return container.camera_service()


def get_detection_service() -> IDetectionService:
    """Get detection service instance."""
    container = get_container()
    return container.detection_service()


def get_depth_service() -> IDepthService:
    """Get depth service instance."""
    container = get_container()
    return container.depth_service()


def get_tracking_service() -> ITrackingService:
    """Get tracking service instance."""
    container = get_container()
    return container.tracking_service()


def get_event_bus() -> IEventBus:
    """Get event bus instance."""
    container = get_container()
    return container.event_bus()


def get_frame_repository() -> IFrameRepository:
    """Get frame repository instance."""
    container = get_container()
    return container.frame_repository()


def get_detection_repository() -> IDetectionRepository:
    """Get detection repository instance."""
    container = get_container()
    return container.detection_repository()


def get_metrics_repository() -> IMetricsRepository:
    """Get metrics repository instance."""
    container = get_container()
    return container.metrics_repository()


def get_cache_repository() -> ICacheRepository:
    """Get cache repository instance."""
    container = get_container()
    return container.cache_repository()


def get_settings() -> Settings:
    """Get settings instance."""
    container = get_container()
    return container.settings()


# FastAPI dependency injection helpers
async def camera_service_dependency() -> ICameraService:
    """FastAPI dependency for camera service."""
    return await get_camera_service()


async def detection_service_dependency() -> IDetectionService:
    """FastAPI dependency for detection service."""
    return get_detection_service()


async def depth_service_dependency() -> IDepthService:
    """FastAPI dependency for depth service."""
    return get_depth_service()


async def frame_repository_dependency() -> IFrameRepository:
    """FastAPI dependency for frame repository."""
    return get_frame_repository()


async def detection_repository_dependency() -> IDetectionRepository:
    """FastAPI dependency for detection repository."""
    return get_detection_repository()


async def metrics_repository_dependency() -> IMetricsRepository:
    """FastAPI dependency for metrics repository."""
    return get_metrics_repository()


async def cache_repository_dependency() -> ICacheRepository:
    """FastAPI dependency for cache repository."""
    return get_cache_repository()


async def settings_dependency() -> Settings:
    """FastAPI dependency for settings."""
    return get_settings()


# Cleanup function
def cleanup_container() -> None:
    """Cleanup container and all services."""
    global _container_instance
    
    if _container_instance:
        try:
            # Cleanup services
            camera_service = _container_instance.camera_service()
            if hasattr(camera_service, 'cleanup'):
                camera_service.cleanup()
            
            detection_service = _container_instance.detection_service()
            if hasattr(detection_service, 'cleanup'):
                detection_service.cleanup()
            
            depth_service = _container_instance.depth_service()
            if hasattr(depth_service, 'cleanup'):
                depth_service.cleanup()
            
            # Clear repositories
            frame_repository = _container_instance.frame_repository()
            if hasattr(frame_repository, 'clear'):
                frame_repository.clear()
            
            detection_repository = _container_instance.detection_repository()
            if hasattr(detection_repository, 'clear'):
                detection_repository.clear()
            
            cache_repository = _container_instance.cache_repository()
            if hasattr(cache_repository, 'clear'):
                cache_repository.clear()
            
            logger.info("[CONTAINER] Container cleaned up successfully")
            
        except Exception as e:
            logger.error(f"[CONTAINER] Error during cleanup: {e}")
        finally:
            _container_instance = None
