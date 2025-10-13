#!/usr/bin/env python3

"""
Detection controller for Jarvis smart CV pipeline.

Handles detection-related API endpoints using use cases.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging

from ...infrastructure.container import (
    detection_service_dependency,
    depth_service_dependency,
    frame_repository_dependency,
    detection_repository_dependency,
    metrics_repository_dependency,
    settings_dependency
)
from ...infrastructure.config.settings import Settings
from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository
from ...domain.entities.frame import FrameId
from ...application.use_cases.analyze_frame import AnalyzeFrameUseCase, AnalyzeFrameRequest, AnalyzeFrameResponse

logger = logging.getLogger(__name__)


class DetectionController:
    """Controller for detection operations."""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup detection-related routes."""
        
        @self.router.post("/analyze", response_model=Dict[str, Any])
        async def analyze_frame(
            frame_id: Optional[str] = None,
            classifiers: List[str] = Query(default=["person"]),
            confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
            include_depth: bool = Query(default=True),
            include_3d_position: bool = Query(default=True),
            max_detections: int = Query(default=10, ge=1, le=100),
            store_results: bool = Query(default=True),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency)
        ):
            """Analyze a frame for detections."""
            try:
                use_case = AnalyzeFrameUseCase(
                    detection_service,
                    depth_service,
                    frame_repository,
                    detection_repository,
                    metrics_repository
                )
                
                request = AnalyzeFrameRequest(
                    frame_id=FrameId(frame_id) if frame_id else None,
                    classifiers=classifiers,
                    confidence_threshold=confidence_threshold,
                    include_depth=include_depth,
                    include_3d_position=include_3d_position,
                    max_detections=max_detections,
                    store_results=store_results
                )
                
                response = await use_case.execute(request)
                
                return {
                    "frame_id": str(response.frame_id),
                    "detections": [detection.to_dict() for detection in response.detections],
                    "detection_count": response.detection_count,
                    "processing_time_ms": response.processing_time_ms,
                    "stored": response.stored
                }
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error analyzing frame: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to analyze frame: {str(e)}"
                )
        
        @self.router.get("/latest", response_model=Dict[str, Any])
        async def analyze_latest_frame(
            classifiers: List[str] = Query(default=["person"]),
            detection_service: IDetectionService = Depends(detection_service_dependency),
            depth_service: IDepthService = Depends(depth_service_dependency),
            frame_repository: IFrameRepository = Depends(frame_repository_dependency),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency),
            metrics_repository: IMetricsRepository = Depends(metrics_repository_dependency)
        ):
            """Analyze the latest frame."""
            try:
                use_case = AnalyzeFrameUseCase(
                    detection_service,
                    depth_service,
                    frame_repository,
                    detection_repository,
                    metrics_repository
                )
                
                response = await use_case.analyze_latest_frame(classifiers)
                
                if not response:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="No frame available for analysis"
                    )
                
                return {
                    "frame_id": str(response.frame_id),
                    "detections": [detection.to_dict() for detection in response.detections],
                    "detection_count": response.detection_count,
                    "processing_time_ms": response.processing_time_ms,
                    "stored": response.stored
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error analyzing latest frame: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to analyze latest frame: {str(e)}"
                )
        
        @self.router.get("/frame/{frame_id}", response_model=Dict[str, Any])
        async def get_detections_by_frame(
            frame_id: str,
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency)
        ):
            """Get detections for a specific frame."""
            try:
                detections = await detection_repository.get_detections_by_frame(FrameId(frame_id))
                
                return {
                    "frame_id": frame_id,
                    "detections": [detection.to_dict() for detection in detections],
                    "detection_count": len(detections)
                }
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error getting detections for frame {frame_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get detections: {str(e)}"
                )
        
        @self.router.get("/class/{class_name}", response_model=Dict[str, Any])
        async def get_detections_by_class(
            class_name: str,
            start_time: Optional[str] = Query(default=None),
            end_time: Optional[str] = Query(default=None),
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency)
        ):
            """Get detections by class name."""
            try:
                from ...domain.value_objects import Timestamp
                
                start_timestamp = Timestamp.from_datetime(start_time) if start_time else None
                end_timestamp = Timestamp.from_datetime(end_time) if end_time else None
                
                detections = await detection_repository.get_detections_by_class(
                    class_name, start_timestamp, end_timestamp
                )
                
                return {
                    "class_name": class_name,
                    "detections": [detection.to_dict() for detection in detections],
                    "detection_count": len(detections)
                }
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error getting detections for class {class_name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get detections: {str(e)}"
                )
        
        @self.router.get("/statistics", response_model=Dict[str, Any])
        async def get_detection_statistics(
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency)
        ):
            """Get detection statistics."""
            try:
                stats = await detection_repository.get_detection_statistics()
                return stats
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error getting detection statistics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get detection statistics: {str(e)}"
                )
        
        @self.router.get("/supported-classes", response_model=List[str])
        async def get_supported_classes(
            detection_service: IDetectionService = Depends(detection_service_dependency)
        ):
            """Get supported detection classes."""
            try:
                classes = await detection_service.get_supported_classes()
                return classes
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error getting supported classes: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get supported classes: {str(e)}"
                )
        
        @self.router.get("/model-info", response_model=Dict[str, Any])
        async def get_model_info(
            detection_service: IDetectionService = Depends(detection_service_dependency)
        ):
            """Get detection model information."""
            try:
                model_info = await detection_service.get_model_info()
                return model_info
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error getting model info: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get model info: {str(e)}"
                )
        
        @self.router.delete("/frame/{frame_id}")
        async def delete_detections_by_frame(
            frame_id: str,
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency)
        ):
            """Delete detections for a frame."""
            try:
                count = await detection_repository.delete_detections_by_frame(FrameId(frame_id))
                return {"message": f"Deleted {count} detections for frame {frame_id}"}
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error deleting detections for frame {frame_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete detections: {str(e)}"
                )
        
        @self.router.delete("/")
        async def clear_detections(
            detection_repository: IDetectionRepository = Depends(detection_repository_dependency)
        ):
            """Clear all stored detections."""
            try:
                await detection_repository.clear()
                return {"message": "All detections cleared"}
                
            except Exception as e:
                logger.error(f"[DETECTION_CONTROLLER] Error clearing detections: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to clear detections: {str(e)}"
                )
