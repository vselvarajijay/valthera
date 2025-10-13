#!/usr/bin/env python3

"""
Analyze frame use case for Jarvis smart CV pipeline.

Orchestrates frame analysis including detection, depth processing,
and result storage.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.repositories.frame_repository import IFrameRepository, IDetectionRepository, IMetricsRepository
from ...domain.entities.frame import Frame, FrameId
from ...domain.entities.detection import Detection
from ...domain.value_objects import ProcessingTime
from ...domain.exceptions import FrameNotFoundError, DetectionError

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeFrameRequest:
    """Request to analyze a frame."""
    frame_id: Optional[FrameId] = None
    classifiers: List[str] = None
    confidence_threshold: float = 0.5
    include_depth: bool = True
    include_3d_position: bool = True
    max_detections: int = 10
    store_results: bool = True


@dataclass
class AnalyzeFrameResponse:
    """Response from frame analysis."""
    frame_id: FrameId
    detections: List[Detection]
    processing_time_ms: float
    detection_count: int
    stored: bool


class AnalyzeFrameUseCase:
    """Use case for analyzing frames."""
    
    def __init__(
        self,
        detection_service: IDetectionService,
        depth_service: IDepthService,
        frame_repository: IFrameRepository,
        detection_repository: IDetectionRepository,
        metrics_repository: IMetricsRepository
    ):
        self._detection_service = detection_service
        self._depth_service = depth_service
        self._frame_repository = frame_repository
        self._detection_repository = detection_repository
        self._metrics_repository = metrics_repository
    
    async def execute(self, request: AnalyzeFrameRequest) -> AnalyzeFrameResponse:
        """Execute frame analysis."""
        import time
        start_time = time.time()
        
        try:
            # Get frame
            frame = await self._get_frame(request.frame_id)
            if not frame:
                raise FrameNotFoundError(f"Frame {request.frame_id} not found")
            
            # Set default classifiers if not specified
            classifiers = request.classifiers or ["person"]
            
            # Run detection
            detections = await self._run_detection(frame, classifiers, request.confidence_threshold)
            
            # Limit detections
            if len(detections) > request.max_detections:
                detections = detections[:request.max_detections]
                logger.debug(f"[ANALYZE_FRAME] Limited detections to {request.max_detections}")
            
            # Add depth information if requested
            if request.include_depth and frame.has_depth_data():
                detections = await self._depth_service.add_depth_to_detections(frame, detections)
                logger.debug(f"[ANALYZE_FRAME] Added depth to {len(detections)} detections")
            
            # Store results if requested
            stored = False
            if request.store_results:
                await self._detection_repository.save_detections(frame.id, detections)
                stored = True
                logger.debug(f"[ANALYZE_FRAME] Stored {len(detections)} detections")
            
            # Record metrics
            processing_time = (time.time() - start_time) * 1000
            await self._record_metrics(frame.id, processing_time, detections)
            
            logger.debug(f"[ANALYZE_FRAME] Analyzed frame {frame.id}: {len(detections)} detections in {processing_time:.1f}ms")
            
            return AnalyzeFrameResponse(
                frame_id=frame.id,
                detections=detections,
                processing_time_ms=processing_time,
                detection_count=len(detections),
                stored=stored
            )
            
        except Exception as e:
            logger.error(f"[ANALYZE_FRAME] Error analyzing frame: {e}")
            raise
    
    async def _get_frame(self, frame_id: Optional[FrameId]) -> Optional[Frame]:
        """Get frame by ID or latest frame."""
        if frame_id:
            return await self._frame_repository.get_by_id(frame_id)
        else:
            return await self._frame_repository.get_latest()
    
    async def _run_detection(
        self, 
        frame: Frame, 
        classifiers: List[str], 
        confidence_threshold: float
    ) -> List[Detection]:
        """Run detection on frame."""
        try:
            all_detections = []
            
            for classifier in classifiers:
                if classifier == "person":
                    detections = await self._detection_service.detect_persons(frame)
                elif classifier == "vehicle":
                    detections = await self._detection_service.detect_vehicles(frame)
                elif classifier == "face":
                    detections = await self._detection_service.detect_faces(frame)
                else:
                    # Generic object detection
                    detections = await self._detection_service.detect_objects(frame, [classifier])
                
                # Filter by confidence threshold
                filtered_detections = [
                    d for d in detections 
                    if d.confidence.value >= confidence_threshold
                ]
                
                all_detections.extend(filtered_detections)
                logger.debug(f"[ANALYZE_FRAME] {classifier}: {len(filtered_detections)} detections")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"[ANALYZE_FRAME] Error running detection: {e}")
            raise DetectionError("detection", str(e))
    
    async def _record_metrics(
        self, 
        frame_id: FrameId, 
        processing_time_ms: float, 
        detections: List[Detection]
    ) -> None:
        """Record analysis metrics."""
        try:
            # Record processing time
            await self._metrics_repository.record_processing_time(
                "analyze_frame",
                processing_time_ms,
                {"frame_id": str(frame_id), "detection_count": len(detections)}
            )
            
            # Record detection counts by class
            detection_counts = {}
            for detection in detections:
                class_name = detection.class_name
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            for class_name, count in detection_counts.items():
                await self._metrics_repository.record_detection_count(class_name, count)
            
            # Record frame processing metrics
            await self._metrics_repository.record_frame_processed(
                frame_id,
                processing_time_ms,
                len(detections)
            )
            
        except Exception as e:
            logger.error(f"[ANALYZE_FRAME] Error recording metrics: {e}")
    
    async def analyze_latest_frame(self, classifiers: List[str] = None) -> Optional[AnalyzeFrameResponse]:
        """Analyze the latest available frame."""
        try:
            request = AnalyzeFrameRequest(
                classifiers=classifiers or ["person"],
                store_results=False
            )
            return await self.execute(request)
            
        except Exception as e:
            logger.error(f"[ANALYZE_FRAME] Error analyzing latest frame: {e}")
            return None
    
    async def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        try:
            return await self._detection_repository.get_detection_statistics()
        except Exception as e:
            logger.error(f"[ANALYZE_FRAME] Error getting detection statistics: {e}")
            return {}
