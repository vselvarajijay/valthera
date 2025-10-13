#!/usr/bin/env python3

"""
Stream frames use case for Jarvis smart CV pipeline.

Orchestrates continuous frame capture, processing, and streaming.
"""

import logging
import asyncio
from typing import AsyncGenerator, Optional, Callable, List
from dataclasses import dataclass

from ...domain.services.camera_service import ICameraService
from ...domain.services.detection_service import IDetectionService
from ...domain.services.depth_service import IDepthService
from ...domain.entities.frame import Frame
from ...domain.entities.detection import Detection
from ...domain.exceptions import CameraUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class StreamFramesRequest:
    """Request to stream frames."""
    enable_detection: bool = True
    enable_depth: bool = True
    classifiers: List[str] = None
    confidence_threshold: float = 0.5
    max_detections: int = 10
    fps_limit: float = 10.0
    store_frames: bool = False
    store_detections: bool = False


@dataclass
class StreamFrameResult:
    """Result from streaming a single frame."""
    frame: Frame
    detections: List[Detection]
    processing_time_ms: float
    frame_count: int


class StreamFramesUseCase:
    """Use case for streaming frames with processing."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
        
        self._is_streaming = False
        self._frame_count = 0
        self._stream_callbacks: List[Callable[[StreamFrameResult], None]] = []
    
    async def execute(self, request: StreamFramesRequest) -> AsyncGenerator[StreamFrameResult, None]:
        """Execute frame streaming."""
        if self._is_streaming:
            logger.warning("[STREAM_FRAMES] Already streaming")
            return
        
        try:
            # Check camera availability
            if not await self._camera_service.is_available():
                raise CameraUnavailableError("Camera service not available")
            
            # Start camera streaming
            await self._camera_service.start_streaming()
            self._is_streaming = True
            
            logger.info("[STREAM_FRAMES] Started frame streaming")
            
            # Stream frames
            async for frame in self._camera_service.stream_frames():
                if not self._is_streaming:
                    break
                
                try:
                    # Process frame
                    result = await self._process_frame(frame, request)
                    
                    # Notify callbacks
                    for callback in self._stream_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"[STREAM_FRAMES] Callback error: {e}")
                    
                    yield result
                    
                    # Control frame rate
                    await asyncio.sleep(1.0 / request.fps_limit)
                    
                except Exception as e:
                    logger.error(f"[STREAM_FRAMES] Error processing frame: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[STREAM_FRAMES] Error in streaming: {e}")
            raise
        finally:
            await self.stop_streaming()
    
    async def _process_frame(self, frame: Frame, request: StreamFramesRequest) -> StreamFrameResult:
        """Process a single frame."""
        import time
        start_time = time.time()
        
        detections = []
        
        try:
            # Run detection if enabled
            if request.enable_detection:
                classifiers = request.classifiers or ["person"]
                
                for classifier in classifiers:
                    if classifier == "person":
                        class_detections = await self._detection_service.detect_persons(frame)
                    elif classifier == "vehicle":
                        class_detections = await self._detection_service.detect_vehicles(frame)
                    elif classifier == "face":
                        class_detections = await self._detection_service.detect_faces(frame)
                    else:
                        class_detections = await self._detection_service.detect_objects(frame, [classifier])
                    
                    # Filter by confidence threshold
                    filtered_detections = [
                        d for d in class_detections 
                        if d.confidence.value >= request.confidence_threshold
                    ]
                    
                    detections.extend(filtered_detections)
                
                # Limit detections
                if len(detections) > request.max_detections:
                    detections = detections[:request.max_detections]
            
            # Add depth information if enabled
            if request.enable_depth and frame.has_depth_data():
                detections = await self._depth_service.add_depth_to_detections(frame, detections)
            
            processing_time = (time.time() - start_time) * 1000
            self._frame_count += 1
            
            return StreamFrameResult(
                frame=frame,
                detections=detections,
                processing_time_ms=processing_time,
                frame_count=self._frame_count
            )
            
        except Exception as e:
            logger.error(f"[STREAM_FRAMES] Error processing frame: {e}")
            # Return result with empty detections on error
            return StreamFrameResult(
                frame=frame,
                detections=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                frame_count=self._frame_count
            )
    
    async def stop_streaming(self) -> None:
        """Stop frame streaming."""
        if not self._is_streaming:
            return
        
        try:
            await self._camera_service.stop_streaming()
            self._is_streaming = False
            logger.info("[STREAM_FRAMES] Stopped frame streaming")
            
        except Exception as e:
            logger.error(f"[STREAM_FRAMES] Error stopping streaming: {e}")
    
    def add_stream_callback(self, callback: Callable[[StreamFrameResult], None]) -> None:
        """Add callback for stream results."""
        self._stream_callbacks.append(callback)
    
    def remove_stream_callback(self, callback: Callable[[StreamFrameResult], None]) -> None:
        """Remove stream callback."""
        if callback in self._stream_callbacks:
            self._stream_callbacks.remove(callback)
    
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
    
    def get_frame_count(self) -> int:
        """Get total frame count."""
        return self._frame_count
    
    def reset_frame_count(self) -> None:
        """Reset frame count."""
        self._frame_count = 0
