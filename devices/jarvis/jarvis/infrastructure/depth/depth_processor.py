#!/usr/bin/env python3

"""
Depth processor for Jarvis smart CV pipeline.

Implements IDepthService to handle depth calculations and 3D positioning.
"""

import logging
from typing import List, Optional

from ...domain.services.depth_service import IDepthService
from ...domain.entities.frame import Frame
from ...domain.entities.detection import Detection, Position3D
from ...domain.entities.camera import CameraIntrinsics
from ...domain.value_objects import DepthValue
from ...domain.exceptions import DepthCalculationError

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DepthProcessor(IDepthService):
    """Service for processing depth data and calculating 3D positions."""
    
    def __init__(self):
        self._is_initialized = True
        logger.info("[DEPTH_PROCESSOR] Initialized")
    
    async def get_depth_at_point(self, frame: Frame, x: int, y: int) -> Optional[DepthValue]:
        """Get depth value at specific pixel coordinates."""
        if not frame.has_depth_data():
            return None
        
        if not NUMPY_AVAILABLE:
            raise DepthCalculationError("NumPy not available")
        
        try:
            # Convert depth data to numpy array
            depth_data = np.frombuffer(frame.depth_data, dtype=np.uint16)
            depth_image = depth_data.reshape((frame.resolution.height, frame.resolution.width))
            
            # Check bounds
            if not (0 <= y < frame.resolution.height and 0 <= x < frame.resolution.width):
                return None
            
            # Get depth value
            depth_mm = float(depth_image[y, x])
            
            # Check if depth is valid (not zero or max value)
            if depth_mm == 0 or depth_mm >= 65535:
                return None
            
            return DepthValue(depth_mm)
            
        except Exception as e:
            logger.error(f"[DEPTH_PROCESSOR] Error getting depth at point ({x}, {y}): {e}")
            return None
    
    async def calculate_3d_position(
        self, 
        frame: Frame, 
        x: int, 
        y: int, 
        depth_mm: float
    ) -> Optional[Position3D]:
        """Calculate 3D position from pixel coordinates and depth."""
        if not frame.camera_intrinsics:
            logger.warning("[DEPTH_PROCESSOR] No camera intrinsics available")
            return None
        
        try:
            # Create CameraIntrinsics object
            intrinsics = CameraIntrinsics.from_dict(frame.camera_intrinsics)
            
            # Calculate 3D position
            x_3d, y_3d, z_3d = intrinsics.pixel_to_3d(x, y, depth_mm)
            
            return Position3D(x=x_3d, y=y_3d, z=z_3d)
            
        except Exception as e:
            logger.error(f"[DEPTH_PROCESSOR] Error calculating 3D position: {e}")
            return None
    
    async def add_depth_to_detections(self, frame: Frame, detections: List[Detection]) -> List[Detection]:
        """Add depth information to detections."""
        if not frame.has_depth_data():
            logger.debug("[DEPTH_PROCESSOR] No depth data available, returning detections unchanged")
            return detections
        
        enhanced_detections = []
        
        for detection in detections:
            try:
                # Get center point of bounding box
                center_x = detection.bounding_box.center_x
                center_y = detection.bounding_box.center_y
                
                # Get depth at center point
                depth = await self.get_depth_at_point(frame, center_x, center_y)
                
                if depth:
                    # Calculate 3D position
                    position_3d = await self.calculate_3d_position(
                        frame, center_x, center_y, depth.value_mm
                    )
                    
                    # Create enhanced detection
                    enhanced_detection = Detection(
                        class_name=detection.class_name,
                        class_id=detection.class_id,
                        confidence=detection.confidence,
                        bounding_box=detection.bounding_box,
                        depth=depth,
                        position_3d=position_3d,
                        classifier_type=detection.classifier_type,
                        processing_time=detection.processing_time,
                        model_version=detection.model_version,
                        attributes=detection.attributes
                    )
                    
                    enhanced_detections.append(enhanced_detection)
                else:
                    # Keep original detection if no depth available
                    enhanced_detections.append(detection)
                    
            except Exception as e:
                logger.error(f"[DEPTH_PROCESSOR] Error enhancing detection: {e}")
                # Keep original detection on error
                enhanced_detections.append(detection)
        
        logger.debug(f"[DEPTH_PROCESSOR] Enhanced {len(enhanced_detections)} detections with depth")
        return enhanced_detections
    
    async def calculate_center_depth(self, frame: Frame, region_size: float = 0.3) -> Optional[DepthValue]:
        """Calculate average depth in center region of frame."""
        if not frame.has_depth_data():
            return None
        
        if not NUMPY_AVAILABLE:
            raise DepthCalculationError("NumPy not available")
        
        try:
            # Convert depth data to numpy array
            depth_data = np.frombuffer(frame.depth_data, dtype=np.uint16)
            depth_image = depth_data.reshape((frame.resolution.height, frame.resolution.width))
            
            # Calculate center region bounds
            center_width = int(frame.resolution.width * region_size)
            center_height = int(frame.resolution.height * region_size)
            
            x_start = (frame.resolution.width - center_width) // 2
            y_start = (frame.resolution.height - center_height) // 2
            x_end = x_start + center_width
            y_end = y_start + center_height
            
            # Extract center region
            center_region = depth_image[y_start:y_end, x_start:x_end]
            
            # Get valid depth values (non-zero and not max value)
            valid_depths = center_region[
                (center_region > 0) & (center_region < 65535)
            ]
            
            if len(valid_depths) == 0:
                return None
            
            # Calculate average depth
            avg_depth_mm = float(np.mean(valid_depths))
            
            return DepthValue(avg_depth_mm)
            
        except Exception as e:
            logger.error(f"[DEPTH_PROCESSOR] Error calculating center depth: {e}")
            return None
    
    async def is_depth_available(self, frame: Frame) -> bool:
        """Check if frame has valid depth data."""
        if not frame.has_depth_data():
            return False
        
        try:
            # Quick validation of depth data
            if not frame.validate_depth_data():
                return False
            
            # Check if depth data contains valid values
            if not NUMPY_AVAILABLE:
                return True  # Assume valid if we can't check
            
            depth_data = np.frombuffer(frame.depth_data, dtype=np.uint16)
            depth_image = depth_data.reshape((frame.resolution.height, frame.resolution.width))
            
            # Check if there are any valid depth values
            valid_pixels = np.sum((depth_image > 0) & (depth_image < 65535))
            return valid_pixels > 0
            
        except Exception as e:
            logger.error(f"[DEPTH_PROCESSOR] Error checking depth availability: {e}")
            return False
