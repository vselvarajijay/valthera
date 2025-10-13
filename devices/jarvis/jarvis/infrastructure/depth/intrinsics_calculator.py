#!/usr/bin/env python3

"""
Intrinsics calculator for Jarvis smart CV pipeline.

Handles camera intrinsic parameter calculations and conversions.
"""

import logging
from typing import Optional, Dict, Any, Tuple

from ...domain.entities.camera import CameraIntrinsics
from ...domain.exceptions import DepthCalculationError

logger = logging.getLogger(__name__)


class IntrinsicsCalculator:
    """Calculator for camera intrinsic parameters."""
    
    @staticmethod
    def calculate_intrinsics_from_resolution(
        width: int, 
        height: int, 
        fov_degrees: float = 60.0
    ) -> CameraIntrinsics:
        """
        Calculate approximate intrinsics from resolution and field of view.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Field of view in degrees
            
        Returns:
            CameraIntrinsics object with calculated parameters
        """
        try:
            import math
            
            # Convert FOV to radians
            fov_rad = math.radians(fov_degrees)
            
            # Calculate focal length (assuming horizontal FOV)
            fx = width / (2 * math.tan(fov_rad / 2))
            fy = fx  # Assume square pixels
            
            # Principal point at image center
            ppx = width / 2
            ppy = height / 2
            
            return CameraIntrinsics(
                fx=fx,
                fy=fy,
                ppx=ppx,
                ppy=ppy,
                width=width,
                height=height
            )
            
        except Exception as e:
            logger.error(f"[INTRINSICS_CALCULATOR] Error calculating intrinsics: {e}")
            raise DepthCalculationError(f"Failed to calculate intrinsics: {e}")
    
    @staticmethod
    def estimate_depth_from_size(
        bounding_box_width: int,
        bounding_box_height: int,
        real_object_width_mm: float,
        real_object_height_mm: float,
        intrinsics: CameraIntrinsics
    ) -> Optional[float]:
        """
        Estimate depth from bounding box size and known object dimensions.
        
        Args:
            bounding_box_width: Width of bounding box in pixels
            bounding_box_height: Height of bounding box in pixels
            real_object_width_mm: Real width of object in mm
            real_object_height_mm: Real height of object in mm
            intrinsics: Camera intrinsic parameters
            
        Returns:
            Estimated depth in mm, or None if calculation fails
        """
        try:
            # Calculate depth from width
            depth_from_width = (real_object_width_mm * intrinsics.fx) / bounding_box_width
            
            # Calculate depth from height
            depth_from_height = (real_object_height_mm * intrinsics.fy) / bounding_box_height
            
            # Use average of both estimates
            estimated_depth = (depth_from_width + depth_from_height) / 2
            
            # Validate depth (reasonable range)
            if 100 <= estimated_depth <= 10000:  # 10cm to 10m
                return estimated_depth
            else:
                logger.warning(f"[INTRINSICS_CALCULATOR] Estimated depth out of range: {estimated_depth}mm")
                return None
                
        except Exception as e:
            logger.error(f"[INTRINSICS_CALCULATOR] Error estimating depth: {e}")
            return None
    
    @staticmethod
    def pixel_to_3d_with_intrinsics(
        x: int, 
        y: int, 
        depth_mm: float, 
        intrinsics: CameraIntrinsics
    ) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates and depth to 3D coordinates using intrinsics.
        
        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            depth_mm: Depth in millimeters
            intrinsics: Camera intrinsic parameters
            
        Returns:
            Tuple of (x_3d, y_3d, z_3d) in meters
        """
        return intrinsics.pixel_to_3d(x, y, depth_mm)
    
    @staticmethod
    def validate_intrinsics(intrinsics: CameraIntrinsics) -> bool:
        """
        Validate camera intrinsic parameters.
        
        Args:
            intrinsics: Camera intrinsic parameters to validate
            
        Returns:
            True if intrinsics are valid, False otherwise
        """
        try:
            # Check focal lengths
            if intrinsics.fx <= 0 or intrinsics.fy <= 0:
                logger.warning("[INTRINSICS_CALCULATOR] Invalid focal lengths")
                return False
            
            # Check principal point
            if intrinsics.ppx < 0 or intrinsics.ppy < 0:
                logger.warning("[INTRINSICS_CALCULATOR] Invalid principal point")
                return False
            
            # Check image dimensions
            if intrinsics.width <= 0 or intrinsics.height <= 0:
                logger.warning("[INTRINSICS_CALCULATOR] Invalid image dimensions")
                return False
            
            # Check reasonable focal length range
            max_focal_length = max(intrinsics.width, intrinsics.height) * 2
            if intrinsics.fx > max_focal_length or intrinsics.fy > max_focal_length:
                logger.warning("[INTRINSICS_CALCULATOR] Focal lengths too large")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[INTRINSICS_CALCULATOR] Error validating intrinsics: {e}")
            return False
    
    @staticmethod
    def get_default_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
        """
        Get default camera intrinsics for common resolutions.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Default CameraIntrinsics object
        """
        # Common focal lengths for different resolutions
        focal_lengths = {
            (640, 480): (525.0, 525.0),
            (1280, 720): (1050.0, 1050.0),
            (1920, 1080): (1575.0, 1575.0)
        }
        
        fx, fy = focal_lengths.get((width, height), (525.0, 525.0))
        
        return CameraIntrinsics(
            fx=fx,
            fy=fy,
            ppx=width / 2,
            ppy=height / 2,
            width=width,
            height=height
        )
