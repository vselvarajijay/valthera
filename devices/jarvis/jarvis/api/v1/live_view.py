"""
Live view API endpoints for real-time tracking visualization
"""
import io
import logging
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Response, Depends
from PIL import Image

from ...tracking.unified_controller import get_tracking_controller, TrackingMode
from ...classifiers.registry import get_registry
from ...infrastructure.container import camera_service_dependency, frame_repository_dependency
from ...domain.services.camera_service import ICameraService
from ...domain.repositories.frame_repository import IFrameRepository

logger = logging.getLogger(__name__)

router = APIRouter()

# Global registry instance
_registry = None

def get_registry_instance():
    """Get registry instance"""
    global _registry
    if _registry is None:
        _registry = get_registry()
    return _registry


@router.get("/annotated")
def get_annotated_frame():
    """Get the latest annotated frame with tracking overlays"""
    try:
        # Get tracking controller
        controller = get_tracking_controller()
        current_mode = controller.get_current_mode()
        
        if current_mode == TrackingMode.STOPPED:
            # Return raw frame if no tracking is active
            return get_raw_frame()
        
        # Get registry and current tracker
        registry = get_registry_instance()
        
        # Get classifier from the active tracker instead of registry
        classifier = None
        if current_mode == TrackingMode.VEHICLES:
            # Get vehicle classifier from vehicle tracker
            vehicle_tracker = controller.vehicle_tracker
            if hasattr(vehicle_tracker, 'vehicle_classifier'):
                classifier = vehicle_tracker.vehicle_classifier
        elif current_mode == TrackingMode.PEOPLE:
            # Get person classifier from people tracker
            people_tracker = controller.people_tracker
            if hasattr(people_tracker, 'person_classifier'):
                classifier = people_tracker.person_classifier
        
        if not classifier:
            logger.warning(f"[LIVE_VIEW] No classifier available for {current_mode}")
            return get_raw_frame()
        
        # Get latest frame from camera API
        try:
            # Use the working camera API endpoint
            import requests
            response = requests.get("http://jarvis-api:8001/api/v1/camera/raw", timeout=2)
            if response.status_code == 200:
                # Convert response to OpenCV format
                image_bytes = response.content
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Run detection and annotation
                    detections = classifier.detect(frame)
                    annotated_frame = classifier.annotate_frame(frame, detections)
                    
                    # Convert back to JPEG
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    return Response(content=buffer.tobytes(), media_type="image/jpeg")
            
        except Exception as e:
            logger.error(f"[LIVE_VIEW] Error getting camera frame: {e}")
        
        # Fallback to raw frame
        return get_raw_frame()
        
    except Exception as e:
        logger.error(f"[LIVE_VIEW] Error generating annotated frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating annotated frame: {str(e)}")


@router.get("/raw")
def get_raw_frame():
    """Get the latest raw frame"""
    try:
        # Use the working camera API endpoint
        import requests
        response = requests.get("http://jarvis-api:8001/api/v1/camera/raw", timeout=2)
        
        if response.status_code == 200:
            return Response(content=response.content, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="No camera frame available")
            
    except Exception as e:
        logger.error(f"[LIVE_VIEW] Error getting raw frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting raw frame: {str(e)}")


@router.get("/status")
def get_live_view_status():
    """Get live view status including tracking mode"""
    try:
        controller = get_tracking_controller()
        current_mode = controller.get_current_mode()
        
        return {
            "tracking_mode": current_mode.value,
            "is_tracking": current_mode != TrackingMode.STOPPED,
            "annotated_available": current_mode != TrackingMode.STOPPED,
            "raw_available": True
        }
        
    except Exception as e:
        logger.error(f"[LIVE_VIEW] Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")
