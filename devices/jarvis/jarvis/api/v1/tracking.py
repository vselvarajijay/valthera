#!/usr/bin/env python3

"""
Unified tracking API for mode switching and control.

This module provides REST API endpoints for managing tracking modes
between vehicles and people, ensuring only one object type tracks at a time.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...tracking.unified_controller import get_tracking_controller, TrackingMode

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/tracking", tags=["tracking"])


# Pydantic models
class TrackingModeRequest(BaseModel):
    object_type: str  # 'vehicles' or 'people'


class TrackingStatusResponse(BaseModel):
    current_mode: str
    is_running: bool
    vehicle_tracker: Dict[str, Any]
    people_tracker: Dict[str, Any]


class TrackingModeResponse(BaseModel):
    message: str
    current_mode: str
    status: str


# Unified tracking endpoints
@router.post("/start", response_model=TrackingModeResponse)
def start_tracking(request: TrackingModeRequest):
    """Start tracking for the specified object type"""
    try:
        # Validate object type
        if request.object_type == "vehicles":
            mode = TrackingMode.VEHICLES
        elif request.object_type == "people":
            mode = TrackingMode.PEOPLE
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid object_type. Must be 'vehicles' or 'people'"
            )
        
        controller = get_tracking_controller()
        
        if controller.start_tracking(mode):
            return TrackingModeResponse(
                message=f"{request.object_type.title()} tracking started",
                current_mode=request.object_type,
                status="running"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to start {request.object_type} tracking"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error starting {request.object_type} tracking: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start {request.object_type} tracking"
        )


@router.post("/stop", response_model=TrackingModeResponse)
def stop_tracking():
    """Stop any currently active tracking"""
    try:
        controller = get_tracking_controller()
        
        if controller.stop_tracking():
            return TrackingModeResponse(
                message="Tracking stopped",
                current_mode="stopped",
                status="stopped"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to stop tracking")
            
    except Exception as e:
        logger.error(f"[API] Error stopping tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop tracking")


@router.get("/status", response_model=TrackingStatusResponse)
def get_tracking_status():
    """Get current tracking status and mode"""
    try:
        controller = get_tracking_controller()
        status = controller.get_status()
        
        return TrackingStatusResponse(
            current_mode=status['current_mode'],
            is_running=status['is_running'],
            vehicle_tracker=status['vehicle_tracker'],
            people_tracker=status['people_tracker']
        )
        
    except Exception as e:
        logger.error(f"[API] Error getting tracking status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tracking status")


@router.post("/switch", response_model=TrackingModeResponse)
def switch_tracking_mode(request: TrackingModeRequest):
    """Switch to a different tracking mode"""
    try:
        # Validate object type
        if request.object_type == "vehicles":
            mode = TrackingMode.VEHICLES
        elif request.object_type == "people":
            mode = TrackingMode.PEOPLE
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid object_type. Must be 'vehicles' or 'people'"
            )
        
        controller = get_tracking_controller()
        
        if controller.switch_mode(mode):
            return TrackingModeResponse(
                message=f"Switched to {request.object_type} tracking",
                current_mode=request.object_type,
                status="running"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to switch to {request.object_type} tracking"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error switching to {request.object_type} tracking: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to switch to {request.object_type} tracking"
        )


@router.post("/initialize")
def initialize_tracking():
    """Initialize all tracking systems"""
    try:
        controller = get_tracking_controller()
        results = controller.initialize_all()
        
        return {
            "message": "Tracking systems initialized",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"[API] Error initializing tracking systems: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize tracking systems")


@router.get("/modes")
def get_available_modes():
    """Get available tracking modes"""
    return {
        "available_modes": [
            {
                "mode": "vehicles",
                "description": "Track vehicles with Re-ID",
                "enabled": True
            },
            {
                "mode": "people", 
                "description": "Track people with Re-ID",
                "enabled": True
            },
            {
                "mode": "stopped",
                "description": "No tracking active",
                "enabled": True
            }
        ],
        "note": "Only one mode can be active at a time"
    }


@router.get("/health")
def health_check():
    """Health check for tracking systems"""
    try:
        controller = get_tracking_controller()
        status = controller.get_status()
        
        # Check if trackers are properly initialized
        vehicle_healthy = status['vehicle_tracker']['status'].get('total_detections', 0) >= 0
        people_healthy = status['people_tracker']['status'].get('total_detections', 0) >= 0
        
        overall_healthy = vehicle_healthy and people_healthy
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "vehicle_tracker": "healthy" if vehicle_healthy else "unhealthy",
                "people_tracker": "healthy" if people_healthy else "unhealthy",
                "current_mode": status['current_mode'],
                "is_running": status['is_running']
            }
        }
        
    except Exception as e:
        logger.error(f"[API] Error in health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
