#!/usr/bin/env python3

"""
FastAPI endpoints for vehicle tracking data.

This module provides REST API endpoints for accessing vehicle detection
data, managing tracking configuration, and generating merged videos.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...db.vehicle_db import (
    get_all_vehicles, get_vehicle_timeline, get_vehicle_stats,
    find_match, add_vehicle, update_vehicle
)
from ...video.clip_manager import merge_clips, get_storage_stats, cleanup_old_clips
from ...tracking.vehicle_tracker import get_tracker, TrackingConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/vehicles", tags=["vehicles"])


# Pydantic models
class VehicleResponse(BaseModel):
    vehicle_id: int
    first_seen: datetime
    last_seen: datetime
    count: int
    thumbnail_paths: List[str]
    clip_paths: List[str]


class VehicleStatsResponse(BaseModel):
    total_vehicles: int
    returning_vehicles: int
    recent_vehicles: int
    avg_sightings_per_vehicle: float
    return_rate: float


class TrackingConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    sample_interval_seconds: Optional[float] = None
    similarity_threshold: Optional[float] = None
    max_detections_per_frame: Optional[int] = None
    cleanup_days_old: Optional[int] = None
    cleanup_interval_hours: Optional[int] = None


class MergeVideoRequest(BaseModel):
    output_filename: Optional[str] = None


# Vehicle endpoints
@router.get("/", response_model=List[VehicleResponse])
async def get_vehicles():
    """Get all tracked vehicles"""
    try:
        vehicles = get_all_vehicles()
        return [
            VehicleResponse(
                vehicle_id=v['vehicle_id'],
                first_seen=v['first_seen'],
                last_seen=v['last_seen'],
                count=v['count'],
                thumbnail_paths=v['thumbnail_paths'],
                clip_paths=v['clip_paths']
            )
            for v in vehicles
        ]
    except Exception as e:
        logger.error(f"[API] Error getting vehicles: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve vehicles")


@router.get("/stats", response_model=VehicleStatsResponse)
async def get_vehicle_stats():
    """Get vehicle tracking statistics"""
    try:
        stats = get_vehicle_stats()
        return VehicleStatsResponse(**stats)
    except Exception as e:
        logger.error(f"[API] Error getting vehicle stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve vehicle statistics")


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: int = Path(..., description="Vehicle ID")):
    """Get details for a specific vehicle"""
    try:
        vehicle = get_vehicle_timeline(vehicle_id)
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        return VehicleResponse(
            vehicle_id=vehicle['vehicle_id'],
            first_seen=vehicle['first_seen'],
            last_seen=vehicle['last_seen'],
            count=vehicle['count'],
            thumbnail_paths=vehicle['thumbnail_paths'],
            clip_paths=vehicle['clip_paths']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error getting vehicle {vehicle_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve vehicle")


@router.get("/{vehicle_id}/clips")
async def get_vehicle_clips(vehicle_id: int = Path(..., description="Vehicle ID")):
    """Get all clips for a specific vehicle"""
    try:
        vehicle = get_vehicle_timeline(vehicle_id)
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        return {
            "vehicle_id": vehicle_id,
            "clips": vehicle['clip_paths'],
            "thumbnails": vehicle['thumbnail_paths'],
            "count": len(vehicle['clip_paths'])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error getting clips for vehicle {vehicle_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve clips")


@router.post("/{vehicle_id}/merge")
async def merge_vehicle_clips(
    vehicle_id: int = Path(..., description="Vehicle ID"),
    request: MergeVideoRequest = MergeVideoRequest()
):
    """Generate merged video for all clips of a vehicle"""
    try:
        vehicle = get_vehicle_timeline(vehicle_id)
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        clip_paths = vehicle['clip_paths']
        if not clip_paths:
            raise HTTPException(status_code=400, detail="No clips found for vehicle")
        
        # Generate output filename
        output_filename = request.output_filename or f"vehicle_{vehicle_id}_merged.mp4"
        output_path = f"/data/vehicle_tracker/merged/{output_filename}"
        
        # Merge clips
        result_path = merge_clips(clip_paths, output_path)
        if not result_path:
            raise HTTPException(status_code=500, detail="Failed to merge clips")
        
        return {
            "vehicle_id": vehicle_id,
            "merged_video_path": result_path,
            "clips_merged": len(clip_paths),
            "message": "Video merged successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error merging clips for vehicle {vehicle_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to merge clips")


@router.get("/{vehicle_id}/merged/{filename}")
async def download_merged_video(
    vehicle_id: int = Path(..., description="Vehicle ID"),
    filename: str = Path(..., description="Merged video filename")
):
    """Download a merged video file"""
    try:
        file_path = f"/data/vehicle_tracker/merged/{filename}"
        
        # Verify vehicle exists
        vehicle = get_vehicle_timeline(vehicle_id)
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Merged video not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="video/mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error downloading merged video {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download video")


# Tracking control endpoints
@router.post("/tracking/start")
async def start_tracking():
    """Start vehicle tracking"""
    try:
        tracker = get_tracker()
        if tracker.start_tracking():
            return {"message": "Vehicle tracking started", "status": "running"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start tracking")
    except Exception as e:
        logger.error(f"[API] Error starting tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to start tracking")


@router.post("/tracking/stop")
async def stop_tracking():
    """Stop vehicle tracking"""
    try:
        tracker = get_tracker()
        tracker.stop_tracking()
        return {"message": "Vehicle tracking stopped", "status": "stopped"}
    except Exception as e:
        logger.error(f"[API] Error stopping tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop tracking")


@router.get("/tracking/status")
async def get_tracking_status():
    """Get current tracking status"""
    try:
        tracker = get_tracker()
        status = tracker.get_status()
        return status
    except Exception as e:
        logger.error(f"[API] Error getting tracking status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tracking status")


@router.get("/tracking/config")
async def get_tracking_config():
    """Get current tracking configuration"""
    try:
        tracker = get_tracker()
        status = tracker.get_status()
        return status['config']
    except Exception as e:
        logger.error(f"[API] Error getting tracking config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tracking configuration")


@router.post("/tracking/config")
async def update_tracking_config(request: TrackingConfigRequest):
    """Update tracking configuration"""
    try:
        tracker = get_tracker()
        current_status = tracker.get_status()
        current_config = current_status['config']
        
        # Update only provided fields
        new_config = TrackingConfig(
            enabled=request.enabled if request.enabled is not None else current_config['enabled'],
            sample_interval_seconds=request.sample_interval_seconds if request.sample_interval_seconds is not None else current_config['sample_interval_seconds'],
            similarity_threshold=request.similarity_threshold if request.similarity_threshold is not None else current_config['similarity_threshold'],
            max_detections_per_frame=request.max_detections_per_frame if request.max_detections_per_frame is not None else current_config['max_detections_per_frame'],
            cleanup_days_old=request.cleanup_days_old if request.cleanup_days_old is not None else current_config['cleanup_days_old'],
            cleanup_interval_hours=request.cleanup_interval_hours if request.cleanup_interval_hours is not None else current_config['cleanup_interval_hours']
        )
        
        tracker.update_config(new_config)
        
        return {
            "message": "Configuration updated successfully",
            "config": new_config.__dict__
        }
    except Exception as e:
        logger.error(f"[API] Error updating tracking config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tracking configuration")


# Storage management endpoints
@router.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics"""
    try:
        stats = get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"[API] Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get storage statistics")


@router.post("/storage/cleanup")
async def run_storage_cleanup(
    days_old: int = Query(7, description="Delete files older than this many days"),
    dry_run: bool = Query(False, description="If true, only count files without deleting")
):
    """Run storage cleanup"""
    try:
        stats = cleanup_old_clips(days_old=days_old, dry_run=dry_run)
        return {
            "message": "Cleanup completed" if not dry_run else "Cleanup simulation completed",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"[API] Error running cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to run storage cleanup")


def main():
    """Test the vehicle API endpoints"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[API] Testing vehicle API endpoints...")
    
    # Test database operations
    try:
        vehicles = get_all_vehicles()
        logger.info(f"[API] Found {len(vehicles)} vehicles")
        
        stats = get_vehicle_stats()
        logger.info(f"[API] Vehicle stats: {stats}")
        
        storage_stats = get_storage_stats()
        logger.info(f"[API] Storage stats: {storage_stats}")
        
    except Exception as e:
        logger.error(f"[API] Error testing API: {e}")
    
    logger.info("[API] Test completed")


if __name__ == "__main__":
    main()
