#!/usr/bin/env python3

"""
FastAPI endpoints for people tracking data.

This module provides REST API endpoints for accessing people detection
data, managing tracking configuration, and generating merged videos.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...db.unified_tracker_db import (
    get_all_objects, get_object_timeline, get_object_stats
)
from ...video.clip_manager import merge_clips, get_storage_stats, cleanup_old_clips
from ...tracking.unified_controller import get_tracking_controller, TrackingMode

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/people", tags=["people"])


# Pydantic models
class PersonResponse(BaseModel):
    person_id: int
    first_seen: datetime
    last_seen: datetime
    count: int
    thumbnail_paths: List[str]
    clip_paths: List[str]
    metadata: Dict[str, Any] = {}


class PersonStatsResponse(BaseModel):
    total_persons: int
    returning_persons: int
    recent_persons: int
    avg_sightings_per_person: float
    return_rate: float


class TrackingStatusResponse(BaseModel):
    is_running: bool
    start_time: Optional[datetime] = None
    total_detections: int = 0


class TrackingConfigResponse(BaseModel):
    sample_interval_seconds: float
    similarity_threshold: float
    max_detections_per_frame: int
    cleanup_days_old: int


class MergeVideoRequest(BaseModel):
    output_filename: Optional[str] = None


# People endpoints
@router.get("/", response_model=List[PersonResponse])
async def get_people():
    """Get all tracked people"""
    try:
        people = get_all_objects('person')
        return [
            PersonResponse(
                person_id=p['person_id'],
                first_seen=p['first_seen'],
                last_seen=p['last_seen'],
                count=p['count'],
                thumbnail_paths=p['thumbnail_paths'],
                clip_paths=p['clip_paths'],
                metadata=p.get('metadata', {})
            )
            for p in people
        ]
    except Exception as e:
        logger.error(f"[API] Error getting people: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve people")


@router.get("/stats", response_model=PersonStatsResponse)
async def get_people_stats():
    """Get people tracking statistics"""
    try:
        stats = get_object_stats('person')
        return PersonStatsResponse(**stats)
    except Exception as e:
        logger.error(f"[API] Error getting people stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve people statistics")


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int = Path(..., description="Person ID")):
    """Get specific person timeline"""
    try:
        person = get_object_timeline(person_id, 'person')
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return PersonResponse(
            person_id=person['person_id'],
            first_seen=person['first_seen'],
            last_seen=person['last_seen'],
            count=person['count'],
            thumbnail_paths=person['thumbnail_paths'],
            clip_paths=person['clip_paths'],
            metadata=person.get('metadata', {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error getting person {person_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve person")


@router.get("/{person_id}/clips")
async def get_person_clips(person_id: int = Path(..., description="Person ID")):
    """Get video clips for a specific person"""
    try:
        person = get_object_timeline(person_id, 'person')
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        clips = person['clip_paths']
        return {"person_id": person_id, "clips": clips}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error getting person clips {person_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve person clips")


@router.post("/{person_id}/merge")
async def merge_person_clips(
    person_id: int = Path(..., description="Person ID"),
    request: MergeVideoRequest = None
):
    """Merge all video clips for a specific person"""
    try:
        person = get_object_timeline(person_id, 'person')
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        clips = person['clip_paths']
        if not clips:
            raise HTTPException(status_code=404, detail="No clips found for this person")
        
        # Generate output filename if not provided
        output_filename = request.output_filename if request and request.output_filename else f"person_{person_id}_merged.mp4"
        
        # Merge clips
        result = merge_clips(clips, output_filename)
        
        if result['success']:
            return {
                "message": "Clips merged successfully",
                "person_id": person_id,
                "merged_video_path": result['output_path'],
                "clips_merged": len(clips)
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to merge clips: {result.get('error', 'Unknown error')}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error merging person clips {person_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to merge person clips")


@router.get("/{person_id}/merged/{filename}")
async def download_merged_video(
    person_id: int = Path(..., description="Person ID"),
    filename: str = Path(..., description="Merged video filename")
):
    """Download merged video for a specific person"""
    try:
        # Construct file path (this would need to be implemented based on your storage structure)
        file_path = f"/data/tracker/people/merged/{filename}"
        
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
        logger.error(f"[API] Error downloading merged video {person_id}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download video")


# Tracking control endpoints
@router.post("/tracking/start")
async def start_tracking():
    """Start people tracking"""
    try:
        controller = get_tracking_controller()
        if controller.start_tracking(TrackingMode.PEOPLE):
            return {"message": "People tracking started", "status": "running"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start people tracking")
    except Exception as e:
        logger.error(f"[API] Error starting people tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to start people tracking")


@router.post("/tracking/stop")
async def stop_tracking():
    """Stop people tracking"""
    try:
        controller = get_tracking_controller()
        if controller.stop_tracking():
            return {"message": "People tracking stopped", "status": "stopped"}
        else:
            raise HTTPException(status_code=500, detail="Failed to stop people tracking")
    except Exception as e:
        logger.error(f"[API] Error stopping people tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop people tracking")


@router.get("/tracking/status", response_model=TrackingStatusResponse)
async def get_tracking_status():
    """Get current people tracking status"""
    try:
        controller = get_tracking_controller()
        status = controller.get_status()
        
        # Extract people-specific status
        people_status = status['people_tracker']['status']
        
        return TrackingStatusResponse(
            is_running=status['people_tracker']['is_running'],
            start_time=None,  # TODO: Track start time
            total_detections=people_status['total_detections']
        )
    except Exception as e:
        logger.error(f"[API] Error getting people tracking status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get people tracking status")


@router.get("/tracking/config", response_model=TrackingConfigResponse)
async def get_tracking_config():
    """Get current people tracking configuration"""
    try:
        controller = get_tracking_controller()
        status = controller.get_status()
        
        # Extract people-specific config
        people_config = status['people_tracker']['status']['config']
        
        return TrackingConfigResponse(**people_config)
    except Exception as e:
        logger.error(f"[API] Error getting people tracking config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get people tracking configuration")


@router.post("/tracking/config")
async def update_tracking_config(config: TrackingConfigResponse):
    """Update people tracking configuration"""
    try:
        controller = get_tracking_controller()
        
        # Update people tracker config
        controller.people_tracker.update_config(
            sample_interval_seconds=config.sample_interval_seconds,
            similarity_threshold=config.similarity_threshold,
            max_detections_per_frame=config.max_detections_per_frame,
            cleanup_days_old=config.cleanup_days_old
        )
        
        return {"message": "People tracking configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"[API] Error updating people tracking config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update people tracking configuration")


# Storage management endpoints
@router.get("/storage/stats")
async def get_storage_stats():
    """Get people storage statistics"""
    try:
        stats = get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"[API] Error getting people storage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve storage statistics")


@router.post("/storage/cleanup")
async def cleanup_storage(
    days_old: int = Query(7, description="Delete files older than this many days"),
    dry_run: bool = Query(True, description="Preview only, don't actually delete")
):
    """Cleanup old people storage files"""
    try:
        result = cleanup_old_clips(days_old, dry_run)
        return {
            "message": "Cleanup completed" if not dry_run else "Cleanup preview completed",
            "dry_run": dry_run,
            "stats": result
        }
    except Exception as e:
        logger.error(f"[API] Error cleaning up people storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup storage")
