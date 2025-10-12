#!/usr/bin/env python3

"""
Video clip management for vehicle tracking.

This module provides utilities for saving frame buffers as video clips,
merging multiple clips, and managing storage.
"""

import logging
import os
from typing import List, Optional
from datetime import datetime, timedelta

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Storage directories
CROPS_DIR = "/data/vehicle_tracker/crops"
CLIPS_DIR = "/data/vehicle_tracker/clips"
MERGED_DIR = "/data/vehicle_tracker/merged"

# Video settings
DEFAULT_FPS = 15
DEFAULT_CODEC = 'mp4v'
DEFAULT_FOURCC = cv2.VideoWriter_fourcc(*DEFAULT_CODEC) if CV2_AVAILABLE else None


def save_clip(frames: List[np.ndarray], output_path: str, fps: int = DEFAULT_FPS) -> Optional[str]:
    """
    Save frame buffer as video clip.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Output video file path
        fps: Frames per second
        
    Returns:
        Output path if successful, None otherwise
    """
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("[VIDEO] Required libraries not available")
        return None
    
    if not frames:
        logger.warning("[VIDEO] No frames to save")
        return None
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get frame dimensions
        height, width, channels = frames[0].shape
        
        # Create video writer
        writer = cv2.VideoWriter(
            output_path,
            DEFAULT_FOURCC,
            fps,
            (width, height)
        )
        
        if not writer.isOpened():
            logger.error(f"[VIDEO] Failed to open video writer: {output_path}")
            return None
        
        # Write frames
        for frame in frames:
            writer.write(frame)
        
        # Release writer
        writer.release()
        
        logger.debug(f"[VIDEO] Saved clip: {output_path} ({len(frames)} frames)")
        return output_path
        
    except Exception as e:
        logger.error(f"[VIDEO] Error saving clip {output_path}: {e}")
        return None


def merge_clips(clip_paths: List[str], output_path: str, fps: int = DEFAULT_FPS) -> Optional[str]:
    """
    Merge multiple video clips into one.
    
    Args:
        clip_paths: List of input clip paths
        output_path: Output merged video path
        fps: Output frames per second
        
    Returns:
        Output path if successful, None otherwise
    """
    if not MOVIEPY_AVAILABLE:
        logger.error("[VIDEO] MoviePy not available for merging")
        return None
    
    if not clip_paths:
        logger.warning("[VIDEO] No clips to merge")
        return None
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Filter existing clips
        valid_clips = []
        for clip_path in clip_paths:
            if os.path.exists(clip_path):
                try:
                    clip = VideoFileClip(clip_path)
                    valid_clips.append(clip)
                except Exception as e:
                    logger.warning(f"[VIDEO] Skipping invalid clip {clip_path}: {e}")
            else:
                logger.warning(f"[VIDEO] Clip not found: {clip_path}")
        
        if not valid_clips:
            logger.error("[VIDEO] No valid clips to merge")
            return None
        
        # Concatenate clips
        final_clip = concatenate_videoclips(valid_clips)
        
        # Write merged video
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            fps=fps,
            audio=False,
            verbose=False,
            logger=None
        )
        
        # Cleanup
        final_clip.close()
        for clip in valid_clips:
            clip.close()
        
        logger.info(f"[VIDEO] Merged {len(valid_clips)} clips: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"[VIDEO] Error merging clips to {output_path}: {e}")
        return None


def cleanup_old_clips(days_old: int = 7, dry_run: bool = False) -> Dict[str, int]:
    """
    Clean up old video clips to save storage space.
    
    Args:
        days_old: Delete clips older than this many days
        dry_run: If True, only count files without deleting
        
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        'clips_deleted': 0,
        'clips_kept': 0,
        'space_freed_mb': 0,
        'errors': 0
    }
    
    cutoff_time = datetime.now() - timedelta(days=days_old)
    
    try:
        # Clean up clips directory
        if os.path.exists(CLIPS_DIR):
            for filename in os.listdir(CLIPS_DIR):
                file_path = os.path.join(CLIPS_DIR, filename)
                
                if os.path.isfile(file_path):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_time:
                            if not dry_run:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats['space_freed_mb'] += file_size / (1024 * 1024)
                                logger.debug(f"[VIDEO] Deleted old clip: {filename}")
                            stats['clips_deleted'] += 1
                        else:
                            stats['clips_kept'] += 1
                            
                    except Exception as e:
                        logger.error(f"[VIDEO] Error processing {filename}: {e}")
                        stats['errors'] += 1
        
        # Clean up merged directory (keep merged videos longer)
        merged_cutoff = datetime.now() - timedelta(days=days_old * 2)
        if os.path.exists(MERGED_DIR):
            for filename in os.listdir(MERGED_DIR):
                file_path = os.path.join(MERGED_DIR, filename)
                
                if os.path.isfile(file_path):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < merged_cutoff:
                            if not dry_run:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats['space_freed_mb'] += file_size / (1024 * 1024)
                                logger.debug(f"[VIDEO] Deleted old merged video: {filename}")
                            stats['clips_deleted'] += 1
                        else:
                            stats['clips_kept'] += 1
                            
                    except Exception as e:
                        logger.error(f"[VIDEO] Error processing merged {filename}: {e}")
                        stats['errors'] += 1
        
        logger.info(f"[VIDEO] Cleanup complete: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"[VIDEO] Error during cleanup: {e}")
        stats['errors'] += 1
        return stats


def get_storage_stats() -> Dict[str, Any]:
    """
    Get storage statistics for video files.
    
    Returns:
        Dictionary with storage statistics
    """
    stats = {
        'total_clips': 0,
        'total_merged': 0,
        'total_size_mb': 0,
        'clips_size_mb': 0,
        'merged_size_mb': 0,
        'crops_size_mb': 0
    }
    
    try:
        # Count clips
        if os.path.exists(CLIPS_DIR):
            for filename in os.listdir(CLIPS_DIR):
                file_path = os.path.join(CLIPS_DIR, filename)
                if os.path.isfile(file_path):
                    stats['total_clips'] += 1
                    stats['clips_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
        
        # Count merged videos
        if os.path.exists(MERGED_DIR):
            for filename in os.listdir(MERGED_DIR):
                file_path = os.path.join(MERGED_DIR, filename)
                if os.path.isfile(file_path):
                    stats['total_merged'] += 1
                    stats['merged_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
        
        # Count crops
        if os.path.exists(CROPS_DIR):
            for filename in os.listdir(CROPS_DIR):
                file_path = os.path.join(CROPS_DIR, filename)
                if os.path.isfile(file_path):
                    stats['crops_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
        
        stats['total_size_mb'] = (
            stats['clips_size_mb'] + 
            stats['merged_size_mb'] + 
            stats['crops_size_mb']
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"[VIDEO] Error getting storage stats: {e}")
        return stats


def main():
    """Test video utilities"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[VIDEO] Testing video utilities...")
    
    # Test with dummy frames
    if CV2_AVAILABLE and NUMPY_AVAILABLE:
        # Create dummy frames
        frames = []
        for i in range(30):  # 1 second at 30fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Test saving clip
        test_clip_path = os.path.join(CLIPS_DIR, "test_clip.mp4")
        result = save_clip(frames, test_clip_path)
        if result:
            logger.info(f"[VIDEO] Test clip saved: {result}")
            
            # Test merging
            test_merge_path = os.path.join(MERGED_DIR, "test_merged.mp4")
            merge_result = merge_clips([test_clip_path], test_merge_path)
            if merge_result:
                logger.info(f"[VIDEO] Test merge successful: {merge_result}")
        
        # Test storage stats
        stats = get_storage_stats()
        logger.info(f"[VIDEO] Storage stats: {stats}")
        
        # Test cleanup (dry run)
        cleanup_stats = cleanup_old_clips(days_old=0, dry_run=True)
        logger.info(f"[VIDEO] Cleanup stats: {cleanup_stats}")
    
    logger.info("[VIDEO] Test completed")


if __name__ == "__main__":
    main()
