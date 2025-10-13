#!/usr/bin/env python3

"""
Vehicle tracking pipeline for continuous detection and re-identification.

This module provides the main tracking loop that integrates with the existing
Jarvis CV pipeline to continuously detect, track, and store vehicle data.
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..classifiers.registry import get_registry
from ..classifiers.vehicle_classifier import VehicleClassifier
from ..db.vehicle_db import init_db, cleanup_db
from ..video.clip_manager import cleanup_old_clips, get_storage_stats

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for vehicle tracking"""
    enabled: bool = True
    sample_interval_seconds: float = 3.0  # How often to process frames
    similarity_threshold: float = 0.7    # ReID matching threshold
    max_detections_per_frame: int = 10    # Limit detections per frame
    cleanup_days_old: int = 7             # Days to keep clips
    cleanup_interval_hours: int = 24       # How often to run cleanup


class VehicleTracker:
    """Main vehicle tracking pipeline"""
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize vehicle tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or TrackingConfig()
        self.registry = get_registry()
        self.vehicle_classifier: Optional[VehicleClassifier] = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._tracking_thread: Optional[threading.Thread] = None
        self._last_cleanup_time = 0
        
        logger.info("[TRACKER] Vehicle tracker initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the tracking pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize database
            if not init_db():
                logger.error("[TRACKER] Failed to initialize database")
                return False
            
            # Create and register vehicle classifier
            self.vehicle_classifier = VehicleClassifier()
            if not self.vehicle_classifier.initialize():
                logger.error("[TRACKER] Failed to initialize vehicle classifier")
                return False
            
            # Register classifier type with registry
            self.registry.register_classifier_type("vehicle", VehicleClassifier)
            
            logger.info("[TRACKER] Vehicle tracking pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to initialize tracking pipeline: {e}")
            return False
    
    def start_tracking(self) -> bool:
        """
        Start the vehicle tracking loop.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("[TRACKER] Tracking already running")
            return True
        
        if not self.vehicle_classifier:
            logger.error("[TRACKER] Vehicle classifier not initialized")
            return False
        
        try:
            self._stop_event.clear()
            self.is_running = True
            
            # Start tracking thread
            self._tracking_thread = threading.Thread(
                target=self._tracking_loop,
                name="VehicleTracker",
                daemon=True
            )
            self._tracking_thread.start()
            
            logger.info("[TRACKER] Vehicle tracking started")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to start tracking: {e}")
            self.is_running = False
            return False
    
    def stop_tracking(self):
        """Stop the vehicle tracking loop"""
        if not self.is_running:
            logger.warning("[TRACKER] Tracking not running")
            return
        
        try:
            self._stop_event.set()
            self.is_running = False
            
            if self._tracking_thread and self._tracking_thread.is_alive():
                self._tracking_thread.join(timeout=5.0)
            
            logger.info("[TRACKER] Vehicle tracking stopped")
            
        except Exception as e:
            logger.error(f"[TRACKER] Error stopping tracking: {e}")
    
    def _tracking_loop(self):
        """Main tracking loop (runs in separate thread)"""
        logger.info("[TRACKER] Tracking loop started")
        
        last_process_time = 0
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time to process
                if current_time - last_process_time >= self.config.sample_interval_seconds:
                    # Process frame (this would integrate with existing camera pipeline)
                    # For now, we'll just log that we're ready to process
                    logger.debug("[TRACKER] Ready to process frame")
                    last_process_time = current_time
                
                # Check if it's time for cleanup
                if current_time - self._last_cleanup_time >= self.config.cleanup_interval_hours * 3600:
                    self._run_cleanup()
                    self._last_cleanup_time = current_time
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[TRACKER] Error in tracking loop: {e}")
                time.sleep(1.0)  # Wait before retrying
        
        logger.info("[TRACKER] Tracking loop ended")
    
    def _run_cleanup(self):
        """Run storage cleanup"""
        try:
            logger.info("[TRACKER] Running storage cleanup...")
            
            # Clean up old clips
            cleanup_stats = cleanup_old_clips(
                days_old=self.config.cleanup_days_old,
                dry_run=False
            )
            
            logger.info(f"[TRACKER] Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"[TRACKER] Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current tracking status.
        
        Returns:
            Status dictionary
        """
        storage_stats = get_storage_stats()
        
        return {
            'is_running': self.is_running,
            'config': {
                'enabled': self.config.enabled,
                'sample_interval_seconds': self.config.sample_interval_seconds,
                'similarity_threshold': self.config.similarity_threshold,
                'max_detections_per_frame': self.config.max_detections_per_frame,
                'cleanup_days_old': self.config.cleanup_days_old,
                'cleanup_interval_hours': self.config.cleanup_interval_hours
            },
            'classifier': {
                'initialized': self.vehicle_classifier is not None and self.vehicle_classifier.is_initialized,
                'frame_buffer_size': self.vehicle_classifier.get_frame_buffer_size() if self.vehicle_classifier else 0
            },
            'storage': storage_stats,
            'last_cleanup_time': self._last_cleanup_time
        }
    
    def update_config(self, new_config: TrackingConfig):
        """
        Update tracking configuration.
        
        Args:
            new_config: New configuration
        """
        self.config = new_config
        logger.info(f"[TRACKER] Configuration updated: {new_config}")
    
    def cleanup(self):
        """Cleanup tracker resources"""
        try:
            self.stop_tracking()
            
            if self.vehicle_classifier:
                self.vehicle_classifier.cleanup()
            
            cleanup_db()
            
            logger.info("[TRACKER] Vehicle tracker cleaned up")
            
        except Exception as e:
            logger.error(f"[TRACKER] Error during cleanup: {e}")


# Global tracker instance
_tracker_instance: Optional[VehicleTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> VehicleTracker:
    """Get the global vehicle tracker instance"""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = VehicleTracker()
        return _tracker_instance


def cleanup_tracker():
    """Cleanup global tracker instance"""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance:
            _tracker_instance.cleanup()
            _tracker_instance = None


def main():
    """Test the vehicle tracker"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[TRACKER] Testing vehicle tracker...")
    
    # Initialize tracker
    tracker = get_tracker()
    
    if not tracker.initialize():
        logger.error("[TRACKER] Failed to initialize tracker")
        return
    
    # Start tracking
    if tracker.start_tracking():
        logger.info("[TRACKER] Tracking started successfully")
        
        # Let it run for a bit
        time.sleep(10)
        
        # Get status
        status = tracker.get_status()
        logger.info(f"[TRACKER] Status: {status}")
        
        # Stop tracking
        tracker.stop_tracking()
    
    # Cleanup
    cleanup_tracker()
    logger.info("[TRACKER] Test completed")


if __name__ == "__main__":
    main()
