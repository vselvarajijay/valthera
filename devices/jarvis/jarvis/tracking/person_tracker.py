#!/usr/bin/env python3

"""
People tracking pipeline for continuous detection and re-identification.

This module provides the main tracking loop that integrates with the existing
Jarvis CV pipeline to continuously detect, track, and store people data.
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
from ..classifiers.person_classifier import PersonClassifier
from ..db.unified_tracker_db import init_db, cleanup_db
from ..video.clip_manager import cleanup_old_clips, get_storage_stats

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for people tracking"""
    enabled: bool = True
    sample_interval_seconds: float = 3.0  # How often to process frames
    similarity_threshold: float = 0.70    # ReID matching threshold
    max_detections_per_frame: int = 10    # Limit detections per frame
    cleanup_days_old: int = 7            # Days to keep clips
    cleanup_interval_hours: int = 24       # How often to run cleanup


class PersonTracker:
    """Main people tracking pipeline"""
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize people tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or TrackingConfig()
        self.registry = get_registry()
        self.person_classifier: Optional[PersonClassifier] = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._tracking_thread: Optional[threading.Thread] = None
        self._last_cleanup_time = 0
        
        logger.info("[TRACKER] People tracker initialized")
    
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
            
            # Create and register person classifier
            self.person_classifier = PersonClassifier()
            if not self.person_classifier.initialize():
                logger.error("[TRACKER] Failed to initialize person classifier")
                return False
            
            # Register classifier type with registry
            self.registry.register_classifier_type("person", PersonClassifier)
            
            logger.info("[TRACKER] People tracking pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to initialize tracking pipeline: {e}")
            return False
    
    def start_tracking(self) -> bool:
        """
        Start the tracking pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("[TRACKER] People tracking already running")
            return True
        
        if not self.person_classifier:
            logger.error("[TRACKER] Person classifier not initialized")
            return False
        
        try:
            self._stop_event.clear()
            self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._tracking_thread.start()
            self.is_running = True
            
            logger.info("[TRACKER] People tracking started")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to start tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """
        Stop the tracking pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_running:
            logger.warning("[TRACKER] People tracking not running")
            return True
        
        try:
            self._stop_event.set()
            
            if self._tracking_thread and self._tracking_thread.is_alive():
                self._tracking_thread.join(timeout=5.0)
            
            self.is_running = False
            logger.info("[TRACKER] People tracking stopped")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to stop tracking: {e}")
            return False
    
    def _tracking_loop(self):
        """Main tracking loop"""
        logger.info("[TRACKER] People tracking loop started")
        
        try:
            while not self._stop_event.is_set():
                # Get frame from camera (placeholder - integrate with actual camera)
                frame = self._get_camera_frame()
                
                if frame is not None:
                    # Detect people
                    detections = self.person_classifier.detect(frame)
                    
                    if detections:
                        logger.debug(f"[TRACKER] Detected {len(detections)} people")
                        
                        # Log detection details
                        for detection in detections:
                            person_id = detection.attributes.get('person_id') if detection.attributes else None
                            logger.debug(f"[TRACKER] Person: {detection.class_name}, confidence: {detection.confidence:.2f}, ID: {person_id}")
                
                # Run cleanup if needed
                self._run_cleanup_if_needed()
                
                # Wait for next sample
                self._stop_event.wait(self.config.sample_interval_seconds)
                
        except Exception as e:
            logger.error(f"[TRACKER] Error in tracking loop: {e}")
        finally:
            logger.info("[TRACKER] People tracking loop ended")
    
    def _get_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get frame from camera (placeholder implementation).
        
        Returns:
            Camera frame as numpy array, or None if failed
        """
        # TODO: Integrate with actual camera system
        # For now, return None to indicate no frame available
        return None
    
    def _run_cleanup_if_needed(self):
        """Run cleanup if enough time has passed"""
        current_time = time.time()
        if current_time - self._last_cleanup_time > self.config.cleanup_interval_hours * 3600:
            try:
                logger.info("[TRACKER] Running cleanup...")
                cleanup_old_clips(self.config.cleanup_days_old)
                self._last_cleanup_time = current_time
            except Exception as e:
                logger.error(f"[TRACKER] Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current tracking status.
        
        Returns:
            Status dictionary
        """
        return {
            "is_running": self.is_running,
            "start_time": None,  # TODO: Track start time
            "total_detections": self.person_classifier.stats.total_detections if self.person_classifier else 0,
            "config": {
                "sample_interval_seconds": self.config.sample_interval_seconds,
                "similarity_threshold": self.config.similarity_threshold,
                "max_detections_per_frame": self.config.max_detections_per_frame,
                "cleanup_days_old": self.config.cleanup_days_old
            }
        }
    
    def update_config(self, **kwargs):
        """
        Update tracking configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"[TRACKER] Updated config: {key} = {value}")
            else:
                logger.warning(f"[TRACKER] Unknown config parameter: {key}")
    
    def cleanup(self):
        """Cleanup tracker resources"""
        try:
            self.stop_tracking()
            
            if self.person_classifier:
                self.person_classifier.cleanup()
            
            cleanup_db()
            logger.info("[TRACKER] People tracker cleaned up")
            
        except Exception as e:
            logger.error(f"[TRACKER] Error during cleanup: {e}")


# Global tracker instance
_tracker_instance: Optional[PersonTracker] = None
_tracker_lock = threading.Lock()


def get_people_tracker() -> PersonTracker:
    """Get the global people tracker instance"""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = PersonTracker()
        return _tracker_instance


def cleanup_people_tracker():
    """Cleanup global people tracker instance"""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance:
            _tracker_instance.cleanup()
            _tracker_instance = None


def main():
    """Test the people tracker"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[TRACKER] Testing PeopleTracker...")
    
    # Initialize tracker
    tracker = PersonTracker()
    
    if not tracker.initialize():
        logger.error("[TRACKER] Failed to initialize tracker")
        return
    
    # Test configuration
    logger.info(f"[TRACKER] Initial config: {tracker.get_status()}")
    
    # Test config update
    tracker.update_config(sample_interval_seconds=1.0, similarity_threshold=0.75)
    logger.info(f"[TRACKER] Updated config: {tracker.get_status()}")
    
    # Test start/stop (without actual camera)
    logger.info("[TRACKER] Testing start/stop...")
    if tracker.start_tracking():
        logger.info("[TRACKER] Tracking started successfully")
        time.sleep(2)  # Let it run briefly
        if tracker.stop_tracking():
            logger.info("[TRACKER] Tracking stopped successfully")
        else:
            logger.error("[TRACKER] Failed to stop tracking")
    else:
        logger.error("[TRACKER] Failed to start tracking")
    
    # Cleanup
    tracker.cleanup()
    logger.info("[TRACKER] Test completed")


if __name__ == "__main__":
    main()
