#!/usr/bin/env python3

"""
Unified tracking controller for managing toggle-based tracking.

This module provides a controller to manage tracking between vehicles and people,
ensuring only one object type can track at a time.
"""

import logging
import threading
from typing import Optional, Dict, Any
from enum import Enum

from .vehicle_tracker import get_tracker as get_vehicle_tracker
from .person_tracker import get_people_tracker

logger = logging.getLogger(__name__)


class TrackingMode(Enum):
    """Supported tracking modes"""
    VEHICLES = "vehicles"
    PEOPLE = "people"
    STOPPED = "stopped"


class UnifiedTrackingController:
    """Controller to manage toggle-based tracking between vehicles and people"""
    
    def __init__(self):
        """Initialize the unified tracking controller"""
        self.current_mode = TrackingMode.STOPPED
        self._lock = threading.Lock()
        
        # Get tracker instances
        self.vehicle_tracker = get_vehicle_tracker()
        self.people_tracker = get_people_tracker()
        
        # Initialize all trackers
        self.initialize_all()
        
        logger.info("[UNIFIED_CONTROLLER] Unified tracking controller initialized")
    
    def start_tracking(self, mode: TrackingMode) -> bool:
        """
        Start tracking for the specified mode.
        Stops any currently running tracker first.
        
        Args:
            mode: Tracking mode to start
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Stop current tracking if running
                if self.current_mode != TrackingMode.STOPPED:
                    logger.info(f"[UNIFIED_CONTROLLER] Stopping current tracking mode: {self.current_mode.value}")
                    self._stop_current_tracking()
                
                # Start new tracking mode
                if mode == TrackingMode.VEHICLES:
                    success = self.vehicle_tracker.start_tracking()
                    if success:
                        self.current_mode = TrackingMode.VEHICLES
                        logger.info("[UNIFIED_CONTROLLER] Vehicle tracking started")
                    else:
                        logger.error("[UNIFIED_CONTROLLER] Failed to start vehicle tracking")
                        return False
                        
                elif mode == TrackingMode.PEOPLE:
                    success = self.people_tracker.start_tracking()
                    if success:
                        self.current_mode = TrackingMode.PEOPLE
                        logger.info("[UNIFIED_CONTROLLER] People tracking started")
                    else:
                        logger.error("[UNIFIED_CONTROLLER] Failed to start people tracking")
                        return False
                        
                elif mode == TrackingMode.STOPPED:
                    self.current_mode = TrackingMode.STOPPED
                    logger.info("[UNIFIED_CONTROLLER] Tracking stopped")
                    return True
                    
                else:
                    logger.error(f"[UNIFIED_CONTROLLER] Unknown tracking mode: {mode}")
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"[UNIFIED_CONTROLLER] Error starting tracking mode {mode}: {e}")
                return False
    
    def stop_tracking(self) -> bool:
        """
        Stop any currently running tracking.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if self.current_mode == TrackingMode.STOPPED:
                    logger.warning("[UNIFIED_CONTROLLER] No tracking currently running")
                    return True
                
                success = self._stop_current_tracking()
                if success:
                    self.current_mode = TrackingMode.STOPPED
                    logger.info("[UNIFIED_CONTROLLER] Tracking stopped")
                
                return success
                
            except Exception as e:
                logger.error(f"[UNIFIED_CONTROLLER] Error stopping tracking: {e}")
                return False
    
    def switch_mode(self, mode: TrackingMode) -> bool:
        """
        Switch to a different tracking mode.
        This is equivalent to calling start_tracking().
        
        Args:
            mode: New tracking mode
            
        Returns:
            True if successful, False otherwise
        """
        return self.start_tracking(mode)
    
    def get_current_mode(self) -> TrackingMode:
        """
        Get the current tracking mode.
        
        Returns:
            Current tracking mode
        """
        with self._lock:
            return self.current_mode
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive tracking status.
        
        Returns:
            Status dictionary with current mode and tracker details
        """
        with self._lock:
            status = {
                "current_mode": self.current_mode.value,
                "is_running": self.current_mode != TrackingMode.STOPPED,
                "vehicle_tracker": {
                    "is_running": self.vehicle_tracker.is_running,
                    "status": self.vehicle_tracker.get_status()
                },
                "people_tracker": {
                    "is_running": self.people_tracker.is_running,
                    "status": self.people_tracker.get_status()
                }
            }
            
            return status
    
    def _stop_current_tracking(self) -> bool:
        """
        Stop the currently running tracker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.current_mode == TrackingMode.VEHICLES:
                return self.vehicle_tracker.stop_tracking()
            elif self.current_mode == TrackingMode.PEOPLE:
                return self.people_tracker.stop_tracking()
            else:
                return True  # Already stopped
                
        except Exception as e:
            logger.error(f"[UNIFIED_CONTROLLER] Error stopping current tracking: {e}")
            return False
    
    def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all trackers.
        
        Returns:
            Dictionary with initialization results
        """
        results = {}
        
        try:
            results["vehicle_tracker"] = self.vehicle_tracker.initialize()
            results["people_tracker"] = self.people_tracker.initialize()
            
            logger.info(f"[UNIFIED_CONTROLLER] Initialization results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"[UNIFIED_CONTROLLER] Error initializing trackers: {e}")
            return {"vehicle_tracker": False, "people_tracker": False}
    
    def cleanup_all(self):
        """Cleanup all trackers"""
        try:
            self.stop_tracking()
            
            self.vehicle_tracker.cleanup()
            self.people_tracker.cleanup()
            
            logger.info("[UNIFIED_CONTROLLER] All trackers cleaned up")
            
        except Exception as e:
            logger.error(f"[UNIFIED_CONTROLLER] Error during cleanup: {e}")


# Global controller instance
_controller_instance: Optional[UnifiedTrackingController] = None
_controller_lock = threading.Lock()


def get_tracking_controller() -> UnifiedTrackingController:
    """Get the global unified tracking controller instance"""
    global _controller_instance
    
    with _controller_lock:
        if _controller_instance is None:
            _controller_instance = UnifiedTrackingController()
        return _controller_instance


def cleanup_tracking_controller():
    """Cleanup global tracking controller instance"""
    global _controller_instance
    
    with _controller_lock:
        if _controller_instance:
            _controller_instance.cleanup_all()
            _controller_instance = None


def main():
    """Test the unified tracking controller"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[UNIFIED_CONTROLLER] Testing UnifiedTrackingController...")
    
    # Initialize controller
    controller = UnifiedTrackingController()
    
    # Test initialization
    init_results = controller.initialize_all()
    logger.info(f"[UNIFIED_CONTROLLER] Initialization results: {init_results}")
    
    # Test status
    status = controller.get_status()
    logger.info(f"[UNIFIED_CONTROLLER] Initial status: {status}")
    
    # Test mode switching
    logger.info("[UNIFIED_CONTROLLER] Testing mode switching...")
    
    # Start vehicle tracking
    if controller.start_tracking(TrackingMode.VEHICLES):
        logger.info("[UNIFIED_CONTROLLER] Vehicle tracking started")
        status = controller.get_status()
        logger.info(f"[UNIFIED_CONTROLLER] Status after starting vehicles: {status['current_mode']}")
        
        # Switch to people tracking
        if controller.switch_mode(TrackingMode.PEOPLE):
            logger.info("[UNIFIED_CONTROLLER] Switched to people tracking")
            status = controller.get_status()
            logger.info(f"[UNIFIED_CONTROLLER] Status after switching to people: {status['current_mode']}")
            
            # Stop tracking
            if controller.stop_tracking():
                logger.info("[UNIFIED_CONTROLLER] Tracking stopped")
                status = controller.get_status()
                logger.info(f"[UNIFIED_CONTROLLER] Final status: {status['current_mode']}")
            else:
                logger.error("[UNIFIED_CONTROLLER] Failed to stop tracking")
        else:
            logger.error("[UNIFIED_CONTROLLER] Failed to switch to people tracking")
    else:
        logger.error("[UNIFIED_CONTROLLER] Failed to start vehicle tracking")
    
    # Cleanup
    controller.cleanup_all()
    logger.info("[UNIFIED_CONTROLLER] Test completed")


if __name__ == "__main__":
    main()
