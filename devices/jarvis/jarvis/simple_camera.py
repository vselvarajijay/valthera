#!/usr/bin/env python3

import logging
import time
import threading
import numpy as np
import cv2
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimpleFrame:
    """Container for simple camera frame data"""
    color_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0

class SimpleCamera:
    """Simple OpenCV-based camera interface that works without RealSense"""
    
    def __init__(self, 
                 width: int = 640, 
                 height: int = 480, 
                 fps: int = 30,
                 camera_index: int = 0):
        
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        
        # OpenCV camera
        self.cap = None
        
        # Processing state
        self.is_running = False
        self.thread = None
        self.frame_count = 0
        
        # Latest frame data
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Callbacks
        self.on_new_frame = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize OpenCV camera"""
        try:
            # Try to find a working camera
            camera_index = self._find_camera()
            if camera_index is None:
                logger.error("[SIMPLE_CAMERA] No camera device found")
                return False
            
            self.camera_index = camera_index
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"[SIMPLE_CAMERA] Failed to open camera at index {camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"[SIMPLE_CAMERA] Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA] Error initializing camera: {e}")
            return False
    
    def _find_camera(self):
        """Find available camera device"""
        try:
            # Try different camera indices
            for i in range(6):  # Try indices 0-5
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        logger.info(f"[SIMPLE_CAMERA] Found camera at index {i}")
                        return i
                    cap.release()
            return None
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA] Error finding camera: {e}")
            return None
    
    def _capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            # Create simple frame container
            simple_frame = SimpleFrame(
                color_frame=frame,
                depth_frame=None,  # No depth data with simple camera
                timestamp=time.time(),
                frame_id=self.frame_count
            )
            
            return simple_frame
            
        except Exception as e:
            logger.error(f"[SIMPLE_CAMERA] Error capturing frame: {e}")
            return None
    
    def _capture_loop(self):
        """Main frame capture loop"""
        logger.info("[SIMPLE_CAMERA] Starting frame capture loop...")
        
        while self.is_running:
            try:
                frame = self._capture_frame()
                if frame is not None:
                    # Update latest frame
                    with self.frame_lock:
                        self.latest_frame = frame
                    
                    # Trigger callback
                    if self.on_new_frame:
                        self.on_new_frame(frame)
                    
                    self.frame_count += 1
                    
                    # Log every 30 frames (about every second at 30fps)
                    if self.frame_count % 30 == 0:
                        logger.info(f"[SIMPLE_CAMERA] Captured frame {self.frame_count}")
                else:
                    logger.warning(f"[SIMPLE_CAMERA] Failed to capture frame")
                
                time.sleep(1.0 / self.fps)  # Control capture rate
                
            except Exception as e:
                logger.error(f"[SIMPLE_CAMERA] Error in capture loop: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start frame capture"""
        if self.is_running:
            logger.warning("[SIMPLE_CAMERA] Camera already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            logger.error("[SIMPLE_CAMERA] Cannot start - camera not initialized")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("[SIMPLE_CAMERA] Camera started")
    
    def stop(self):
        """Stop frame capture"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("[SIMPLE_CAMERA] Camera stopped")
    
    def get_latest_frame(self) -> Optional[SimpleFrame]:
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.latest_frame
    
    def capture_single_frame(self) -> Optional[SimpleFrame]:
        """Capture a single frame (blocking)"""
        return self._capture_frame()
    
    def set_frame_callback(self, callback):
        """Set callback for new frame events"""
        self.on_new_frame = callback
    
    def get_camera_info(self) -> Dict[str, any]:
        """Get camera information"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'is_running': self.is_running,
            'camera_index': self.camera_index,
            'opencv_available': True
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        logger.info("[SIMPLE_CAMERA] Camera cleaned up")


def main():
    """Test the simple camera interface"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[SIMPLE_CAMERA] Testing SimpleCamera interface...")
    
    # Initialize simple camera
    simple_cam = SimpleCamera(width=640, height=480, fps=30)
    
    try:
        # Start capturing frames
        simple_cam.start()
        logger.info("[SIMPLE_CAMERA] Camera started, capturing frames...")
        
        # Test for 10 seconds
        for i in range(10):
            time.sleep(1)
            
            # Get latest frame
            frame = simple_cam.get_latest_frame()
            if frame:
                logger.info(f"[SIMPLE_CAMERA] Frame {frame.frame_id}: "
                           f"Color {frame.color_frame.shape}, "
                           f"Timestamp {frame.timestamp:.3f}")
            else:
                logger.warning("[SIMPLE_CAMERA] No frame available")
        
        # Test single frame capture
        logger.info("[SIMPLE_CAMERA] Testing single frame capture...")
        single_frame = simple_cam.capture_single_frame()
        if single_frame:
            logger.info(f"[SIMPLE_CAMERA] Single frame captured: {single_frame.frame_id}")
        
        # Get camera info
        info = simple_cam.get_camera_info()
        logger.info(f"[SIMPLE_CAMERA] Camera info: {info}")
        
    except KeyboardInterrupt:
        logger.info("[SIMPLE_CAMERA] Stopping camera...")
    finally:
        simple_cam.cleanup()
        logger.info("[SIMPLE_CAMERA] Camera test completed")


if __name__ == "__main__":
    main()
