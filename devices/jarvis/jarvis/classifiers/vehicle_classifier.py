#!/usr/bin/env python3

"""
Vehicle classifier for detection and re-identification.

This classifier extends BaseClassifier to detect vehicles using YOLOv8,
extract crops, generate ReID embeddings, and save video clips.
"""

import logging
import time
import collections
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

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
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from ..classifiers.registry import BaseClassifier, ModelConfig
from ..models.base import UnifiedDetection
from ..reid.osnet_extractor import get_embedding
from ..video.clip_manager import save_clip, CROPS_DIR, CLIPS_DIR
from ..db.vehicle_db import find_match, add_vehicle, update_vehicle

logger = logging.getLogger(__name__)

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}

# Frame buffer for video clips (3 seconds at 30fps)
FRAME_BUFFER_SIZE = 90


class VehicleClassifier(BaseClassifier):
    """YOLO-based vehicle detector with ReID integration"""
    
    def __init__(self, 
                 name: str = "vehicle",
                 config: Optional[ModelConfig] = None,
                 vehicle_classes: Optional[List[int]] = None):
        """
        Initialize the vehicle classifier.
        
        Args:
            name: Classifier name
            config: Model configuration
            vehicle_classes: List of COCO class IDs to detect (default: all vehicles)
        """
        if config is None:
            config = ModelConfig(
                name="vehicle",
                path="yolov8n.pt",
                model_type="yolo",
                classes=list(VEHICLE_CLASSES.keys()),
                confidence_threshold=0.5
            )
        
        super().__init__(name, config)
        self.vehicle_classes = vehicle_classes or list(VEHICLE_CLASSES.keys())
        self.stats.model_version = config.version or "8.0"
        
        # Frame buffer for video clips
        self.frame_buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)
        
        logger.info(f"[CLASSIFIER] Vehicle classifier initialized for classes: {self.vehicle_classes}")
    
    def _load_model(self) -> Any:
        """Load the YOLO model"""
        try:
            if YOLO_AVAILABLE:
                model = YOLO(self.config.path)
                logger.info(f"[CLASSIFIER] YOLOv8n model loaded: {self.config.path}")
                return model
            else:
                logger.error("[CLASSIFIER] YOLO not available - vehicle detection disabled")
                return None
        except Exception as e:
            logger.error(f"[CLASSIFIER] Error loading YOLO model: {e}")
            return None
    
    def add_frame_to_buffer(self, frame: np.ndarray):
        """Add frame to buffer for video clip generation"""
        if NUMPY_AVAILABLE and frame is not None:
            self.frame_buffer.append(frame.copy())
    
    def detect(self, frame: np.ndarray) -> List[UnifiedDetection]:
        """
        Detect vehicles in the given frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of UnifiedDetection objects with bounding boxes and embeddings
        """
        if not self.is_initialized or not YOLO_AVAILABLE or not self.model:
            return []
        
        if not NUMPY_AVAILABLE or frame is None:
            return []
        
        start_time = time.time()
        
        try:
            # Add frame to buffer
            self.add_frame_to_buffer(frame)
            
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Check if detected object is a vehicle
                        class_id = int(box.cls[0])
                        if class_id in self.vehicle_classes:
                            confidence = float(box.conf[0])
                            
                            # Check confidence threshold
                            if confidence >= self.config.confidence_threshold:
                                # Extract bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                
                                # Extract crop
                                crop = frame[y1:y2, x1:x2]
                                
                                # Generate ReID embedding
                                embedding = get_embedding(crop)
                                
                                # Save thumbnail and clip
                                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                                thumbnail_path = f"{CROPS_DIR}/{timestamp}_{len(detections)}.jpg"
                                clip_path = f"{CLIPS_DIR}/{timestamp}_{len(detections)}.mp4"
                                
                                # Save thumbnail
                                cv2.imwrite(thumbnail_path, crop)
                                
                                # Save video clip from buffer
                                if len(self.frame_buffer) > 0:
                                    save_clip(list(self.frame_buffer), clip_path)
                                
                                # Try to match with existing vehicles
                                vehicle_id = None
                                if embedding is not None:
                                    vehicle_id = find_match(embedding)
                                    
                                    if vehicle_id:
                                        # Update existing vehicle
                                        update_vehicle(vehicle_id, thumbnail_path, clip_path)
                                        logger.debug(f"[CLASSIFIER] Updated vehicle {vehicle_id}")
                                    else:
                                        # Add new vehicle
                                        vehicle_id = add_vehicle(embedding, thumbnail_path, clip_path)
                                        logger.debug(f"[CLASSIFIER] Added new vehicle {vehicle_id}")
                                
                                # Create detection object
                                detection = UnifiedDetection(
                                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                                    confidence=confidence,
                                    class_id=class_id,
                                    class_name=VEHICLE_CLASSES.get(class_id, "vehicle"),
                                    classifier_type=self.name,
                                    depth_mm=None,  # Will be filled by pipeline
                                    position_3d=None,  # Will be filled by pipeline
                                    attributes={
                                        'embedding': embedding.tolist() if embedding is not None else None,
                                        'vehicle_id': vehicle_id,
                                        'thumbnail_path': thumbnail_path,
                                        'clip_path': clip_path
                                    },
                                    processing_time_ms=None,  # Will be calculated
                                    model_version=self.stats.model_version
                                )
                                detections.append(detection)
            
            # Update performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.stats.update_stats(len(detections), processing_time)
            
            # Update processing time in detections
            for detection in detections:
                detection.processing_time_ms = processing_time / len(detections) if detections else processing_time
            
            return detections
            
        except Exception as e:
            logger.error(f"[CLASSIFIER] Error detecting vehicles: {e}")
            return []
    
    def annotate_frame(self, frame: np.ndarray, detections: List[UnifiedDetection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input image as numpy array
            detections: List of UnifiedDetection objects
            
        Returns:
            Annotated frame with bounding boxes drawn
        """
        if not CV2_AVAILABLE or frame is None:
            return frame
        
        annotated_frame = frame.copy()
        
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Get vehicle ID from attributes
                vehicle_id = detection.attributes.get('vehicle_id') if detection.attributes else None
                
                # Draw bounding box (different colors for new vs returning vehicles)
                color = (0, 255, 0) if vehicle_id else (0, 0, 255)  # Green for returning, Red for new
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence and vehicle ID
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                if vehicle_id:
                    label += f" (ID: {vehicle_id})"
                else:
                    label += " (NEW)"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), 
                             color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, 
                           (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[CLASSIFIER] Error annotating frame: {e}")
            return frame
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.config.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"[CLASSIFIER] Confidence threshold set to {self.config.confidence_threshold}")
    
    def set_vehicle_classes(self, classes: List[int]):
        """Update vehicle classes to detect"""
        self.vehicle_classes = classes
        logger.info(f"[CLASSIFIER] Vehicle classes set to: {classes}")
    
    def get_frame_buffer_size(self) -> int:
        """Get current frame buffer size"""
        return len(self.frame_buffer)
    
    def clear_frame_buffer(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
        logger.debug("[CLASSIFIER] Frame buffer cleared")


def main():
    """Test the vehicle classifier"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[CLASSIFIER] Testing VehicleClassifier...")
    
    # Initialize classifier
    classifier = VehicleClassifier()
    
    if not classifier.initialize():
        logger.error("[CLASSIFIER] Failed to initialize classifier")
        return
    
    # Test with webcam if available
    if not CV2_AVAILABLE:
        logger.error("[CLASSIFIER] OpenCV not available for camera test")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("[CLASSIFIER] No camera available for testing")
        return
    
    logger.info("[CLASSIFIER] Starting detection test... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[CLASSIFIER] Failed to read frame")
                continue
            
            # Detect vehicles
            detections = classifier.detect(frame)
            
            # Annotate frame
            annotated_frame = classifier.annotate_frame(frame, detections)
            
            # Display frame
            cv2.imshow('Vehicle Detection Test', annotated_frame)
            
            # Log detections
            if detections:
                logger.info(f"[CLASSIFIER] Detected {len(detections)} vehicle(s)")
                for det in detections:
                    vehicle_id = det.attributes.get('vehicle_id') if det.attributes else None
                    logger.info(f"[CLASSIFIER] - {det.class_name}: {det.confidence:.2f}, ID: {vehicle_id}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("[CLASSIFIER] Stopping test...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        classifier.cleanup()
        logger.info("[CLASSIFIER] Test completed")


if __name__ == "__main__":
    main()
