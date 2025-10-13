#!/usr/bin/env python3

"""
Enhanced person classifier with Re-ID integration.

This classifier extends BaseClassifier to detect people using YOLOv8,
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
from ..reid.person_extractor import get_person_embedding
from ..video.clip_manager import save_clip
from ..db.unified_tracker_db import find_match, add_object, update_object, get_storage_dirs

logger = logging.getLogger(__name__)

# Person class ID in COCO dataset
PERSON_CLASS_ID = 0

# Frame buffer for video clips (3 seconds at 30fps)
FRAME_BUFFER_SIZE = 90


class PersonClassifier(BaseClassifier):
    """YOLO-based person detector with ReID integration"""
    
    def __init__(self, 
                 name: str = "person",
                 config: Optional[ModelConfig] = None,
                 person_class_id: int = 0):
        """
        Initialize the person classifier with Re-ID capabilities.
        
        Args:
            name: Classifier name
            config: Model configuration
            person_class_id: COCO class ID for person (default: 0)
        """
        if config is None:
            config = ModelConfig(
                name="person",
                path="yolov8n.pt",
                model_type="yolo",
                classes=[person_class_id],
                confidence_threshold=0.5
            )
        
        super().__init__(name, config)
        self.person_class_id = person_class_id
        self.stats.model_version = config.version or "8.0"
        
        # Frame buffer for video clips
        self.frame_buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)
        
        logger.info(f"[CLASSIFIER] Person classifier with Re-ID initialized")
    
    def _load_model(self) -> Any:
        """Load the YOLO model"""
        try:
            if YOLO_AVAILABLE:
                # Load YOLOv8n model (nano - fastest for Jetson)
                model = YOLO(self.config.path)
                logger.info(f"[CLASSIFIER] YOLOv8n model loaded: {self.config.path}")
                return model
            else:
                logger.error("[CLASSIFIER] YOLO not available - person detection disabled")
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
        Detect people in the given frame with Re-ID.
        
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
                        # Check if detected object is a person
                        class_id = int(box.cls[0])
                        if class_id == self.person_class_id:
                            confidence = float(box.conf[0])
                            
                            # Check confidence threshold
                            if confidence >= self.config.confidence_threshold:
                                # Extract bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                
                                # Extract crop
                                crop = frame[y1:y2, x1:x2]
                                
                                # Generate ReID embedding
                                embedding = None
                                try:
                                    embedding = get_person_embedding(crop)
                                except Exception as e:
                                    logger.debug(f"[CLASSIFIER] ReID not available, continuing without embedding: {e}")
                                    embedding = None
                                
                                # Get storage directories for people
                                storage_dirs = get_storage_dirs('person')
                                
                                # Save thumbnail and clip
                                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                                thumbnail_path = f"{storage_dirs['crops_dir']}/{timestamp}_{len(detections)}.jpg"
                                clip_path = f"{storage_dirs['clips_dir']}/{timestamp}_{len(detections)}.mp4"
                                
                                # Save thumbnail
                                cv2.imwrite(thumbnail_path, crop)
                                
                                # Save video clip from buffer
                                if len(self.frame_buffer) > 0:
                                    save_clip(list(self.frame_buffer), clip_path)
                                
                                # Try to match with existing people
                                person_id = None
                                if embedding is not None:
                                    person_id = find_match(embedding, 'person', threshold=0.70)
                                    
                                    if person_id:
                                        # Update existing person
                                        update_object(person_id, 'person', thumbnail_path, clip_path)
                                        logger.debug(f"[CLASSIFIER] Updated person {person_id}")
                                    else:
                                        # Add new person
                                        person_id = add_object(embedding, 'person', thumbnail_path, clip_path)
                                        logger.debug(f"[CLASSIFIER] Added new person {person_id}")
                                else:
                                    # No embedding available, create new person without matching
                                    person_id = add_object(None, 'person', thumbnail_path, clip_path)
                                    logger.debug(f"[CLASSIFIER] Added new person {person_id} without embedding")
                                
                                # Create detection object
                                detection = UnifiedDetection(
                                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                                    confidence=confidence,
                                    class_id=class_id,
                                    class_name="person",
                                    classifier_type=self.name,
                                    depth_mm=None,  # Will be filled by pipeline
                                    position_3d=None,  # Will be filled by pipeline
                                    attributes={
                                        'embedding': embedding.tolist() if embedding is not None else None,
                                        'person_id': person_id,
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
            logger.error(f"[CLASSIFIER] Error detecting people: {e}")
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
                
                # Get person ID from attributes
                person_id = detection.attributes.get('person_id') if detection.attributes else None
                
                # Draw bounding box (different colors for new vs returning people)
                color = (0, 255, 0) if person_id else (0, 0, 255)  # Green for returning, Red for new
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence and person ID
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                if person_id:
                    label += f" (ID: {person_id})"
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
    
    def get_frame_buffer_size(self) -> int:
        """Get current frame buffer size"""
        return len(self.frame_buffer)
    
    def clear_frame_buffer(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
        logger.debug("[CLASSIFIER] Frame buffer cleared")


def main():
    """Test the person classifier"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[CLASSIFIER] Testing PersonClassifier...")
    
    # Initialize classifier
    classifier = PersonClassifier()
    
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
            
            # Detect people
            detections = classifier.detect(frame)
            
            # Annotate frame
            annotated_frame = classifier.annotate_frame(frame, detections)
            
            # Display frame
            cv2.imshow('Person Detection Test', annotated_frame)
            
            # Log detections
            if detections:
                logger.info(f"[CLASSIFIER] Detected {len(detections)} person(s)")
            
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
