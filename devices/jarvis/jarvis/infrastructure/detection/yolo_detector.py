#!/usr/bin/env python3

"""
Unified YOLO detector for Jarvis smart CV pipeline.

Consolidates person, object, and vehicle detection into a single
YOLO-based detection service.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import asyncio

from ...domain.services.detection_service import IDetectionService
from ...domain.entities.frame import Frame
from ...domain.entities.detection import Detection, BoundingBox, DetectionClass
from ...domain.value_objects import Confidence, ProcessingTime
from ...domain.exceptions import ModelLoadError, DetectionError
from ...classifiers.person_classifier import PersonClassifier

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

try:
    import numpy as np
    import cv2
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class YOLODetector(IDetectionService):
    """Unified YOLO-based detection service."""
    
    # COCO class mapping
    COCO_CLASSES = {
        0: "person",
        1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
        9: "traffic_light", 11: "stop_sign", 13: "bench", 14: "bird",
        15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 27: "handbag", 28: "tie", 31: "suitcase", 32: "frisbee",
        33: "skis", 34: "snowboard", 35: "sports_ball", 36: "kite", 37: "baseball_bat",
        38: "baseball_glove", 39: "skateboard", 40: "surfboard", 41: "tennis_racket",
        42: "bottle", 43: "wine_glass", 44: "cup", 45: "fork", 46: "knife",
        47: "spoon", 48: "bowl", 49: "banana", 50: "apple", 51: "sandwich",
        52: "orange", 53: "broccoli", 54: "carrot", 55: "hot_dog", 56: "pizza",
        57: "donut", 58: "cake", 59: "chair", 60: "couch", 61: "potted_plant",
        62: "bed", 63: "dining_table", 64: "toilet", 65: "tv", 66: "laptop",
        67: "mouse", 68: "remote", 69: "keyboard", 70: "cell_phone", 71: "microwave",
        72: "oven", 73: "toaster", 74: "sink", 75: "refrigerator", 76: "book",
        77: "clock", 78: "vase", 79: "scissors", 80: "teddy_bear", 81: "hair_drier",
        82: "toothbrush"
    }
    
    # Vehicle classes
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.person_classifier: Optional[PersonClassifier] = None
        self._is_initialized = False
        self._model_info: Dict[str, Any] = {}
        
        logger.info(f"[YOLO_DETECTOR] Initializing with model: {model_path}")
    
    async def detect_objects(self, frame: Frame, class_names: List[str] = None) -> List[Detection]:
        """Detect objects in frame with optional class filtering."""
        if not self._is_initialized:
            raise DetectionError("detector", "Detector not initialized")
        
        try:
            start_time = time.time()
            
            # Convert frame to OpenCV format
            image = self._frame_to_cv2(frame)
            
            # Run detection
            results = self.model(image, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection data
                        conf = float(box.conf[0])
                        if conf < self.confidence_threshold:
                            continue
                        
                        class_id = int(box.cls[0])
                        class_name = self.COCO_CLASSES.get(class_id, "unknown")
                        
                        # Apply class filtering if specified
                        if class_names and class_name not in class_names:
                            continue
                        
                        # Extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = BoundingBox(
                            x1=int(x1), y1=int(y1),
                            x2=int(x2), y2=int(y2)
                        )
                        
                        # Create detection
                        detection = Detection(
                            class_name=class_name,
                            class_id=class_id,
                            confidence=Confidence(conf),
                            bounding_box=bbox,
                            classifier_type="yolo",
                            processing_time=ProcessingTime.from_seconds(time.time() - start_time),
                            model_version=self._model_info.get("version", "unknown")
                        )
                        
                        detections.append(detection)
            
            logger.debug(f"[YOLO_DETECTOR] Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"[YOLO_DETECTOR] Detection failed: {e}")
            raise DetectionError("yolo", str(e))
    
    async def detect_persons(self, frame: Frame) -> List[Detection]:
        """Detect persons in frame using person classifier with ReID."""
        if not self._is_initialized:
            raise DetectionError("detector", "Detector not initialized")
        
        # Use person classifier if available (with ReID and database storage)
        if self.person_classifier:
            try:
                # Convert frame to numpy array for person classifier
                image = self._frame_to_cv2(frame)
                
                # Use person classifier for detection with ReID
                unified_detections = self.person_classifier.detect(image)
                
                # Convert UnifiedDetection to Detection objects
                detections = []
                for unified_detection in unified_detections:
                    # Create bounding box
                    x1, y1, x2, y2 = unified_detection.bbox
                    bbox = BoundingBox(
                        x1=int(x1), y1=int(y1),
                        x2=int(x2), y2=int(y2)
                    )
                    
                    # Create detection
                    detection = Detection(
                        class_name=unified_detection.class_name,
                        class_id=unified_detection.class_id,
                        confidence=Confidence(unified_detection.confidence),
                        bounding_box=bbox,
                        classifier_type=unified_detection.classifier_type,
                        processing_time=ProcessingTime.from_milliseconds(unified_detection.processing_time_ms or 0),
                        model_version=unified_detection.model_version
                    )
                    
                    detections.append(detection)
                
                logger.debug(f"[YOLO_DETECTOR] Person classifier detected {len(detections)} persons")
                return detections
                
            except Exception as e:
                logger.error(f"[YOLO_DETECTOR] Person classifier detection failed: {e}")
                # Fall back to basic detection
                logger.info("[YOLO_DETECTOR] Falling back to basic person detection")
        
        # Fall back to basic YOLO detection if person classifier is not available
        return await self.detect_objects(frame, class_names=["person"])
    
    async def detect_vehicles(self, frame: Frame) -> List[Detection]:
        """Detect vehicles in frame."""
        vehicle_detections = await self.detect_objects(frame, class_names=list(self.VEHICLE_CLASSES))
        
        # Filter to only vehicle classes
        filtered_detections = [
            d for d in vehicle_detections 
            if d.class_name in self.VEHICLE_CLASSES
        ]
        
        return filtered_detections
    
    async def detect_faces(self, frame: Frame) -> List[Detection]:
        """Detect faces in frame."""
        # YOLO doesn't have face detection, return empty list
        # This could be extended with a dedicated face detection model
        logger.warning("[YOLO_DETECTOR] Face detection not supported by YOLO")
        return []
    
    async def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        return list(self.COCO_CLASSES.values())
    
    async def is_initialized(self) -> bool:
        """Check if detection service is initialized."""
        return self._is_initialized
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return self._model_info.copy()
    
    async def initialize(self) -> bool:
        """Initialize the YOLO model and person classifier."""
        if not YOLO_AVAILABLE:
            raise ModelLoadError(self.model_path, "YOLO library not available")
        
        try:
            logger.info(f"[YOLO_DETECTOR] Loading YOLO model: {self.model_path}")
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Initialize person classifier for ReID functionality
            logger.info("[YOLO_DETECTOR] Initializing person classifier with ReID")
            self.person_classifier = PersonClassifier()
            if not self.person_classifier.initialize():
                logger.warning("[YOLO_DETECTOR] Failed to initialize person classifier, falling back to basic detection")
                self.person_classifier = None
            
            # Get model info
            self._model_info = {
                "model_path": self.model_path,
                "model_type": "yolo",
                "version": "8.0",  # YOLOv8
                "classes": len(self.COCO_CLASSES),
                "confidence_threshold": self.confidence_threshold,
                "person_classifier_enabled": self.person_classifier is not None
            }
            
            self._is_initialized = True
            logger.info("[YOLO_DETECTOR] Successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"[YOLO_DETECTOR] Initialization failed: {e}")
            self._is_initialized = False
            raise ModelLoadError(self.model_path, str(e))
    
    async def cleanup(self) -> None:
        """Cleanup detector resources."""
        try:
            # Cleanup person classifier
            if self.person_classifier:
                self.person_classifier.cleanup()
                self.person_classifier = None
            
            self.model = None
            self._is_initialized = False
            logger.info("[YOLO_DETECTOR] Cleaned up")
            
        except Exception as e:
            logger.error(f"[YOLO_DETECTOR] Error during cleanup: {e}")
    
    def _frame_to_cv2(self, frame: Frame) -> 'np.ndarray':
        """Convert domain Frame to OpenCV image format."""
        if not NUMPY_AVAILABLE:
            raise DetectionError("yolo", "NumPy not available")
        
        try:
            # Convert bytes to numpy array
            # Assuming RGB format (3 bytes per pixel)
            image_data = np.frombuffer(frame.color_data, dtype=np.uint8)
            image = image_data.reshape((frame.resolution.height, frame.resolution.width, 3))
            
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"[YOLO_DETECTOR] Error converting frame to OpenCV format: {e}")
            raise DetectionError("yolo", f"Frame conversion failed: {e}")
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detections."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        logger.info(f"[YOLO_DETECTOR] Confidence threshold set to {threshold}")
