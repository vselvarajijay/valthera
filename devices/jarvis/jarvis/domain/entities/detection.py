#!/usr/bin/env python3

"""
Detection entity for Jarvis smart CV pipeline.

Represents detected objects with bounding boxes, confidence scores,
and optional 3D positioning information.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from ..value_objects import Confidence, DepthValue, ProcessingTime


class DetectionClass(Enum):
    """Supported detection classes."""
    PERSON = "person"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    FACE = "face"
    OBJECT = "object"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Represents a bounding box with integer coordinates."""
    
    x1: int
    y1: int
    x2: int
    y2: int
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x1 >= self.x2:
            raise ValueError(f"x1 ({self.x1}) must be less than x2 ({self.x2})")
        if self.y1 >= self.y2:
            raise ValueError(f"y1 ({self.y1}) must be less than y2 ({self.y2})")
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Coordinates must be non-negative")
    
    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def center_x(self) -> int:
        """Get center X coordinate."""
        return (self.x1 + self.x2) // 2
    
    @property
    def center_y(self) -> int:
        """Get center Y coordinate."""
        return (self.y1 + self.y2) // 2
    
    def center(self) -> tuple[int, int]:
        """Get center coordinates as tuple."""
        return (self.center_x, self.center_y)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def intersection_area(self, other: 'BoundingBox') -> int:
        """Calculate intersection area with another bounding box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        return (x2 - x1) * (y2 - y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def scale(self, factor: float) -> 'BoundingBox':
        """Scale bounding box by factor."""
        return BoundingBox(
            x1=int(self.x1 * factor),
            y1=int(self.y1 * factor),
            x2=int(self.x2 * factor),
            y2=int(self.y2 * factor)
        )
    
    def to_list(self) -> List[int]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[int]) -> 'BoundingBox':
        """Create from list format [x1, y1, x2, y2]."""
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        return cls(coords[0], coords[1], coords[2], coords[3])


@dataclass(frozen=True)
class Position3D:
    """Represents a 3D position in meters."""
    
    x: float
    y: float
    z: float
    
    def __post_init__(self):
        """Validate 3D position."""
        if self.z <= 0:
            raise ValueError(f"Z coordinate must be positive (depth), got {self.z}")
    
    @property
    def distance_from_origin(self) -> float:
        """Calculate distance from origin."""
        import math
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate distance to another position."""
        import math
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position3D':
        """Create from dictionary."""
        return cls(data["x"], data["y"], data["z"])


@dataclass(frozen=True)
class Detection:
    """Represents a detected object with all associated information."""
    
    # Core detection data
    class_name: str
    class_id: int
    confidence: Confidence
    bounding_box: BoundingBox
    
    # Optional 3D information
    depth: Optional[DepthValue] = None
    position_3d: Optional[Position3D] = None
    
    # Detection metadata
    classifier_type: str = "unknown"
    processing_time: Optional[ProcessingTime] = None
    model_version: Optional[str] = None
    
    # Extended attributes (pose, emotion, etc.)
    attributes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate detection data."""
        if not self.class_name:
            raise ValueError("Class name cannot be empty")
        
        if self.class_id < 0:
            raise ValueError("Class ID must be non-negative")
    
    def has_3d_info(self) -> bool:
        """Check if detection has 3D information."""
        return self.depth is not None or self.position_3d is not None
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if detection has high confidence."""
        return self.confidence.is_high(threshold)
    
    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if detection has low confidence."""
        return self.confidence.is_low(threshold)
    
    def is_close(self, max_distance_mm: float = 2000.0) -> bool:
        """Check if detection is close (within distance threshold)."""
        if not self.depth:
            return False
        return self.depth.value_mm <= max_distance_mm
    
    def is_far(self, min_distance_mm: float = 5000.0) -> bool:
        """Check if detection is far (beyond distance threshold)."""
        if not self.depth:
            return False
        return self.depth.value_mm >= min_distance_mm
    
    def overlaps_with(self, other: 'Detection', threshold: float = 0.5) -> bool:
        """Check if detection overlaps with another detection."""
        return self.bounding_box.iou(other.bounding_box) >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for serialization."""
        result = {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": float(self.confidence.value),
            "bounding_box": self.bounding_box.to_list(),
            "classifier_type": self.classifier_type,
            "has_3d_info": self.has_3d_info()
        }
        
        if self.depth:
            result["depth_mm"] = self.depth.value_mm
        
        if self.position_3d:
            result["position_3d"] = self.position_3d.to_dict()
        
        if self.processing_time:
            result["processing_time_ms"] = self.processing_time.value_ms
        
        if self.model_version:
            result["model_version"] = self.model_version
        
        if self.attributes:
            result["attributes"] = self.attributes
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """Create detection from dictionary."""
        return cls(
            class_name=data["class_name"],
            class_id=data["class_id"],
            confidence=Confidence(data["confidence"]),
            bounding_box=BoundingBox.from_list(data["bounding_box"]),
            classifier_type=data.get("classifier_type", "unknown"),
            depth=DepthValue(data["depth_mm"]) if "depth_mm" in data else None,
            position_3d=Position3D.from_dict(data["position_3d"]) if "position_3d" in data else None,
            processing_time=ProcessingTime(data["processing_time_ms"]) if "processing_time_ms" in data else None,
            model_version=data.get("model_version"),
            attributes=data.get("attributes")
        )
