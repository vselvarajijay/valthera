#!/usr/bin/env python3

"""
Detection service interface for Jarvis smart CV pipeline.

Defines the contract for object detection operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..entities.frame import Frame
from ..entities.detection import Detection


class IDetectionService(ABC):
    """Interface for object detection operations."""
    
    @abstractmethod
    async def detect_objects(self, frame: Frame, class_names: List[str] = None) -> List[Detection]:
        """Detect objects in frame with optional class filtering."""
        pass
    
    @abstractmethod
    async def detect_persons(self, frame: Frame) -> List[Detection]:
        """Detect persons in frame."""
        pass
    
    @abstractmethod
    async def detect_vehicles(self, frame: Frame) -> List[Detection]:
        """Detect vehicles in frame."""
        pass
    
    @abstractmethod
    async def detect_faces(self, frame: Frame) -> List[Detection]:
        """Detect faces in frame."""
        pass
    
    @abstractmethod
    async def get_supported_classes(self) -> List[str]:
        """Get list of supported detection classes."""
        pass
    
    @abstractmethod
    async def is_initialized(self) -> bool:
        """Check if detection service is initialized."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        pass
