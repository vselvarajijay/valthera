#!/usr/bin/env python3

"""
Detector registry for Jarvis smart CV pipeline.

Simplified registry for managing detection services with better
resource management than the original classifier registry.
"""

import logging
from typing import Dict, List, Optional, Type
from dataclasses import dataclass

from ...domain.services.detection_service import IDetectionService
from ...domain.exceptions import ModelLoadError

from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for a detector."""
    name: str
    detector_type: str
    model_path: str
    confidence_threshold: float = 0.5
    enabled: bool = True


class DetectorRegistry:
    """Registry for managing detection services."""
    
    def __init__(self):
        self._detectors: Dict[str, IDetectionService] = {}
        self._configs: Dict[str, DetectorConfig] = {}
        self._detector_types: Dict[str, Type[IDetectionService]] = {}
        
        # Register default detector types
        self._register_default_types()
        
        logger.info("[DETECTOR_REGISTRY] Initialized")
    
    def _register_default_types(self):
        """Register default detector types."""
        self._detector_types["yolo"] = YOLODetector
        logger.info("[DETECTOR_REGISTRY] Registered default detector types")
    
    def register_detector_type(self, detector_type: str, detector_class: Type[IDetectionService]):
        """Register a new detector type."""
        self._detector_types[detector_type] = detector_class
        logger.info(f"[DETECTOR_REGISTRY] Registered detector type: {detector_type}")
    
    async def create_detector(self, config: DetectorConfig) -> Optional[IDetectionService]:
        """Create a detector with given configuration."""
        if config.detector_type not in self._detector_types:
            logger.error(f"[DETECTOR_REGISTRY] Unknown detector type: {config.detector_type}")
            return None
        
        try:
            detector_class = self._detector_types[config.detector_type]
            
            # Create detector instance
            if config.detector_type == "yolo":
                detector = detector_class(
                    model_path=config.model_path,
                    confidence_threshold=config.confidence_threshold
                )
            else:
                # Generic creation for other detector types
                detector = detector_class()
            
            # Initialize detector
            success = await detector.initialize()
            if not success:
                logger.error(f"[DETECTOR_REGISTRY] Failed to initialize detector: {config.name}")
                return None
            
            # Store detector and config
            self._detectors[config.name] = detector
            self._configs[config.name] = config
            
            logger.info(f"[DETECTOR_REGISTRY] Created detector: {config.name}")
            return detector
            
        except Exception as e:
            logger.error(f"[DETECTOR_REGISTRY] Failed to create detector {config.name}: {e}")
            return None
    
    async def get_detector(self, name: str) -> Optional[IDetectionService]:
        """Get detector by name."""
        return self._detectors.get(name)
    
    async def get_enabled_detectors(self) -> List[IDetectionService]:
        """Get all enabled detectors."""
        enabled_detectors = []
        for name, config in self._configs.items():
            if config.enabled and name in self._detectors:
                enabled_detectors.append(self._detectors[name])
        return enabled_detectors
    
    async def enable_detector(self, name: str) -> bool:
        """Enable a detector."""
        if name in self._configs:
            self._configs[name].enabled = True
            logger.info(f"[DETECTOR_REGISTRY] Enabled detector: {name}")
            return True
        return False
    
    async def disable_detector(self, name: str) -> bool:
        """Disable a detector."""
        if name in self._configs:
            self._configs[name].enabled = False
            logger.info(f"[DETECTOR_REGISTRY] Disabled detector: {name}")
            return True
        return False
    
    async def remove_detector(self, name: str) -> bool:
        """Remove a detector."""
        if name in self._detectors:
            try:
                await self._detectors[name].cleanup()
                del self._detectors[name]
                del self._configs[name]
                logger.info(f"[DETECTOR_REGISTRY] Removed detector: {name}")
                return True
            except Exception as e:
                logger.error(f"[DETECTOR_REGISTRY] Error removing detector {name}: {e}")
        return False
    
    async def get_registry_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        return {
            "total_detectors": len(self._detectors),
            "enabled_detectors": len([c for c in self._configs.values() if c.enabled]),
            "detector_types": list(self._detector_types.keys()),
            "detectors": {
                name: {
                    "type": config.detector_type,
                    "enabled": config.enabled,
                    "model_path": config.model_path,
                    "confidence_threshold": config.confidence_threshold
                }
                for name, config in self._configs.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup all detectors."""
        for name, detector in self._detectors.items():
            try:
                await detector.cleanup()
            except Exception as e:
                logger.error(f"[DETECTOR_REGISTRY] Error cleaning up detector {name}: {e}")
        
        self._detectors.clear()
        self._configs.clear()
        logger.info("[DETECTOR_REGISTRY] Cleaned up all detectors")


# Global registry instance
_registry_instance: Optional[DetectorRegistry] = None


def get_registry() -> DetectorRegistry:
    """Get the global detector registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DetectorRegistry()
    return _registry_instance


async def cleanup_registry():
    """Cleanup the global registry."""
    global _registry_instance
    if _registry_instance:
        await _registry_instance.cleanup()
        _registry_instance = None
