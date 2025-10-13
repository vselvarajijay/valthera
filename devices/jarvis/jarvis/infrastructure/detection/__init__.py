#!/usr/bin/env python3

"""
Detection infrastructure package for Jarvis smart CV pipeline.

This package contains implementations of detection services that consolidate
the existing classifier implementations into a unified detection system.
"""

from .yolo_detector import YOLODetector
from .detector_registry import DetectorRegistry
from .model_manager import ModelManager

__all__ = [
    'YOLODetector',
    'DetectorRegistry',
    'ModelManager'
]
