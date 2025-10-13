#!/usr/bin/env python3

"""
Domain layer for Jarvis smart CV pipeline.

This package contains the core business logic, entities, value objects,
and domain services interfaces. It has no external dependencies and
represents the pure business domain.
"""

from .entities.frame import Frame, FrameId
from .entities.detection import Detection, BoundingBox, Position3D
from .entities.camera import CameraConfig, CameraStatus
from .value_objects import Confidence, Timestamp, Resolution
from .exceptions import (
    JarvisError,
    FrameNotFoundError,
    CameraUnavailableError,
    ModelLoadError,
    InvalidConfigurationError
)

__all__ = [
    # Entities
    'Frame',
    'FrameId', 
    'Detection',
    'BoundingBox',
    'Position3D',
    'CameraConfig',
    'CameraStatus',
    
    # Value Objects
    'Confidence',
    'Timestamp',
    'Resolution',
    
    # Exceptions
    'JarvisError',
    'FrameNotFoundError',
    'CameraUnavailableError',
    'ModelLoadError',
    'InvalidConfigurationError'
]
