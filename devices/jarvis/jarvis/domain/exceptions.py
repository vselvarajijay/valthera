#!/usr/bin/env python3

"""
Domain-specific exceptions for Jarvis smart CV pipeline.

These exceptions represent business logic errors and are used
throughout the domain layer to maintain clean error handling.
"""


class JarvisError(Exception):
    """Base exception for Jarvis domain errors."""
    pass


class FrameNotFoundError(JarvisError):
    """Raised when a requested frame cannot be found."""
    
    def __init__(self, frame_id: str):
        self.frame_id = frame_id
        super().__init__(f"Frame {frame_id} not found")


class CameraUnavailableError(JarvisError):
    """Raised when camera hardware is not available or accessible."""
    
    def __init__(self, device_id: str = None):
        self.device_id = device_id
        message = f"Camera not available"
        if device_id:
            message += f" (device: {device_id})"
        super().__init__(message)


class ModelLoadError(JarvisError):
    """Raised when AI model fails to load or initialize."""
    
    def __init__(self, model_path: str, reason: str = None):
        self.model_path = model_path
        self.reason = reason
        message = f"Failed to load model: {model_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class InvalidConfigurationError(JarvisError):
    """Raised when configuration values are invalid."""
    
    def __init__(self, config_key: str, value, reason: str = None):
        self.config_key = config_key
        self.value = value
        self.reason = reason
        message = f"Invalid configuration for {config_key}: {value}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class DetectionError(JarvisError):
    """Raised when object detection fails."""
    
    def __init__(self, classifier_type: str, reason: str = None):
        self.classifier_type = classifier_type
        self.reason = reason
        message = f"Detection failed for {classifier_type}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class DepthCalculationError(JarvisError):
    """Raised when depth calculation fails."""
    
    def __init__(self, reason: str = None):
        self.reason = reason
        message = "Depth calculation failed"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class TrackingError(JarvisError):
    """Raised when object tracking fails."""
    
    def __init__(self, reason: str = None):
        self.reason = reason
        message = "Object tracking failed"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
