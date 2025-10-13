#!/usr/bin/env python3

"""
Event bus interface for Jarvis smart CV pipeline.

Defines the contract for domain event publishing and subscription.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
from datetime import datetime
import uuid


class DomainEvent(ABC):
    """Base class for domain events."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.event_id = str(uuid.uuid4())


class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass


class IEventBus(ABC):
    """Interface for domain event publishing."""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: type, handler: EventHandler) -> None:
        """Subscribe to domain events."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        """Unsubscribe from domain events."""
        pass


# Common domain events
class FrameReceivedEvent(DomainEvent):
    """Event published when a new frame is received."""
    
    def __init__(self, frame_id: str, timestamp: datetime = None):
        super().__init__()
        self.frame_id = frame_id
        if timestamp:
            self.timestamp = timestamp


class FrameProcessedEvent(DomainEvent):
    """Event published when a frame is processed."""
    
    def __init__(self, frame_id: str, detection_count: int, processing_time_ms: float):
        super().__init__()
        self.frame_id = frame_id
        self.detection_count = detection_count
        self.processing_time_ms = processing_time_ms


class DetectionFoundEvent(DomainEvent):
    """Event published when detections are found."""
    
    def __init__(self, frame_id: str, detections: list, class_names: list):
        super().__init__()
        self.frame_id = frame_id
        self.detections = detections
        self.class_names = class_names


class CameraErrorEvent(DomainEvent):
    """Event published when camera errors occur."""
    
    def __init__(self, error_message: str, device_id: str = None):
        super().__init__()
        self.error_message = error_message
        self.device_id = device_id


class PipelineStartedEvent(DomainEvent):
    """Event published when pipeline starts."""
    pass


class PipelineStoppedEvent(DomainEvent):
    """Event published when pipeline stops."""
    pass
