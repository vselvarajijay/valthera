#!/usr/bin/env python3

"""
Value objects for Jarvis smart CV pipeline.

Value objects are immutable objects that represent concepts
in the domain without identity. They encapsulate validation
and behavior related to their values.
"""

from dataclasses import dataclass
from typing import Union
from datetime import datetime


@dataclass(frozen=True)
class Confidence:
    """Represents a confidence score between 0.0 and 1.0."""
    
    value: float
    
    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.value}")
    
    def __str__(self) -> str:
        return f"{self.value:.3f}"
    
    def is_high(self, threshold: float = 0.8) -> bool:
        """Check if confidence is above threshold."""
        return self.value >= threshold
    
    def is_low(self, threshold: float = 0.3) -> bool:
        """Check if confidence is below threshold."""
        return self.value <= threshold


@dataclass(frozen=True)
class Timestamp:
    """Represents a timestamp with validation."""
    
    value: float  # Unix timestamp
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Timestamp must be non-negative, got {self.value}")
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """Create timestamp for current time."""
        import time
        return cls(time.time())
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'Timestamp':
        """Create timestamp from datetime object."""
        return cls(dt.timestamp())
    
    def to_datetime(self) -> datetime:
        """Convert to datetime object."""
        return datetime.fromtimestamp(self.value)
    
    def age_seconds(self) -> float:
        """Get age in seconds from now."""
        import time
        return time.time() - self.value
    
    def __str__(self) -> str:
        return self.to_datetime().isoformat()


@dataclass(frozen=True)
class Resolution:
    """Represents image resolution with width and height."""
    
    width: int
    height: int
    
    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        return self.width / self.height
    
    @property
    def total_pixels(self) -> int:
        """Get total number of pixels."""
        return self.width * self.height
    
    def scale(self, factor: float) -> 'Resolution':
        """Scale resolution by factor."""
        return Resolution(
            width=int(self.width * factor),
            height=int(self.height * factor)
        )
    
    def fit_within(self, max_width: int, max_height: int) -> 'Resolution':
        """Scale to fit within maximum dimensions."""
        scale_w = max_width / self.width
        scale_h = max_height / self.height
        scale = min(scale_w, scale_h)
        return self.scale(scale)
    
    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class DepthValue:
    """Represents a depth measurement in millimeters."""
    
    value_mm: float
    
    def __post_init__(self):
        if self.value_mm < 0:
            raise ValueError(f"Depth must be non-negative, got {self.value_mm}")
    
    @property
    def value_meters(self) -> float:
        """Get depth in meters."""
        return self.value_mm / 1000.0
    
    @classmethod
    def from_meters(cls, meters: float) -> 'DepthValue':
        """Create depth value from meters."""
        return cls(meters * 1000.0)
    
    def is_valid(self, max_depth_mm: float = 10000.0) -> bool:
        """Check if depth is within valid range."""
        return 0 < self.value_mm <= max_depth_mm
    
    def __str__(self) -> str:
        return f"{self.value_mm:.1f}mm"


@dataclass(frozen=True)
class ProcessingTime:
    """Represents processing time in milliseconds."""
    
    value_ms: float
    
    def __post_init__(self):
        if self.value_ms < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.value_ms}")
    
    @property
    def value_seconds(self) -> float:
        """Get processing time in seconds."""
        return self.value_ms / 1000.0
    
    @classmethod
    def from_seconds(cls, seconds: float) -> 'ProcessingTime':
        """Create processing time from seconds."""
        return cls(seconds * 1000.0)
    
    def is_fast(self, threshold_ms: float = 100.0) -> bool:
        """Check if processing is fast."""
        return self.value_ms <= threshold_ms
    
    def is_slow(self, threshold_ms: float = 1000.0) -> bool:
        """Check if processing is slow."""
        return self.value_ms >= threshold_ms
    
    def __str__(self) -> str:
        return f"{self.value_ms:.1f}ms"
