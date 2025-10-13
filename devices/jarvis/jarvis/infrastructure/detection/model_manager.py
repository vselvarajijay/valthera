#!/usr/bin/env python3

"""
Model manager for Jarvis smart CV pipeline.

Manages shared model instances to reduce memory usage and improve
performance across multiple detection services.
"""

import logging
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model_type: str
    model_path: str
    version: Optional[str] = None
    loaded_at: Optional[float] = None
    access_count: int = 0
    last_access: Optional[float] = None


class ModelManager:
    """Manages shared model instances."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._lock = threading.RLock()
        
        logger.info("[MODEL_MANAGER] Initialized")
    
    def register_model(
        self, 
        name: str, 
        model: Any, 
        model_type: str, 
        model_path: str,
        version: Optional[str] = None
    ) -> None:
        """Register a model instance."""
        with self._lock:
            import time
            self._models[name] = model
            self._model_info[name] = ModelInfo(
                name=name,
                model_type=model_type,
                model_path=model_path,
                version=version,
                loaded_at=time.time(),
                access_count=0,
                last_access=time.time()
            )
            logger.info(f"[MODEL_MANAGER] Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a model instance."""
        with self._lock:
            if name in self._models:
                import time
                self._model_info[name].access_count += 1
                self._model_info[name].last_access = time.time()
                return self._models[name]
            return None
    
    def has_model(self, name: str) -> bool:
        """Check if model is registered."""
        with self._lock:
            return name in self._models
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get model information."""
        with self._lock:
            return self._model_info.get(name)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all registered models."""
        with self._lock:
            return self._models.copy()
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        with self._lock:
            import time
            current_time = time.time()
            
            stats = {
                "total_models": len(self._models),
                "models": {}
            }
            
            for name, info in self._model_info.items():
                age_seconds = current_time - info.loaded_at if info.loaded_at else 0
                last_access_seconds = current_time - info.last_access if info.last_access else 0
                
                stats["models"][name] = {
                    "model_type": info.model_type,
                    "model_path": info.model_path,
                    "version": info.version,
                    "age_seconds": age_seconds,
                    "access_count": info.access_count,
                    "last_access_seconds_ago": last_access_seconds
                }
            
            return stats
    
    def remove_model(self, name: str) -> bool:
        """Remove a model."""
        with self._lock:
            if name in self._models:
                del self._models[name]
                del self._model_info[name]
                logger.info(f"[MODEL_MANAGER] Removed model: {name}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all models."""
        with self._lock:
            self._models.clear()
            self._model_info.clear()
            logger.info("[MODEL_MANAGER] Cleared all models")


# Global model manager instance
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance


def cleanup_model_manager():
    """Cleanup the global model manager."""
    global _model_manager_instance
    if _model_manager_instance:
        _model_manager_instance.clear()
        _model_manager_instance = None
