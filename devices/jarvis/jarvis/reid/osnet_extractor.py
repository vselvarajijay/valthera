#!/usr/bin/env python3

"""
OSNet feature extractor for vehicle re-identification.

This module provides OSNet-based feature extraction for generating
vehicle embeddings used in re-identification tasks.
"""

import logging
import os
from typing import Union, Optional
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from torchreid.utils import FeatureExtractor
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False

logger = logging.getLogger(__name__)


class OSNetExtractor:
    """OSNet feature extractor for vehicle embeddings"""
    
    def __init__(self, model_name: str = 'osnet_x0_25', device: Optional[str] = None):
        """
        Initialize OSNet feature extractor.
        
        Args:
            model_name: OSNet model variant (default: osnet_x0_25)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.extractor = None
        self._lock = threading.Lock()
        
        logger.info(f"[REID] Initializing OSNet extractor: {model_name} on {self.device}")
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'
    
    def _ensure_initialized(self) -> bool:
        """Ensure extractor is initialized"""
        with self._lock:
            if self.extractor is not None:
                return True
            
            if not TORCHREID_AVAILABLE:
                logger.error("[REID] torchreid not available - cannot initialize OSNet")
                return False
            
            try:
                self.extractor = FeatureExtractor(
                    model_name=self.model_name,
                    device=self.device,
                    verbose=False
                )
                logger.info(f"[REID] OSNet extractor initialized successfully")
                return True
            except Exception as e:
                logger.error(f"[REID] Failed to initialize OSNet extractor: {e}")
                return False
    
    def get_embedding(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract embedding from image.
        
        Args:
            image: Image path (str) or numpy array (BGR format)
            
        Returns:
            Feature embedding as numpy array, or None if failed
        """
        if not self._ensure_initialized():
            return None
        
        if not NUMPY_AVAILABLE:
            logger.error("[REID] NumPy not available")
            return None
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not CV2_AVAILABLE:
                    logger.error("[REID] OpenCV not available for image loading")
                    return None
                img = cv2.imread(image)
                if img is None:
                    logger.error(f"[REID] Failed to load image: {image}")
                    return None
            else:
                img = image
            
            # Extract features
            features = self.extractor(img)
            
            if features is None or len(features) == 0:
                logger.warning("[REID] No features extracted from image")
                return None
            
            # Convert to numpy array
            embedding = features[0].cpu().numpy()
            
            logger.debug(f"[REID] Extracted embedding shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"[REID] Error extracting embedding: {e}")
            return None
    
    def cleanup(self):
        """Cleanup extractor resources"""
        with self._lock:
            if self.extractor:
                try:
                    # Most models don't need explicit cleanup
                    self.extractor = None
                    logger.info("[REID] OSNet extractor cleaned up")
                except Exception as e:
                    logger.error(f"[REID] Error cleaning up extractor: {e}")


# Global extractor instance
_extractor_instance: Optional[OSNetExtractor] = None
_extractor_lock = threading.Lock()


def get_embedding(image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Convenience function to get embedding from image.
    
    Args:
        image: Image path (str) or numpy array (BGR format)
        
    Returns:
        Feature embedding as numpy array, or None if failed
    """
    global _extractor_instance
    
    with _extractor_lock:
        if _extractor_instance is None:
            _extractor_instance = OSNetExtractor()
    
    return _extractor_instance.get_embedding(image)


def cleanup_extractor():
    """Cleanup global extractor instance"""
    global _extractor_instance
    
    with _extractor_lock:
        if _extractor_instance:
            _extractor_instance.cleanup()
            _extractor_instance = None


def main():
    """Test the OSNet extractor"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[REID] Testing OSNet extractor...")
    
    # Test with sample image if available
    test_image_path = "test_vehicle.jpg"
    
    if os.path.exists(test_image_path):
        embedding = get_embedding(test_image_path)
        if embedding is not None:
            logger.info(f"[REID] Successfully extracted embedding: shape={embedding.shape}")
        else:
            logger.error("[REID] Failed to extract embedding")
    else:
        logger.info("[REID] No test image found - skipping test")
    
    cleanup_extractor()
    logger.info("[REID] Test completed")


if __name__ == "__main__":
    main()
