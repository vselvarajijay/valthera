#!/usr/bin/env python3

"""
Person Re-ID feature extractor using OSNet.

This module provides OSNet-based feature extraction optimized for person
re-identification tasks, using the fastest model for real-time performance.
"""

import logging
import threading
from typing import Union, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from torchreid.utils import FeatureExtractor
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False

logger = logging.getLogger(__name__)


class PersonReIDExtractor:
    """OSNet feature extractor optimized for person Re-ID"""
    
    def __init__(self, model_name: str = 'osnet_x0_25', device: Optional[str] = None):
        """
        Initialize person Re-ID extractor.
        
        Args:
            model_name: OSNet model variant (osnet_x0_25 for fastest performance)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.extractor = None
        self._lock = threading.Lock()
        
        logger.info(f"[PERSON_REID] Initializing person Re-ID extractor: {model_name} on {self.device}")
    
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
                logger.error("[PERSON_REID] torchreid not available - cannot initialize OSNet")
                return False
            
            try:
                self.extractor = FeatureExtractor(
                    model_name=self.model_name,
                    device=self.device,
                    verbose=False
                )
                logger.info(f"[PERSON_REID] Person Re-ID extractor initialized successfully")
                return True
            except Exception as e:
                logger.error(f"[PERSON_REID] Failed to initialize person Re-ID extractor: {e}")
                return False
    
    def get_embedding(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract person embedding from image.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Person embedding vector (512-dim) or None if failed
        """
        if not self._ensure_initialized():
            return None
        
        try:
            # Extract features using torchreid
            features = self.extractor(image)
            
            if features is not None and len(features) > 0:
                # Convert to numpy array and normalize
                embedding = features[0].cpu().numpy()
                # L2 normalization for better similarity matching
                embedding = embedding / np.linalg.norm(embedding)
                logger.debug(f"[PERSON_REID] Extracted embedding shape: {embedding.shape}")
                return embedding
            else:
                logger.warning("[PERSON_REID] No features extracted from image")
                return None
                
        except Exception as e:
            logger.error(f"[PERSON_REID] Error extracting person embedding: {e}")
            return None
    
    def cleanup(self):
        """Cleanup extractor resources"""
        with self._lock:
            if self.extractor:
                try:
                    # Most models don't need explicit cleanup
                    self.extractor = None
                    logger.info("[PERSON_REID] Person Re-ID extractor cleaned up")
                except Exception as e:
                    logger.error(f"[PERSON_REID] Error cleaning up extractor: {e}")


# Global instance for singleton pattern
_person_extractor_instance: Optional[PersonReIDExtractor] = None
_extractor_lock = threading.Lock()


def get_person_embedding(image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Get person embedding using singleton extractor instance.
    
    Args:
        image: Image path or numpy array
        
    Returns:
        Person embedding vector or None if failed
    """
    global _person_extractor_instance
    
    with _extractor_lock:
        if _person_extractor_instance is None:
            _person_extractor_instance = PersonReIDExtractor()
    
    return _person_extractor_instance.get_embedding(image)


def cleanup_person_extractor():
    """Cleanup person extractor instance"""
    global _person_extractor_instance
    
    with _extractor_lock:
        if _person_extractor_instance:
            _person_extractor_instance.cleanup()
            _person_extractor_instance = None


def main():
    """Test the person Re-ID extractor"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[PERSON_REID] Testing person Re-ID extractor...")
    
    # Test with dummy numpy array
    if NUMPY_AVAILABLE:
        dummy_image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        embedding = get_person_embedding(dummy_image)
        if embedding is not None:
            logger.info(f"[PERSON_REID] Successfully extracted embedding: shape={embedding.shape}")
        else:
            logger.error("[PERSON_REID] Failed to extract embedding")
    else:
        logger.info("[PERSON_REID] NumPy not available - skipping test")
    
    cleanup_person_extractor()
    logger.info("[PERSON_REID] Test completed")


if __name__ == "__main__":
    main()
