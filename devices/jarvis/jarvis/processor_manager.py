#!/usr/bin/env python3

"""
Processor initialization utilities for Jarvis.

This module provides functions to initialize camera processors on-demand
to avoid circular imports.
"""

import logging
import threading
import asyncio
from typing import Optional

from .center_depth_processor import CenterDepthProcessor
from .core.smart_pipeline import SmartCVPipeline

logger = logging.getLogger(__name__)

# Global processors
center_depth_processor: Optional[CenterDepthProcessor] = None
smart_pipeline: Optional[SmartCVPipeline] = None

def ensure_processors_initialized():
    """Initialize processors on-demand if not already initialized"""
    global center_depth_processor, smart_pipeline
    
    if center_depth_processor is None or smart_pipeline is None:
        logger.info("Initializing camera processors on-demand...")
        
        try:
            # Initialize center depth processor
            logger.info("Initializing center depth processor...")
            center_depth_processor = CenterDepthProcessor()
            center_depth_processor.start()
            logger.info("Center depth processor started")
            
            # Initialize smart CV pipeline with existing depth camera
            logger.info("Initializing smart CV pipeline...")
            existing_depth_camera = center_depth_processor.get_depth_camera()
            smart_pipeline = SmartCVPipeline(depth_camera=existing_depth_camera)
            smart_pipeline.start()
            logger.info("Smart CV pipeline started")
            
            logger.info("Camera processors initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise

def get_processors():
    """Get the current processor instances"""
    return center_depth_processor, smart_pipeline

def cleanup_processors():
    """Cleanup processors on shutdown"""
    global center_depth_processor, smart_pipeline
    
    try:
        if smart_pipeline:
            logger.info("Stopping smart CV pipeline...")
            smart_pipeline.stop()
            smart_pipeline = None
            
        if center_depth_processor:
            logger.info("Stopping center depth processor...")
            center_depth_processor.stop()
            center_depth_processor = None
            
        logger.info("Processors cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during processor cleanup: {e}")




