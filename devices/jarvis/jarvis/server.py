#!/usr/bin/env python3

"""
Main server file for Jarvis smart CV pipeline.

Refactored server that uses the clean architecture with dependency injection,
proper error handling, and structured logging.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import clean architecture components
from jarvis.infrastructure.container import (
    get_container,
    configure_container,
    wire_container,
    cleanup_container
)
from jarvis.infrastructure.config.settings import get_settings
from jarvis.infrastructure.config.config_loader import get_config_loader
from jarvis.api.controllers import create_api_router
from jarvis.api.websocket.stream_handler import StreamHandler, get_stream_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("[SERVER] Starting Jarvis smart CV pipeline...")
    
    try:
        # Load configuration
        config_loader = get_config_loader()
        settings = config_loader.load_config()
        
        # Configure dependency injection container
        configure_container(settings.to_dict())
        
        # Wire container to application packages
        wire_container([
            "jarvis.api.controllers",
            "jarvis.application.use_cases"
        ])
        
        logger.info("[SERVER] Dependency injection configured")
        
        # Initialize pipeline components
        await _initialize_pipeline()
        
        logger.info("[SERVER] Jarvis pipeline started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"[SERVER] Error during startup: {e}")
        raise
    finally:
        # Cleanup
        logger.info("[SERVER] Shutting down Jarvis pipeline...")
        cleanup_container()
        logger.info("[SERVER] Shutdown complete")


async def _initialize_pipeline():
    """Initialize pipeline components."""
    try:
        from jarvis.infrastructure.container import get_camera_service, get_detection_service
        from jarvis.tracking.vehicle_tracker import get_tracker
        
        # Initialize camera service
        camera_service = await get_camera_service()
        if not await camera_service.is_available():
            logger.warning("[SERVER] Camera service not available")
        
        # Initialize detection service
        detection_service = get_detection_service()
        if not await detection_service.is_initialized():
            logger.warning("[SERVER] Detection service not initialized")
        
        # Initialize vehicle tracker
        tracker = get_tracker()
        if not tracker.initialize():
            logger.warning("[SERVER] Vehicle tracker not initialized")
        
        logger.info("[SERVER] Pipeline components initialized")
            
    except Exception as e:
        logger.error(f"[SERVER] Error initializing pipeline: {e}")
        raise


# Create FastAPI application
app = FastAPI(
    title="Jarvis Smart CV Pipeline",
    description="Intelligent computer vision pipeline for Jetson Nano",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=settings.server.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
api_router = create_api_router()
app.include_router(api_router)

# WebSocket endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming."""
    try:
        stream_handler = get_stream_handler()
        await stream_handler.handle_websocket(websocket)
    except WebSocketDisconnect:
        logger.info("[SERVER] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[SERVER] WebSocket error: {e}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"[SERVER] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Jarvis Smart CV Pipeline",
        "version": "2.0.0",
        "status": "running",
        "architecture": "clean_architecture",
        "endpoints": {
            "api": "/api/v1",
            "websocket": "/ws/stream",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        from jarvis.infrastructure.container import get_camera_service, get_detection_service
        
        camera_service = await get_camera_service()
        camera_available = await camera_service.is_available()
        detection_service = get_detection_service()
        detection_initialized = await detection_service.is_initialized()
        
        status = "healthy" if camera_available and detection_initialized else "degraded"
        
        return {
            "status": status,
            "camera_available": camera_available,
            "detection_initialized": detection_initialized
        }
        
    except Exception as e:
        logger.error(f"[SERVER] Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/debug")
async def debug_info():
    """Debug information endpoint."""
    try:
        from jarvis.infrastructure.container import get_camera_service, get_detection_service
        
        camera_service = await get_camera_service()
        detection_service = get_detection_service()
        
        return {
            "status": "ok",
            "camera_service": {
                "type": type(camera_service).__name__,
                "available": await camera_service.is_available()
            },
            "detection_service": {
                "type": type(detection_service).__name__,
                "initialized": await detection_service.is_initialized()
            },
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"[SERVER] Debug info error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    try:
        from .infrastructure.container import get_metrics_repository
        
        metrics_repo = get_metrics_repository()
        metrics_data = await metrics_repo.get_system_metrics()
        
        # Convert to Prometheus format (simplified)
        prometheus_metrics = []
        
        # Add basic metrics
        prometheus_metrics.append("# HELP jarvis_up Service status")
        prometheus_metrics.append("# TYPE jarvis_up gauge")
        prometheus_metrics.append("jarvis_up 1")
        
        # Add processing time metrics
        if "processing_stats" in metrics_data:
            for operation, stats in metrics_data["processing_stats"].items():
                prometheus_metrics.append(f"# HELP jarvis_processing_time_ms Processing time in milliseconds")
                prometheus_metrics.append(f"# TYPE jarvis_processing_time_ms gauge")
                prometheus_metrics.append(f'jarvis_processing_time_ms{{operation="{operation}"}} {stats.get("average_ms", 0)}')
        
        return "\n".join(prometheus_metrics)
        
    except Exception as e:
        logger.error(f"[SERVER] Metrics error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Configuration endpoint
@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    try:
        config_loader = get_config_loader()
        config_summary = config_loader.get_config_summary()
        return config_summary
        
    except Exception as e:
        logger.error(f"[SERVER] Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("[SERVER] Additional startup tasks completed")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks."""
    logger.info("[SERVER] Additional shutdown tasks completed")


def main():
    """Main entry point."""
    settings = get_settings()
    
    logger.info(f"[SERVER] Starting server on {settings.server.host}:{settings.server.port}")
    logger.info(f"[SERVER] Environment: {settings.environment}")
    logger.info(f"[SERVER] Debug mode: {settings.debug}")
    
    uvicorn.run(
        "jarvis.server:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
        reload=settings.debug,
        access_log=True
    )


if __name__ == "__main__":
    main()