#!/usr/bin/env python3

import asyncio
import logging
import os
import sys
import time
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from .center_depth_processor import CenterDepthProcessor
from .core.smart_pipeline import SmartCVPipeline
from .api import api_v1_router
from .api.v1.stream import start_websocket_manager, stop_websocket_manager, broadcast_analysis_result
from .processor_manager import ensure_processors_initialized, get_processors, cleanup_processors
from .tracking.vehicle_tracker import get_tracker, cleanup_tracker

# Configuration
HTTP_PORT = int(os.environ.get("JARVIS_HTTP_PORT", "8001"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/jarvis.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jarvis Smart CV Pipeline API",
    description="""
    ## Intelligent Computer Vision Pipeline API
    
    The Jarvis Smart CV Pipeline provides a comprehensive API for intelligent computer vision processing with multi-classifier support, real-time streaming, and comprehensive system control.
    
    ### Key Features:
    - **Multi-Classifier Support**: Person detection, object classification, face recognition, and vehicle detection
    - **Real-time Streaming**: WebSocket-based live camera feeds with depth data
    - **Intel RealSense Integration**: RGB and depth camera support
    - **Vehicle Tracking**: Advanced vehicle detection and tracking capabilities
    - **Pipeline Management**: Dynamic pipeline control and configuration
    - **Health Monitoring**: Comprehensive system status and diagnostics
    
    ### Camera Capabilities:
    - Raw RGB camera feeds
    - Depth map visualization
    - Real-time object detection
    - Vehicle counting and tracking
    - Mock camera support for testing
    
    ### API Endpoints:
    - **Analysis**: Unified analysis with multiple classifiers
    - **Streaming**: Real-time WebSocket data streams
    - **Pipeline**: System control and configuration
    - **Classifiers**: Individual classifier management
    - **Frames**: Frame access and processing
    - **Camera**: Direct camera control and feeds
    - **Vehicles**: Vehicle-specific operations
    
    ### Authentication:
    No authentication required for local network access.
    
    ### Base URLs:
    - Development: `http://localhost:8001`
    - Production: `http://jarvis.local:8001`
    """,
    version="2.0.0",
    contact={
        "name": "Valthera Development Team",
        "email": "dev@valthera.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "analysis",
            "description": "Unified analysis operations with multiple classifiers",
        },
        {
            "name": "streaming", 
            "description": "Real-time WebSocket streaming endpoints",
        },
        {
            "name": "pipeline",
            "description": "Pipeline control and configuration management",
        },
        {
            "name": "classifiers",
            "description": "Individual classifier management and configuration",
        },
        {
            "name": "frames",
            "description": "Frame access, processing, and retrieval",
        },
        {
            "name": "camera",
            "description": "Direct camera control, feeds, and status",
        },
        {
            "name": "vehicles",
            "description": "Vehicle detection, tracking, and statistics",
        },
        {
            "name": "health",
            "description": "System health monitoring and diagnostics",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS for easy local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global processors (managed by processor_manager)
center_depth_processor: Optional[CenterDepthProcessor] = None
smart_pipeline: Optional[SmartCVPipeline] = None
vehicle_tracker = None


@app.on_event("startup")
async def startup_event():
    """Initialize basic services on startup (without camera processing)"""
    global center_depth_processor, smart_pipeline, vehicle_tracker
    
    logger.info("Starting Jarvis Smart CV Pipeline API...")
    
    try:
        # Start WebSocket manager
        await start_websocket_manager()
        logger.info("WebSocket manager started")
        
        # Initialize processors as None - will be created on-demand
        center_depth_processor = None
        smart_pipeline = None
        
        # Initialize vehicle tracker
        vehicle_tracker = get_tracker()
        if vehicle_tracker.initialize():
            logger.info("Vehicle tracker initialized")
            # Start tracking automatically
            if vehicle_tracker.start_tracking():
                logger.info("Vehicle tracking started")
            else:
                logger.warning("Failed to start vehicle tracking")
        else:
            logger.error("Failed to initialize vehicle tracker")
        
        logger.info("API server ready. Camera processing will start on first request.")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup processors on shutdown"""
    global center_depth_processor, smart_pipeline, vehicle_tracker
    
    logger.info("Shutting down Jarvis Smart CV Pipeline...")
    
    try:
        # Stop WebSocket manager
        await stop_websocket_manager()
        logger.info("WebSocket manager stopped")
        
        # Stop vehicle tracker
        if vehicle_tracker:
            vehicle_tracker.cleanup()
            logger.info("Vehicle tracker stopped")
        
        if smart_pipeline:
            smart_pipeline.cleanup()
            logger.info("Smart CV pipeline stopped")
            
        if center_depth_processor:
            center_depth_processor.cleanup()
            logger.info("Center depth processor stopped")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Mount API v1 router
app.include_router(api_v1_router)


@app.get("/health", tags=["health"], summary="System Health Check", description="Get comprehensive system health status including processor states and availability")
async def health():
    """
    Get comprehensive system health status.
    
    Returns detailed information about:
    - Overall system status
    - Center depth processor state
    - Smart CV pipeline state
    - Timestamp and version information
    
    This endpoint is useful for monitoring system health and debugging issues.
    """
    # Get current processor status from processor manager
    center_depth_processor, smart_pipeline = get_processors()
    
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "center_depth_processor": {
            "running": center_depth_processor.is_running if center_depth_processor else False,
            "available": center_depth_processor is not None
        },
        "smart_cv_pipeline": {
            "running": smart_pipeline.is_running if smart_pipeline else False,
            "available": smart_pipeline is not None
        }
    }
    
    logger.info(f"Health check: {status}")
    return JSONResponse(status)


@app.post("/api/v1/pipeline/initialize", tags=["pipeline"], summary="Initialize Pipeline", description="Manually initialize camera processors and pipeline components")
async def initialize_pipeline():
    """
    Manually initialize camera processors and pipeline components.
    
    This endpoint forces initialization of:
    - Center depth processor
    - Smart CV pipeline
    - Vehicle tracker
    
    Useful for debugging or when automatic initialization fails.
    Returns the initialization status of each component.
    """
    try:
        ensure_processors_initialized()
        return {
            "status": "success",
            "message": "Camera processors initialized successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize processors: {str(e)}")


@app.get("/logs")
async def get_logs():
    """Get recent log entries"""
    try:
        with open('/tmp/jarvis.log', 'r') as f:
            lines = f.readlines()
            # Return last 100 lines
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            return JSONResponse({
                "log_entries": recent_lines,
                "total_lines": len(lines),
                "showing_last": len(recent_lines)
            })
    except FileNotFoundError:
        return JSONResponse({"error": "Log file not found", "log_entries": []})
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return JSONResponse({"error": str(e), "log_entries": []})


@app.get("/debug", response_class=HTMLResponse, tags=["health"], summary="Debug View", description="Interactive debug page showing camera feeds, depth maps, and vehicle tracking")
async def debug_view():
    """
    Interactive debug page for camera and depth visualization.
    
    This endpoint returns an HTML page that provides:
    - Real-time camera feed display
    - Depth map visualization with color mapping
    - Vehicle detection and counting
    - WebSocket connection status
    - Interactive controls for testing
    
    The page automatically connects to WebSocket streams and displays
    live data from the camera system.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jarvis Camera Debug</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #1a1a1a;
                font-family: Arial, sans-serif;
                color: white;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .status {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
            }
            .status.error {
                background: #4a1a1a;
                border: 1px solid #ff4444;
            }
            .status.success {
                background: #1a4a1a;
                border: 1px solid #44ff44;
            }
            .feeds {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 15px;
            }
            
            .car-counter {
                background: #3a3a3a;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                color: #4ade80;
            }
            
            @media (max-width: 1200px) {
                .feeds {
                    grid-template-columns: 1fr 1fr;
                }
            }
            .feed {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 8px;
            }
            .feed h2 {
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 18px;
            }
            img {
                width: 100%;
                height: auto;
                border-radius: 4px;
                background: #000;
            }
            .info {
                margin-top: 10px;
                font-size: 12px;
                color: #888;
            }
            .controls {
                text-align: center;
                margin-bottom: 20px;
            }
            button {
                background: #4a4a4a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                margin: 0 10px;
            }
            button:hover {
                background: #5a5a5a;
            }
            button:disabled {
                background: #2a2a2a;
                color: #666;
                cursor: not-allowed;
            }
            @media (max-width: 768px) {
                .feeds {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Jarvis Camera Debug View</h1>
            
            <div class="controls">
                <button onclick="checkStatus()" id="status-btn">Check Camera Status</button>
            </div>
            
            <div id="status" class="status">
                <div>Status: <span id="status-text">Checking...</span></div>
                <div id="status-details"></div>
            </div>
            
            <div class="feeds">
                <div class="feed">
                    <h2>Raw Camera Feed</h2>
                    <img id="raw-feed" src="/api/v1/camera/raw" alt="Raw camera feed" onerror="handleImageError('raw')">
                    <div class="info">WebSocket streaming (~10 FPS)</div>
                </div>
                <div class="feed">
                    <h2>Depth Map</h2>
                    <canvas id="depth-canvas" width="640" height="480" style="width: 100%; height: auto; border-radius: 4px; background: #000;"></canvas>
                    <div class="info">WebSocket streaming (~10 FPS) - Raw depth data</div>
                </div>
                <div class="feed">
                    <h2>Car Tracking</h2>
                    <div class="car-counter">
                        Cars Detected: <span id="car-count">0</span>
                    </div>
                    <img id="tracking-feed" src="/api/v1/camera/raw" alt="Car tracking" onerror="handleImageError('tracking')">
                    <div class="info">YOLO car detection with bounding boxes</div>
                </div>
            </div>
        </div>
        <script>
            let isInitialized = false;
            let refreshInterval = null;
            let wsConnection = null;
            
            function updateStatus(message, isError = false) {
                const statusEl = document.getElementById('status');
                const statusText = document.getElementById('status-text');
                const statusDetails = document.getElementById('status-details');
                
                statusText.textContent = message;
                statusDetails.innerHTML = '';
                
                if (isError) {
                    statusEl.className = 'status error';
                } else {
                    statusEl.className = 'status success';
                }
            }
            
            function handleImageError(type) {
                updateStatus('Camera feed not available - camera may be initializing', true);
            }
            
            function renderDepthData(depthData, metadata) {
                console.log('renderDepthData called with:', metadata);
                console.log('Depth data length:', depthData ? depthData.length : 'null');
                const canvas = document.getElementById('depth-canvas');
                const ctx = canvas.getContext('2d');
                
                if (!depthData || !metadata) {
                    console.log('Missing depth data or metadata');
                    return;
                }
                
                console.log('Starting depth rendering...');
                
                const width = metadata.width;
                const height = metadata.height;
                console.log('Rendering depth data:', width, 'x', height);
                
                // Resize canvas if needed
                canvas.width = width;
                canvas.height = height;
                
                // Decode base64 depth data to Uint16Array
                const binaryString = atob(depthData);
                const depth = new Uint16Array(binaryString.length / 2);
                for (let i = 0; i < binaryString.length; i += 2) {
                    depth[i / 2] = binaryString.charCodeAt(i) | (binaryString.charCodeAt(i + 1) << 8);
                }
                
                // Create image data for canvas
                const img = ctx.createImageData(width, height);
                
                // Full depth image with RED=close, GREEN=far color mapping
                const maxDepth = 5000; // 5 meters in mm
                const minDepth = 100;   // 10cm minimum depth
                
                let minVal = 65535;
                let maxVal = 0;
                
                // Find actual min/max values for better contrast
                for (let i = 0; i < depth.length; i++) {
                    if (depth[i] > 0 && depth[i] < 65535) {
                        minVal = Math.min(minVal, depth[i]);
                        maxVal = Math.max(maxVal, depth[i]);
                    }
                }
                
                // Use actual range with reasonable bounds
                let actualMin = Math.max(minDepth, minVal);
                let actualMax = Math.min(maxDepth, maxVal);
                
                // Ensure we have a reasonable range
                let range = actualMax - actualMin;
                if (range < 500) {
                    const center = (actualMin + actualMax) / 2;
                    actualMin = Math.max(minDepth, center - 1000);
                    actualMax = Math.min(maxDepth, center + 1000);
                    range = actualMax - actualMin;
                }
                
                console.log('Depth range:', actualMin, 'to', actualMax, 'mm (range:', range, 'mm)');
                
                for (let i = 0, j = 0; i < depth.length; i++, j += 4) {
                    const v16 = depth[i];
                    
                    if (v16 === 0 || v16 >= 65535) {
                        // Invalid depth - show as black
                        img.data[j] = 0;     // R
                        img.data[j+1] = 0;   // G
                        img.data[j+2] = 0;   // B
                        img.data[j+3] = 255; // A
                    } else {
                        // Map depth to color: RED=close, GREEN=far
                        const normalized = Math.max(0, Math.min(1, (v16 - actualMin) / range));
                        
                        // Apply gamma correction for better visibility
                        const gamma = 0.6;
                        const contrastNormalized = Math.pow(normalized, gamma);
                        
                        // Color mapping: RED (close) -> YELLOW (mid) -> GREEN (far)
                        let r, g, b;
                        
                        if (contrastNormalized < 0.5) {
                            // Close to mid: RED to YELLOW
                            const t = contrastNormalized * 2;
                            r = 255;
                            g = Math.floor(t * 255);
                            b = 0;
                        } else {
                            // Mid to far: YELLOW to GREEN
                            const t = (contrastNormalized - 0.5) * 2;
                            r = Math.floor((1 - t) * 255);
                            g = 255;
                            b = 0;
                        }
                        
                        img.data[j] = r;     // R
                        img.data[j+1] = g;   // G
                        img.data[j+2] = b;   // B
                        img.data[j+3] = 255; // A
                    }
                }
                
                // Draw the depth image to canvas
                ctx.putImageData(img, 0, 0);
            }
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/camera/stream`;
                
                wsConnection = new WebSocket(wsUrl);
                
                wsConnection.onopen = function() {
                    updateStatus('WebSocket connected - streaming camera data');
                    isInitialized = true;
                };
                
                wsConnection.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        console.log('Received WebSocket message:', data.type, 'Frame:', data.frame_count);
                        
                        if (data.type === 'camera_frame') {
                            // Update raw camera feed
                            if (data.raw_frame) {
                                const rawImg = document.getElementById('raw-feed');
                                rawImg.src = 'data:image/jpeg;base64,' + data.raw_frame;
                                console.log('Updated raw camera feed');
                            }
                            
                            // Update depth map with raw depth data
                    if (data.depth_data && data.depth_data.data && data.depth_data.metadata) {
                        console.log('✅ DEPTH DATA FOUND - Rendering depth data:', data.depth_data.metadata);
                        console.log('Depth data size:', data.depth_data.data.length, 'bytes');
                        renderDepthData(data.depth_data.data, data.depth_data.metadata);
                    } else {
                        console.log('❌ NO DEPTH DATA - Available data:', data.depth_data);
                    }
                            
                            // Update car tracking view
                            if (data.tracking_frame) {
                                const trackingImg = document.getElementById('tracking-feed');
                                trackingImg.src = 'data:image/jpeg;base64,' + data.tracking_frame;
                            }
                            
                            // Update car count
                            if (data.car_count !== undefined) {
                                document.getElementById('car-count').textContent = data.car_count;
                            }
                            
                            // Update status with frame and car info
                            updateStatus(`Streaming - Frame ${data.frame_count} - Cars: ${data.car_count || 0}`);
                        } else if (data.action === 'pong') {
                            // Handle pong responses
                            console.log('Received pong from server');
                        }
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                wsConnection.onclose = function() {
                    updateStatus('WebSocket disconnected', true);
                    isInitialized = false;
                    
                    // Try to reconnect after 3 seconds
                    setTimeout(() => {
                        if (!wsConnection || wsConnection.readyState === WebSocket.CLOSED) {
                            updateStatus('Attempting to reconnect...');
                            connectWebSocket();
                        }
                    }, 3000);
                };
                
                wsConnection.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    updateStatus('WebSocket connection error', true);
                };
            }
            
            async function checkStatus() {
                try {
                    const response = await fetch('/api/v1/camera/status');
                    const data = await response.json();
                    
                    if (response.ok) {
                        const status = data.running ? 'Running' : 'Not Running';
                        const available = data.available ? 'Available' : 'Not Available';
                        const wsConnections = data.websocket_connections || 0;
                        const streamingActive = data.streaming_active || false;
                        
                        updateStatus(`Camera Status: ${status} (${available}) - WS Connections: ${wsConnections} - Streaming: ${streamingActive ? 'Active' : 'Inactive'}`);
                        
                        if (data.available && data.running && !wsConnection) {
                            connectWebSocket();
                        }
                    } else {
                        updateStatus('Failed to check camera status', true);
                    }
                } catch (error) {
                    updateStatus('Error checking camera status: ' + error.message, true);
                }
            }
            
            function startRefresh() {
                // No longer needed with WebSocket streaming
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                    refreshInterval = null;
                }
            }
            
            // Check status on page load and connect WebSocket
            window.onload = function() {
                checkStatus();
                // Connect WebSocket after a short delay
                setTimeout(() => {
                    if (!wsConnection || wsConnection.readyState === WebSocket.CLOSED) {
                        connectWebSocket();
                    }
                }, 1000);
            };
            
            // Cleanup on page unload
            window.onbeforeunload = function() {
                if (wsConnection) {
                    wsConnection.close();
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/", tags=["health"], summary="API Information", description="Get basic API information and available endpoints")
async def root():
    """
    Get basic API information and available endpoints.
    
    Returns:
    - API name and version
    - Description of capabilities
    - Available endpoint categories
    
    This is the root endpoint for discovering API capabilities.
    """
    return {
        "name": "Jarvis Smart CV Pipeline",
        "version": "2.0.0",
        "description": "Intelligent computer vision pipeline with multi-classifier support",
        "endpoints": {
            "health": "/health",
            "logs": "/logs",
            "debug": "/debug",
            "api_v1": "/api/v1",
            "analyze": "/api/v1/analyze",
            "stream": "/api/v1/stream",
            "pipeline": "/api/v1/pipeline",
            "classifiers": "/api/v1/classifiers",
            "frames": "/api/v1/frames",
            "camera": "/api/v1/camera",
            "camera_stream": "WS /api/v1/camera/stream"
        }
    }


def main():
    """Main function for local testing"""
    import uvicorn
    logger.info(f"Starting Jarvis Smart CV Pipeline on port {HTTP_PORT}")
    uvicorn.run("jarvis.server:app", host="0.0.0.0", port=HTTP_PORT, reload=False)


if __name__ == "__main__":
    main()