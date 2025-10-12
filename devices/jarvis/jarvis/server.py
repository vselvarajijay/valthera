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

app = FastAPI(title="Jarvis Smart CV Pipeline", version="2.0.0")

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


@app.on_event("startup")
async def startup_event():
    """Initialize basic services on startup (without camera processing)"""
    global center_depth_processor, smart_pipeline
    
    logger.info("Starting Jarvis Smart CV Pipeline API...")
    
    try:
        # Start WebSocket manager
        await start_websocket_manager()
        logger.info("WebSocket manager started")
        
        # Initialize processors as None - will be created on-demand
        center_depth_processor = None
        smart_pipeline = None
        
        logger.info("API server ready. Camera processing will start on first request.")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup processors on shutdown"""
    global center_depth_processor, smart_pipeline
    
    logger.info("Shutting down Jarvis Smart CV Pipeline...")
    
    try:
        # Stop WebSocket manager
        await stop_websocket_manager()
        logger.info("WebSocket manager stopped")
        
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


@app.get("/health")
async def health():
    """Health check endpoint"""
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


@app.post("/api/v1/pipeline/initialize")
async def initialize_pipeline():
    """Manually initialize camera processors"""
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


@app.get("/debug", response_class=HTMLResponse)
async def debug_view():
    """Debug HTML page showing camera and depth views"""
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
                    <img id="depth-feed" src="/api/v1/camera/depth" alt="Depth map" onerror="handleImageError('depth')">
                    <div class="info">WebSocket streaming (~10 FPS)</div>
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
                        
                        if (data.type === 'camera_frame') {
                            // Update raw camera feed
                            if (data.raw_frame) {
                                const rawImg = document.getElementById('raw-feed');
                                rawImg.src = 'data:image/jpeg;base64,' + data.raw_frame;
                            }
                            
                            // Update depth map
                            if (data.depth_frame) {
                                const depthImg = document.getElementById('depth-feed');
                                depthImg.src = 'data:image/jpeg;base64,' + data.depth_frame;
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


    """Root endpoint with basic info"""
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