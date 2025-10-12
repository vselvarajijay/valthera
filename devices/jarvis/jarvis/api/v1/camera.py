#!/usr/bin/env python3

import asyncio
import base64
import json
import logging
import time
import cv2
from typing import Optional, Set
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ...depth_camera import DepthCamera
from ...classifiers.object_classifier import ObjectClassifier

logger = logging.getLogger(__name__)

router = APIRouter(tags=["camera"])

# Global simple camera instance
simple_camera: Optional[DepthCamera] = None

# Global car classifier instance
car_classifier: Optional[ObjectClassifier] = None

# WebSocket connections for streaming
active_connections: Set[WebSocket] = set()
streaming_task: Optional[asyncio.Task] = None

@router.get("/raw")
async def get_camera_raw():
    """Get raw RGB camera frame directly (no CV pipeline needed)"""
    global simple_camera
    
    # Lazy initialization
    if not simple_camera:
        logger.info("[CAMERA_API] Initializing simple camera...")
        simple_camera = DepthCamera(width=640, height=480, fps=30)
        simple_camera.start()
        time.sleep(0.5)  # Let camera stabilize
        logger.info("[CAMERA_API] Simple camera initialized")
    
    frame = simple_camera.get_latest_frame()
    if not frame or frame.color_frame is None:
        raise HTTPException(status_code=404, detail="No camera frame available")
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame.color_frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/depth")
async def get_camera_depth():
    """Get depth map directly (no CV pipeline needed)"""
    global simple_camera
    
    # Lazy initialization
    if not simple_camera:
        logger.info("[CAMERA_API] Initializing simple camera...")
        simple_camera = DepthCamera(width=640, height=480, fps=30)
        simple_camera.start()
        time.sleep(0.5)  # Let camera stabilize
        logger.info("[CAMERA_API] Simple camera initialized")
    
    frame = simple_camera.get_latest_frame()
    if not frame or frame.depth_frame is None:
        raise HTTPException(status_code=404, detail="No depth data available")
    
    # Normalize and colorize depth map
    depth_normalized = cv2.normalize(frame.depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', depth_colored)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/status")
async def get_camera_status():
    """Get simple camera status"""
    global simple_camera
    return {
        "available": simple_camera is not None,
        "running": simple_camera.is_running if simple_camera else False,
        "frame_count": simple_camera.frame_count if simple_camera else 0,
        "websocket_connections": len(active_connections),
        "streaming_active": streaming_task is not None and not streaming_task.done()
    }

@router.websocket("/stream")
async def websocket_camera_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming camera frames and depth data"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"[CAMERA_WS] Client connected. Total connections: {len(active_connections)}")
    
    # Start streaming task if not already running
    global streaming_task
    if not streaming_task or streaming_task.done():
        streaming_task = asyncio.create_task(stream_camera_data())
    
    try:
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                try:
                    message = json.loads(data)
                    if message.get("action") == "ping":
                        await websocket.send_text(json.dumps({"action": "pong", "timestamp": time.time()}))
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"[CAMERA_WS] WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        logger.info(f"[CAMERA_WS] Client disconnected. Total connections: {len(active_connections)}")
        
        # Stop streaming if no more connections
        if not active_connections and streaming_task and not streaming_task.done():
            streaming_task.cancel()

async def stream_camera_data():
    """Background task to stream camera data to all connected WebSocket clients"""
    global simple_camera, car_classifier
    
    logger.info("[CAMERA_STREAM] Starting camera streaming task")
    
    # Initialize camera if needed
    if not simple_camera:
        logger.info("[CAMERA_STREAM] Initializing camera for streaming...")
        simple_camera = DepthCamera(width=640, height=480, fps=30)
        simple_camera.start()
        await asyncio.sleep(0.5)  # Let camera stabilize
    
    # Initialize car classifier if needed
    if not car_classifier:
        logger.info("[CAMERA_STREAM] Initializing car classifier...")
        car_classifier = ObjectClassifier()
        car_classifier.initialize()
        logger.info("[CAMERA_STREAM] Car classifier initialized")
    
    frame_count = 0
    
    try:
        while active_connections:
            try:
                # Get latest frame
                frame = simple_camera.get_latest_frame()
                if not frame:
                    await asyncio.sleep(0.1)
                    continue
                
                # Encode frames as JPEG
                raw_data = None
                depth_data = None
                tracking_data = None
                car_count = 0
                car_detections = []
                
                if frame.color_frame is not None:
                    # Raw frame
                    _, raw_buffer = cv2.imencode('.jpg', frame.color_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    raw_data = base64.b64encode(raw_buffer).decode('utf-8')
                    
                    # Car detection and tracking frame
                    if car_classifier and car_classifier.is_initialized:
                        car_detections = car_classifier.detect_by_class(frame.color_frame, ["car"])
                        car_count = len(car_detections)
                        
                        # Draw bounding boxes on tracking frame
                        tracking_frame = frame.color_frame.copy()
                        for detection in car_detections:
                            x1, y1, x2, y2 = detection.bbox
                            cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Car {detection.confidence:.2f}"
                            cv2.putText(tracking_frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Encode tracking frame
                        _, tracking_buffer = cv2.imencode('.jpg', tracking_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        tracking_data = base64.b64encode(tracking_buffer).decode('utf-8')
                
                if frame.depth_frame is not None:
                    # Normalize and colorize depth map
                    depth_normalized = cv2.normalize(frame.depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    _, depth_buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    depth_data = base64.b64encode(depth_buffer).decode('utf-8')
                
                # Create message
                message = {
                    "type": "camera_frame",
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "raw_frame": raw_data,
                    "depth_frame": depth_data,
                    "tracking_frame": tracking_data,
                    "car_count": car_count,
                    "car_detections": [
                        {
                            "bbox": det.bbox,
                            "confidence": det.confidence
                        } for det in car_detections
                    ],
                    "frame_count": frame_count
                }
                
                # Send to all connected clients
                disconnected = set()
                for websocket in list(active_connections):
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"[CAMERA_STREAM] Error sending to client: {e}")
                        disconnected.add(websocket)
                
                # Remove disconnected clients
                for websocket in disconnected:
                    active_connections.discard(websocket)
                
                frame_count += 1
                
                # Control frame rate (target ~10 FPS for WebSocket)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[CAMERA_STREAM] Error in streaming loop: {e}")
                await asyncio.sleep(0.1)
                
    except asyncio.CancelledError:
        logger.info("[CAMERA_STREAM] Streaming task cancelled")
    except Exception as e:
        logger.error(f"[CAMERA_STREAM] Streaming task error: {e}")
    finally:
        logger.info("[CAMERA_STREAM] Camera streaming task ended")
