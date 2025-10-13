#!/usr/bin/env python3

import asyncio
import base64
import json
import logging
import time
import cv2
import numpy as np
from typing import Optional, Set, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...depth_camera import DepthCamera
from ...simple_camera import SimpleCamera
from ...classifiers.object_classifier import ObjectClassifier
from ...classifiers.vehicle_classifier import VehicleClassifier
from ...center_depth_processor import CenterDepthProcessor

logger = logging.getLogger(__name__)

# Pydantic models for API documentation
class VehicleDetection(BaseModel):
    """Vehicle detection result"""
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score (0.0-1.0)")
    class_name: str = Field(..., description="Detected class name (e.g., 'car', 'truck')")
    class_id: int = Field(..., description="Class ID from the model")

class DepthMetadata(BaseModel):
    """Depth data metadata"""
    width: int = Field(..., description="Depth image width in pixels")
    height: int = Field(..., description="Depth image height in pixels")
    format: str = Field(..., description="Depth data format (e.g., 'z16_le')")
    dtype: str = Field(..., description="Data type (e.g., 'uint16')")

class DepthData(BaseModel):
    """Depth data with metadata"""
    data: str = Field(..., description="Base64-encoded depth data")
    metadata: DepthMetadata = Field(..., description="Depth data metadata")

class CameraStatus(BaseModel):
    """Camera system status"""
    available: bool = Field(..., description="Whether camera hardware is available")
    running: bool = Field(..., description="Whether camera processing is active")
    frame_count: int = Field(..., description="Total frames processed")
    websocket_connections: int = Field(..., description="Number of active WebSocket clients")
    streaming_active: bool = Field(..., description="Whether real-time streaming is running")

class CameraFrameMessage(BaseModel):
    """WebSocket camera frame message"""
    type: str = Field(..., description="Message type ('camera_frame')")
    frame_id: int = Field(..., description="Unique frame identifier")
    timestamp: float = Field(..., description="Frame timestamp")
    raw_frame: Optional[str] = Field(None, description="Base64-encoded raw RGB frame")
    tracking_frame: Optional[str] = Field(None, description="Base64-encoded tracking frame")
    car_count: int = Field(..., description="Number of detected vehicles")
    car_detections: List[VehicleDetection] = Field(..., description="List of vehicle detections")
    depth_data: Optional[DepthData] = Field(None, description="Depth data with metadata")
    frame_count: int = Field(..., description="Sequential frame counter")

class PingMessage(BaseModel):
    """WebSocket ping message"""
    action: str = Field(..., description="Action type ('ping')")

class PongMessage(BaseModel):
    """WebSocket pong response"""
    action: str = Field(..., description="Action type ('pong')")
    timestamp: float = Field(..., description="Response timestamp")

router = APIRouter(tags=["camera"])

# Global simple camera instance
simple_camera: Optional[SimpleCamera] = None

# Global car classifier instance
car_classifier: Optional[VehicleClassifier] = None

# Global center depth processor instance
center_depth_processor: Optional[CenterDepthProcessor] = None

# WebSocket connections for streaming
active_connections: Set[WebSocket] = set()
streaming_task: Optional[asyncio.Task] = None

@router.get("/raw", summary="Get Raw Camera Feed", description="Get raw RGB camera frame from Intel RealSense camera or fallback to mock camera")
async def get_camera_raw():
    """
    Get raw RGB camera frame from Intel RealSense camera.
    
    This endpoint provides access to the latest RGB frame from the camera system.
    It automatically initializes the center depth processor if not already running.
    
    **Camera Sources (in order of preference):**
    1. Intel RealSense D435i camera (if available)
    2. SimpleCamera fallback (if RealSense unavailable)
    3. Mock camera (for testing when no real camera available)
    
    **Response:**
    - Returns JPEG image data
    - Content-Type: image/jpeg
    - Quality: 80% compression for optimal performance
    
    **Error Handling:**
    - Falls back to mock camera if real camera fails
    - Logs all errors for debugging
    - Always returns a valid image response
    """
    global center_depth_processor
    
    try:
        # Initialize center depth processor if not already done
        if not center_depth_processor:
            logger.info("[CAMERA_API] Initializing center depth processor...")
            center_depth_processor = CenterDepthProcessor()
            center_depth_processor.start()
            logger.info("[CAMERA_API] Center depth processor started")
        
        # Get the depth camera from the center depth processor
        if center_depth_processor and center_depth_processor.depth_camera:
            depth_camera = center_depth_processor.depth_camera
            
            # Get latest frame from depth camera (which also has RGB)
            frame = depth_camera.get_latest_frame()
            if frame and frame.color_frame is not None:
                # Use real RGB data from Intel RealSense
                _, buffer = cv2.imencode('.jpg', frame.color_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
        # Fallback to mock camera if no real camera available
        logger.warning("[CAMERA_API] No Intel RealSense camera available, using mock camera")
        return await get_mock_camera()
        
    except Exception as e:
        logger.error(f"[CAMERA_API] Error getting RGB data: {e}")
        return await get_mock_camera()

@router.get("/depth", summary="Get Depth Map", description="Get depth map visualization from CenterDepthProcessor with color mapping")
async def get_camera_depth():
    """
    Get depth map visualization from CenterDepthProcessor.
    
    This endpoint provides access to the latest depth frame from the camera system,
    processed and colorized for visualization.
    
    **Depth Processing:**
    1. Retrieves raw depth data from Intel RealSense camera
    2. Normalizes depth values to 0-255 range
    3. Applies JET colormap for visualization
    4. Encodes as JPEG for web display
    
    **Color Mapping:**
    - Red/Orange: Close objects (near camera)
    - Yellow: Medium distance
    - Blue/Purple: Far objects (far from camera)
    - Black: Invalid/no depth data
    
    **Response:**
    - Returns JPEG image data with colorized depth
    - Content-Type: image/jpeg
    - Quality: 80% compression
    
    **Fallback:**
    - Uses mock depth data if real camera unavailable
    - Always returns a valid depth visualization
    """
    global center_depth_processor
    
    try:
        # Initialize center depth processor if not already done
        if not center_depth_processor:
            logger.info("[CAMERA_API] Initializing center depth processor...")
            center_depth_processor = CenterDepthProcessor()
            center_depth_processor.start()
            logger.info("[CAMERA_API] Center depth processor started")
        
        # Get the depth camera from the center depth processor
        if center_depth_processor and center_depth_processor.depth_camera:
            depth_camera = center_depth_processor.depth_camera
            
            # Get latest frame from depth camera
            frame = depth_camera.get_latest_frame()
            if frame and frame.depth_frame is not None:
                # Use real depth data
                depth_normalized = cv2.normalize(frame.depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
        # Fallback to mock depth if no real depth camera available
        logger.warning("[CAMERA_API] No depth camera available, using mock depth")
        return await get_mock_depth()
        
    except Exception as e:
        logger.error(f"[CAMERA_API] Error getting depth data: {e}")
        return await get_mock_depth()

async def get_mock_depth():
    """Get a mock depth map for testing when no real depth camera is available"""
    # Create a simple mock depth map
    height, width = 480, 640
    depth_map = np.zeros((height, width), dtype=np.uint16)
    
    # Create a simple depth pattern
    for i in range(height):
        for j in range(width):
            # Create a simple depth pattern
            depth_value = int(500 + (i + j) * 0.5)
            depth_map[i, j] = min(depth_value, 2000)
    
    # Normalize and colorize depth map
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/mock", summary="Get Mock Camera Feed", description="Get a mock camera frame for testing when no real camera is available")
async def get_mock_camera():
    """
    Get a mock camera frame for testing when no real camera is available.
    
    This endpoint generates a synthetic test pattern for development and testing.
    Useful when no physical camera is connected or for debugging purposes.
    
    **Mock Pattern Features:**
    - Blue horizontal stripes (moving pattern)
    - Red vertical stripes (moving pattern)
    - "MOCK CAMERA" text overlay
    - "No Real Camera Connected" status message
    
    **Response:**
    - Returns JPEG image data
    - Content-Type: image/jpeg
    - 640x480 resolution
    - Consistent test pattern for reliable testing
    """
    import numpy as np
    
    # Create a simple test pattern
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some test pattern
    for i in range(0, height, 20):
        frame[i:i+10, :] = [100, 150, 200]  # Blue stripes
    
    for j in range(0, width, 30):
        frame[:, j:j+15] = [200, 100, 150]  # Red stripes
    
    # Add text
    import cv2
    cv2.putText(frame, "MOCK CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "No Real Camera Connected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/status", response_model=CameraStatus, summary="Get Camera Status", description="Get comprehensive camera system status including availability, connections, and streaming state")
async def get_camera_status():
    """
    Get comprehensive camera system status.
    
    This endpoint provides detailed information about the camera system state,
    including processor status, connection counts, and streaming activity.
    
    **Status Information:**
    - Camera availability and running state
    - Frame processing count
    - Active WebSocket connections
    - Streaming task status
    
    **Response Fields:**
    - `available`: Whether camera hardware is available
    - `running`: Whether camera processing is active
    - `frame_count`: Total frames processed
    - `websocket_connections`: Number of active WebSocket clients
    - `streaming_active`: Whether real-time streaming is running
    
    **Use Cases:**
    - Health monitoring and diagnostics
    - Debugging connection issues
    - Monitoring system performance
    - Client connection management
    """
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
    """
    Real-time WebSocket streaming of camera data.
    
    This WebSocket endpoint provides live streaming of:
    - Raw RGB camera frames
    - Depth data (raw uint16 arrays)
    - Vehicle detection results
    - Frame metadata and timestamps
    
    **Stream Features:**
    - Real-time camera feed (~10 FPS)
    - Base64-encoded image data
    - Raw depth data with metadata
    - Vehicle detection with bounding boxes
    - Automatic reconnection handling
    
    **Message Format:**
    ```json
    {
        "type": "camera_frame",
        "frame_id": 123,
        "timestamp": 1697034190.123,
        "raw_frame": "base64_encoded_jpeg",
        "tracking_frame": "base64_encoded_jpeg",
        "car_count": 2,
        "car_detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "class_name": "car",
                "class_id": 2
            }
        ],
        "depth_data": {
            "data": "base64_encoded_uint16_array",
            "metadata": {
                "width": 640,
                "height": 480,
                "format": "z16_le",
                "dtype": "uint16"
            }
        },
        "frame_count": 1234
    }
    ```
    
    **Client Messages:**
    - Send `{"action": "ping"}` for keepalive
    - Server responds with `{"action": "pong", "timestamp": ...}`
    
    **Connection Management:**
    - Automatic connection cleanup on disconnect
    - Streaming stops when no clients connected
    - Handles multiple concurrent connections
    """
    
    try:
        logger.info(f"[CAMERA_WS] Accepting WebSocket connection...")
        await websocket.accept()
        logger.info(f"[CAMERA_WS] WebSocket accepted successfully")
        
        active_connections.add(websocket)
        logger.info(f"[CAMERA_WS] Client connected. Total connections: {len(active_connections)}")
        
        # Start streaming task if not already running
        global streaming_task
        if not streaming_task or streaming_task.done():
            logger.info(f"[CAMERA_WS] Starting streaming task...")
            streaming_task = asyncio.create_task(stream_camera_data())
        else:
            logger.info(f"[CAMERA_WS] Streaming task already running")
    except Exception as e:
        logger.error(f"[CAMERA_WS] Error in WebSocket setup: {e}")
        import traceback
        logger.error(f"[CAMERA_WS] Traceback: {traceback.format_exc()}")
        return
    
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
                logger.info(f"[CAMERA_WS] Client disconnected normally")
                break
            except Exception as e:
                logger.error(f"[CAMERA_WS] Error in message handling: {e}")
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
    global simple_camera, car_classifier, center_depth_processor
    
    logger.info("[CAMERA_STREAM] Starting camera streaming task...")
    
    # Initialize center depth processor if not already done
    if not center_depth_processor:
        try:
            logger.info("[CAMERA_STREAM] Initializing center depth processor...")
            center_depth_processor = CenterDepthProcessor()
            logger.info(f"[CAMERA_STREAM] CenterDepthProcessor created, depth_camera={center_depth_processor.depth_camera is not None}")
            
            if center_depth_processor.depth_camera:
                logger.info(f"[CAMERA_STREAM] Depth camera initialized: {center_depth_processor.depth_camera.is_initialized()}")
                center_depth_processor.start()
                logger.info("[CAMERA_STREAM] Center depth processor started")
                
                # Wait a moment for it to start capturing
                await asyncio.sleep(2)
                logger.info("[CAMERA_STREAM] Center depth processor should be capturing now")
            else:
                logger.error("[CAMERA_STREAM] Depth camera not available")
                center_depth_processor = None
                
        except Exception as e:
            logger.error(f"[CAMERA_STREAM] Failed to initialize center depth processor: {e}")
            import traceback
            logger.error(f"[CAMERA_STREAM] Traceback: {traceback.format_exc()}")
            center_depth_processor = None
    
    # Initialize real camera
    try:
        simple_camera = SimpleCamera(width=640, height=480, fps=30, camera_index=0)  # Use camera 0
        simple_camera.start()
        await asyncio.sleep(1.0)  # Let camera stabilize
        logger.info("[CAMERA_STREAM] Real camera initialized successfully")
    except Exception as e:
        logger.error(f"[CAMERA_STREAM] Failed to initialize real camera: {e}")
        logger.info("[CAMERA_STREAM] Falling back to mock camera")
        simple_camera = None
    
    # Initialize car classifier
    try:
        car_classifier = VehicleClassifier()
        if car_classifier.initialize():
            logger.info("[CAMERA_STREAM] Car classifier initialized successfully")
        else:
            logger.warning("[CAMERA_STREAM] Failed to initialize car classifier")
            car_classifier = None
    except Exception as e:
        logger.error(f"[CAMERA_STREAM] Error initializing car classifier: {e}")
        car_classifier = None
    
    frame_count = 0
    
    try:
        logger.info("[CAMERA_STREAM] Starting main streaming loop...")
        while True:  # Keep running forever, not just while active_connections
            try:
                # Skip if no connections
                if not active_connections:
                    await asyncio.sleep(0.1)
                    continue
                
                logger.debug(f"[CAMERA_STREAM] Processing frame {frame_count}, connections: {len(active_connections)}")
                
                # Get real camera frame or fallback to mock
                frame = None
                if center_depth_processor and center_depth_processor.depth_camera:
                    # Use RealSense camera for both RGB and depth
                    depth_camera = center_depth_processor.depth_camera
                    frame = depth_camera.get_latest_frame()
                    if frame:
                        logger.debug(f"[CAMERA_STREAM] Using RealSense frame: color={frame.color_frame is not None}, depth={frame.depth_frame is not None}")
                
                if not frame and simple_camera:
                    # Fallback to SimpleCamera if available
                    frame = simple_camera.get_latest_frame()
                    if frame:
                        logger.debug(f"[CAMERA_STREAM] Using SimpleCamera frame")
                
                if not frame:
                    # Fallback to mock frame
                    frame = create_mock_frame(frame_count)
                    logger.debug(f"[CAMERA_STREAM] Using mock frame")
                
                # Encode frames as JPEG
                raw_data = None
                tracking_data = None
                
                if frame.color_frame is not None:
                    # Raw frame
                    _, raw_buffer = cv2.imencode('.jpg', frame.color_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    raw_data = base64.b64encode(raw_buffer).decode('utf-8')
                    
                    # Use same frame for tracking
                    tracking_data = raw_data
                
                # Handle depth data - try to get from CenterDepthProcessor first
                depth_data = None
                depth_metadata = None
                if center_depth_processor and center_depth_processor.depth_camera:
                    depth_camera = center_depth_processor.depth_camera
                    depth_frame = depth_camera.get_latest_frame()
                    if depth_frame and depth_frame.depth_frame is not None:
                        # Send raw depth data as uint16 array
                        depth_array = depth_frame.depth_frame.astype(np.uint16)
                        depth_bytes = depth_array.tobytes()
                        depth_data = base64.b64encode(depth_bytes).decode('utf-8')
                        depth_metadata = {
                            "width": depth_array.shape[1],
                            "height": depth_array.shape[0],
                            "format": "z16_le",
                            "dtype": "uint16"
                        }
                        logger.debug(f"[CAMERA_STREAM] Using real depth data: {depth_array.shape}")
                    else:
                        logger.warning(f"[CAMERA_STREAM] No depth frame available from depth camera")
                else:
                    logger.warning(f"[CAMERA_STREAM] CenterDepthProcessor not available: processor={center_depth_processor is not None}, camera={center_depth_processor.depth_camera is not None if center_depth_processor else False}")
                
                # Fallback to mock depth if no real depth data available
                if depth_data is None and frame.color_frame is not None:
                    # Generate mock depth from color frame
                    height, width = frame.color_frame.shape[:2]
                    depth_map = np.zeros((height, width), dtype=np.uint16)
                    
                    # Convert color frame to grayscale for depth estimation
                    gray = cv2.cvtColor(frame.color_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Create depth based on image features
                    # Use edge detection to find object boundaries
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Use bilateral filter to smooth while preserving edges
                    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
                    
                    # Create depth map based on brightness and edges
                    for i in range(height):
                        for j in range(width):
                            # Base depth on brightness (darker = closer)
                            brightness = smoothed[i, j]
                            
                            # Add depth variation based on edges
                            if edges[i, j] > 0:
                                # Edges are typically closer (lower depth value)
                                depth_value = int(300 + (255 - brightness) * 1.5)
                            else:
                                # Smooth areas are farther (higher depth value)
                                depth_value = int(800 + (255 - brightness) * 1.2)
                            
                            # Add some noise for realism (using deterministic noise based on position)
                            noise = ((i + j) % 100 - 50)  # Deterministic noise
                            depth_value = max(200, min(2000, depth_value + noise))
                            depth_map[i, j] = depth_value
                    
                    # Send raw depth data as uint16 array
                    depth_bytes = depth_map.tobytes()
                    depth_data = base64.b64encode(depth_bytes).decode('utf-8')
                    depth_metadata = {
                        "width": width,
                        "height": height,
                        "format": "z16_le",
                        "dtype": "uint16"
                    }
                
                # Perform car detection if classifier is available
                car_count = 0
                car_detections = []
                
                if car_classifier and frame.color_frame is not None:
                    try:
                        detections = car_classifier.detect(frame.color_frame)
                        car_count = len(detections)
                        car_detections = []
                        
                        for detection in detections:
                            car_detections.append({
                                "bbox": [int(detection.bbox[0]), int(detection.bbox[1]), 
                                        int(detection.bbox[2]), int(detection.bbox[3])],
                                "confidence": float(detection.confidence),
                                "class_name": detection.class_name,
                                "class_id": int(detection.class_id)
                            })
                    except Exception as e:
                        logger.error(f"[CAMERA_STREAM] Error in car detection: {e}")
                        car_count = 0
                        car_detections = []
                
                # Create message with actual image data
                message = {
                    "type": "camera_frame",
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "raw_frame": raw_data,
                    "tracking_frame": tracking_data,
                    "car_count": car_count,
                    "car_detections": car_detections,
                    "frame_count": frame_count
                }
                
                # Add depth data if available
                if depth_data and depth_metadata:
                    message["depth_data"] = {
                        "data": depth_data,
                        "metadata": depth_metadata
                    }
                
                # Send to all connected clients
                disconnected = set()
                for websocket in list(active_connections):
                    try:
                        await websocket.send_text(json.dumps(message))
                        logger.debug(f"[CAMERA_STREAM] Sent frame {frame_count} to client")
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


def create_mock_frame(frame_count):
    """Create a mock camera frame for testing"""
    import numpy as np
    
    # Create a simple test pattern
    width, height = 640, 480
    color_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add animated test pattern
    time_offset = frame_count * 0.1
    
    # Moving stripes
    for i in range(0, height, 20):
        stripe_color = int(128 + 127 * np.sin(time_offset + i * 0.1))
        color_frame[i:i+10, :] = [stripe_color//3, stripe_color//2, stripe_color]
    
    # Moving vertical stripes
    for j in range(0, width, 30):
        stripe_color = int(128 + 127 * np.sin(time_offset + j * 0.05))
        color_frame[:, j:j+15] = [stripe_color, stripe_color//2, stripe_color//3]
    
    # Add animated text
    cv2.putText(color_frame, "MOCK CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(color_frame, f"Frame: {frame_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(color_frame, "WebSocket Streaming Active", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create a simple mock depth frame
    depth_frame = np.zeros((height, width), dtype=np.uint16)
    for i in range(height):
        for j in range(width):
            depth_frame[i, j] = int(1000 + 500 * np.sin(i * 0.01) * np.cos(j * 0.01))
    
    # Create mock frame object
    class MockFrame:
        def __init__(self, color_frame, depth_frame, frame_id, timestamp):
            self.color_frame = color_frame
            self.depth_frame = depth_frame
            self.frame_id = frame_id
            self.timestamp = timestamp
    
    return MockFrame(color_frame, depth_frame, frame_count, time.time())
