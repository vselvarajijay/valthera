#!/usr/bin/env python3

"""
WebSocket handlers for Jarvis smart CV pipeline.

Clean WebSocket handlers that use streaming use cases
and manage connection lifecycle properly.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from jarvis.infrastructure.container import (
    camera_service_dependency,
    detection_service_dependency,
    depth_service_dependency
)
from jarvis.domain.services.camera_service import ICameraService
from jarvis.domain.services.detection_service import IDetectionService
from jarvis.domain.services.depth_service import IDepthService
from jarvis.application.use_cases.stream_frames import StreamFramesUseCase, StreamFramesRequest, StreamFrameResult

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"[CONNECTION_MANAGER] Client connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"[CONNECTION_MANAGER] Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send message to a specific WebSocket."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
        except Exception as e:
            logger.error(f"[CONNECTION_MANAGER] Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: str) -> None:
        """Broadcast message to all connected WebSockets."""
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_text(message)
                    else:
                        disconnected.append(connection)
                except Exception as e:
                    logger.error(f"[CONNECTION_MANAGER] Error broadcasting to connection: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


class StreamHandler:
    """Handles WebSocket streaming operations."""
    
    def __init__(
        self,
        camera_service: ICameraService,
        detection_service: IDetectionService,
        depth_service: IDepthService
    ):
        self._camera_service = camera_service
        self._detection_service = detection_service
        self._depth_service = depth_service
        self._connection_manager = ConnectionManager()
        self._streaming_task: Optional[asyncio.Task] = None
        self._is_streaming = False
    
    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection."""
        await self._connection_manager.connect(websocket)
        
        try:
            while True:
                # Wait for client message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await self._handle_message(websocket, message)
                
        except WebSocketDisconnect:
            logger.info("[STREAM_HANDLER] WebSocket disconnected")
        except Exception as e:
            logger.error(f"[STREAM_HANDLER] WebSocket error: {e}")
        finally:
            await self._connection_manager.disconnect(websocket)
    
    async def _handle_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        
        if message_type == "start_stream":
            await self._handle_start_stream(websocket, message)
        elif message_type == "stop_stream":
            await self._handle_stop_stream(websocket, message)
        elif message_type == "get_status":
            await self._handle_get_status(websocket, message)
        else:
            await self._send_error(websocket, f"Unknown message type: {message_type}")
    
    async def _handle_start_stream(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle start stream request."""
        try:
            if self._is_streaming:
                await self._send_error(websocket, "Streaming already active")
                return
            
            # Parse stream configuration
            config = message.get("config", {})
            request = StreamFramesRequest(
                enable_detection=config.get("enable_detection", True),
                enable_depth=config.get("enable_depth", True),
                classifiers=config.get("classifiers", ["person"]),
                confidence_threshold=config.get("confidence_threshold", 0.5),
                max_detections=config.get("max_detections", 10),
                fps_limit=config.get("fps_limit", 10.0),
                store_frames=config.get("store_frames", False),
                store_detections=config.get("store_detections", False)
            )
            
            # Start streaming
            await self._start_streaming(request)
            
            await self._send_success(websocket, "Streaming started")
            
        except Exception as e:
            logger.error(f"[STREAM_HANDLER] Error starting stream: {e}")
            await self._send_error(websocket, f"Failed to start stream: {str(e)}")
    
    async def _handle_stop_stream(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle stop stream request."""
        try:
            await self._stop_streaming()
            await self._send_success(websocket, "Streaming stopped")
            
        except Exception as e:
            logger.error(f"[STREAM_HANDLER] Error stopping stream: {e}")
            await self._send_error(websocket, f"Failed to stop stream: {str(e)}")
    
    async def _handle_get_status(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle get status request."""
        try:
            status = {
                "streaming": self._is_streaming,
                "connections": self._connection_manager.get_connection_count(),
                "camera_available": await self._camera_service.is_available(),
                "detection_initialized": await self._detection_service.is_initialized()
            }
            
            await self._send_status(websocket, status)
            
        except Exception as e:
            logger.error(f"[STREAM_HANDLER] Error getting status: {e}")
            await self._send_error(websocket, f"Failed to get status: {str(e)}")
    
    async def _start_streaming(self, request: StreamFramesRequest) -> None:
        """Start the streaming process."""
        self._is_streaming = True
        
        # Create streaming use case
        use_case = StreamFramesUseCase(
            self._camera_service,
            self._detection_service,
            self._depth_service
        )
        
        # Start streaming task
        self._streaming_task = asyncio.create_task(
            self._stream_loop(use_case, request)
        )
        
        logger.info("[STREAM_HANDLER] Streaming started")
    
    async def _stop_streaming(self) -> None:
        """Stop the streaming process."""
        self._is_streaming = False
        
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None
        
        logger.info("[STREAM_HANDLER] Streaming stopped")
    
    async def _stream_loop(self, use_case: StreamFramesUseCase, request: StreamFramesRequest) -> None:
        """Main streaming loop."""
        try:
            async for result in use_case.execute(request):
                if not self._is_streaming:
                    break
                
                # Convert result to JSON-serializable format
                stream_data = {
                    "type": "frame_data",
                    "data": {
                        "frame_id": str(result.frame.id),
                        "timestamp": str(result.frame.timestamp),
                        "resolution": {
                            "width": result.frame.resolution.width,
                            "height": result.frame.resolution.height
                        },
                        "detections": [detection.to_dict() for detection in result.detections],
                        "detection_count": len(result.detections),
                        "processing_time_ms": result.processing_time_ms,
                        "frame_count": result.frame_count
                    }
                }
                
                # Broadcast to all connected clients
                await self._connection_manager.broadcast(json.dumps(stream_data))
                
        except asyncio.CancelledError:
            logger.info("[STREAM_HANDLER] Streaming loop cancelled")
        except Exception as e:
            logger.error(f"[STREAM_HANDLER] Error in streaming loop: {e}")
            # Notify clients of error
            error_message = {
                "type": "error",
                "message": f"Streaming error: {str(e)}"
            }
            await self._connection_manager.broadcast(json.dumps(error_message))
        finally:
            self._is_streaming = False
    
    async def _send_success(self, websocket: WebSocket, message: str) -> None:
        """Send success message."""
        response = {
            "type": "success",
            "message": message
        }
        await self._connection_manager.send_personal_message(json.dumps(response), websocket)
    
    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send error message."""
        response = {
            "type": "error",
            "message": message
        }
        await self._connection_manager.send_personal_message(json.dumps(response), websocket)
    
    async def _send_status(self, websocket: WebSocket, status: Dict[str, Any]) -> None:
        """Send status message."""
        response = {
            "type": "status",
            "data": status
        }
        await self._connection_manager.send_personal_message(json.dumps(response), websocket)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return self._connection_manager.get_connection_count()
    
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming


# Global stream handler instance
_stream_handler_instance: Optional[StreamHandler] = None


def get_stream_handler() -> StreamHandler:
    """Get the global stream handler instance."""
    global _stream_handler_instance
    if _stream_handler_instance is None:
        # This would be injected via DI in a real implementation
        from jarvis.infrastructure.container import get_camera_service, get_detection_service, get_depth_service
        _stream_handler_instance = StreamHandler(
            get_camera_service(),
            get_detection_service(),
            get_depth_service()
        )
    return _stream_handler_instance
