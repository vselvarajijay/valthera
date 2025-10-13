#!/usr/bin/env python3

"""
WebSocket connection manager for Jarvis smart CV pipeline.

Manages WebSocket connection lifecycle and broadcasting.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Enhanced WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Accept a WebSocket connection with optional metadata."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            self._connection_metadata[websocket] = metadata or {}
        
        logger.info(f"[WS_CONNECTION_MANAGER] Client connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if websocket in self._connection_metadata:
                del self._connection_metadata[websocket]
        
        logger.info(f"[WS_CONNECTION_MANAGER] Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket) -> bool:
        """Send message to a specific WebSocket. Returns success status."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
                return True
            else:
                await self.disconnect(websocket)
                return False
        except Exception as e:
            logger.error(f"[WS_CONNECTION_MANAGER] Error sending personal message: {e}")
            await self.disconnect(websocket)
            return False
    
    async def send_personal_json(self, data: Dict[str, Any], websocket: WebSocket) -> bool:
        """Send JSON message to a specific WebSocket."""
        import json
        return await self.send_personal_message(json.dumps(data), websocket)
    
    async def broadcast(self, message: str) -> int:
        """Broadcast message to all connected WebSockets. Returns number of successful sends."""
        async with self._lock:
            disconnected = []
            successful_sends = 0
            
            for connection in self.active_connections:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_text(message)
                        successful_sends += 1
                    else:
                        disconnected.append(connection)
                except Exception as e:
                    logger.error(f"[WS_CONNECTION_MANAGER] Error broadcasting to connection: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
                if connection in self._connection_metadata:
                    del self._connection_metadata[connection]
            
            return successful_sends
    
    async def broadcast_json(self, data: Dict[str, Any]) -> int:
        """Broadcast JSON message to all connected WebSockets."""
        import json
        return await self.broadcast(json.dumps(data))
    
    async def broadcast_to_group(self, message: str, group_name: str) -> int:
        """Broadcast message to connections in a specific group."""
        async with self._lock:
            successful_sends = 0
            disconnected = []
            
            for connection in self.active_connections:
                metadata = self._connection_metadata.get(connection, {})
                if metadata.get("group") == group_name:
                    try:
                        if connection.client_state == WebSocketState.CONNECTED:
                            await connection.send_text(message)
                            successful_sends += 1
                        else:
                            disconnected.append(connection)
                    except Exception as e:
                        logger.error(f"[WS_CONNECTION_MANAGER] Error broadcasting to group {group_name}: {e}")
                        disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
                if connection in self._connection_metadata:
                    del self._connection_metadata[connection]
            
            return successful_sends
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_connection_metadata(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get metadata for a specific connection."""
        return self._connection_metadata.get(websocket, {})
    
    def set_connection_metadata(self, websocket: WebSocket, metadata: Dict[str, Any]) -> None:
        """Set metadata for a specific connection."""
        if websocket in self.active_connections:
            self._connection_metadata[websocket] = metadata
    
    def get_connections_by_group(self, group_name: str) -> List[WebSocket]:
        """Get all connections in a specific group."""
        connections = []
        for connection in self.active_connections:
            metadata = self._connection_metadata.get(connection, {})
            if metadata.get("group") == group_name:
                connections.append(connection)
        return connections
    
    def get_group_count(self, group_name: str) -> int:
        """Get number of connections in a specific group."""
        return len(self.get_connections_by_group(group_name))
    
    def get_all_groups(self) -> List[str]:
        """Get list of all active groups."""
        groups = set()
        for metadata in self._connection_metadata.values():
            group = metadata.get("group")
            if group:
                groups.add(group)
        return list(groups)
    
    async def cleanup_disconnected(self) -> int:
        """Clean up disconnected connections. Returns number cleaned."""
        async with self._lock:
            disconnected = []
            
            for connection in self.active_connections:
                if connection.client_state != WebSocketState.CONNECTED:
                    disconnected.append(connection)
            
            for connection in disconnected:
                self.active_connections.remove(connection)
                if connection in self._connection_metadata:
                    del self._connection_metadata[connection]
            
            if disconnected:
                logger.info(f"[WS_CONNECTION_MANAGER] Cleaned up {len(disconnected)} disconnected connections")
            
            return len(disconnected)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        async with self._lock:
            groups = {}
            for metadata in self._connection_metadata.values():
                group = metadata.get("group", "default")
                groups[group] = groups.get(group, 0) + 1
            
            return {
                "total_connections": len(self.active_connections),
                "groups": groups,
                "active_groups": len(groups)
            }
