#!/usr/bin/env python3

"""
DuckDB interface for vehicle tracking data.

This module provides database operations for storing and retrieving
vehicle detection data, embeddings, and metadata.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import threading

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "/data/vehicle_tracker/vehicles.duckdb"
DATA_DIR = "/data/vehicle_tracker"
CROPS_DIR = os.path.join(DATA_DIR, "crops")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MERGED_DIR = os.path.join(DATA_DIR, "merged")

# Global connection
_connection: Optional[duckdb.DuckDBPyConnection] = None
_connection_lock = threading.Lock()


def _ensure_data_dirs():
    """Ensure data directories exist"""
    for directory in [DATA_DIR, CROPS_DIR, CLIPS_DIR, MERGED_DIR]:
        os.makedirs(directory, exist_ok=True)


def init_db() -> Optional[duckdb.DuckDBPyConnection]:
    """
    Initialize DuckDB connection and create schema.
    
    Returns:
        DuckDB connection object, or None if failed
    """
    global _connection
    
    if not DUCKDB_AVAILABLE:
        logger.error("[DB] DuckDB not available")
        return None
    
    with _connection_lock:
        if _connection is not None:
            return _connection
        
        try:
            # Ensure data directories exist
            _ensure_data_dirs()
            
            # Connect to database
            _connection = duckdb.connect(DB_PATH)
            
            # Create schema
            _connection.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    vehicle_id INTEGER PRIMARY KEY,
                    embedding FLOAT[],
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    count INTEGER DEFAULT 1,
                    thumbnail_paths TEXT DEFAULT '',
                    clip_paths TEXT DEFAULT ''
                )
            """)
            
            # Create index on embedding for faster similarity search
            try:
                _connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_vehicles_embedding 
                    ON vehicles USING ivfflat (embedding) WITH (metric = 'cosine')
                """)
            except Exception as e:
                # Index creation might fail on some DuckDB versions
                logger.warning(f"[DB] Could not create embedding index: {e}")
            
            logger.info(f"[DB] Database initialized: {DB_PATH}")
            return _connection
            
        except Exception as e:
            logger.error(f"[DB] Failed to initialize database: {e}")
            _connection = None
            return None


def find_match(embedding: np.ndarray, threshold: float = 0.7) -> Optional[int]:
    """
    Find matching vehicle by embedding similarity.
    
    Args:
        embedding: Vehicle embedding vector
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Vehicle ID if match found, None otherwise
    """
    if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("[DB] Required libraries not available for similarity search")
        return None
    
    conn = init_db()
    if not conn:
        return None
    
    try:
        # Get all embeddings
        result = conn.execute("SELECT vehicle_id, embedding FROM vehicles").fetchall()
        
        if not result:
            return None
        
        # Extract embeddings and IDs
        vehicle_ids = [row[0] for row in result]
        embeddings = [row[1] for row in result]
        
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings)
        query_embedding = embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            logger.debug(f"[DB] Found match: vehicle_id={vehicle_ids[best_idx]}, similarity={best_similarity:.3f}")
            return vehicle_ids[best_idx]
        else:
            logger.debug(f"[DB] No match found: best similarity={best_similarity:.3f} < threshold={threshold}")
            return None
            
    except Exception as e:
        logger.error(f"[DB] Error finding match: {e}")
        return None


def add_vehicle(embedding: np.ndarray, thumbnail_path: str, clip_path: str) -> Optional[int]:
    """
    Add new vehicle to database.
    
    Args:
        embedding: Vehicle embedding vector
        thumbnail_path: Path to thumbnail image
        clip_path: Path to video clip
        
    Returns:
        New vehicle ID, or None if failed
    """
    conn = init_db()
    if not conn:
        return None
    
    try:
        now = datetime.now()
        
        # Insert new vehicle
        result = conn.execute("""
            INSERT INTO vehicles (embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths)
            VALUES (?, ?, ?, 1, ?, ?)
            RETURNING vehicle_id
        """, (
            embedding.tolist(),
            now,
            now,
            thumbnail_path,
            clip_path
        )).fetchone()
        
        vehicle_id = result[0]
        logger.info(f"[DB] Added new vehicle: ID={vehicle_id}")
        return vehicle_id
        
    except Exception as e:
        logger.error(f"[DB] Error adding vehicle: {e}")
        return None


def update_vehicle(vehicle_id: int, thumbnail_path: str, clip_path: str) -> bool:
    """
    Update existing vehicle with new sighting.
    
    Args:
        vehicle_id: Vehicle ID to update
        thumbnail_path: Path to new thumbnail image
        clip_path: Path to new video clip
        
    Returns:
        True if successful, False otherwise
    """
    conn = init_db()
    if not conn:
        return False
    
    try:
        now = datetime.now()
        
        # Update vehicle record
        conn.execute("""
            UPDATE vehicles
            SET last_seen = ?,
                count = count + 1,
                thumbnail_paths = thumbnail_paths || ',' || ?,
                clip_paths = clip_paths || ',' || ?
            WHERE vehicle_id = ?
        """, (now, thumbnail_path, clip_path, vehicle_id))
        
        logger.info(f"[DB] Updated vehicle: ID={vehicle_id}")
        return True
        
    except Exception as e:
        logger.error(f"[DB] Error updating vehicle {vehicle_id}: {e}")
        return False


def get_all_vehicles() -> List[Dict[str, Any]]:
    """
    Get all vehicles from database.
    
    Returns:
        List of vehicle dictionaries
    """
    conn = init_db()
    if not conn:
        return []
    
    try:
        result = conn.execute("""
            SELECT vehicle_id, embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths
            FROM vehicles
            ORDER BY last_seen DESC
        """).fetchall()
        
        vehicles = []
        for row in result:
            vehicle = {
                'vehicle_id': row[0],
                'embedding': row[1],
                'first_seen': row[2],
                'last_seen': row[3],
                'count': row[4],
                'thumbnail_paths': row[5].split(',') if row[5] else [],
                'clip_paths': row[6].split(',') if row[6] else []
            }
            vehicles.append(vehicle)
        
        logger.debug(f"[DB] Retrieved {len(vehicles)} vehicles")
        return vehicles
        
    except Exception as e:
        logger.error(f"[DB] Error getting all vehicles: {e}")
        return []


def get_vehicle_timeline(vehicle_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed timeline for a specific vehicle.
    
    Args:
        vehicle_id: Vehicle ID
        
    Returns:
        Vehicle timeline dictionary, or None if not found
    """
    conn = init_db()
    if not conn:
        return None
    
    try:
        result = conn.execute("""
            SELECT vehicle_id, embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths
            FROM vehicles
            WHERE vehicle_id = ?
        """, (vehicle_id,)).fetchone()
        
        if not result:
            return None
        
        vehicle = {
            'vehicle_id': result[0],
            'embedding': result[1],
            'first_seen': result[2],
            'last_seen': result[3],
            'count': result[4],
            'thumbnail_paths': result[5].split(',') if result[5] else [],
            'clip_paths': result[6].split(',') if result[6] else []
        }
        
        return vehicle
        
    except Exception as e:
        logger.error(f"[DB] Error getting vehicle timeline {vehicle_id}: {e}")
        return None


def get_vehicle_stats() -> Dict[str, Any]:
    """
    Get summary statistics for all vehicles.
    
    Returns:
        Statistics dictionary
    """
    conn = init_db()
    if not conn:
        return {}
    
    try:
        # Total vehicles
        total_result = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()
        total_vehicles = total_result[0] if total_result else 0
        
        # Returning vehicles (seen more than once)
        returning_result = conn.execute("SELECT COUNT(*) FROM vehicles WHERE count > 1").fetchone()
        returning_vehicles = returning_result[0] if returning_result else 0
        
        # Recent activity (last 24 hours)
        recent_result = conn.execute("""
            SELECT COUNT(*) FROM vehicles 
            WHERE last_seen >= datetime('now', '-1 day')
        """).fetchone()
        recent_vehicles = recent_result[0] if recent_result else 0
        
        # Average sightings per vehicle
        avg_result = conn.execute("SELECT AVG(count) FROM vehicles").fetchone()
        avg_sightings = avg_result[0] if avg_result else 0
        
        stats = {
            'total_vehicles': total_vehicles,
            'returning_vehicles': returning_vehicles,
            'recent_vehicles': recent_vehicles,
            'avg_sightings_per_vehicle': round(avg_sightings, 2) if avg_sightings else 0,
            'return_rate': round((returning_vehicles / total_vehicles * 100), 2) if total_vehicles > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"[DB] Error getting vehicle stats: {e}")
        return {}


def cleanup_db():
    """Cleanup database connection"""
    global _connection
    
    with _connection_lock:
        if _connection:
            try:
                _connection.close()
                logger.info("[DB] Database connection closed")
            except Exception as e:
                logger.error(f"[DB] Error closing database: {e}")
            finally:
                _connection = None


def main():
    """Test the database operations"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[DB] Testing database operations...")
    
    # Initialize database
    conn = init_db()
    if not conn:
        logger.error("[DB] Failed to initialize database")
        return
    
    # Test with dummy data
    dummy_embedding = np.random.rand(512)  # Typical OSNet embedding size
    
    # Add test vehicle
    vehicle_id = add_vehicle(dummy_embedding, "test_thumb.jpg", "test_clip.mp4")
    if vehicle_id:
        logger.info(f"[DB] Added test vehicle: ID={vehicle_id}")
        
        # Test finding match
        match_id = find_match(dummy_embedding)
        if match_id == vehicle_id:
            logger.info("[DB] Match test passed")
        else:
            logger.error(f"[DB] Match test failed: expected {vehicle_id}, got {match_id}")
        
        # Test update
        if update_vehicle(vehicle_id, "test_thumb2.jpg", "test_clip2.mp4"):
            logger.info("[DB] Update test passed")
        else:
            logger.error("[DB] Update test failed")
        
        # Test retrieval
        vehicles = get_all_vehicles()
        logger.info(f"[DB] Retrieved {len(vehicles)} vehicles")
        
        # Test stats
        stats = get_vehicle_stats()
        logger.info(f"[DB] Stats: {stats}")
    
    cleanup_db()
    logger.info("[DB] Test completed")


if __name__ == "__main__":
    main()
