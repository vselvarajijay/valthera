#!/usr/bin/env python3

"""
Unified DuckDB interface for vehicle and people tracking data.

This module provides database operations for storing and retrieving
tracked object data (vehicles and people), embeddings, and metadata
using a single unified database schema.
"""

import logging
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import threading
from enum import Enum

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

class ObjectType(Enum):
    """Supported object types for tracking"""
    VEHICLE = "vehicle"
    PERSON = "person"

# Database configuration
DB_PATH = "/data/tracker/objects.duckdb"
DATA_DIR = "/data/tracker"
VEHICLE_CROPS_DIR = os.path.join(DATA_DIR, "vehicles", "crops")
VEHICLE_CLIPS_DIR = os.path.join(DATA_DIR, "vehicles", "clips")
VEHICLE_MERGED_DIR = os.path.join(DATA_DIR, "vehicles", "merged")
PERSON_CROPS_DIR = os.path.join(DATA_DIR, "people", "crops")
PERSON_CLIPS_DIR = os.path.join(DATA_DIR, "people", "clips")
PERSON_MERGED_DIR = os.path.join(DATA_DIR, "people", "merged")

# Global connection
_connection: Optional[duckdb.DuckDBPyConnection] = None
_connection_lock = threading.Lock()


def _ensure_data_dirs():
    """Ensure data directories exist"""
    directories = [
        DATA_DIR, VEHICLE_CROPS_DIR, VEHICLE_CLIPS_DIR, VEHICLE_MERGED_DIR,
        PERSON_CROPS_DIR, PERSON_CLIPS_DIR, PERSON_MERGED_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def init_db() -> Optional[duckdb.DuckDBPyConnection]:
    """
    Initialize DuckDB connection and create unified schema.
    
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
            
            # Create unified schema
            _connection.execute("""
                CREATE TABLE IF NOT EXISTS tracked_objects (
                    object_id INTEGER PRIMARY KEY,
                    object_type TEXT NOT NULL,
                    embedding FLOAT[],
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    count INTEGER DEFAULT 1,
                    thumbnail_paths TEXT DEFAULT '',
                    clip_paths TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create index on embedding for faster similarity search
            try:
                _connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_objects_embedding 
                    ON tracked_objects USING ivfflat (embedding) WITH (metric = 'cosine')
                """)
            except Exception as e:
                # Index creation might fail on some DuckDB versions
                logger.warning(f"[DB] Could not create embedding index: {e}")
            
            # Create index on object_type for faster filtering
            try:
                _connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_objects_type 
                    ON tracked_objects (object_type)
                """)
            except Exception as e:
                logger.warning(f"[DB] Could not create type index: {e}")
            
            logger.info(f"[DB] Unified database initialized: {DB_PATH}")
            return _connection
            
        except Exception as e:
            logger.error(f"[DB] Failed to initialize unified database: {e}")
            _connection = None
            return None


def find_match(embedding: np.ndarray, object_type: str, threshold: float = 0.7) -> Optional[int]:
    """
    Find matching object by embedding similarity.
    
    Args:
        embedding: Object embedding vector
        object_type: Type of object ('vehicle' or 'person')
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Object ID if match found, None otherwise
    """
    if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("[DB] Required libraries not available for similarity search")
        return None
    
    conn = init_db()
    if not conn:
        return None
    
    try:
        # Get all embeddings for the specific object type
        result = conn.execute("""
            SELECT object_id, embedding 
            FROM tracked_objects 
            WHERE object_type = ?
        """, (object_type,)).fetchall()
        
        if not result:
            return None
        
        # Extract embeddings and IDs
        object_ids = [row[0] for row in result]
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
            logger.debug(f"[DB] Found match: object_id={object_ids[best_idx]}, similarity={best_similarity:.3f}")
            return object_ids[best_idx]
        else:
            logger.debug(f"[DB] No match found: best similarity={best_similarity:.3f} < threshold={threshold}")
            return None
            
    except Exception as e:
        logger.error(f"[DB] Error finding match: {e}")
        return None


def add_object(embedding: Optional[np.ndarray], object_type: str, thumbnail_path: str, clip_path: str, metadata: Optional[Dict] = None) -> Optional[int]:
    """
    Add new object to database.
    
    Args:
        embedding: Object embedding vector
        object_type: Type of object ('vehicle' or 'person')
        thumbnail_path: Path to thumbnail image
        clip_path: Path to video clip
        metadata: Optional metadata dictionary
        
    Returns:
        New object ID, or None if failed
    """
    conn = init_db()
    if not conn:
        return None
    
    try:
        now = datetime.now()
        metadata_json = json.dumps(metadata or {})
        
        # Insert new object
        result = conn.execute("""
            INSERT INTO tracked_objects (object_type, embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths, metadata)
            VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            RETURNING object_id
        """, (
            object_type,
            embedding.tolist() if embedding is not None else None,
            now,
            now,
            thumbnail_path,
            clip_path,
            metadata_json
        )).fetchone()
        
        object_id = result[0]
        logger.info(f"[DB] Added new {object_type}: ID={object_id}")
        return object_id
        
    except Exception as e:
        logger.error(f"[DB] Error adding {object_type}: {e}")
        return None


def update_object(object_id: int, object_type: str, thumbnail_path: str, clip_path: str) -> bool:
    """
    Update existing object with new sighting.
    
    Args:
        object_id: Object ID to update
        object_type: Type of object ('vehicle' or 'person')
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
        
        # Update object record
        conn.execute("""
            UPDATE tracked_objects
            SET last_seen = ?,
                count = count + 1,
                thumbnail_paths = thumbnail_paths || ',' || ?,
                clip_paths = clip_paths || ',' || ?
            WHERE object_id = ? AND object_type = ?
        """, (now, thumbnail_path, clip_path, object_id, object_type))
        
        logger.info(f"[DB] Updated {object_type}: ID={object_id}")
        return True
        
    except Exception as e:
        logger.error(f"[DB] Error updating {object_type} {object_id}: {e}")
        return False


def get_all_objects(object_type: str) -> List[Dict[str, Any]]:
    """
    Get all objects of a specific type from database.
    
    Args:
        object_type: Type of objects to retrieve ('vehicle' or 'person')
        
    Returns:
        List of object dictionaries
    """
    conn = init_db()
    if not conn:
        return []
    
    try:
        result = conn.execute("""
            SELECT object_id, embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths, metadata
            FROM tracked_objects
            WHERE object_type = ?
            ORDER BY last_seen DESC
        """, (object_type,)).fetchall()
        
        objects = []
        for row in result:
            obj = {
                f'{object_type}_id': row[0],  # Dynamic field name for API compatibility
                'object_id': row[0],
                'object_type': object_type,
                'embedding': row[1],
                'first_seen': row[2],
                'last_seen': row[3],
                'count': row[4],
                'thumbnail_paths': row[5].split(',') if row[5] else [],
                'clip_paths': row[6].split(',') if row[6] else [],
                'metadata': json.loads(row[7]) if row[7] else {}
            }
            objects.append(obj)
        
        logger.debug(f"[DB] Retrieved {len(objects)} {object_type}s")
        return objects
        
    except Exception as e:
        logger.error(f"[DB] Error getting all {object_type}s: {e}")
        return []


def get_object_timeline(object_id: int, object_type: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed timeline for a specific object.
    
    Args:
        object_id: Object ID
        object_type: Type of object ('vehicle' or 'person')
        
    Returns:
        Object timeline dictionary, or None if not found
    """
    conn = init_db()
    if not conn:
        return None
    
    try:
        result = conn.execute("""
            SELECT object_id, embedding, first_seen, last_seen, count, thumbnail_paths, clip_paths, metadata
            FROM tracked_objects
            WHERE object_id = ? AND object_type = ?
        """, (object_id, object_type)).fetchone()
        
        if not result:
            return None
        
        obj = {
            f'{object_type}_id': result[0],  # Dynamic field name for API compatibility
            'object_id': result[0],
            'object_type': object_type,
            'embedding': result[1],
            'first_seen': result[2],
            'last_seen': result[3],
            'count': result[4],
            'thumbnail_paths': result[5].split(',') if result[5] else [],
            'clip_paths': result[6].split(',') if result[6] else [],
            'metadata': json.loads(result[7]) if result[7] else {}
        }
        
        return obj
        
    except Exception as e:
        logger.error(f"[DB] Error getting {object_type} timeline {object_id}: {e}")
        return None


def get_object_stats(object_type: str) -> Dict[str, Any]:
    """
    Get summary statistics for objects of a specific type.
    
    Args:
        object_type: Type of objects ('vehicle' or 'person')
        
    Returns:
        Statistics dictionary
    """
    conn = init_db()
    if not conn:
        return {}
    
    try:
        # Total objects
        total_result = conn.execute("""
            SELECT COUNT(*) FROM tracked_objects WHERE object_type = ?
        """, (object_type,)).fetchone()
        total_objects = total_result[0] if total_result else 0
        
        # Returning objects (seen more than once)
        returning_result = conn.execute("""
            SELECT COUNT(*) FROM tracked_objects 
            WHERE object_type = ? AND count > 1
        """, (object_type,)).fetchone()
        returning_objects = returning_result[0] if returning_result else 0
        
        # Recent activity (last 24 hours)
        recent_result = conn.execute("""
            SELECT COUNT(*) FROM tracked_objects 
            WHERE object_type = ? AND last_seen >= current_timestamp - interval '1 day'
        """, (object_type,)).fetchone()
        recent_objects = recent_result[0] if recent_result else 0
        
        # Average sightings per object
        avg_result = conn.execute("""
            SELECT AVG(count) FROM tracked_objects WHERE object_type = ?
        """, (object_type,)).fetchone()
        avg_sightings = avg_result[0] if avg_result else 0
        
        stats = {
            f'total_{object_type}s': total_objects,
            f'returning_{object_type}s': returning_objects,
            f'recent_{object_type}s': recent_objects,
            f'avg_sightings_per_{object_type}': round(avg_sightings, 2) if avg_sightings else 0,
            'return_rate': round((returning_objects / total_objects * 100), 2) if total_objects > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"[DB] Error getting {object_type} stats: {e}")
        return {}


def get_storage_dirs(object_type: str) -> Dict[str, str]:
    """
    Get storage directories for a specific object type.
    
    Args:
        object_type: Type of object ('vehicle' or 'person')
        
    Returns:
        Dictionary with directory paths
    """
    if object_type == ObjectType.VEHICLE.value:
        return {
            'crops_dir': VEHICLE_CROPS_DIR,
            'clips_dir': VEHICLE_CLIPS_DIR,
            'merged_dir': VEHICLE_MERGED_DIR
        }
    elif object_type == ObjectType.PERSON.value:
        return {
            'crops_dir': PERSON_CROPS_DIR,
            'clips_dir': PERSON_CLIPS_DIR,
            'merged_dir': PERSON_MERGED_DIR
        }
    else:
        raise ValueError(f"Unknown object type: {object_type}")


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
    """Test the unified database operations"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("[DB] Testing unified database operations...")
    
    # Initialize database
    conn = init_db()
    if not conn:
        logger.error("[DB] Failed to initialize database")
        return
    
    # Test with dummy data
    dummy_embedding = np.random.rand(512)  # Typical OSNet embedding size
    
    # Test vehicle operations
    vehicle_id = add_object(dummy_embedding, ObjectType.VEHICLE.value, "test_vehicle_thumb.jpg", "test_vehicle_clip.mp4")
    if vehicle_id:
        logger.info(f"[DB] Added test vehicle: ID={vehicle_id}")
        
        # Test finding match
        match_id = find_match(dummy_embedding, ObjectType.VEHICLE.value)
        if match_id == vehicle_id:
            logger.info("[DB] Vehicle match test passed")
        else:
            logger.error(f"[DB] Vehicle match test failed: expected {vehicle_id}, got {match_id}")
    
    # Test person operations
    person_id = add_object(dummy_embedding, ObjectType.PERSON.value, "test_person_thumb.jpg", "test_person_clip.mp4")
    if person_id:
        logger.info(f"[DB] Added test person: ID={person_id}")
        
        # Test finding match
        match_id = find_match(dummy_embedding, ObjectType.PERSON.value)
        if match_id == person_id:
            logger.info("[DB] Person match test passed")
        else:
            logger.error(f"[DB] Person match test failed: expected {person_id}, got {match_id}")
    
    # Test retrieval
    vehicles = get_all_objects(ObjectType.VEHICLE.value)
    people = get_all_objects(ObjectType.PERSON.value)
    logger.info(f"[DB] Retrieved {len(vehicles)} vehicles and {len(people)} people")
    
    # Test stats
    vehicle_stats = get_object_stats(ObjectType.VEHICLE.value)
    person_stats = get_object_stats(ObjectType.PERSON.value)
    logger.info(f"[DB] Vehicle stats: {vehicle_stats}")
    logger.info(f"[DB] Person stats: {person_stats}")
    
    cleanup_db()
    logger.info("[DB] Test completed")


if __name__ == "__main__":
    main()
