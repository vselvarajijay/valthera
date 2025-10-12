#!/usr/bin/env python3

"""
Database layer for vehicle tracking using DuckDB.
"""

from .vehicle_db import (
    init_db, find_match, add_vehicle, update_vehicle,
    get_all_vehicles, get_vehicle_timeline, get_vehicle_stats,
    cleanup_db
)

__all__ = [
    'init_db', 'find_match', 'add_vehicle', 'update_vehicle',
    'get_all_vehicles', 'get_vehicle_timeline', 'get_vehicle_stats',
    'cleanup_db'
]
