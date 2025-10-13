#!/usr/bin/env python3

"""
Cleanup script for vehicle tracking storage maintenance.

This script can be run as a cron job to automatically clean up
old video clips and manage storage space.
"""

import argparse
import logging
import sys
from datetime import datetime

# Add jarvis module to path
sys.path.insert(0, '/app')

from jarvis.video.clip_manager import cleanup_old_clips, get_storage_stats

def main():
    """Main cleanup script"""
    parser = argparse.ArgumentParser(description="Clean up old vehicle tracking files")
    parser.add_argument(
        "--days", 
        type=int, 
        default=7, 
        help="Delete files older than this many days (default: 7)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting cleanup: days={args.days}, dry_run={args.dry_run}")
    
    try:
        # Get storage stats before cleanup
        stats_before = get_storage_stats()
        logger.info(f"Storage before cleanup: {stats_before}")
        
        # Run cleanup
        cleanup_stats = cleanup_old_clips(
            days_old=args.days,
            dry_run=args.dry_run
        )
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        
        # Get storage stats after cleanup
        if not args.dry_run:
            stats_after = get_storage_stats()
            logger.info(f"Storage after cleanup: {stats_after}")
            
            space_freed = stats_before['total_size_mb'] - stats_after['total_size_mb']
            logger.info(f"Space freed: {space_freed:.1f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
