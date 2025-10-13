#!/usr/bin/env python3

"""
Video utilities for saving and merging vehicle detection clips.
"""

from .clip_manager import (
    save_clip, merge_clips, cleanup_old_clips,
    CROPS_DIR, CLIPS_DIR, MERGED_DIR
)

__all__ = [
    'save_clip', 'merge_clips', 'cleanup_old_clips',
    'CROPS_DIR', 'CLIPS_DIR', 'MERGED_DIR'
]
