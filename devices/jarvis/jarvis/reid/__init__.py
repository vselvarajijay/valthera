#!/usr/bin/env python3

"""
ReID module for vehicle re-identification using OSNet embeddings.
"""

from .osnet_extractor import OSNetExtractor, get_embedding

__all__ = ['OSNetExtractor', 'get_embedding']
