"""
Root wrapper for FruitSegmenter and DetectedInstance.
Re-exports classes from core.segmentation for module resolution convenience.
"""

from core.segmentation import FruitSegmenter, DetectedInstance

__all__ = ["FruitSegmenter", "DetectedInstance"]
