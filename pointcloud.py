"""
Root wrapper for PointCloudProcessor.
Re-exports classes from core.pointcloud for module resolution convenience.
"""

from core.pointcloud import PointCloudProcessor

__all__ = ["PointCloudProcessor"]
