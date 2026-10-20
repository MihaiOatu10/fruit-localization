"""
Stereo Perception Module Interface.
Provides stereo disparity estimation and sky suppression routines.
"""

from core.stereo_matcher import StereoMatcherSGBM

StereoMatcher = StereoMatcherSGBM

__all__ = ["StereoMatcherSGBM", "StereoMatcher"]
