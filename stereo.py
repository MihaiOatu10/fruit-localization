"""
Root wrapper for StereoMatcherSGBM.
Re-exports classes from core.stereo_matcher for module resolution convenience.
"""

from core.stereo_matcher import StereoMatcherSGBM

StereoMatcher = StereoMatcherSGBM

__all__ = ["StereoMatcherSGBM", "StereoMatcher"]
