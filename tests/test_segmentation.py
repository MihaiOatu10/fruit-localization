"""
Unit test for FruitSegmenter mask coordinate mapping.
"""

import cv2
import numpy as np
from core.segmentation import FruitSegmenter


def test_unletterbox_mask_preserves_vertical_position():
    """A 720x1280 mask letterboxed to 384x640 (12 px pad top/bottom) must map back without vertical squeeze."""
    h, w = 720, 1280
    truth = np.zeros((h, w), dtype=np.float32)
    truth[20:80, 600:660] = 1.0     # near top edge
    truth[640:700, 600:660] = 1.0   # near bottom edge

    letterboxed = np.zeros((384, 640), dtype=np.float32)
    letterboxed[12:372] = cv2.resize(truth, (640, 360), interpolation=cv2.INTER_LINEAR)

    restored = FruitSegmenter._unletterbox_mask(letterboxed, h, w) > 0.5
    for y0, y1 in [(20, 80), (640, 700)]:
        ys = np.nonzero(restored[:, 630])[0]
        ys = ys[(ys > y0 - 30) & (ys < y1 + 30)]
        assert abs(ys.min() - y0) <= 2 and abs(ys.max() - (y1 - 1)) <= 2
