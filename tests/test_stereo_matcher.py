"""
Unit test for StereoMatcherSGBM and Disparity Estimation.
"""

from pathlib import Path
import cv2
import torch
import numpy as np
import pytest
import yaml
from core.stereo_matcher import StereoMatcherSGBM

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "stereo_sgbm.yaml"
RAFT_CKPT = Path(__file__).resolve().parent.parent / "weights" / "raftstereo-middlebury.pth"
HAS_RAFT = RAFT_CKPT.is_file() and torch.cuda.is_available()


def make_matcher(engine: str) -> StereoMatcherSGBM:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["engine"] = engine
    matcher = StereoMatcherSGBM(cfg)
    if engine == "raft":
        assert matcher.raft_engine is not None and matcher.raft_engine.initialized
    return matcher


def test_stereo_matcher_disparity_computation():
    """Validates that StereoMatcherSGBM computes disparity and returns expected shapes and types."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "stereo_sgbm.yaml"
    matcher = StereoMatcherSGBM(config_path)

    # Test with dummy stereo pair
    img_l = np.zeros((512, 512, 3), dtype=np.uint8)
    img_r = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add a bright square shifted horizontally by 10 pixels
    cv2_rect_l = img_l.copy()
    cv2_rect_l[100:200, 100:200] = 255
    cv2_rect_r = img_r.copy()
    cv2_rect_r[100:200, 90:190] = 255

    disp, valid_mask = matcher.compute_disparity(cv2_rect_l, cv2_rect_r)

    assert disp.shape == (512, 512)
    assert valid_mask.shape == (512, 512)
    assert disp.dtype == np.float32
    assert valid_mask.dtype == bool


def test_stereo_matcher_depth_conversion():
    """Validates conversion from disparity to metric depth."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "stereo_sgbm.yaml"
    matcher = StereoMatcherSGBM(config_path)

    disp = np.array([[20.0, 0.0], [-1.0, 40.0]], dtype=np.float32)
    focal_length = 1000.0  # px
    baseline_mm = 75.0     # mm
    depth = matcher.compute_depth(disp, focal_length_px=focal_length, baseline_mm=baseline_mm)

    # Z = (f * B) / d
    # (1000 * 75) / 20 = 3750 mm
    assert np.isclose(depth[0, 0], 3750.0)
    assert np.isnan(depth[0, 1])  # 0 disparity -> NaN
    assert np.isnan(depth[1, 0])  # negative disparity -> NaN
    assert np.isclose(depth[1, 1], 1875.0)


@pytest.mark.parametrize(
    "engine",
    ["sgbm", pytest.param("raft", marks=pytest.mark.skipif(not HAS_RAFT, reason="needs CUDA + RAFT-Stereo checkpoint"))],
)
@pytest.mark.parametrize(
    "shift_px, sky_level",
    [
        (2, 250),   # Glare (>= luminance_threshold) over a far, near-zero-disparity scene
        (20, 254),  # Whiteout (>= 253) is suppressed regardless of surrounding disparity
    ],
)
def test_stereo_matcher_sky_suppression(shift_px, sky_level, engine):
    """Validates that overexposed saturated sky regions are invalidated while textured scene survives."""
    if engine == "raft" and shift_px < 4:
        pytest.skip("near-zero shift on iid synthetic noise is out of distribution for RAFT-Stereo")
    matcher = make_matcher(engine)

    # Width must exceed num_disparities (256) or SGBM yields no valid pixels and the test is vacuous
    rng = np.random.default_rng(0)
    texture = np.repeat(np.repeat(rng.integers(0, 200, (100, 280), dtype=np.uint8), 2, 0), 2, 1)
    img_l = np.dstack([texture[:, :512]] * 3)
    img_r = np.dstack([texture[:, shift_px:512 + shift_px]] * 3)
    img_l[:50, :] = sky_level
    img_r[:50, :] = sky_level

    disp, valid_mask = matcher.compute_disparity(img_l, img_r)
    # Saturated white region should have valid_mask == False
    assert not np.any(valid_mask[:45, :])
    # Textured scene below the sky keeps valid disparity at the true shift
    scene_valid = valid_mask[60:, 300:]
    assert scene_valid.mean() > 0.9
    assert np.isclose(np.median(disp[60:, 300:][scene_valid]), shift_px, atol=0.5)


def test_stereo_matcher_narrow_frame_skips_wls():
    """Frames narrower than the disparity search range must not crash the WLS filter."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "stereo_sgbm.yaml"
    matcher = StereoMatcherSGBM(config_path)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    disp, valid_mask = matcher.compute_disparity(img, img)
    assert disp.shape == (200, 200)
    assert valid_mask.dtype == bool


def test_stereo_matcher_mask_refinement():
    """Validates that mask refinement smooths and fills disparity inside fruit mask."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "stereo_sgbm.yaml"
    matcher = StereoMatcherSGBM(config_path)

    img_l = np.ones((128, 128, 3), dtype=np.uint8) * 120
    img_r = np.ones((128, 128, 3), dtype=np.uint8) * 120

    # Add circle feature shifted by 15 pixels
    cv2.circle(img_l, (64, 64), 25, (200, 50, 50), -1)
    cv2.circle(img_r, (49, 64), 25, (200, 50, 50), -1)

    fruit_mask = np.zeros((128, 128), dtype=bool)
    fruit_mask[44:84, 44:84] = True

    disp, valid_mask = matcher.compute_disparity(img_l, img_r, protected_mask=fruit_mask)
    assert disp.shape == (128, 128)
    assert np.any(valid_mask[fruit_mask])



@pytest.mark.skipif(not HAS_RAFT, reason="needs CUDA + RAFT-Stereo checkpoint")
def test_raft_stereo_runs_on_cuda():
    """Validates the official RAFT-Stereo checkpoint loads strictly onto the GPU and recovers a known shift."""
    matcher = make_matcher("raft")
    assert next(matcher.raft_engine.model.parameters()).is_cuda

    rng = np.random.default_rng(1)
    texture = np.repeat(np.repeat(rng.integers(0, 255, (160, 320), dtype=np.uint8), 2, 0), 2, 1)
    shift_px = 16
    # x_R = x_L - d: right view shows left content shifted left by d
    img_l = np.dstack([texture[:, :512]] * 3)
    img_r = np.dstack([texture[:, shift_px:shift_px + 512]] * 3)

    disp, valid_mask = matcher.compute_disparity(img_l, img_r)
    assert disp.shape == (320, 512) and disp.dtype == np.float32
    # Columns left of the shift have no correspondence in the right view
    assert not valid_mask[:, :shift_px - 1].any()
    core_valid = valid_mask[20:-20, 64:-20]
    assert core_valid.mean() > 0.95
    assert np.isclose(np.median(disp[20:-20, 64:-20][core_valid]), shift_px, atol=0.5)
