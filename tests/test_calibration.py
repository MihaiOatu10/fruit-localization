"""
Unit Test for Stereo Calibration and Reprojection Matrix Calculation.
"""

import numpy as np
import pytest
from core.calibration import StereoCalibrator


def test_calibration_q_matrix():
    """Validates the structure and properties of the Q matrix from camera calibration."""
    calibrator = StereoCalibrator("configs/camera_calib_zheng2023.yaml")
    Q = calibrator.get_q_matrix()

    assert Q.shape == (4, 4), f"Unexpected Q shape: {Q.shape}"
    # In standardized Q matrix, Q[3, 2] is -1/B
    baseline = calibrator.get_baseline_mm()
    assert baseline > 70.0 and baseline < 80.0, f"Expected baseline near 74.68mm, got {baseline}"

    # Verify that focal length is positive
    focal_length = calibrator.get_focal_length()
    assert focal_length > 500.0, f"Expected focal length > 500, got {focal_length}"


def test_calibration_matrix_dimensions():
    """Validates that all internal matrices are strictly coerced to expected shapes and types."""
    calibrator = StereoCalibrator("configs/camera_calib.yaml")
    assert calibrator.K1.shape == (3, 3)
    assert calibrator.K2.shape == (3, 3)
    assert calibrator.R.shape == (3, 3)
    assert calibrator.T.shape == (3, 1)
    assert calibrator.D1.ndim == 2 and calibrator.D1.shape[1] == 1
    assert calibrator.D2.ndim == 2 and calibrator.D2.shape[1] == 1
    assert calibrator.K1.dtype == np.float64
    assert calibrator.K2.dtype == np.float64
    assert calibrator.R.dtype == np.float64
    assert calibrator.T.dtype == np.float64
    assert calibrator.D1.dtype == np.float64
    assert calibrator.D2.dtype == np.float64


def test_calibration_mock_flat_fixtures():
    """
    Validates that StereoCalibrator handles flat (1D) or uncoerced camera matrices,
    rotation matrices, translation vectors, and distortion coefficients without
    OpenCV assertion errors.
    """
    mock_config = {
        "image_width": 1024,
        "image_height": 1024,
        "left_camera": {
            "camera_matrix": [1122.7, 0.0, 325.9, 0.0, 1122.5, 234.4, 0.0, 0.0, 1.0],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "right_camera": {
            "camera_matrix": [1130.6, 0.0, 330.5, 0.0, 1130.4, 259.7, 0.0, 0.0, 1.0],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "extrinsics": {
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "T": [-74.676, -0.168, 0.490],
            "baseline_mm": 74.676,
        },
    }
    calibrator = StereoCalibrator(mock_config)
    assert calibrator.K1.shape == (3, 3)
    assert calibrator.K2.shape == (3, 3)
    assert calibrator.R.shape == (3, 3)
    assert calibrator.T.shape == (3, 1)
    assert calibrator.D1.shape == (5, 1)
    assert calibrator.D2.shape == (5, 1)

    Q = calibrator.get_q_matrix()
    assert Q.shape == (4, 4)


def test_calibration_rectify_pair():
    """Validates epipolar rectification on dummy image pairs and resizing behavior."""
    calibrator = StereoCalibrator("configs/camera_calib_zheng2023.yaml")
    dummy_l = np.zeros((1024, 1024, 3), dtype=np.uint8)
    dummy_r = np.zeros((1024, 1024, 3), dtype=np.uint8)
    rect_l, rect_r = calibrator.rectify_pair(dummy_l, dummy_r)
    assert rect_l.shape == (1024, 1024, 3)
    assert rect_r.shape == (1024, 1024, 3)

    # Test automatic resize when input dimension differs
    non_standard_l = np.zeros((720, 1280, 3), dtype=np.uint8)
    non_standard_r = np.zeros((720, 1280, 3), dtype=np.uint8)
    rect_l2, rect_r2 = calibrator.rectify_pair(non_standard_l, non_standard_r)
    assert rect_l2.shape == (1024, 1024, 3)
    assert rect_r2.shape == (1024, 1024, 3)


def test_calibration_size_6_coercion():
    """
    Specifically validates that StereoCalibrator smoothly handles arrays of size 6
    (e.g., 2x3 affine camera matrices or 2x3 rotation matrices) without raising
    'ValueError: cannot reshape array of size 6 into shape (3,3)'.
    """
    config_size_6 = {
        "image_width": 1024,
        "image_height": 1024,
        "left_camera": {
            # 2x3 matrix (size 6)
            "camera_matrix": [
                [1122.7, 0.0, 325.9],
                [0.0, 1122.5, 234.4],
            ],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "right_camera": {
            # Flat list of 6 elements
            "camera_matrix": [1130.6, 0.0, 330.5, 0.0, 1130.4, 259.7],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "extrinsics": {
            # 2x3 rotation matrix (size 6)
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "T": [-74.676, -0.168, 0.490],
            "baseline_mm": 74.676,
        },
    }

    calibrator = StereoCalibrator(config_size_6)
    assert calibrator.K1.shape == (3, 3)
    assert calibrator.K2.shape == (3, 3)
    assert calibrator.R.shape == (3, 3)
    assert calibrator.T.shape == (3, 1)

    # Validate that bottom row was properly constructed
    assert np.allclose(calibrator.K1[2], [0.0, 0.0, 1.0])
    assert np.allclose(calibrator.K2[2], [0.0, 0.0, 1.0])
    assert np.allclose(calibrator.R[2], [0.0, 0.0, 1.0])

    Q = calibrator.get_q_matrix()
    assert Q.shape == (4, 4)


def test_calibration_prerectified_zed2i():
    """Validates passthrough of already rectified ZED 2i frames and the Q built from Table 2 of the kiwi paper."""
    calibrator = StereoCalibrator("configs/camera_calib.yaml")
    assert calibrator.already_rectified

    img_l = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    img_r = img_l[:, ::-1].copy()
    rect_l, rect_r = calibrator.rectify_pair(img_l, img_r)
    assert rect_l is img_l and rect_r is img_r

    with pytest.raises(ValueError):
        calibrator.rectify_pair(np.zeros((1024, 1024, 3), np.uint8), np.zeros((1024, 1024, 3), np.uint8))

    # A point at known (X, Y, Z) must reproject exactly through Q
    fx, fy, cx, cy = calibrator.K1[0, 0], calibrator.K1[1, 1], calibrator.K1[0, 2], calibrator.K1[1, 2]
    B = calibrator.get_baseline_mm()
    assert np.isclose(B, 119.671) and np.isclose(fx, 614.925)
    X, Y, Z = 120.0, -80.0, 700.0
    u, v, d = cx + fx * X / Z, cy + fy * Y / Z, fx * B / Z
    xyzw = calibrator.get_q_matrix() @ np.array([u, v, d, 1.0])
    assert np.allclose(xyzw[:3] / xyzw[3], [X, Y, Z], atol=1e-6)
