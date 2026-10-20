"""
Unit and Integration Test for End-to-End Tomato Localization Pipeline.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from pipeline import TomatoLocalizationPipeline


def test_pipeline_synthetic_dry_run():
    """Validates that pipeline runs to completion on synthetic stereo frames."""
    repo_root = Path(__file__).resolve().parent.parent
    pipeline = TomatoLocalizationPipeline(
        calib_config_path=str(repo_root / "configs" / "camera_calib.yaml"),
        stereo_config_path=str(repo_root / "configs" / "stereo_sgbm.yaml"),
        geometry_config_path=str(repo_root / "configs" / "geometry_config.yaml"),
        model_weights=None,
        device="cpu",
    )

    img_l = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_r = np.zeros((720, 1280, 3), dtype=np.uint8)

    results = pipeline.process_pair(img_l, img_r)

    assert "rectified_left" in results
    assert "rectified_right" in results
    assert "disparity_map" in results
    assert "valid_mask" in results
    assert "instances" in results
    assert "localized_fruits" in results
    assert "telemetry" in results

    assert results["rectified_left"].shape == (720, 1280, 3)
    assert results["disparity_map"].shape == (720, 1280)
    assert isinstance(results["telemetry"]["total_latency_ms"], float)


def test_pipeline_holdout_dry_run():
    """Validates end-to-end execution on actual stereo image pair from test_holdout."""
    repo_root = Path(__file__).resolve().parent.parent
    left_path = repo_root / "data" / "test_holdout" / "left" / "images" / "00400.png"
    right_path = repo_root / "data" / "test_holdout" / "right" / "images" / "00400.png"
    if not left_path.is_file() or not right_path.is_file():
        # Same raw stereo pair as shipped in the repository
        left_path = repo_root / "stereo-original-image" / "left" / "00400.png"
        right_path = repo_root / "stereo-original-image" / "right" / "00400.png"

    if not left_path.is_file() or not right_path.is_file():
        pytest.skip(f"Holdout images not found at {left_path}")

    img_l = cv2.imread(str(left_path))
    img_r = cv2.imread(str(right_path))
    assert img_l is not None and img_r is not None

    pipeline = TomatoLocalizationPipeline(
        calib_config_path=str(repo_root / "configs" / "camera_calib.yaml"),
        stereo_config_path=str(repo_root / "configs" / "stereo_sgbm.yaml"),
        geometry_config_path=str(repo_root / "configs" / "geometry_config.yaml"),
        model_weights=None,
        device="cpu",
    )

    results = pipeline.process_pair(img_l, img_r)

    assert "rectified_left" in results
    assert "rectified_right" in results
    assert "disparity_map" in results
    assert "localized_fruits" in results
    assert "telemetry" in results
    assert results["rectified_left"].shape == (720, 1280, 3)
    assert results["disparity_map"].shape == (720, 1280)
    assert results["telemetry"]["total_latency_ms"] > 0
