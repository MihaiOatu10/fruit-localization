"""
Unit test for PointCloudProcessor.
"""

from pathlib import Path
import numpy as np
import pytest
from core.pointcloud import PointCloudProcessor


def test_pointcloud_processor_reproject():
    """Validates 3D coordinate reprojection from disparity map and Q matrix."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "geometry_config.yaml"
    processor = PointCloudProcessor(config_path)

    disp = np.full((128, 128), 25.0, dtype=np.float32)
    Q = np.eye(4, dtype=np.float64)
    Q[3, 2] = -1.0 / 74.676

    pts_3d = processor.reproject_disparity_to_3d(disp, Q)
    assert pts_3d.shape == (128, 128, 3)
    assert np.isfinite(pts_3d).any()


def test_pointcloud_voxel_and_instance():
    """Validates mask gating and voxel downsampling."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "geometry_config.yaml"
    processor = PointCloudProcessor(config_path)

    points = np.random.uniform(low=[10.0, 10.0, 300.0], high=[50.0, 50.0, 350.0], size=(200, 3)).astype(np.float32)
    downsampled = processor.voxel_grid_downsample(points, voxel_size=2.0)

    assert len(downsampled) > 0
    assert len(downsampled) <= len(points)
    assert downsampled.shape[1] == 3


def test_pointcloud_lof_and_threshold():
    """Validates LOF outlier removal and minimum point thresholding."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "geometry_config.yaml"
    processor = PointCloudProcessor(config_path)

    # 100 inlier points on sphere surface
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    x = 35.0 * np.sin(theta) * np.cos(phi)
    y = 35.0 * np.sin(theta) * np.sin(phi)
    z = 350.0 + 35.0 * np.cos(theta)
    pts = np.column_stack([x, y, z]).astype(np.float32)

    # Add 5 distant outliers
    outliers = np.array([[100.0, 100.0, 1000.0], [-100.0, -100.0, 50.0]] * 3, dtype=np.float32)
    pts_with_noise = np.vstack([pts, outliers])

    cleaned = processor.filter_outliers_lof(pts_with_noise)
    assert len(cleaned) <= len(pts_with_noise)
    assert len(cleaned) >= 10



def test_pointcloud_fallback_envelope_reads_config():
    """The broad depth fallback must use pointcloud.fallback_z_*_mm, not hardcoded limits."""
    points_3d = np.zeros((10, 10, 3), dtype=np.float32)
    points_3d[..., 2] = 4000.0  # Beyond z_max_mm, so the strict envelope keeps nothing
    mask = np.ones((10, 10), dtype=bool)

    narrow = PointCloudProcessor({"pointcloud": {"z_max_mm": 1000.0, "fallback_z_max_mm": 3500.0}})
    assert len(narrow.extract_instance_points(points_3d, mask)) == 0

    wide = PointCloudProcessor({"pointcloud": {"z_max_mm": 1000.0, "fallback_z_max_mm": 4500.0}})
    assert len(wide.extract_instance_points(points_3d, mask)) == 100


def test_pointcloud_surface_band_removes_background_bleed():
    """A large far cluster (leaves behind the mask edge) must be gated out, unlike with 5% LOF contamination."""
    rng = np.random.default_rng(0)
    fruit = rng.normal([0.0, 0.0, 600.0], [15.0, 15.0, 5.0], (700, 3))
    leaves = rng.normal([10.0, 0.0, 850.0], [20.0, 20.0, 30.0], (300, 3))
    processor = PointCloudProcessor({"pointcloud": {"surface_band_mm": 40.0, "surface_band_pad_mm": 5.0}})

    gated = processor.gate_surface_band(np.vstack([fruit, leaves]).astype(np.float32))
    assert len(gated) >= 0.95 * len(fruit)
    assert gated[:, 2].max() < 650.0

    assert len(PointCloudProcessor({"pointcloud": {"surface_band_mm": 0.0}}).gate_surface_band(leaves)) == len(leaves)
