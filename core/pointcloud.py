"""
3D Point Cloud Processing & Filtering Module.
Reprojects 2D disparity to camera coordinates, applies mask gating,
voxel grid downsampling, and Local Outlier Factor (LOF) filtering.
"""

from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import yaml
import numpy as np
import cv2
from sklearn.neighbors import LocalOutlierFactor


class PointCloudProcessor:
    """
    Handles 3D coordinate reconstruction and conditioning.
    Replicates equations (3), (4), and (5) from Zheng et al. (Agronomy 2023).
    """

    def __init__(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None):
        if config_source is None:
            self.config = {}
        elif isinstance(config_source, (str, Path)):
            config_path = Path(config_source)
            if not config_path.is_file():
                alt_path = Path(__file__).resolve().parent.parent / config_path
                if alt_path.is_file():
                    config_path = alt_path
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_source or {}

        pc_cfg = self.config.get("pointcloud", {})
        self.z_min_mm = float(pc_cfg.get("z_min_mm", 150.0))
        self.z_max_mm = float(pc_cfg.get("z_max_mm", 2500.0))
        self.fallback_z_min_mm = float(pc_cfg.get("fallback_z_min_mm", 100.0))
        self.fallback_z_max_mm = float(pc_cfg.get("fallback_z_max_mm", 3500.0))
        self.voxel_size_mm = float(pc_cfg.get("voxel_size_mm", 2.0))
        self.min_points = int(pc_cfg.get("min_points_threshold", 20))
        # Visible fruit surface spans at most ~one radius along the viewing ray; anything outside the densest
        # such band is leaf/sky bleed behind the mask edge or an occluder in front (0 disables)
        self.surface_band_mm = float(pc_cfg.get("surface_band_mm", 40.0))
        self.surface_band_pad_mm = float(pc_cfg.get("surface_band_pad_mm", 5.0))

        outlier_cfg = self.config.get("outlier_removal", {})
        self.outlier_method = outlier_cfg.get("method", "lof")
        self.lof_neighbors = int(outlier_cfg.get("lof_neighbors", 20))
        self.lof_contamination = float(outlier_cfg.get("lof_contamination", 0.05))

    def reproject_disparity_to_3d(
        self, 
        disparity_map: np.ndarray, 
        Q_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Reprojects 2D disparity map to full-scene 3D camera coordinates (X, Y, Z in mm).
        Uses OpenCV reprojectImageTo3D or explicit homogeneous perspective transform.
        """
        disp = np.asarray(disparity_map, dtype=np.float32)
        Q = np.asarray(Q_matrix, dtype=np.float64).reshape((4, 4))
        points_3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
        return points_3d

    def extract_instance_points(
        self,
        points_3d: np.ndarray,
        instance_mask: np.ndarray,
        valid_disp_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extracts 3D coordinates falling inside the instance mask and depth envelope.
        Handles mask alignment, missing value sentinels, optical axis sign, and depth bounds.
        """
        inst_mask = np.asarray(instance_mask, dtype=bool)
        h, w = points_3d.shape[:2]

        if inst_mask.shape != (h, w):
            inst_mask = cv2.resize(
                inst_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Intersect with valid disparity mask if sufficient points exist
        if valid_disp_mask is not None:
            val_mask = np.asarray(valid_disp_mask, dtype=bool)
            if val_mask.shape != (h, w):
                val_mask = cv2.resize(
                    val_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            combined_mask = inst_mask & val_mask
            # If combined mask retains at least 8 pixels, use it; otherwise fall back to inst_mask
            target_mask = combined_mask if np.sum(combined_mask) >= 8 else inst_mask
        else:
            target_mask = inst_mask

        # Index points
        pts = points_3d[target_mask]  # Shape: (N, 3)

        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 1. Filter out NaNs and Infs
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 2. Filter out OpenCV missing value sentinels (10000.0) and near-zero noise
        z_vals = pts[:, 2]
        sentinel_mask = (np.abs(z_vals) < 9990.0) & (np.abs(z_vals) > 10.0)
        pts = pts[sentinel_mask]
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 3. Handle camera coordinate axis sign: ensure positive depth (Z > 0)
        if np.median(pts[:, 2]) < 0:
            pts[:, 2] = -pts[:, 2]

        # 4. Filter within configured depth envelope
        z_coords = pts[:, 2]
        depth_mask = (z_coords >= self.z_min_mm) & (z_coords <= self.z_max_mm)
        valid_pts = pts[depth_mask]

        # Fallback to broader realistic picking envelope if strict bounds eliminated points
        if len(valid_pts) < 10 and len(pts) >= 10:
            broad_mask = (z_coords >= self.fallback_z_min_mm) & (z_coords <= self.fallback_z_max_mm)
            valid_pts = pts[broad_mask]

        return valid_pts

    def gate_surface_band(self, points: np.ndarray, band_mm: Optional[float] = None) -> np.ndarray:
        """
        Keeps points whose range along the viewing ray falls in the densest window of width band_mm
        (plus padding). Unlike LOF, this removes contaminating clusters that make up a large share of the mask.
        """
        band = self.surface_band_mm if band_mm is None else band_mm
        if band <= 0.0 or len(points) < self.min_points:
            return points

        ranges = np.linalg.norm(points, axis=1)
        sorted_r = np.sort(ranges)
        counts = np.searchsorted(sorted_r, sorted_r + band, side="right") - np.arange(len(sorted_r))
        start = sorted_r[int(np.argmax(counts))]
        keep = (ranges >= start - self.surface_band_pad_mm) & (ranges <= start + band + self.surface_band_pad_mm)
        return points[keep]

    def voxel_grid_downsample(self, points: np.ndarray, voxel_size: Optional[float] = None) -> np.ndarray:
        """
        Replicates Paper Equation (3):
        Divides points into cubic voxels and computes the centroid (X_c, Y_c, Z_c)
        for every non-empty voxel.
        """
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        v_size = voxel_size if voxel_size is not None else self.voxel_size_mm
        if v_size <= 0.0 or len(points) < 10:
            return points

        # Discretize coordinates into integer voxel indices
        min_bound = np.min(points, axis=0)
        voxel_coords = np.floor((points - min_bound) / v_size).astype(np.int64)

        # Hash coordinates for grouping
        # Map 3D index (vx, vy, vz) to unique integer key
        vx, vy, vz = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
        # Quantized voxel bounds
        vx_range = np.max(vx) + 1
        vy_range = np.max(vy) + 1
        keys = vx + vy * vx_range + vz * (vx_range * vy_range)

        # Compute centroid per unique voxel key
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        downsampled = np.zeros((len(unique_keys), 3), dtype=np.float32)

        # Sum coordinates and counts per voxel
        np.add.at(downsampled, inverse_indices, points)
        counts = np.bincount(inverse_indices, minlength=len(unique_keys))[:, np.newaxis]
        downsampled /= counts

        return downsampled

    def filter_outliers_lof(
        self, 
        points: np.ndarray, 
        n_neighbors: Optional[int] = None, 
        contamination: Optional[float] = None
    ) -> np.ndarray:
        """
        Replicates Paper Equations (4) and (5):
        Filters sparse outliers using Local Outlier Factor (LOF).
        """
        if len(points) < self.min_points:
            return points

        k = n_neighbors or self.lof_neighbors
        contam = contamination or self.lof_contamination

        # Ensure k is smaller than total points
        k = min(k, len(points) - 1)
        if k < 5:
            return points

        # kd_tree: 'auto' may pick brute force, whose BLAS/OpenMP kNN kernel segfaults on some Windows builds
        lof = LocalOutlierFactor(n_neighbors=k, contamination=contam, algorithm="kd_tree")
        y_pred = lof.fit_predict(points)

        # Inlier points have label 1; outliers have label -1
        inliers = points[y_pred == 1]
        return inliers

    def process_instance(
        self,
        points_3d: np.ndarray,
        instance_mask: np.ndarray,
        valid_disp_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        End-to-end instance point cloud processing:
        Extracts masked points -> Surface depth-band gating -> Voxel downsampling -> LOF outlier removal.
        Returns:
            cleaned_points: (M, 3) float32 array
            is_valid: bool indicating if point density is sufficient for fitting
        """
        raw_pts = self.extract_instance_points(points_3d, instance_mask, valid_disp_mask)
        if len(raw_pts) < 10:
            return raw_pts, False

        gated = self.gate_surface_band(raw_pts)
        if len(gated) >= 10:
            raw_pts = gated

        # Voxel downsampling (Eq. 3)
        downsampled = self.voxel_grid_downsample(raw_pts)
        if len(downsampled) < 10:
            downsampled = raw_pts

        # LOF outlier filtering (Eq. 4 & 5)
        cleaned = self.filter_outliers_lof(downsampled)
        if len(cleaned) < 10:
            cleaned = downsampled

        is_valid = len(cleaned) >= 10
        return cleaned, is_valid
