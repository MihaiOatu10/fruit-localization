"""
Spherical Fitting Module for Fruit Depth Point Clouds.
Implements dual-phase least-squares sphere fitting:
Algebraic linear solve for initialization and non-linear Levenberg-Marquardt refinement.
"""

from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
from scipy.optimize import least_squares


@dataclass
class SphereFitResult:
    """Encapsulates fitted sphere parameters and quality metrics."""
    centroid: np.ndarray       # [xc, yc, zc] in mm
    radius_mm: float           # radius r in mm
    rmse_mm: float             # Root Mean Square Error of surface residuals
    num_points: int            # Number of points used in fit
    is_valid: bool             # Whether fitting passed error tolerance checks


class SphereFitter:
    """
    Fits 3D point cloud of fruit surfaces to a spherical model.
    Replicates Paper Equation (6) from Zheng et al. (Agronomy 2023).
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

        fit_cfg = self.config.get("sphere_fitting", {})
        self.max_iter = int(fit_cfg.get("max_iterations", 150))
        self.ftol = float(fit_cfg.get("ftol", 1.0e-7))
        self.xtol = float(fit_cfg.get("xtol", 1.0e-7))
        self.radius_bounds = tuple(fit_cfg.get("radius_bounds_mm", [15.0, 75.0]))
        self.max_rmse = float(fit_cfg.get("max_allowed_rmse_mm", 4.5))
        # Soft prior pulling r toward the cap's lateral radius; dominates only when depth curvature is
        # below disparity noise (shallow caps), which otherwise drives r onto the upper bound.
        self.radius_prior_weight = float(fit_cfg.get("radius_prior_weight", 0.3))
        # soft_l1 scale for surface residuals (mask-edge background bleed); the prior stays quadratic
        self.loss_scale_mm = float(fit_cfg.get("loss_scale_mm", 2.0))
        # Soft pull of the centre onto the viewing ray through the 2D mask centroid (a sphere's centre projects
        # to its silhouette centre); stops lateral drift onto same-depth leaves or neighbouring fruit
        self.ray_prior_weight = float(fit_cfg.get("ray_prior_weight", 0.5))
        # Physical plausibility gates applied on top of the RMSE check
        self.valid_radius_mm = tuple(float(r) for r in fit_cfg.get("valid_radius_mm", self.radius_bounds))
        self.max_ray_offset_ratio = float(fit_cfg.get("max_ray_offset_ratio", 0.5))
        # Stereo depth noise grows as Z^2: a disparity error of delta px maps to delta * Z^2 / (f * B) mm.
        # That term is added in quadrature to max_rmse; needs set_stereo_geometry() (0 = fixed threshold)
        self.rmse_disparity_noise_px = float(fit_cfg.get("rmse_disparity_noise_px", 0.0))
        self.focal_px: Optional[float] = None
        self.baseline_mm: Optional[float] = None

        # Centroid depth envelope shares the point cloud depth limits (same defaults as PointCloudProcessor)
        pc_cfg = self.config.get("pointcloud", {})
        self.z_min_mm = float(pc_cfg.get("z_min_mm", 150.0))
        self.z_max_mm = float(pc_cfg.get("z_max_mm", 2500.0))
        # Physical plausibility window applied to the fitted centroid depth by the pipeline
        self.valid_centroid_z_mm = tuple(float(z) for z in fit_cfg.get("valid_centroid_z_mm", [200.0, 5000.0]))

    def set_stereo_geometry(self, focal_px: float, baseline_mm: float) -> None:
        """Rectified focal length (px) and baseline (mm) used for depth-scaled RMSE gating."""
        self.focal_px = float(focal_px)
        self.baseline_mm = float(baseline_mm)

    def max_rmse_at(self, z_mm: float) -> float:
        """RMSE gate at centroid depth Z: sqrt(max_rmse^2 + (delta * Z^2 / (f * B))^2)."""
        if self.rmse_disparity_noise_px <= 0.0 or not self.focal_px or not self.baseline_mm:
            return self.max_rmse
        sigma_z = self.rmse_disparity_noise_px * z_mm ** 2 / (self.focal_px * self.baseline_mm)
        return float(np.hypot(self.max_rmse, sigma_z))

    def fit_algebraic(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Phase 1: Fast Algebraic Linear Least-Squares Sphere Fit.
        Expands (x - xc)^2 + (y - yc)^2 + (z - zc)^2 = r^2 into:
        x^2 + y^2 + z^2 + A*x + B*y + C*z + D = 0
        Solves [x, y, z, 1] * [A, B, C, D]^T = -(x^2 + y^2 + z^2)
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        A_mat = np.column_stack([x, y, z, np.ones_like(x)])
        f_vec = -(x**2 + y**2 + z**2)

        # Solve linear system using SVD / least squares
        sol, residuals, rank, s = np.linalg.lstsq(A_mat, f_vec, rcond=None)
        A, B, C, D = sol

        xc = -A / 2.0
        yc = -B / 2.0
        zc = -C / 2.0
        r_sq = xc**2 + yc**2 + zc**2 - D

        r = np.sqrt(r_sq) if r_sq > 0 else 35.0
        return np.array([xc, yc, zc], dtype=np.float64), float(r)

    @staticmethod
    def _residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        True geometric orthogonal distance residuals:
        d_i = ||p_i - c|| - r
        """
        xc, yc, zc, r = params
        dists = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2 + (points[:, 2] - zc)**2)
        return dists - r

    @staticmethod
    def lateral_radius(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viewing direction and radius of the visible cap measured perpendicular to the line of sight.
        Lateral extent is resolved at pixel scale, unlike depth relief which is limited by disparity noise.
        """
        surface_center = np.median(points, axis=0)
        view_dir = surface_center / max(np.linalg.norm(surface_center), 1e-9)
        offsets = points - surface_center
        lateral = offsets - np.outer(offsets @ view_dir, view_dir)
        radial = np.linalg.norm(lateral - np.median(lateral, axis=0), axis=1)
        return view_dir, float(np.percentile(radial, 95))

    @staticmethod
    def _ray_offset(center: np.ndarray, ray_dir: np.ndarray) -> np.ndarray:
        """Component of the centre perpendicular to the camera ray (mm)."""
        return center - (center @ ray_dir) * ray_dir

    def _objective(
        self,
        params: np.ndarray,
        points: np.ndarray,
        r_prior: float,
        prior_scale: float,
        ray_dir: Optional[np.ndarray] = None,
        ray_scale: float = 0.0,
    ) -> np.ndarray:
        """
        Robust surface residuals plus quadratic radius and ray priors. soft_l1 is applied here rather than via
        least_squares(loss=...) so it does not also cap the priors' pull on large deviations.
        """
        d = self._residuals(params, points)
        c = self.loss_scale_mm
        robust = np.sign(d) * c * np.sqrt(2.0 * (np.sqrt(1.0 + (d / c) ** 2) - 1.0))
        out = np.append(robust, prior_scale * (params[3] - r_prior))
        if ray_dir is not None and ray_scale > 0.0:
            out = np.concatenate([out, ray_scale * self._ray_offset(params[:3], ray_dir)])
        return out

    def fit(self, points: np.ndarray, ray_dir: Optional[np.ndarray] = None) -> SphereFitResult:
        """
        Dual-phase fitting:
        1. Algebraic linear initialization.
        2. Non-linear Levenberg-Marquardt geometric optimization.

        ray_dir: optional camera ray through the 2D mask centroid; enables the ray prior and drift gate.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return SphereFitResult(
                centroid=np.zeros(3, dtype=np.float64),
                radius_mm=0.0,
                rmse_mm=999.0,
                num_points=0,
                is_valid=False,
            )

        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 4:
            return SphereFitResult(
                centroid=np.zeros(3, dtype=np.float64),
                radius_mm=0.0,
                rmse_mm=999.0,
                num_points=len(pts),
                is_valid=False,
            )

        r_min, r_max = float(self.radius_bounds[0]), float(self.radius_bounds[1])
        view_dir, r_lateral = self.lateral_radius(pts)
        if ray_dir is not None:
            ray_dir = np.asarray(ray_dir, dtype=np.float64).reshape(3)
            ray_dir = ray_dir / np.linalg.norm(ray_dir) if np.all(np.isfinite(ray_dir)) and np.linalg.norm(ray_dir) > 0 else None
        r_prior = float(np.clip(r_lateral, r_min, r_max))

        # Centroid lies behind the visible surface, so allow up to one max radius past the far plane
        lower_bounds = np.array([-np.inf, -np.inf, self.z_min_mm, r_min])
        upper_bounds = np.array([np.inf, np.inf, self.z_max_mm + r_max, r_max])

        # 1. Algebraic initialization, falling back to the surface pushed back by r_prior along the ray
        try:
            init_center, init_radius = self.fit_algebraic(pts)
        except Exception:
            init_center, init_radius = np.full(3, np.nan), np.nan
        # A visible cap is convex toward the camera: the centre lies ~r behind the front of the cap along the
        # view direction (~0.5 r for a side-on, partly occluded view). Shallow noisy caps also admit the mirrored
        # concave solution (centre at or in front of the cap), which the algebraic fit and the solver can land on
        # depending on noise. The front (5th percentile) is used rather than the median, which in-band background
        # bleed pushes deeper. x = [xc, yc, zc, r].
        surface_center = np.median(pts, axis=0)
        cap_front = float(np.percentile(pts @ view_dir, 5))
        behind = lambda x: float(np.asarray(x[:3]) @ view_dir) - cap_front > 0.25 * float(x[3])
        behind_init = np.append(surface_center + r_prior * view_dir, r_prior)

        algebraic_ok = (
            np.all(np.isfinite(init_center)) and r_min <= init_radius <= r_max
            and lower_bounds[2] <= init_center[2] <= upper_bounds[2] and behind(np.append(init_center, init_radius))
        )
        if not algebraic_ok:
            init_center, init_radius = behind_init[:3], r_prior

        # least_squares rejects any x0 outside the bounds, so nudge it strictly inside
        margin = 1e-6 * (upper_bounds - lower_bounds)
        margin[~np.isfinite(margin)] = 0.0
        clip_x0 = lambda x: np.clip(x, lower_bounds + margin, upper_bounds - margin)
        x0 = clip_x0(np.array([init_center[0], init_center[1], init_center[2], init_radius]))

        # 2. Non-linear refinement (geometric residuals + soft radius prior)
        prior_scale = self.radius_prior_weight * np.sqrt(len(pts))
        ray_scale = self.ray_prior_weight * np.sqrt(len(pts))

        solve = lambda start: least_squares(
            self._objective,
            start,
            args=(pts, r_prior, prior_scale, ray_dir, ray_scale),
            bounds=(lower_bounds, upper_bounds),
            method="trf",  # Trust Region Reflective handles parameter bounds
            ftol=self.ftol,
            xtol=self.xtol,
            max_nfev=self.max_iter,
        )

        try:
            res = solve(x0)
            if not behind(res.x):
                # Converged to the concave mirror solution: restart from behind the surface
                retry = solve(clip_x0(behind_init))
                if behind(retry.x):
                    res = retry
            opt_xc, opt_yc, opt_zc, opt_r = res.x
            residuals = self._residuals(res.x, pts)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            radius_ok = self.valid_radius_mm[0] <= opt_r <= self.valid_radius_mm[1]
            ray_ok = True
            if ray_dir is not None:
                ray_ok = np.linalg.norm(self._ray_offset(res.x[:3], ray_dir)) <= self.max_ray_offset_ratio * opt_r
            is_valid = bool(
                res.success and behind(res.x) and (rmse <= self.max_rmse_at(opt_zc)) and radius_ok and ray_ok
            )
        except Exception:
            opt_xc, opt_yc, opt_zc = init_center
            opt_r = init_radius
            residuals = self._residuals(x0, pts)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            is_valid = False

        return SphereFitResult(
            centroid=np.array([opt_xc, opt_yc, opt_zc], dtype=np.float64),
            radius_mm=float(opt_r),
            rmse_mm=rmse,
            num_points=len(pts),
            is_valid=is_valid,
        )
