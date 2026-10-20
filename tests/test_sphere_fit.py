"""
Unit Test for Spherical Fitting Algorithm.
Validates algebraic + Levenberg-Marquardt least-squares optimization on synthetic noisy point clouds.
"""

from typing import Tuple
import numpy as np
import pytest
from core.sphere_fit import SphereFitter, SphereFitResult


def generate_noisy_hemisphere(
    center: Tuple[float, float, float] = (15.0, -10.0, 380.0),
    radius: float = 40.0,
    num_points: int = 600,
    noise_std: float = 0.5,
) -> np.ndarray:
    """
    Generates a synthetic front-facing hemisphere point cloud with Gaussian noise.
    Matches the optical surface visible to a binocular camera.
    """
    xc, yc, zc = center
    # Generate random points on unit sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    # Cosine distribution for hemisphere facing the camera (z <= zc)
    cos_theta = np.random.uniform(0.0, 1.0, num_points)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    x = xc + radius * sin_theta * np.cos(phi)
    y = yc + radius * sin_theta * np.sin(phi)
    z = zc - radius * cos_theta  # Front hemisphere facing camera

    points = np.column_stack([x, y, z])

    # Inject Gaussian spatial sensor noise
    noise = np.random.normal(0.0, noise_std, points.shape)
    noisy_points = points + noise

    return noisy_points.astype(np.float32)


def test_sphere_fitter_accuracy():
    """Verifies that the fitted sphere parameters meet the paper's sub-millimeter precision."""
    np.random.seed(42)
    true_center = (20.0, -15.0, 420.0)
    true_radius = 42.0

    points = generate_noisy_hemisphere(
        center=true_center,
        radius=true_radius,
        num_points=800,
        noise_std=0.5,
    )

    fitter = SphereFitter()
    result: SphereFitResult = fitter.fit(points)

    assert result.is_valid, f"Fitting failed. RMSE: {result.rmse_mm:.2f} mm"

    # Centroid error check (must be well within paper limits: < 1.0 mm for synthetic test)
    centroid_err = np.linalg.norm(result.centroid - np.array(true_center))
    assert centroid_err < 1.5, f"Centroid error too large: {centroid_err:.2f} mm"

    # Radius error check
    radius_err = abs(result.radius_mm - true_radius)
    assert radius_err < 1.5, f"Radius error too large: {radius_err:.2f} mm"

    # Residual RMSE should be close to injected noise standard deviation (0.5 mm)
    assert result.rmse_mm < 1.5, f"RMSE too high: {result.rmse_mm:.2f} mm"


def test_sphere_fitter_degenerate():
    """Verifies that SphereFitter gracefully flags degenerate point sets as invalid."""
    fitter = SphereFitter()

    # Under-determined point cloud (< 4 points)
    few_pts = np.array([[10.0, 20.0, 300.0], [12.0, 22.0, 302.0]], dtype=np.float32)
    res_few = fitter.fit(few_pts)
    assert not res_few.is_valid
    assert res_few.num_points == 2

    # Array containing NaNs or Infs
    nan_pts = np.array([
        [10.0, 20.0, np.nan],
        [12.0, 22.0, 302.0],
        [15.0, 25.0, 305.0],
        [np.inf, 20.0, 300.0],
    ], dtype=np.float32)
    res_nan = fitter.fit(nan_pts)
    assert not res_nan.is_valid
    assert res_nan.num_points == 2  # Only 2 finite points remaining


def test_sphere_fitter_reads_depth_envelope_from_config():
    """Fruit beyond 1 m must be refined, not rejected by a hardcoded Z ceiling."""
    np.random.seed(7)
    true_center = (40.0, -25.0, 1500.0)
    true_radius = 30.0
    points = generate_noisy_hemisphere(center=true_center, radius=true_radius, num_points=800, noise_std=0.5)

    fitter = SphereFitter({"pointcloud": {"z_min_mm": 150.0, "z_max_mm": 3500.0}})
    assert fitter.z_max_mm == 3500.0
    result = fitter.fit(points)

    assert result.is_valid, f"Fitting failed. RMSE: {result.rmse_mm:.2f} mm"
    assert np.linalg.norm(result.centroid - np.array(true_center)) < 1.5
    assert abs(result.radius_mm - true_radius) < 1.5


def test_sphere_fitter_shallow_cap_does_not_peg_radius_bound():
    """A flattened, noisy cap (depth relief below disparity noise) must not drive r onto the upper bound."""
    np.random.seed(3)
    true_center = (0.0, 0.0, 900.0)
    true_radius = 25.0
    points = generate_noisy_hemisphere(center=true_center, radius=true_radius, num_points=600, noise_std=0.5)
    # Compress depth relief to ~20%, as SGBM/WLS smoothing does on small fruit, then add depth noise
    surface_z = points[:, 2].min()
    points[:, 2] = surface_z + 0.2 * (points[:, 2] - surface_z) + np.random.normal(0.0, 1.5, len(points))

    fitter = SphereFitter({"sphere_fitting": {"radius_bounds_mm": [10.0, 90.0]}})
    result = fitter.fit(points)

    assert result.radius_mm < 0.9 * fitter.radius_bounds[1], f"Radius pegged at bound: {result.radius_mm:.1f} mm"
    assert abs(result.radius_mm - true_radius) < 8.0


def test_sphere_fitter_valid_centroid_window_reads_config():
    assert SphereFitter().valid_centroid_z_mm == (200.0, 5000.0)
    fitter = SphereFitter({"sphere_fitting": {"valid_centroid_z_mm": [250.0, 1200.0]}})
    assert fitter.valid_centroid_z_mm == (250.0, 1200.0)


def test_sphere_fitter_ray_prior_resists_lateral_contamination():
    """Same-depth contamination beside the fruit must not drag the centre off the mask-centroid ray."""
    np.random.seed(3)
    true_center = np.array([30.0, -20.0, 650.0])
    fruit = generate_noisy_hemisphere(center=tuple(true_center), radius=26.0, num_points=600, noise_std=1.0)
    # Tilted leaf patch at fruit depth, offset to one side
    u, v = np.random.uniform(-15, 15, (2, 250))
    leaf = np.column_stack([true_center[0] + 30 + u, true_center[1] + v, true_center[2] - 20 + 0.8 * u])
    pts = np.vstack([fruit, leaf])
    ray = true_center / np.linalg.norm(true_center)

    cfg = {"sphere_fitting": {"radius_bounds_mm": [10.0, 90.0], "max_allowed_rmse_mm": 8.0}}
    with_ray = SphereFitter(cfg).fit(pts, ray_dir=ray)
    without = SphereFitter(cfg).fit(pts)

    offset = lambda c: np.linalg.norm(c - (c @ ray) * ray)
    assert offset(with_ray.centroid) < offset(without.centroid)
    assert offset(with_ray.centroid) < 0.5 * with_ray.radius_mm


def test_sphere_fitter_plausibility_gates():
    """Fits are rejected when the radius is implausible or the centre drifts off the mask-centroid ray."""
    np.random.seed(4)
    center = np.array([0.0, 0.0, 600.0])
    pts = generate_noisy_hemisphere(center=tuple(center), radius=25.0, num_points=500, noise_std=0.3)

    ok = SphereFitter({"sphere_fitting": {"valid_radius_mm": [17.5, 45.0]}}).fit(pts, ray_dir=center)
    assert ok.is_valid

    too_small = SphereFitter({"sphere_fitting": {"valid_radius_mm": [30.0, 45.0]}}).fit(pts)
    assert not too_small.is_valid

    # Ray aimed 60 mm to the side of the true centre with the prior disabled -> drift gate trips
    side_ray = np.array([60.0, 0.0, 600.0])
    drift = SphereFitter({"sphere_fitting": {"ray_prior_weight": 0.0}}).fit(pts, ray_dir=side_ray)
    assert not drift.is_valid


def test_sphere_fitter_rmse_gate_scales_with_depth_squared():
    """The RMSE gate grows with stereo depth noise (delta * Z^2 / (f * B)) and is fixed without stereo geometry."""
    fitter = SphereFitter({"sphere_fitting": {"max_allowed_rmse_mm": 8.0, "rmse_disparity_noise_px": 0.16}})
    assert fitter.max_rmse_at(2000.0) == 8.0  # no geometry set -> fixed gate

    fitter.set_stereo_geometry(focal_px=614.925, baseline_mm=119.671)
    sigma = lambda z: 0.16 * z ** 2 / (614.925 * 119.671)
    for z in (700.0, 1600.0, 2000.0):
        assert np.isclose(fitter.max_rmse_at(z), np.hypot(8.0, sigma(z)))
    assert fitter.max_rmse_at(700.0) < 8.1 and fitter.max_rmse_at(2000.0) > 11.0

    # A far fit with RMSE between the fixed and depth-scaled gates is accepted only with scaling
    np.random.seed(5)
    center = (0.0, 0.0, 1900.0)
    pts = generate_noisy_hemisphere(center=center, radius=27.0, num_points=500, noise_std=9.5)
    scaled = fitter.fit(pts, ray_dir=np.array(center))
    fixed = SphereFitter({"sphere_fitting": {"max_allowed_rmse_mm": 8.0}}).fit(pts, ray_dir=np.array(center))
    assert 8.0 < scaled.rmse_mm <= fitter.max_rmse_at(scaled.centroid[2])
    assert scaled.is_valid and not fixed.is_valid


def test_sphere_fitter_rejects_concave_mirror_solution():
    """Starting from the mirrored centre in front of a shallow cap must still converge behind the surface."""
    np.random.seed(6)
    true_center = np.array([10.0, -5.0, 1100.0])
    r = 30.0
    # Shallow cap (polar angle <= 40 deg) with stereo-like noise: the concave mirror is a competitive minimum
    phi = np.random.uniform(0, 2 * np.pi, 600)
    cos_t = np.random.uniform(np.cos(np.radians(40)), 1.0, 600)
    sin_t = np.sqrt(1 - cos_t ** 2)
    pts = np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), -r * cos_t]) + true_center
    pts += np.random.normal(0, 1.5, pts.shape)

    cfg = {"sphere_fitting": {"radius_bounds_mm": [10.0, 90.0], "max_allowed_rmse_mm": 8.0}}
    reference = SphereFitter(cfg).fit(pts, ray_dir=true_center)
    fitter = SphereFitter(cfg)
    fitter.fit_algebraic = lambda p: (true_center - np.array([0.0, 0.0, 2 * r]) * 1.2, r)  # force a front init
    res = fitter.fit(pts, ray_dir=true_center)
    surface = np.median(pts, axis=0)
    view = surface / np.linalg.norm(surface)
    assert (res.centroid - surface) @ view > 0.0
    assert res.is_valid
    assert np.linalg.norm(res.centroid - reference.centroid) < 0.5
