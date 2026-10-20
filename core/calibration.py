"""
Stereo Calibration and Epipolar Rectification Module.
Provides coordinate transformations to align binocular images onto a common epipolar plane.
"""

from typing import Tuple, Dict, Any, Union
from pathlib import Path
import yaml
import numpy as np
import cv2


def _coerce_camera_matrix(mat: Any, image_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Coerces various camera matrix representations into a strictly valid (3, 3) float64 intrinsic matrix.
    Handles:
    - (3, 3) or flat 9 elements: standard pinhole K matrix
    - (2, 3) or flat 6 elements: 2x3 affine matrix (appends [0, 0, 1] row) or [fx, fy, cx, cy, ...]
    - (4,) elements: [fx, fy, cx, cy]
    - (3,) elements: [f, cx, cy]
    - (1,) elements: focal length f with principal point at image center
    """
    if mat is None:
        w, h = image_size
        return np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr

    if arr.size == 9:
        return arr.reshape((3, 3))

    if arr.size == 6:
        flat = arr.flatten()
        # Case A: 2x3 matrix representing [[fx, 0, cx], [0, fy, cy]]
        if arr.shape == (2, 3) or np.isclose(flat[1], 0.0) or np.isclose(flat[3], 0.0):
            mat_2x3 = flat.reshape((2, 3))
            return np.vstack([mat_2x3, np.array([[0.0, 0.0, 1.0]], dtype=np.float64)])
        # Case B: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = flat[0], flat[1], flat[2], flat[3]
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    if arr.size == 4:
        flat = arr.flatten()
        fx, fy, cx, cy = flat[0], flat[1], flat[2], flat[3]
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    if arr.size == 3:
        flat = arr.flatten()
        f, cx, cy = flat[0], flat[1], flat[2]
        return np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    if arr.size == 1:
        f = float(arr.item())
        w, h = image_size
        return np.array([
            [f, 0.0, w / 2.0],
            [0.0, f, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    if arr.size > 9:
        return arr.flatten()[:9].reshape((3, 3))

    w, h = image_size
    return np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _coerce_rotation_matrix(rot: Any) -> np.ndarray:
    """
    Coerces various rotation representations into a strictly valid (3, 3) float64 orthogonal matrix.
    Handles:
    - (3, 3) or flat 9 elements: standard 3x3 rotation matrix
    - (2, 3) or flat 6 elements: 2 rows of rotation matrix (reconstructs 3rd via cross product)
      or 6-DoF pose [rx, ry, rz, tx, ty, tz]
    - (3,) elements: Rodrigues angle-axis vector or Euler angles
    - (4,) elements: Quaternion [x, y, z, w]
    """
    if rot is None:
        return np.eye(3, dtype=np.float64)

    arr = np.asarray(rot, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr

    if arr.size == 9:
        return arr.reshape((3, 3))

    if arr.size == 6:
        flat = arr.flatten()
        norm1 = np.linalg.norm(flat[:3])
        norm2 = np.linalg.norm(flat[3:6])
        if norm1 > 1e-4 and norm2 > 1e-4 and abs(norm1 - 1.0) < 0.2 and abs(norm2 - 1.0) < 0.2:
            r1 = flat[:3] / norm1
            r2 = flat[3:6] - np.dot(r1, flat[3:6]) * r1
            norm_r2 = np.linalg.norm(r2)
            if norm_r2 > 1e-6:
                r2 = r2 / norm_r2
            else:
                r2 = np.array([0.0, 1.0, 0.0])
            r3 = np.cross(r1, r2)
            return np.vstack([r1, r2, r3]).astype(np.float64)
        else:
            # 6-DoF pose vector: first 3 are Rodrigues rotation vector
            rvec = flat[:3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            return R.astype(np.float64)

    if arr.size == 3:
        rvec = arr.flatten().reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        return R.astype(np.float64)

    if arr.size == 4:
        x, y, z, w = arr.flatten()
        q_norm = np.sqrt(x*x + y*y + z*z + w*w)
        if q_norm > 1e-6:
            x, y, z, w = x/q_norm, y/q_norm, z/q_norm, w/q_norm
        R = np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)],
            [2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)]
        ], dtype=np.float64)
        return R

    if arr.size > 9:
        return arr.flatten()[:9].reshape((3, 3))

    return np.eye(3, dtype=np.float64)


def _coerce_translation_vector(trans: Any) -> np.ndarray:
    """
    Coerces translation into (3, 1) float64 vector.
    """
    if trans is None:
        return np.zeros((3, 1), dtype=np.float64)
    arr = np.asarray(trans, dtype=np.float64)
    if arr.size == 3:
        return arr.reshape((3, 1))
    if arr.size == 6:
        # Pose vector: last 3 are translation
        return arr.flatten()[3:6].reshape((3, 1))
    if arr.size > 3:
        return arr.flatten()[:3].reshape((3, 1))
    t = np.zeros((3, 1), dtype=np.float64)
    if arr.size > 0:
        t[:arr.size, 0] = arr.flatten()
    return t


def _coerce_dist_coeffs(dist: Any) -> np.ndarray:
    """
    Coerces distortion coefficients into (N, 1) float64 column vector.
    """
    if dist is None:
        return np.zeros((5, 1), dtype=np.float64)
    arr = np.asarray(dist, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((5, 1), dtype=np.float64)
    return arr.reshape((-1, 1))


class StereoCalibrator:
    """
    Manages camera calibration parameters, rectification transforms,
    and perspective disparity-to-depth reprojection matrix Q.
    """

    def __init__(self, config_source: Union[str, Path, Dict[str, Any]]):
        """
        Initializes calibrator with calibration configuration file or dictionary.
        """
        if isinstance(config_source, (str, Path)):
            config_path = Path(config_source)
            if not config_path.is_file():
                alt_path = Path(__file__).resolve().parent.parent / config_path
                if alt_path.is_file():
                    config_path = alt_path
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_source

        self.image_width = int(self.config.get("image_width", 1024))
        self.image_height = int(self.config.get("image_height", 1024))
        self.image_size = (self.image_width, self.image_height)
        # Frames already rectified upstream (e.g. ZED SDK output): pass through, no remap
        self.already_rectified = bool(self.config.get("already_rectified", False))

        # Load intrinsics and extrinsics with flexible, strict dimension coercion
        left_cfg = self.config.get("left_camera", {})
        self.K1 = _coerce_camera_matrix(left_cfg.get("camera_matrix"), self.image_size)
        self.D1 = _coerce_dist_coeffs(left_cfg.get("dist_coeffs"))

        right_cfg = self.config.get("right_camera", {})
        self.K2 = _coerce_camera_matrix(right_cfg.get("camera_matrix"), self.image_size)
        self.D2 = _coerce_dist_coeffs(right_cfg.get("dist_coeffs"))

        extrinsics_cfg = self.config.get("extrinsics", {})
        self.R = _coerce_rotation_matrix(extrinsics_cfg.get("R"))
        self.T = _coerce_translation_vector(extrinsics_cfg.get("T"))
        self.baseline_mm = float(extrinsics_cfg.get("baseline_mm", np.linalg.norm(self.T)))
        if self.baseline_mm <= 0.0:
            self.baseline_mm = float(np.linalg.norm(self.T))
        if self.baseline_mm <= 0.0:
            self.baseline_mm = 74.676

        # Compute rectification matrices
        self._compute_rectification()

    def _compute_rectification(self) -> None:
        """
        Computes rectification transforms R1, R2, P1, P2, and reprojection matrix Q
        using cv2.stereoRectify, and constructs remap lookup tables.
        """
        # Ensure exact matrix dimensions and types for OpenCV
        K1 = _coerce_camera_matrix(self.K1, self.image_size)
        D1 = _coerce_dist_coeffs(self.D1)
        K2 = _coerce_camera_matrix(self.K2, self.image_size)
        D2 = _coerce_dist_coeffs(self.D2)
        R = _coerce_rotation_matrix(self.R)
        T = _coerce_translation_vector(self.T)
        self.K1, self.K2, self.D1, self.D2, self.R, self.T = K1, K2, D1, D2, R, T

        if self.already_rectified:
            self._init_prerectified()
            return

        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.roi1,
            self.roi2,
        ) = cv2.stereoRectify(
            cameraMatrix1=K1,
            distCoeffs1=D1,
            cameraMatrix2=K2,
            distCoeffs2=D2,
            imageSize=self.image_size,
            R=R,
            T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
            newImageSize=self.image_size,
        )

        # Precompute remap coordinate arrays for real-time rectification
        self.map_Lx, self.map_Ly = cv2.initUndistortRectifyMap(
            K1, D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        self.map_Rx, self.map_Ry = cv2.initUndistortRectifyMap(
            K2, D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1
        )

    def _init_prerectified(self) -> None:
        """
        Builds P1, P2 and Q for an already rectified pair from the left intrinsics and baseline.
        Q maps (u, v, d) to X = (u - cx) B / d, Y = (v - cy) Z / fy, Z = fx B / d.
        """
        fx, fy = float(self.K1[0, 0]), float(self.K1[1, 1])
        cx, cy = float(self.K1[0, 2]), float(self.K1[1, 2])
        B = self.baseline_mm

        self.R1 = np.eye(3, dtype=np.float64)
        self.R2 = np.eye(3, dtype=np.float64)
        self.P1 = np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        self.P2 = self.P1.copy()
        self.P2[0, 3] = -fx * B
        aspect = fx / fy
        self.Q = np.array(
            [
                [1.0, 0.0, 0.0, -cx],
                [0.0, aspect, 0.0, -cy * aspect],
                [0.0, 0.0, 0.0, fx],
                [0.0, 0.0, 1.0 / B, 0.0],
            ],
            dtype=np.float64,
        )
        self.roi1 = self.roi2 = (0, 0, self.image_width, self.image_height)
        self.map_Lx = self.map_Ly = self.map_Rx = self.map_Ry = None

    def rectify_pair(
        self, 
        image_left: np.ndarray, 
        image_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies epipolar rectification to left and right images.
        Output images share horizontal epipolar scanlines (y_l == y_r).
        """
        if self.already_rectified:
            # Resizing would silently invalidate fx/cx/cy, so pre-rectified input must match the calibration
            for name, img in (("left", image_left), ("right", image_right)):
                if img.shape[:2] != (self.image_height, self.image_width):
                    raise ValueError(
                        f"Pre-rectified {name} image is {img.shape[1]}x{img.shape[0]}, "
                        f"calibration expects {self.image_width}x{self.image_height}."
                    )
            return image_left, image_right

        if image_left.shape[:2] != (self.image_height, self.image_width):
            image_left = cv2.resize(image_left, self.image_size)
        if image_right.shape[:2] != (self.image_height, self.image_width):
            image_right = cv2.resize(image_right, self.image_size)

        rect_left = cv2.remap(
            image_left, self.map_Lx, self.map_Ly, interpolation=cv2.INTER_LINEAR
        )
        rect_right = cv2.remap(
            image_right, self.map_Rx, self.map_Ry, interpolation=cv2.INTER_LINEAR
        )
        return rect_left, rect_right

    def get_q_matrix(self) -> np.ndarray:
        """Returns the 4x4 disparity-to-depth perspective reprojection matrix Q."""
        return self.Q.copy()

    def get_baseline_mm(self) -> float:
        """Returns the physical baseline in millimeters."""
        return self.baseline_mm

    def get_focal_length(self) -> float:
        """Returns the focal length of the rectified left camera."""
        return float(self.P1[0, 0])
