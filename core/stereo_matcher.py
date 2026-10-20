"""
Stereo Disparity and Depth Estimation Module.
Implements Tuned OpenCV StereoSGBM, Weighted Least Squares (WLS) Disparity Filtering,
Mask-Based Continuous Fruit Stereo Correspondence, and Deep RAFT-Stereo Integration.
"""

from typing import Tuple, Dict, Any, Union, Optional, List
from pathlib import Path
from types import SimpleNamespace
import yaml
import numpy as np
import cv2

# Optional PyTorch import for RAFT-Stereo deep correspondence
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RAFTStereoEngine:
    """
    Official RAFT-Stereo (Lipson et al., 3DV 2021) disparity engine, as used in Zheng et al. 2023 (Sec. 2.4.2).
    Runs the pretrained checkpoint as-is (no fine-tuning) on rectified pairs.
    """

    def __init__(self, raft_cfg: Optional[Dict[str, Any]] = None):
        cfg = raft_cfg or {}
        self.model = None
        self.initialized = False
        self.device = "cpu"
        self.iters = int(cfg.get("iters", 32))

        if not HAS_TORCH:
            print("[RAFT-Stereo] torch not available; engine disabled.", flush=True)
            return

        self.device = self._resolve_device(str(cfg.get("device", "cuda")))
        self.mixed_precision = bool(cfg.get("mixed_precision", True)) and self.device.startswith("cuda")

        model_path = Path(str(cfg.get("model_path", "weights/raftstereo-middlebury.pth")))
        if not model_path.is_file():
            alt_path = Path(__file__).resolve().parent.parent / model_path
            if alt_path.is_file():
                model_path = alt_path
        if not model_path.is_file():
            print(f"[RAFT-Stereo] Checkpoint not found: {model_path}; engine disabled.", flush=True)
            return

        # Architecture args must match the checkpoint; defaults match middlebury/eth3d/sceneflow
        # (raftstereo-realtime.pth needs shared_backbone, n_downsample=3, n_gru_layers=2, slow_fast_gru)
        args = SimpleNamespace(
            hidden_dims=list(cfg.get("hidden_dims", [128, 128, 128])),
            corr_implementation=str(cfg.get("corr_implementation", "reg")),
            shared_backbone=bool(cfg.get("shared_backbone", False)),
            corr_levels=int(cfg.get("corr_levels", 4)),
            corr_radius=int(cfg.get("corr_radius", 4)),
            n_downsample=int(cfg.get("n_downsample", 2)),
            context_norm=str(cfg.get("context_norm", "batch")),
            slow_fast_gru=bool(cfg.get("slow_fast_gru", False)),
            n_gru_layers=int(cfg.get("n_gru_layers", 3)),
            mixed_precision=self.mixed_precision,
        )

        try:
            from third_party.raft_stereo import RAFTStereo

            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Upstream checkpoints were saved from nn.DataParallel
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}

            model = RAFTStereo(args)
            model.load_state_dict(state_dict, strict=True)
            self.model = model.to(self.device).eval()
            self.initialized = True
            gpu = f" ({torch.cuda.get_device_name(self.device)})" if self.device.startswith("cuda") else ""
            print(
                f"[RAFT-Stereo] Loaded {model_path.name} on {self.device}{gpu}, "
                f"iters={self.iters}, mixed_precision={self.mixed_precision}.",
                flush=True,
            )
        except Exception as e:
            print(f"[RAFT-Stereo] Failed to initialize: {e}", flush=True)
            self.model = None
            self.initialized = False

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            device = "cuda"
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[RAFT-Stereo] WARNING: CUDA requested but unavailable; falling back to CPU (slow).", flush=True)
            return "cpu"
        return device

    def compute_disparity(
        self, 
        rectified_left: np.ndarray, 
        rectified_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes dense disparity (pixels, left frame). RAFT-Stereo predicts horizontal flow
        x_R - x_L = -d, so disparity d = -flow_x.
        """
        if not self.initialized or self.model is None or not HAS_TORCH:
            raise RuntimeError("RAFT-Stereo engine is not initialized or torch is unavailable.")

        from third_party.raft_stereo import InputPadder

        h, w = rectified_left.shape[:2]

        # Upstream expects RGB float in [0, 255]; normalization happens inside the model
        img1 = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB) if rectified_left.ndim == 3 else cv2.cvtColor(rectified_left, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB) if rectified_right.ndim == 3 else cv2.cvtColor(rectified_right, cv2.COLOR_GRAY2RGB)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        padder = InputPadder(t1.shape, divis_by=32)
        t1, t2 = padder.pad(t1, t2)

        with torch.inference_mode():
            _, flow_up = self.model(t1, t2, iters=self.iters, test_mode=True)
            flow_x = padder.unpad(flow_up.float())[0, 0].cpu().numpy()

        disparity_map = (-flow_x[:h, :w]).astype(np.float32)
        # Matches that land left of the right image border are unobservable
        x_coords = np.arange(w, dtype=np.float32)[None, :]
        valid_mask = np.isfinite(disparity_map) & (disparity_map > 0.5) & ((x_coords - disparity_map) >= 0.0)
        disparity_map[~valid_mask] = -1.0

        return disparity_map, valid_mask


class StereoMatcherSGBM:
    """
    High-density stereo matcher with tuned StereoSGBM, WLS regularizer,
    mask-guided continuous fruit disparity correspondence, and RAFT integration.
    """

    def __init__(self, config_source: Union[str, Path, Dict[str, Any]]):
        """
        Initializes stereo matcher and WLS filter from configuration.
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
            self.config = config_source or {}

        self.engine_type = str(self.config.get("engine", "sgbm")).lower()

        # SGBM Parameters (tuned for uniform fruit textures)
        sgbm_cfg = self.config.get("stereo_sgbm", {})
        self.min_disparity = int(sgbm_cfg.get("min_disparity", 0))
        self.num_disparities = int(sgbm_cfg.get("num_disparities", 256))
        # Ensure num_disparities is divisible by 16
        if self.num_disparities % 16 != 0:
            self.num_disparities = int(np.ceil(self.num_disparities / 16.0) * 16)

        # Block size tuned to 11-15 to encompass surface gradients across uniform fruit skins
        self.block_size = int(sgbm_cfg.get("block_size", 11))
        if self.block_size % 2 == 0:
            self.block_size += 1

        # Smoothness penalties P1 and P2 dynamically scaled to block size
        channels = 3
        default_p1 = 8 * channels * (self.block_size ** 2)
        default_p2 = 32 * channels * (self.block_size ** 2)
        self.p1 = int(sgbm_cfg.get("p1", default_p1))
        self.p2 = int(sgbm_cfg.get("p2", default_p2))

        self.disp12_max_diff = int(sgbm_cfg.get("disp12_max_diff", 2))
        self.pre_filter_cap = int(sgbm_cfg.get("pre_filter_cap", 31))
        # uniqueness_ratio tuned to 5-8% to prevent premature rejection on smooth fruit textures
        self.uniqueness_ratio = int(sgbm_cfg.get("uniqueness_ratio", 5))
        # speckle_window_size increased to 120 to eliminate mosaic fragmentation
        self.speckle_window_size = int(sgbm_cfg.get("speckle_window_size", 120))
        self.speckle_range = int(sgbm_cfg.get("speckle_range", 2))
        self.mode = int(sgbm_cfg.get("mode", cv2.STEREO_SGBM_MODE_SGBM_3WAY))

        # Build primary left SGBM matcher
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=self.disp12_max_diff,
            preFilterCap=self.pre_filter_cap,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=self.mode,
        )

        # WLS Disparity Filter
        wls_cfg = self.config.get("wls_filter", {})
        self.wls_enabled = bool(wls_cfg.get("enabled", True))
        if self.wls_enabled and hasattr(cv2, "ximgproc"):
            self.wls_lambda = float(wls_cfg.get("lambda", 10000.0))
            self.wls_sigma = float(wls_cfg.get("sigma", 1.8))
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(self.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma)
        else:
            self.wls_enabled = False
            self.right_matcher = None
            self.wls_filter = None

        # Mask-guided continuous surface refinement
        mask_cfg = self.config.get("mask_refinement", {})
        self.mask_refinement_enabled = bool(mask_cfg.get("enabled", True))
        self.bilateral_diameter = int(mask_cfg.get("bilateral_diameter", 9))
        self.bilateral_sigma_color = float(mask_cfg.get("bilateral_sigma_color", 15.0))
        self.bilateral_sigma_space = float(mask_cfg.get("bilateral_sigma_space", 15.0))

        # Deep RAFT-Stereo Engine
        self.raft_engine = None
        if self.engine_type in ["raft", "auto"]:
            self.raft_engine = RAFTStereoEngine(self.config.get("raft_stereo", {}))
            if not self.raft_engine.initialized:
                print(f"[Stereo] engine='{self.engine_type}' but RAFT-Stereo is unavailable; using StereoSGBM.", flush=True)

        # Backlit greenhouse sky suppression
        sky_cfg = self.config.get("sky_suppression", {})
        self.sky_suppression_enabled = bool(sky_cfg.get("enabled", True))
        self.sky_threshold = int(sky_cfg.get("luminance_threshold", 245))

    def _compute_sgbm_wls_disparity(
        self, 
        rectified_left: np.ndarray, 
        rectified_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes dense sub-pixel disparity via tuned StereoSGBM + WLS regularizer.
        """
        if rectified_left.ndim == 3:
            gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = rectified_left
            gray_right = rectified_right

        disp_left_raw = self.left_matcher.compute(gray_left, gray_right)

        # WLS derives its ROI as width - (min_disparity + num_disparities + block_size); narrower frames
        # yield a negative ROI and cv2.error, so fall back to raw SGBM there
        wls_fits = gray_left.shape[1] > self.min_disparity + self.num_disparities + self.block_size
        if self.wls_enabled and wls_fits and self.right_matcher is not None and self.wls_filter is not None:
            disp_right_raw = self.right_matcher.compute(gray_right, gray_left)
            disp_filtered_raw = self.wls_filter.filter(
                disparity_map_left=disp_left_raw,
                left_view=rectified_left,
                disparity_map_right=disp_right_raw,
            )
            disparity_wls = disp_filtered_raw.astype(np.float32) / 16.0
            disparity_raw = disp_left_raw.astype(np.float32) / 16.0

            # Blend WLS filtered with raw SGBM: prioritize WLS smoothness on fruit surfaces
            wls_valid = disparity_wls > (self.min_disparity + 0.1)
            raw_valid = disparity_raw > (self.min_disparity + 0.1)
            disparity_px = np.where(wls_valid, disparity_wls, np.where(raw_valid, disparity_raw, -1.0))
        else:
            disparity_px = disp_left_raw.astype(np.float32) / 16.0

        valid_mask = (disparity_px > (self.min_disparity + 0.1)) & (
            disparity_px <= (self.min_disparity + self.num_disparities + 10)
        )
        return disparity_px, valid_mask

    def _refine_fruit_mask_disparity(
        self,
        disparity_map: np.ndarray,
        valid_mask: np.ndarray,
        rectified_left: np.ndarray,
        rectified_right: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies mask-guided stereo correspondence and continuous surface smoothing
        to eliminate broken mosaic artifacts and specular voids on uniform fruit skins.
        """
        mask_bool = np.asarray(fruit_mask, dtype=bool)
        if np.sum(mask_bool) < 8:
            return disparity_map, valid_mask

        h, w = disparity_map.shape
        y_idxs, x_idxs = np.where(mask_bool)
        margin = 6
        ymin, ymax = max(0, int(np.min(y_idxs)) - margin), min(h, int(np.max(y_idxs)) + margin + 1)
        xmin, xmax = max(0, int(np.min(x_idxs)) - margin), min(w, int(np.max(x_idxs)) + margin + 1)

        local_mask = mask_bool[ymin:ymax, xmin:xmax]
        local_disp = disparity_map[ymin:ymax, xmin:xmax].copy()
        local_valid = (local_disp > (self.min_disparity + 0.1)) & (local_disp <= (self.min_disparity + self.num_disparities)) & local_mask

        valid_pixels_in_fruit = np.sum(local_valid)
        total_pixels_in_fruit = np.sum(local_mask)

        # Case A: Partial valid disparities exist within the fruit mask
        if valid_pixels_in_fruit >= 8:
            try:
                vals = local_disp[local_valid]
                med_val = float(np.median(vals))
                # Outlier rejection using Interquartile Range (IQR)
                q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
                iqr = max(1.0, q75 - q25)
                inlier_submask = local_valid & (local_disp >= (q25 - 2.0 * iqr)) & (local_disp <= (q75 + 2.0 * iqr))

                # Missing or mosaic outlier pixels inside fruit mask
                missing_or_outlier = local_mask & (~inlier_submask)

                if np.sum(missing_or_outlier) > 0:
                    work_disp = local_disp.copy()
                    work_disp[~inlier_submask] = med_val

                    # cv2.inpaint requires uint8 1-channel (CV_8UC1)
                    inlier_vals = work_disp[inlier_submask]
                    if len(inlier_vals) > 0:
                        d_min = float(np.min(inlier_vals))
                        d_max = float(np.max(inlier_vals))
                    else:
                        d_min, d_max = float(self.min_disparity), float(self.min_disparity + self.num_disparities)
                    d_range = max(1.0, d_max - d_min)

                    disp_u8 = np.clip((work_disp - d_min) / d_range * 235.0 + 10.0, 0, 255).astype(np.uint8)
                    inpaint_mask_u8 = (missing_or_outlier.astype(np.uint8) * 255)
                    inpainted_u8 = cv2.inpaint(disp_u8, inpaint_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    smoothed_patch = (inpainted_u8.astype(np.float32) - 10.0) / 235.0 * d_range + d_min
                else:
                    smoothed_patch = local_disp.copy()

                # Apply bilateral filter to eliminate discrete mosaic staircases while preserving curvature
                smoothed_patch = cv2.bilateralFilter(
                    smoothed_patch.astype(np.float32), 
                    d=self.bilateral_diameter, 
                    sigmaColor=self.bilateral_sigma_color, 
                    sigmaSpace=self.bilateral_sigma_space
                )

                # Update disparity and validity inside fruit mask
                local_disp[local_mask] = smoothed_patch[local_mask]
                disparity_map[ymin:ymax, xmin:xmax] = local_disp
                valid_mask[ymin:ymax, xmin:xmax] |= local_mask
            except Exception:
                pass

        # Case B: Severe textureless dropout (< 8 valid pixels) -> Epipolar Template NCC Correspondence
        elif total_pixels_in_fruit >= 10:
            try:
                crop_l = rectified_left[ymin:ymax, xmin:xmax]
                if crop_l.ndim == 3:
                    crop_l_gray = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
                    full_r_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
                else:
                    crop_l_gray = crop_l
                    full_r_gray = rectified_right

                strip_r = full_r_gray[ymin:ymax, :]
                if strip_r.shape[1] >= crop_l_gray.shape[1] and strip_r.shape[0] == crop_l_gray.shape[0] and crop_l_gray.shape[0] > 0 and crop_l_gray.shape[1] > 0:
                    res = cv2.matchTemplate(strip_r, crop_l_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if max_val > 0.30:
                        best_x_right = max_loc[0]
                        inferred_disparity = float(xmin - best_x_right)
                        if self.min_disparity < inferred_disparity <= (self.min_disparity + self.num_disparities):
                            local_disp[local_mask] = inferred_disparity
                            disparity_map[ymin:ymax, xmin:xmax] = local_disp
                            valid_mask[ymin:ymax, xmin:xmax] |= local_mask
            except Exception:
                pass

        return disparity_map, valid_mask

    def compute_disparity(
        self, 
        rectified_left: np.ndarray, 
        rectified_right: np.ndarray,
        protected_mask: Optional[np.ndarray] = None,
        instances: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes dense, continuous disparity field for rectified stereo pair.
        Supports deep RAFT-Stereo, tuned StereoSGBM + WLS, and mask-guided continuous refinement.

        Returns:
            disparity_map: (H, W) float32 array in pixels.
            valid_mask: (H, W) boolean mask of valid disparity pixels.
        """
        disparity_px = None
        valid_mask = None
        used_raft = False

        # 1. Try RAFT-Stereo engine if configured
        if self.engine_type in ["raft", "auto"] and self.raft_engine is not None and self.raft_engine.initialized:
            try:
                disparity_px, valid_mask = self.raft_engine.compute_disparity(rectified_left, rectified_right)
                used_raft = True
            except Exception as e:
                # Graceful fallback to tuned StereoSGBM
                print(f"[Stereo] RAFT-Stereo inference failed ({e}); falling back to StereoSGBM.", flush=True)
                disparity_px = None
                valid_mask = None

        # 2. Compute via Tuned StereoSGBM + WLS Disparity Filter
        if disparity_px is None or valid_mask is None:
            disparity_px, valid_mask = self._compute_sgbm_wls_disparity(rectified_left, rectified_right)

        # 3. Mask-Guided Continuous Surface Refinement (eliminating mosaic artifacts on fruits)
        # SGBM-only: RAFT output is already dense, and the refinement's num_disparities bound would clip it
        if self.mask_refinement_enabled and not used_raft:
            # If explicit YOLO instances are passed
            if instances is not None and len(instances) > 0:
                for inst in instances:
                    mask = getattr(inst, "sanitized_mask", None)
                    if mask is None:
                        mask = getattr(inst, "binary_mask", None)
                    if mask is not None:
                        disparity_px, valid_mask = self._refine_fruit_mask_disparity(
                            disparity_px, valid_mask, rectified_left, rectified_right, mask
                        )
            # If protected fruit union mask is provided
            elif protected_mask is not None:
                prot = np.asarray(protected_mask, dtype=np.uint8)
                if np.sum(prot) > 0:
                    num_labels, labels = cv2.connectedComponents(prot)
                    for lbl in range(1, num_labels):
                        comp_mask = (labels == lbl)
                        disparity_px, valid_mask = self._refine_fruit_mask_disparity(
                            disparity_px, valid_mask, rectified_left, rectified_right, comp_mask
                        )

        # 4. Greenhouse Backlit Sky Suppression (strictly preserving protected fruit surfaces)
        if self.sky_suppression_enabled:
            if rectified_left.ndim == 3:
                gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
            else:
                gray_left = rectified_left
                gray_right = rectified_right

            saturated_glare = (gray_left >= self.sky_threshold) & (gray_right >= self.sky_threshold)
            # RAFT interpolates plausible disparity across textureless glare, so the near-zero test only fits SGBM
            sky_noise_mask = saturated_glare if used_raft else saturated_glare & (disparity_px < 3.0)
            whiteout_mask = (gray_left >= 253) & (gray_right >= 253)
            sky_suppress = sky_noise_mask | whiteout_mask

            if protected_mask is not None:
                prot = np.asarray(protected_mask, dtype=bool)
                if prot.shape == sky_suppress.shape:
                    sky_suppress = sky_suppress & (~prot)

            valid_mask = valid_mask & (~sky_suppress)
            disparity_px[sky_suppress] = -1.0

        return disparity_px.astype(np.float32), valid_mask.astype(bool)

    def compute_depth(
        self, 
        disparity_map: np.ndarray, 
        focal_length_px: float, 
        baseline_mm: float
    ) -> np.ndarray:
        """
        Converts disparity map to metric depth map Z in millimeters:
        Z = (baseline_mm * focal_length_px) / disparity_map
        """
        depth_map = np.full(disparity_map.shape, np.nan, dtype=np.float32)
        valid = disparity_map > 0.0
        depth_map[valid] = (baseline_mm * focal_length_px) / disparity_map[valid]
        return depth_map


# Class alias for seamless backward compatibility
StereoMatcher = StereoMatcherSGBM
