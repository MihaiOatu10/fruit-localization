"""
End-to-End Localization Pipeline.
Coordinates epipolar rectification, instance segmentation, RAFT-Stereo / StereoSGBM disparity matching,
3D point cloud conditioning, and least-squares sphere fitting.
"""

import time
import traceback
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

from core.calibration import StereoCalibrator
from core.stereo_matcher import StereoMatcherSGBM
from core.segmentation import FruitSegmenter, DetectedInstance
from core.pointcloud import PointCloudProcessor
from core.sphere_fit import SphereFitter, SphereFitResult


class TomatoLocalizationPipeline:
    """
    Main orchestration pipeline reproducing the 3D fruit localization framework.
    """

    def __init__(
        self,
        calib_config_path: str = "configs/camera_calib.yaml",
        stereo_config_path: str = "configs/stereo_sgbm.yaml",
        geometry_config_path: str = "configs/geometry_config.yaml",
        model_config_path: str = "configs/model_config.yaml",
        model_weights: Optional[str] = None,
        conf_threshold: float = 0.10,
        device: str = "cpu",
    ):
        # 1. Initialize calibration & rectification
        self.calibrator = StereoCalibrator(calib_config_path)

        # 2. Initialize stereo matcher
        self.matcher = StereoMatcherSGBM(stereo_config_path)

        # 3. Resolve segmentation model weights and parameters
        resolved_weights = model_weights
        target_classes = None
        iou_threshold = 0.45
        erosion_kernel_size = 3
        min_mask_area = 150

        m_cfg_path = Path(model_config_path)
        if not m_cfg_path.is_file():
            alt_cfg = Path(__file__).resolve().parent / m_cfg_path
            if alt_cfg.is_file():
                m_cfg_path = alt_cfg

        if m_cfg_path.is_file():
            try:
                import yaml
                with open(m_cfg_path, "r", encoding="utf-8") as f:
                    m_cfg = yaml.safe_load(f)
                seg_cfg = m_cfg.get("segmentation", {})
                mask_cfg = m_cfg.get("mask_sanitation", {})
                if resolved_weights is None:
                    cand = seg_cfg.get("model_weights")
                    if cand and Path(cand).is_file():
                        resolved_weights = cand
                    elif seg_cfg.get("fallback_weights") and Path(seg_cfg.get("fallback_weights")).is_file():
                        resolved_weights = seg_cfg.get("fallback_weights")
                target_classes = seg_cfg.get("target_classes", None)
                iou_threshold = float(seg_cfg.get("iou_threshold", 0.45))
                erosion_kernel_size = int(mask_cfg.get("erosion_kernel_size", 3))
                min_mask_area = int(mask_cfg.get("min_mask_area_pixels", 150))
            except Exception as e:
                print(f"[Pipeline] Notice: Failed to read model config '{m_cfg_path}': {e}")

        if resolved_weights is None:
            for cand in [
                "weights/best.pt",
                "runs/train-seg/fruit_segmentation/weights/best.pt",
                "yolov8n-seg.pt",
            ]:
                cand_p = Path(cand)
                if not cand_p.is_file():
                    alt_cand = Path(__file__).resolve().parent / cand_p
                    if alt_cand.is_file():
                        cand_p = alt_cand
                if cand_p.is_file():
                    resolved_weights = str(cand_p)
                    break

        # 4. Initialize segmentation model
        self.segmenter = FruitSegmenter(
            weights_path=resolved_weights,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes,
            device=device,
            erosion_kernel_size=erosion_kernel_size,
            min_mask_area=min_mask_area,
        )

        # 5. Initialize point cloud processor
        self.pc_processor = PointCloudProcessor(geometry_config_path)

        # 6. Initialize sphere fitter
        self.sphere_fitter = SphereFitter(geometry_config_path)
        self.sphere_fitter.set_stereo_geometry(self.calibrator.get_focal_length(), self.calibrator.get_baseline_mm())

    def process_pair(
        self,
        image_left: np.ndarray,
        image_right: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Processes a single binocular stereo pair with protected instance iteration.
        """
        timing = {}

        # Stage 1: Stereo Rectification
        t0 = time.perf_counter()
        rect_left, rect_right = self.calibrator.rectify_pair(image_left, image_right)
        timing["rectification_ms"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2A: Instance Segmentation
        t0 = time.perf_counter()
        instances: List[DetectedInstance] = self.segmenter.segment(rect_left)
        timing["segmentation_ms"] = (time.perf_counter() - t0) * 1000.0
        print(f"[Pipeline] Segmentation phase produced {len(instances)} instances.", flush=True)

        # Stage 2B: Disparity Estimation
        t0 = time.perf_counter()
        h, w = rect_left.shape[:2]
        fruit_union_mask = np.zeros((h, w), dtype=bool)
        for inst in instances:
            fruit_union_mask |= inst.binary_mask

        disparity_map, valid_mask = self.matcher.compute_disparity(
            rect_left, rect_right, protected_mask=fruit_union_mask, instances=instances
        )
        timing["disparity_ms"] = (time.perf_counter() - t0) * 1000.0

        num_valid_disp = int(np.sum(valid_mask))
        total_pixels = valid_mask.size
        fruit_valid_disp = int(np.sum(valid_mask & fruit_union_mask))
        fruit_total_px = int(np.sum(fruit_union_mask))
        pct_scene = (num_valid_disp / total_pixels) * 100.0
        pct_fruit = (fruit_valid_disp / max(1, fruit_total_px)) * 100.0
        print(
            f"[Pipeline] Disparity computed: {num_valid_disp}/{total_pixels} scene px valid ({pct_scene:.1f}%), "
            f"{fruit_valid_disp}/{fruit_total_px} fruit px valid ({pct_fruit:.1f}%).",
            flush=True
        )

        # Stage 3: Reproject Full Scene to 3D Camera Coordinates
        t0 = time.perf_counter()
        Q_matrix = self.calibrator.get_q_matrix()
        points_3d = self.pc_processor.reproject_disparity_to_3d(disparity_map, Q_matrix)
        K_rect_inv = np.linalg.inv(self.calibrator.P1[:3, :3])

        localized_fruits = []
        for idx, inst in enumerate(instances, 1):
            # 1. Verificare primară a măștii
            if inst.sanitized_mask is None or np.count_nonzero(inst.sanitized_mask) < 10:
                print(f"[Pipeline] Instance {inst.instance_id}: Mască invalidă/prea mică; skipping.", flush=True)
                continue

            try:
                # 2. Conditionare Point Cloud
                clean_pts, has_enough_pts = self.pc_processor.process_instance(
                    points_3d=points_3d,
                    instance_mask=inst.sanitized_mask,
                    valid_disp_mask=valid_mask,
                )
            except Exception as e:
                print(f"[Pipeline Warning] Crash mecanic la PointCloud pentru Instance {inst.instance_id}: {e}", flush=True)
                continue

            if clean_pts is None or len(clean_pts) < 10:
                print(
                    f"[Pipeline] Instance {inst.instance_id} ('{inst.class_name}', conf={inst.confidence:.2f}): "
                    f"mask={inst.area_pixels} px -> {0 if clean_pts is None else len(clean_pts)} clean 3D pts. Skipping sphere fit.",
                    flush=True
                )
                continue

            # Planar clouds (constant-disparity fill, no depth relief) make the sphere fit degenerate
            z_relief = float(np.ptp(clean_pts[:, 2]))
            if z_relief < 1.0:
                print(
                    f"[Pipeline] Instance {inst.instance_id}: degenerate planar cloud "
                    f"(Z relief {z_relief:.2f} mm at Z~{np.median(clean_pts[:, 2]):.0f} mm); skipping sphere fit.",
                    flush=True
                )
                continue

            try:
                # 3. Spherical Fitting, centre constrained toward the camera ray through the 2D mask centroid
                ys, xs = np.nonzero(inst.sanitized_mask)
                ray_dir = K_rect_inv @ np.array([xs.mean(), ys.mean(), 1.0])
                fit_res: SphereFitResult = self.sphere_fitter.fit(clean_pts, ray_dir=ray_dir)
            except Exception as e:
                print(f"[Pipeline Warning] Fitting sferic eșuat la Instance {inst.instance_id}: {e}", flush=True)
                continue

            # 4. Validare fizică Z (între 20 cm și 5 metri)
            if fit_res is None or fit_res.centroid is None:
                continue

            z_depth = fit_res.centroid[2]
            z_valid_min, z_valid_max = self.sphere_fitter.valid_centroid_z_mm
            if z_depth < z_valid_min or z_depth > z_valid_max:
                print(
                    f"[Pipeline] Instance {inst.instance_id}: Z aberant ({z_depth:.1f}mm); skipping invalid fit.",
                    flush=True
                )
                continue

            print(
                f"[Pipeline] Instance {inst.instance_id} ('{inst.class_name}', conf={inst.confidence:.2f}): "
                f"mask={inst.area_pixels} px -> {len(clean_pts)} clean 3D pts. Fitted XYZ=[{fit_res.centroid[0]:.1f},{fit_res.centroid[1]:.1f},{fit_res.centroid[2]:.1f}]mm",
                flush=True
            )

            fruit_record = {
                "instance_id": inst.instance_id,
                "class_name": inst.class_name,
                "confidence": inst.confidence,
                "bbox": inst.bbox_xyxy,
                "mask": inst.sanitized_mask,
                "centroid_mm": fit_res.centroid,
                "radius_mm": fit_res.radius_mm,
                "diameter_mm": fit_res.radius_mm * 2.0,
                "fit_rmse_mm": fit_res.rmse_mm,
                "point_count": fit_res.num_points,
                "is_valid": fit_res.is_valid and has_enough_pts,
            }
            localized_fruits.append(fruit_record)

        timing["geometry_and_fit_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["total_latency_ms"] = sum(timing.values())

        return {
            "rectified_left": rect_left,
            "rectified_right": rect_right,
            "disparity_map": disparity_map,
            "valid_mask": valid_mask,
            "instances": instances,
            "localized_fruits": localized_fruits,
            "telemetry": timing,
        }