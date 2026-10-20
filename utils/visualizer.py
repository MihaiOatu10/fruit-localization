"""
Visualization Utilities for 2D Instance Overlays, Disparity Maps, and 3D Spheres.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
import cv2


class Visualizer:
    """
    Renders 2D annotations, colored disparity maps, and 3D point cloud overlays.
    """

    @staticmethod
    def draw_detections_2d(
        image: np.ndarray,
        localized_fruits: List[dict],
        instances: Optional[List[Any]] = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlays segmented fruit masks, bounding boxes, centroid [Xc, Yc, Zc],
        and radius annotations onto the 2D image.
        """
        vis_img = image.copy()
        mask_overlay = image.copy()

        # If localized_fruits is empty, show detected 2D instances from segmentation
        if len(localized_fruits) == 0 and instances is not None and len(instances) > 0:
            for inst in instances:
                mask = getattr(inst, "sanitized_mask", None)
                if mask is None:
                    mask = getattr(inst, "binary_mask", None)
                bbox = getattr(inst, "bbox_xyxy", None)
                conf = getattr(inst, "confidence", 0.0)
                iid = getattr(inst, "instance_id", 1)

                if mask is not None and np.any(mask):
                    mask_overlay[mask] = (0, 165, 255)  # Orange for 2D detection without 3D fit

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    txt = f"ID:{iid} conf:{conf:.2f} (2D only)"
                    cv2.putText(vis_img, txt, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

            cv2.putText(
                vis_img, 
                f"Detections: {len(instances)} (Points Insufficient for 3D Sphere)", 
                (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 165, 255), 
                2
            )
        else:
            for fruit in localized_fruits:
                mask = fruit.get("mask")
                bbox = fruit.get("bbox")
                centroid = fruit.get("centroid_mm")
                radius = fruit.get("radius_mm")
                rmse = fruit.get("fit_rmse_mm", 0.0)
                fruit_id = fruit.get("instance_id", 0)

                # Draw mask overlay in bright green/orange
                if mask is not None and mask.any():
                    color = (0, 200, 50) if fruit.get("is_valid", True) else (0, 0, 255)
                    mask_overlay[mask] = color

                # Draw bounding box
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 200, 0), 2)

                    # Format text banner
                    if centroid is not None and radius is not None:
                        txt1 = f"ID:{fruit_id} R:{radius:.1f}mm"
                        txt2 = f"X:{centroid[0]:.1f} Y:{centroid[1]:.1f} Z:{centroid[2]:.1f}mm"
                        txt3 = f"RMSE:{rmse:.2f}mm"

                        cv2.putText(vis_img, txt1, (x1, max(15, y1 - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                        cv2.putText(vis_img, txt2, (x1, max(30, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                        cv2.putText(vis_img, txt3, (x1, max(45, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            status_color = (0, 255, 0) if len(localized_fruits) > 0 else (0, 0, 255)
            status_text = f"Localized: {len(localized_fruits)} Fruits" if len(localized_fruits) > 0 else "0 Fruits Detected"
            cv2.putText(vis_img, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Blend mask overlay
        cv2.addWeighted(mask_overlay, alpha, vis_img, 1.0 - alpha, 0, vis_img)
        return vis_img

    @staticmethod
    def colorize_disparity(
        disparity_map: np.ndarray, 
        min_disp: Optional[float] = None, 
        max_disp: Optional[float] = None
    ) -> np.ndarray:
        """
        Converts floating-point disparity map to JET colormap representation for visual inspection.
        """
        valid = (disparity_map > 0) & np.isfinite(disparity_map)
        if not np.any(valid):
            return np.zeros((*disparity_map.shape, 3), dtype=np.uint8)

        d_min = min_disp if min_disp is not None else float(np.min(disparity_map[valid]))
        d_max = max_disp if max_disp is not None else float(np.max(disparity_map[valid]))

        denom = max(1.0e-5, (d_max - d_min))
        normalized = np.clip((disparity_map - d_min) / denom, 0.0, 1.0)
        disp_uint8 = (normalized * 255).astype(np.uint8)
        color_disp = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)

        # Black out invalid pixels
        color_disp[~valid] = [0, 0, 0]
        return color_disp
