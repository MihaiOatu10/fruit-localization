"""
Instance Segmentation Module for Fruit Detection and Mask Extraction.
Wraps Ultralytics YOLO-seg with morphological boundary sanitation.
"""

import os
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
import cv2


@dataclass
class DetectedInstance:
    """Represents a detected fruit instance with masks and spatial attributes."""
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray        # [x1, y1, x2, y2]
    binary_mask: np.ndarray      # (H, W) bool, raw predicted mask
    sanitized_mask: np.ndarray   # (H, W) bool, post-erosion clean mask
    area_pixels: int


class FruitSegmenter:
    """
    Executes instance segmentation on rectified left camera frames.
    Extracts high-resolution masks and performs morphological edge erosion.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
        target_classes: Optional[Union[List[Union[int, str]], int, str]] = None,
        device: str = "cpu",
        erosion_kernel_size: int = 3,
        min_mask_area: int = 150,
    ):
        # Auto-discover fine-tuned weights if not explicitly passed
        if weights_path is None:
            for cand in [
                "weights/best.pt",
                "runs/train-seg/fruit_segmentation/weights/best.pt",
                "yolov8n-seg.pt",
            ]:
                cand_path = Path(cand)
                if cand_path.is_file():
                    weights_path = str(cand_path)
                    break

        self.weights_path = weights_path or "weights/best.pt"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Standardize target classes list
        if target_classes is None:
            self.target_classes = None
        elif isinstance(target_classes, (list, tuple, set)):
            self.target_classes = list(target_classes)
        else:
            self.target_classes = [target_classes]

        self.device = device
        self.erosion_kernel_size = erosion_kernel_size
        self.min_mask_area = min_mask_area

        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempts to load Ultralytics YOLO model if weights exist on disk or can be downloaded."""
        try:
            if not self.weights_path:
                self.model = None
                return
            weights_file = Path(self.weights_path)
            if not weights_file.is_file():
                alt_weights = Path(__file__).resolve().parent.parent / weights_file
                if alt_weights.is_file():
                    self.weights_path = str(alt_weights)
                elif not str(self.weights_path).endswith(".pt"):
                    self.model = None
                    return
            # Skip ultralytics' import-time DNS probe (is_online -> getaddrinfo), which segfaults on some
            # Windows hosts; weights are local, so no network is needed. Override with YOLO_OFFLINE=0.
            os.environ.setdefault("YOLO_OFFLINE", "1")
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            print(f"[FruitSegmenter] Successfully loaded model weights: {self.weights_path}")
        except Exception as e:
            print(f"[FruitSegmenter] Warning: Failed to load YOLO weights '{self.weights_path}': {e}")
            self.model = None

    def _format_input_image(self, image: np.ndarray) -> np.ndarray:
        """
        Formats and validates the rectified image for the YOLO model:
        1. Ensures shape is 3-channel (H, W, 3).
        2. Handles grayscale (H, W) or (H, W, 1) and converts to 3-channel BGR.
        3. Handles 4-channel BGRA (H, W, 4) and converts to 3-channel BGR.
        4. Handles single-image batch dimension (1, H, W, 3) or channels-first (3, H, W).
        5. Handles normalization: converts float in [0.0, 1.0] to [0, 255] uint8.
        6. Ensures C-contiguous array layout.
        """
        if image is None:
            raise ValueError("[FruitSegmenter] Received None input image")

        img = np.asarray(image)

        # Remove single-image batch dimension if present (1, H, W, C)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Transpose channels-first (3, H, W) to (H, W, 3)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        # Check and coerce channel count to 3
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"[FruitSegmenter] Unsupported channel dimension: {img.shape[2]}")
        else:
            raise ValueError(f"[FruitSegmenter] Unsupported image dimensions: {img.shape}")

        # Normalization and dtype validation: ensure uint8 in range [0, 255]
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Ensure C-contiguous memory layout
        img = np.ascontiguousarray(img)
        return img

    def _is_target_class(self, class_id: int, class_name: str) -> bool:
        """
        Validates whether a detected class index or name matches target criteria.
        Rule 1: Class index 0 is ALWAYS accepted regardless of class name strings.
        """
        # Strictly accept class index 0 regardless of class name strings
        if class_id == 0:
            return True

        # If no target classes specified, accept all classes
        if self.target_classes is None:
            return True

        # Check target classes against class_id and class_name
        for tc in self.target_classes:
            if isinstance(tc, int) and tc == class_id:
                return True
            if isinstance(tc, str):
                if tc.isdigit() and int(tc) == class_id:
                    return True
                if tc.lower() == str(class_name).lower():
                    return True

        return False

    def sanitize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological erosion using an elliptical structuring element
        to strip boundary disparity bleeding caused by background foliage.
        """
        if self.erosion_kernel_size <= 1:
            return mask.copy()

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erosion_kernel_size, self.erosion_kernel_size)
        )
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        # Ensure erosion did not completely destroy the mask
        if np.sum(eroded) < (0.2 * np.sum(mask)):
            return mask.copy()
        return eroded.astype(bool)

    def segment(self, image_rectified_left: np.ndarray) -> List[DetectedInstance]:
        """
        Performs instance segmentation on the rectified left image.
        Returns list of DetectedInstance records.
        """
        # Step 1: Format rectified image correctly (shape, channels, normalization, contiguous)
        formatted_img = self._format_input_image(image_rectified_left)
        h, w = formatted_img.shape[:2]
        instances: List[DetectedInstance] = []

        if self.model is not None:
            # Predict using formatted BGR image (OpenCV standard expected by YOLO)
            # Note: Do not pass classes filter to YOLO predict so we can inspect and log all raw detections
            try:
                results = self.model.predict(
                    source=formatted_img,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                    retina_masks=True,
                )
            except Exception as e:
                if str(self.device).lower() != "cpu":
                    print(f"[FruitSegmenter] Notice: Device '{self.device}' failed ({e}). Retrying on 'cpu'...")
                    self.device = "cpu"
                    results = self.model.predict(
                        source=formatted_img,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device="cpu",
                        verbose=False,
                        retina_masks=True,
                    )
                else:
                    raise e

            # Fallback color check: if 0 detections with BGR, test RGB conversion
            num_raw_init = (
                len(results[0].boxes)
                if (results and len(results) > 0 and results[0].boxes is not None)
                else 0
            )
            if num_raw_init == 0:
                img_rgb = cv2.cvtColor(formatted_img, cv2.COLOR_BGR2RGB)
                try:
                    results_rgb = self.model.predict(
                        source=img_rgb,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device=self.device,
                        verbose=False,
                        retina_masks=True,
                    )
                except Exception:
                    results_rgb = self.model.predict(
                        source=img_rgb,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device="cpu",
                        verbose=False,
                        retina_masks=True,
                    )
                num_raw_rgb = (
                    len(results_rgb[0].boxes)
                    if (results_rgb and len(results_rgb) > 0 and results_rgb[0].boxes is not None)
                    else 0
                )
                if num_raw_rgb > 0:
                    print(
                        f"[FruitSegmenter] Found {num_raw_rgb} detections using RGB color formatting."
                    )
                    results = results_rgb

            # Step 2: Print debug logs of raw YOLO detections BEFORE any filtering is applied
            num_raw = (
                len(results[0].boxes)
                if (results and len(results) > 0 and results[0].boxes is not None)
                else 0
            )
            print(
                f"[FruitSegmenter] Raw YOLO detections: {num_raw} boxes detected before any filtering (conf >= {self.conf_threshold})."
            )

            if results and len(results) > 0 and results[0].boxes is not None and num_raw > 0:
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)

                # Log details for every raw detection
                for idx in range(num_raw):
                    c_id = int(clss[idx])
                    c_name = (
                        r.names[c_id]
                        if (hasattr(r, "names") and isinstance(r.names, dict) and c_id in r.names)
                        else str(c_id)
                    )
                    b_coords = [round(float(coord), 1) for coord in boxes[idx]]
                    print(
                        f"  [Raw Box {idx + 1}] class_id={c_id} ('{c_name}'), conf={float(confs[idx]):.4f}, bbox={b_coords}"
                    )

                # Extract mask prototypes if available
                has_masks = (
                    r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0
                )
                if has_masks:
                    masks_data = r.masks.data.cpu().numpy()
                else:
                    print(
                        "[FruitSegmenter] Notice: YOLO model produced bounding boxes but no mask prototypes. Generating geometry masks from bounding boxes."
                    )

                for idx in range(num_raw):
                    c_id = int(clss[idx])
                    c_name = (
                        r.names[c_id]
                        if (hasattr(r, "names") and isinstance(r.names, dict) and c_id in r.names)
                        else str(c_id)
                    )

                    # Class filtering: strictly accept class index 0 regardless of class name strings
                    if not self._is_target_class(c_id, c_name):
                        print(
                            f"[FruitSegmenter] Filtered out box {idx + 1}: class_id={c_id} ('{c_name}') not in target classes"
                        )
                        continue

                    # Mask extraction: from YOLO segmentation head or bounding box ellipse
                    if has_masks and idx < len(masks_data):
                        raw_m = masks_data[idx]
                        if raw_m.shape != (h, w):
                            raw_m = self._unletterbox_mask(raw_m, h, w)
                        binary_mask = raw_m > 0.5
                    else:
                        bx1, by1, bx2, by2 = map(int, boxes[idx])
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(w, bx2), min(h, by2)
                        center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                        axes = (max(1, (bx2 - bx1) // 2), max(1, (by2 - by1) // 2))
                        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 1, -1)
                        binary_mask = ellipse_mask.astype(bool)

                    area = int(np.sum(binary_mask))
                    if area < self.min_mask_area:
                        print(
                            f"[FruitSegmenter] Filtered out box {idx + 1}: mask area {area} < min_mask_area {self.min_mask_area}"
                        )
                        continue

                    sanitized = self.sanitize_mask(binary_mask)
                    if int(np.sum(sanitized)) == 0:
                        sanitized = binary_mask

                    instances.append(
                        DetectedInstance(
                            instance_id=len(instances) + 1,
                            class_id=c_id,
                            class_name=c_name,
                            confidence=float(confs[idx]),
                            bbox_xyxy=boxes[idx],
                            binary_mask=binary_mask,
                            sanitized_mask=sanitized,
                            area_pixels=area,
                        )
                    )

                if len(instances) > 0:
                    print(
                        f"[FruitSegmenter] Segmentation successfully retained {len(instances)} instances."
                    )
                    return instances

        # Fallback heuristic detector (color-based segmentation in greenhouse/fruit hues)
        # Used if YOLO model weights are uninitialized, detected 0 objects, or in test environments
        print("[FruitSegmenter] YOLO returned 0 instances; invoking heuristic fallback segmentation.")
        return self._fallback_color_segment(formatted_img)

    @staticmethod
    def _unletterbox_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Maps a mask at YOLO's letterboxed inference size back to (h, w): crop the centred padding first,
        then resize. A plain resize stretches the padding into the mask (e.g. 384x640 -> 720x1280 squeezes
        masks vertically by up to ~20 px at the frame edges).
        """
        mh, mw = mask.shape[:2]
        gain = min(mh / h, mw / w)
        pad_x, pad_y = (mw - w * gain) / 2.0, (mh - h * gain) / 2.0
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        bottom, right = int(round(mh - pad_y + 0.1)), int(round(mw - pad_x + 0.1))
        cropped = mask[top:bottom, left:right].astype(np.float32)
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _fallback_color_segment(self, image_bgr: np.ndarray) -> List[DetectedInstance]:
        """
        Heuristic fallback segmentation:
        1. HSV color thresholding for ripe red/orange tomatoes and kiwifruit brown/olive.
        2. Backlit canopy silhouette segmentation for dark rounded fruits against bright sky.
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 1. Broad color mask covering tomato red/orange and kiwifruit brown/olive
        mask1 = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([25, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([150, 20, 20]), np.array([180, 255, 255]))
        mask_kiwi = cv2.inRange(hsv, np.array([10, 15, 20]), np.array([45, 255, 220]))
        color_mask = cv2.bitwise_or(mask1, mask2)
        color_mask = cv2.bitwise_or(color_mask, mask_kiwi)

        # 2. Backlit silhouette detection:
        # Detect dark rounded bodies in canopy when background sky is overexposed
        sky_mask = (hsv[:, :, 2] > 230)
        has_backlit_sky = float(np.mean(sky_mask)) > 0.03
        if has_backlit_sky:
            # Contrast mask: dark canopy elements against backlit sky
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # Exclude sky and deep background shadows
            canopy_mask = ((gray < 160) & (gray > 15)).astype(np.uint8) * 255
            # Morphological opening to break thin leaf stems
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            canopy_opened = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, k_open)
            color_mask = cv2.bitwise_or(color_mask, canopy_opened)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        instances = []
        inst_id = 1

        for c in contours:
            area = cv2.contourArea(c)
            # Bound area between min threshold and max reasonable fruit size (prevent whole tree branches)
            if area < self.min_mask_area or area > 35000:
                continue

            # Check convexity / compactness
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = float(area) / hull_area
            # Fruits are predominantly convex blobs (solidity > 0.65)
            if solidity < 0.65:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / bh
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                continue

            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [c], -1, 1, thickness=-1)
            bin_mask = m.astype(bool)
            sanitized = self.sanitize_mask(bin_mask)

            instances.append(
                DetectedInstance(
                    instance_id=inst_id,
                    class_id=0,
                    class_name="fruit",
                    confidence=0.85,
                    bbox_xyxy=np.array([x, y, x + bw, y + bh], dtype=np.float32),
                    binary_mask=bin_mask,
                    sanitized_mask=sanitized,
                    area_pixels=int(area),
                )
            )
            inst_id += 1

        return instances
