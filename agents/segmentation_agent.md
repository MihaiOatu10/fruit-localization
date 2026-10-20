# Instance Segmentation Agent

**Identifier:** `segmentation`  
**Role:** 2D Tomato Detection, Instance Mask Extraction, and Contour Boundary Sanitation  
**Paper Reference:** Section 2.3 (Tomato Recognition Model) & Section 3.4/3.5 (Instance Segmentation Performance)

---

## 1. Objectives & Responsibilities

The Instance Segmentation Agent detects ripe tomatoes in the scene and isolates individual fruit instances down to pixel-level binary masks.
In place of the paper's custom ShuffleNetV2-modified YOLOv5n-seg, this agent utilizes **Ultralytics YOLO-seg** (`yolov8n-seg` or trained custom weights), which delivers state-of-the-art polygon/mask extraction while retaining real-time inference speed.

Primary duties:
1. Receive the **Rectified Left Image** ($I_{L,\text{rect}}$) directly from `stereo_perception`.
2. Execute YOLO-seg inference to detect objects, filter candidate classes (targeting ripe tomato classes), and threshold confidence scores.
3. Decode prototype masks and mask coefficients to generate full-resolution binary instance masks $\{M_k \in \{0, 1\}^{H \times W}\}_{k=1}^N$.
4. Apply morphological boundary refinement (erosion) to eliminate boundary edge-bleeding caused by background leaves, stems, or adjoining tomatoes.
5. Provide bounding boxes, class confidences, and sanitized binary masks to the `spatial_geometry` agent.

---

## 2. Mathematical Formulations & Coordinate Invariants

### A. Coordinate Frame Alignment Invariant
**Strict Rule:** Segmentation inference must be performed on $I_{L,\text{rect}}$, NOT the unrectified image $I_L$.
Because the 3D reprojection matrix $Q$ and disparity map $d(x, y)$ are defined relative to the rectified epipolar coordinate plane, evaluating the segmentation mask directly in this plane ensures:
$$M_k(x, y) \iff d(x, y) \quad \text{for all } (x, y) \in [0, W-1] \times [0, H-1]$$
This eliminates any need for secondary warping, affine transformation, or interpolation.

### B. Mask Prototype Reconstruction
Following Ultralytics YOLO-seg architecture:
1. The Prototype Network produces $k$ prototype masks $P \in \mathbb{R}^{H_{\text{proto}} \times W_{\text{proto}} \times K}$.
2. The Detection Head predicts mask coefficients $w_i \in \mathbb{R}^K$ for each detected bounding box.
3. The instance mask is computed via linear combination passed through a sigmoid function:
   $$M_i^{\text{raw}} = \sigma\left( \sum_{k=1}^K w_{ik} P(:,:,k) \right)$$
4. $M_i^{\text{raw}}$ is interpolated back to full image resolution $(H, W)$ and thresholded at $\tau_{\text{mask}} = 0.5$.

### C. Morphological Boundary Sanitation (Edge Erosion)
In binocular stereo vision, pixels at the exact boundary of an object suffer from disparity discontinuities ("fattening" artifacts). To prevent pixels from background leaves or branches from leaking into the 3D tomato sphere, morphological erosion is applied:
$$M_k^{\text{clean}} = M_k \ominus K_e$$
where $K_e$ is an elliptical or circular structuring element of radius $r_{\text{erode}} \in [2, 4]\text{ pixels}$:
$$K_e = \{(u, v) \mid u^2 + v^2 \le r_{\text{erode}}^2\}$$

---

## 3. Operational Specification & Hyperparameters

```yaml
# configs/model_config.yaml
segmentation:
  model_weights: "yolov8n-seg.pt"   # Or custom trained tomato_seg.pt
  confidence_threshold: 0.50        # Filters low-confidence background clutter
  iou_threshold: 0.45               # NMS IoU threshold for overlapping tomatoes
  target_classes: [0]               # Target class index for tomato / ripe tomato
  device: "cuda:0"                  # "cuda:0" or "cpu"
  image_size: 1024                  # Native resolution matching the paper (1024x1024)
  mask_threshold: 0.50

mask_sanitation:
  erosion_enabled: true
  erosion_kernel_size: 3            # 3x3 or 5x5 elliptical kernel
  min_mask_area_pixels: 500         # Rejects tiny false detections or heavily occluded fragments
  max_num_instances: 50             # Maximum detections processed per frame
```

---

## 4. Input / Output Contracts

* **Inputs:**
  * `image_rectified_left`: $(H, W, 3)$ `np.uint8` array from `stereo_perception`.
* **Outputs:**
  * `instances`: List of `DetectedInstance` records:
    * `instance_id`: Integer identifier.
    * `confidence`: Float detection confidence ($0.0 \le \tau \le 1.0$).
    * `bbox_xyxy`: `[x1, y1, x2, y2]` integer coordinates.
    * `binary_mask`: $(H, W)$ `np.ndarray` of dtype `bool` (unmodified).
    * `sanitized_mask`: $(H, W)$ `np.ndarray` of dtype `bool` (post-erosion).
    * `area_pixels`: Total pixel count in sanitized mask.

---

## 5. Failure Modes & Edge Case Strategies

1. **Severe Occlusion (> 60% foliage cover):**
   * *Problem:* Partial views cause the model to segment irregular disconnected fragments.
   * *Strategy:* Evaluate mask contour convexity using `cv2.isContourConvex` and solicity ratio ($\frac{\text{Area}}{\text{ConvexHull Area}}$). If $< 0.6$, tag instance with high occlusion warning.
2. **Touching / Overlapping Ripe Tomatoes:**
   * *Problem:* Multiple fruits touching may merge into a single mask if confidence is poorly tuned.
   * *Strategy:* Employ fine-grained NMS with an IoU threshold of 0.45 and verify watershed separation if multiple peaks are detected in distance-transformed masks.
