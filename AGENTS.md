# Multi-Agent Architecture for 3D Tomato Detection and Localization

This document defines the Multi-Agent System (MAS) governing the implementation, orchestration, and continuous validation of the 3D tomato recognition and localization platform based on binocular vision and Ultralytics YOLO-seg (reproducing Zheng et al., *Agronomy* 2023, 13, 2339).

---

## 1. Executive Multi-Agent Topology

The system decomposes the physical and computational pipeline into five autonomous, specialized agents orchestrated by a deterministic event-driven pipeline.

```
                                  +------------------------------------+
                                  |   Pipeline Orchestrator Agent      |
                                  |   (pipeline_orchestrator_agent.md) |
                                  +-----------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +-------------------------------+                     +-------------------------------+
          | Stereo Perception Agent       |                     | Instance Segmentation Agent   |
          | (stereo_perception_agent.md)  |                     | (segmentation_agent.md)       |
          +---------------+---------------+                     +---------------+---------------+
                          |                                                     |
                          | Rectified Disparity d(x,y)                          | Binary Masks M_k
                          | & Reprojection Matrix Q                             | & Confidence BBoxes
                          +--------------------------+--------------------------+
                                                     |
                                                     v
                                      +-------------------------------+
                                      | Spatial Geometry Agent        |
                                      | (spatial_geometry_agent.md)   |
                                      +---------------+---------------+
                                                      |
                                                      v
                                        Tomato Spheres: (xc, yc, zc, r)
                                                      |
                                                      v
                                      +-------------------------------+
                                      | QA & Benchmark Agent          |
                                      | (qa_benchmark_agent.md)       |
                                      +-------------------------------+
```

---

## 2. Agent Registry & Roles

| Agent File | Agent Identifier | Domain Responsibility | Primary Stack / Libraries |
| :--- | :--- | :--- | :--- |
| [`agents/pipeline_orchestrator_agent.md`](file:///mnt/c/users/themi/documents/projects/tomatolocalization/agents/pipeline_orchestrator_agent.md) | `orchestrator` | Pipeline lifecycle, task dispatch, state synchronization, and robotic payload output | Python `asyncio` / standard library, ROS2-ready payload contracts |
| [`agents/stereo_perception_agent.md`](file:///mnt/c/users/themi/documents/projects/tomatolocalization/agents/stereo_perception_agent.md) | `stereo_perception` | Binocular calibration, epipolar rectification, OpenCV StereoSGBM + WLS disparity estimation | OpenCV (`cv2.stereoRectify`, `cv2.StereoSGBM`, `cv2.ximgproc`), NumPy |
| [`agents/segmentation_agent.md`](file:///mnt/c/users/themi/documents/projects/tomatolocalization/agents/segmentation_agent.md) | `segmentation` | 2D fruit detection, instance mask extraction on rectified left frames, morphological edge sanitation | Ultralytics (`ultralytics.YOLO`), PyTorch, OpenCV |
| [`agents/spatial_geometry_agent.md`](file:///mnt/c/users/themi/documents/projects/tomatolocalization/agents/spatial_geometry_agent.md) | `spatial_geometry` | $Q$-matrix reprojection, mask gating, voxel downsampling, LOF outlier filtering, least-squares sphere fitting | NumPy, SciPy (`scipy.optimize.least_squares`), Scikit-learn (`LocalOutlierFactor`) |
| [`agents/qa_benchmark_agent.md`](file:///mnt/c/users/themi/documents/projects/tomatolocalization/agents/qa_benchmark_agent.md) | `qa_benchmark` | Synthetic geometry testing, error regression tracking against paper benchmarks ($e_{\text{pos}} \le \pm 5.0\text{ mm}, e_r \le \pm 3.9\text{ mm}$) | Pytest, NumPy, Matplotlib |

---

## 3. Global Invariants & Coordinate Consistency

To guarantee mathematical consistency across all agent handoffs, the following invariants must never be violated:

1. **Epipolar Frame Synchronization:** Instance segmentation MUST operate strictly on the **Rectified Left Image** ($I_{L,\text{rect}}$). Inference must never run on raw distorted left frames ($I_L$) to ensure the pixel index $(u, v)$ in mask $M_k$ matches $(u, v)$ in disparity map $d$ with zero geometric parallax.
2. **Standardized Camera Coordinate Frame ($C_L$):**
   * Origin: Rectified optical center of the Left camera.
   * $+X$: Points horizontally towards the right camera along baseline $B$.
   * $+Y$: Points vertically downward perpendicular to the optical axis.
   * $+Z$: Points forward into the 3D scene along the optical axis (depth in millimeters).
3. **Disparity Representation:** Disparity values must be floating-point values in pixels (computed by dividing raw 16-bit integer SGBM disparity by 16.0). Invalid or occluded disparities must be marked as `NaN` or negative.
4. **Physical Dimension Scale:** All spatial outputs (baseline $B$, camera coordinates $[X, Y, Z]$, sphere centroid $[x_c, y_c, z_c]$, and radius $r$) are in **millimeters (mm)**.

---

## 4. Inter-Agent Communication Schemas

Data exchanged between agents uses explicit dictionary or dataclass payloads:

```python
# Shared Payload Structures
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class RectificationResult:
    left_rectified: np.ndarray      # (H, W, 3) uint8
    right_rectified: np.ndarray     # (H, W, 3) uint8
    Q_matrix: np.ndarray             # (4, 4) float64 reprojection matrix
    roi_left: tuple                  # (x, y, w, h) valid rectangle

@dataclass
class DisparityResult:
    disparity_map: np.ndarray       # (H, W) float32 in pixels
    valid_mask: np.ndarray          # (H, W) bool

@dataclass
class DetectedInstance:
    instance_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple                # (x1, y1, x2, y2)
    binary_mask: np.ndarray         # (H, W) uint8 / bool (rectified coordinates)
    sanitized_mask: np.ndarray      # (H, W) uint8 after morphological erosion

@dataclass
class LocalizedTomato:
    instance_id: int
    confidence: float
    centroid_camera_frame: np.ndarray # [xc, yc, zc] in mm
    radius_mm: float                  # fitted radius in mm
    num_points_fitted: int
    fit_rmse_mm: float               # Root Mean Square Error of sphere residuals
    occlusion_flag: bool             # True if point density or coverage < 50%
```

---

## 5. Execution Workflow & Step Transition Matrix

```
[Trigger: Stereo Frame Pair (I_L, I_R)]
    |
    v
[1. stereo_perception]: stereo_rectify(I_L, I_R) -> (I_L_rect, I_R_rect, Q)
    |
    +---------------------------------------+
    |                                       |
    v                                       v
[2. segmentation]:                     [3. stereo_perception]:
   detect_and_segment(I_L_rect)            compute_disparity(I_L_rect, I_R_rect)
   -> Instances [M_1, ... M_k]             -> Disparity d(x, y)
    |                                       |
    +-------------------+-------------------+
                        |
                        v
[4. spatial_geometry]:
   For each instance M_k:
     a. Reproject masked points using Q matrix -> P_k
     b. Voxel grid downsample -> P_k_down
     c. Local Outlier Factor (LOF) filtering -> P_k_clean
     d. Algebraic + Non-linear Least Squares Sphere Fitting -> (xc, yc, zc, r, rmse)
    |
    v
[5. qa_benchmark]:
   Validate bounds: 280mm <= zc <= 480mm, r in [20mm, 60mm], rmse < 3.0mm
    |
    v
[6. orchestrator]:
   Dispatch finalized LocalizedTomato payload to picking controller or visualizer
```
