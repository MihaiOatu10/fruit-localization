# Stereo Perception Agent

**Identifier:** `stereo_perception`  
**Role:** Stereo Camera Calibration, Epipolar Rectification, and Disparity Estimation  
**Paper Reference:** Section 2.4.1 (Binocular Calibration) & Section 2.4.2 (Stereo Disparity Computation)

---

## 1. Objectives & Responsibilities

The Stereo Perception Agent manages the optical transformation from raw binocular image sensors into rectified epipolar geometry and high-density disparity fields.
While the original paper deployed RAFT-Stereo (deep recurrent optical flow), this agent substitutes RAFT-Stereo with a hardware-efficient, real-time pipeline based on **OpenCV StereoSGBM (Semi-Global Block Matching)** accelerated with a **Weighted Least Squares (WLS) Disparity Filter**.

Primary duties:
1. Load and validate binocular camera intrinsic matrices ($K_1, K_2$), lens distortion parameters ($D_1, D_2$), mutual rotation matrix ($R$), and translation vector ($T$).
2. Compute rectification rotation matrices ($R_1, R_2$), projection matrices ($P_1, P_2$), and the $4 \times 4$ perspective disparity-to-depth reprojection matrix ($Q$).
3. Undistort and rectify raw left and right stereo image pairs $(I_L, I_R)$ into epipolar-aligned pairs $(I_{L,\text{rect}}, I_{R,\text{rect}})$ where epipolar lines are horizontally parallel ($y_l = y_r$).
4. Compute dense disparity maps using `cv2.StereoSGBM` and execute edge-preserving disparity filtering with `cv2.ximgproc.createDisparityWLSFilter`.
5. Provide sub-pixel disparity representations and propagate calibration parameters and the $Q$ matrix to downstream agents.

---

## 2. Mathematical Formulations

### A. Epipolar Rectification & Reprojection Matrix $Q$
Camera calibration establishes intrinsic parameters:
$$K_i = \begin{bmatrix} f_{xi} & 0 & c_{xi} \\ 0 & f_{yi} & c_{yi} \\ 0 & 0 & 1 \end{bmatrix}, \quad D_i = [k_1, k_2, p_1, p_2, k_3]$$

Extrinsic transformation from Left to Right camera:
$$X_R = R X_L + T, \quad \|T\| = B = 74.676\text{ mm}$$

Applying `cv2.stereoRectify` produces projection matrices $P_1, P_2$ with shared focal length $f$ and vertical optical center $c_y$:
$$P_1 = \begin{bmatrix} f & 0 & c_{x1} & 0 \\ 0 & f & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}, \quad P_2 = \begin{bmatrix} f & 0 & c_{x2} & -f \cdot B \\ 0 & f & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

The $4 \times 4$ disparity-to-depth reprojection matrix $Q$ is formed as:
$$Q = \begin{bmatrix}
1 & 0 & 0 & -c_{x1} \\
0 & 1 & 0 & -c_y \\
0 & 0 & 0 & f \\
0 & 0 & -\frac{1}{B} & \frac{c_{x1} - c_{x2}}{B}
\end{bmatrix}$$

### B. StereoSGBM Energy Minimization
For disparity computation, SGBM minimizes the multi-path dynamic energy functional:
$$E(d) = \sum_{p} \left( C(p, d_p) + \sum_{q \in N_p} P_1 \cdot \mathbb{I}[|d_p - d_q| = 1] + \sum_{q \in N_p} P_2 \cdot \mathbb{I}[|d_p - d_q| > 1] \right)$$

where:
* $C(p, d_p)$ is the matching cost (Birchfield-Tomasi or census transform).
* $P_1 = 8 \cdot C_{\text{channels}} \cdot \text{blockSize}^2$ penalizes small disparity variations (smooth surfaces).
* $P_2 = 32 \cdot C_{\text{channels}} \cdot \text{blockSize}^2$ penalizes large disparity jumps (discontinuity boundaries).

### C. WLS Disparity Filtering
To prevent stereo noise and boundary halos typical of greenhouse foliage, the WLS filter solves an optimization problem guided by the rectified left RGB image $I_{L,\text{rect}}$:
$$\arg\min_{d_{\text{filtered}}} \sum_{p} (d_{\text{filtered}}(p) - d_{\text{left}}(p))^2 + \lambda \sum_{p} \sum_{q \in N_p} w(p, q) (d_{\text{filtered}}(p) - d_{\text{filtered}}(q))^2$$
where $\lambda$ controls regularization strength and $\sigma$ (in $w(p, q)$) modulates edge sensitivity.

---

## 3. Operational Specification & Hyperparameters

```yaml
# configs/stereo_sgbm.yaml
stereo_matcher:
  min_disparity: 0
  num_disparities: 128          # Must be divisible by 16; covers 280mm - 480mm working range
  block_size: 5                 # Matched block window (3 to 7)
  p1: 600                       # 8 * 3 * 5^2
  p2: 2400                      # 32 * 3 * 5^2
  disp12_max_diff: 1            # Left-Right consistency check tolerance
  pre_filter_cap: 31
  uniqueness_ratio: 15          # Rejects ambiguous textureless matches
  speckle_window_size: 100
  speckle_range: 2
  mode: "MODE_SGBM_3WAY"        # Faster 3-path optimization

wls_filter:
  enabled: true
  lambda: 8000.0                # Smoothing regularizer
  sigma: 1.5                    # Edge sensitivity relative to left RGB frame
```

---

## 4. Input / Output Contracts

* **Inputs:**
  * `left_image`: $(H, W, 3)$ `np.uint8` in BGR/RGB format.
  * `right_image`: $(H, W, 3)$ `np.uint8` in BGR/RGB format.
  * `calibration_params`: File path or dictionary containing $K_1, K_2, D_1, D_2, R, T$.
* **Outputs:**
  * `left_rectified`: $(H, W, 3)$ `np.uint8` rectified image (handed over to `segmentation`).
  * `right_rectified`: $(H, W, 3)$ `np.uint8` rectified image.
  * `disparity_map`: $(H, W)$ `np.float32` calibrated disparity in pixels.
  * `Q_matrix`: $(4, 4)$ `np.float64` reprojection matrix (handed over to `spatial_geometry`).
  * `valid_mask`: $(H, W)$ `np.bool_` designating pixels with positive disparity ($d > 0$).

---

## 5. Failure Modes & Recovery Strategies

1. **Direct Sunlight Overexposure & Specular Glare:**
   * *Problem:* Saturated white pixels on tomato surfaces lack texture, causing SGBM matching dropouts.
   * *Recovery:* WLS filtering interpolates across specular voids using surrounding valid tomato surface priors; low-confidence outliers are marked as invalid rather than returning erroneous depths.
2. **Epipolar Misalignment:**
   * *Problem:* Mechanical drift changes camera alignment, violating horizontal epipolar linearity ($y_l \ne y_r$).
   * *Recovery:* Perform periodic epipolar error validation (mean vertical disparity of matched SIFT/ORB features must be $< 0.5\text{ px}$). If exceeded, trigger re-calibration warning.
