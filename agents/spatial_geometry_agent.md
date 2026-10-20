# Spatial Geometry Agent

**Identifier:** `spatial_geometry`  
**Role:** 3D Point Cloud Reprojection, Point Cloud Preprocessing (Voxel + LOF), and Least-Squares Sphere Fitting  
**Paper Reference:** Section 2.4.3 (Depth Point Cloud Generation), Section 2.4.4 (Preprocessing), Section 2.4.5 (Spherical Fitting)

---

## 1. Objectives & Responsibilities

The Spatial Geometry Agent converts 2D pixel coordinates and disparity values into metric 3D point clouds, isolates tomato surfaces, prunes noise, and fits a geometric spherical model to determine the exact centroid position and physical radius of each tomato.

Primary duties:
1. Reconstruct full-scene 3D coordinates using disparity map $d(x, y)$ and the perspective transformation matrix $Q$.
2. Gate and clip 3D points belonging to each tomato instance using its sanitized mask $M_k^{\text{clean}}$.
3. Filter invalid depths outside the operational picking envelope ($250\text{ mm} \le Z \le 550\text{ mm}$).
4. Perform **Voxel Grid Downsampling** to homogenize point density.
5. Apply **Local Outlier Factor (LOF)** or Statistical Outlier Removal to eliminate stray points and stereo edge artifacts.
6. Fit a spherical mathematical model via a dual-phase approach: **Algebraic Linear Least Squares** (for robust initialization) followed by **Non-linear Levenberg-Marquardt Optimization** minimizing orthogonal surface residuals.
7. Return metric coordinates $(x_c, y_c, z_c)$ and radius $r$ in millimeters with error metrics (RMSE).

---

## 2. Mathematical Formulations

### A. Reprojection to 3D Space ($Q$ Matrix)
For every image point $(x, y)$ within tomato mask $M_k$ with valid disparity $d(x, y) > 0$:
$$\begin{bmatrix} X' \\ Y' \\ Z' \\ W' \end{bmatrix} = Q \begin{bmatrix} x \\ y \\ d(x, y) \\ 1 \end{bmatrix} \implies \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} X'/W' \\ Y'/W' \\ Z'/W' \end{bmatrix}$$
where $Q$ is obtained from stereo calibration. This produces the raw instance point cloud $P_k = \{ (X_i, Y_i, Z_i) \}_{i=1}^{N}$.

### B. Voxel Downsampling (Paper Eq. 3)
The bounding volume of $P_k$ is discretized into cubic voxels of side length $s_v$ (e.g., $2.0\text{ mm}$). For all points within voxel $V_j$:
$$\mathbf{p}_{\text{centroid}}^{(j)} = \frac{1}{m_j} \sum_{i=1}^{m_j} \mathbf{p}_i = \left( \frac{\sum X_i}{m_j}, \frac{\sum Y_i}{m_j}, \frac{\sum Z_i}{m_j} \right)$$
This yields a uniformly distributed point set $P_k^{\text{down}}$.

### C. Local Outlier Factor (LOF) Filtering (Paper Eq. 4 & 5)
To eliminate sparse noise points near foliage borders, each point $p \in P_k^{\text{down}}$ is evaluated:
1. $k$-nearest neighbors $N_k(p)$ are determined using a $k$-d tree.
2. The reachability distance of point $p$ with respect to neighbor $o$:
   $$rd_k(p, o) = \max\left( \text{dist}_k(o), \|\mathbf{p} - \mathbf{o}\| \right)$$
3. The local reachability density of $p$:
   $$lrd_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} rd_k(p, o)}$$
4. The Local Outlier Factor:
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{lrd_k(o)}{lrd_k(p)}}{|N_k(p)|}$$
Points with $\text{LOF}_k(p) > \tau_{\text{LOF}}$ (typically $1.2 - 1.5$) are rejected, yielding $P_k^{\text{clean}}$.

### D. Dual-Phase Least Squares Sphere Fitting (Paper Eq. 6)
A tomato surface satisfies:
$$(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 = r^2$$

#### Phase 1: Algebraic Linear Initialization
Expanding the sphere equation:
$$x^2 + y^2 + z^2 - 2x_c x - 2y_c y - 2z_c z + (x_c^2 + y_c^2 + z_c^2 - r^2) = 0$$
Let $A = -2x_c$, $B = -2y_c$, $C = -2z_c$, and $D = x_c^2 + y_c^2 + z_c^2 - r^2$.
We solve the overdetermined linear system $\mathbf{M} \mathbf{u} = \mathbf{w}$:
$$\begin{bmatrix}
x_1 & y_1 & z_1 & 1 \\
\vdots & \vdots & \vdots & \vdots \\
x_n & y_n & z_n & 1
\end{bmatrix} \begin{bmatrix} A \\ B \\ C \\ D \end{bmatrix} = \begin{bmatrix} -(x_1^2 + y_1^2 + z_1^2) \\ \vdots \\ -(x_n^2 + y_n^2 + z_n^2) \end{bmatrix}$$
Using Singular Value Decomposition (SVD):
$$\mathbf{u} = (\mathbf{M}^T \mathbf{M})^{-1} \mathbf{M}^T \mathbf{w} \implies x_c^{(0)} = -\frac{A}{2}, \quad y_c^{(0)} = -\frac{B}{2}, \quad z_c^{(0)} = -\frac{C}{2}, \quad r^{(0)} = \sqrt{{x_c^{(0)}}^2 + {y_c^{(0)}}^2 + {z_c^{(0)}}^2 - D}$$

#### Phase 2: Non-Linear Geometric Optimization (Levenberg-Marquardt)
With initial parameters $[x_c^{(0)}, y_c^{(0)}, z_c^{(0)}, r^{(0)}]$, optimize the true geometric Euclidean distance residual:
$$\min_{x_c, y_c, z_c, r} \sum_{i=1}^n \left( \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2 + (z_i - z_c)^2} - r \right)^2$$
This guarantees unbiased parameter estimation on partial hemispherical shells.

---

## 3. Operational Specification & Hyperparameters

```yaml
# configs/geometry_config.yaml
pointcloud:
  z_min_mm: 200.0              # Sensor near plane
  z_max_mm: 600.0              # Sensor far plane (paper evaluated 280mm - 480mm)
  voxel_size_mm: 2.0           # Grid downsampling step
  min_points_threshold: 100    # Minimum points required for valid fit

outlier_removal:
  method: "lof"                # "lof" or "statistical"
  lof_neighbors: 20            # k nearest neighbors
  lof_contamination: 0.05      # Proportion of expected outliers
  statistical_std_ratio: 1.5

sphere_fitting:
  max_iterations: 100
  tolerance: 1.0e-6
  radius_bounds_mm: [15.0, 75.0] # Plausible ripe tomato radius
  max_allowed_rmse_mm: 4.5       # Quality check limit
```

---

## 4. Input / Output Contracts

* **Inputs:**
  * `disparity_map`: $(H, W)$ `np.float32` disparity from `stereo_perception`.
  * `Q_matrix`: $(4, 4)$ `np.float64` reprojection matrix from `stereo_perception`.
  * `detected_instances`: List of `DetectedInstance` containing sanitized masks $M_k^{\text{clean}}$ from `segmentation`.
* **Outputs:**
  * `localized_tomatoes`: List of `LocalizedTomato`:
    * `instance_id`: Target ID.
    * `centroid`: `[xc, yc, zc]` in mm (left camera frame).
    * `radius`: $r$ in mm.
    * `fit_rmse`: Residual error in mm.
    * `point_count`: Final number of points utilized in fit.
    * `is_valid`: Boolean flag checking all geometric assertions.

---

## 5. Verification Against Paper Benchmarks

The agent strictly evaluates each estimate against the paper's experimental findings (Table 5):
* $\text{Centroid Error} \le 5.0\text{ mm}$ (Direct sunlight with occlusion)
* $\text{Centroid Error} \le 2.8\text{ mm}$ (Optimal shading without occlusion)
* $\text{Radius Error} \le 3.9\text{ mm}$
If `fit_rmse > 4.5 mm` or point cloud coverage spans $< 30^\circ$ of solid angle, an occlusion/low-confidence flag is explicitly emitted.
