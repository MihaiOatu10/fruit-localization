# QA & Benchmark Agent

**Identifier:** `qa_benchmark`  
**Role:** Automated Verification, Unit Testing, Geometric Validation, and Accuracy Benchmarking  
**Paper Reference:** Section 3.1 to Section 3.6 (Experimental Results & Accuracy Benchmarks)

---

## 1. Objectives & Responsibilities

The QA & Benchmark Agent ensures that the prototype faithfully reproduces the localization accuracy and geometric fidelity reported in the paper. It maintains a suite of synthetic and real tests, validates coordinate transforms, monitors residual fitting errors, and guards against spatial regressions.

Primary duties:
1. Provide unit tests for individual mathematical modules (rectification mapping, SGBM parameters, mask erosion, LOF filtering, least-squares sphere fitting).
2. Generate synthetic ground-truth 3D hemisphere datasets with controlled noise and occlusion to verify that the algebraic + non-linear sphere fitter achieves sub-millimeter precision.
3. Validate coordinate transformations against the paper's physical calibration matrix (Table 1: baseline $B = 74.676\text{ mm}$, focal length $f \approx 1122.6\text{ px}$).
4. Benchmark pipeline performance against the paper's target error bounds (Table 5: centroid error $\le \pm 5.0\text{ mm}$, radius error $\le \pm 3.9\text{ mm}$).
5. Provide continuous profiling of end-to-end latency and memory utilization.

---

## 2. Benchmark Reference Standards (from Paper)

The agent uses the paper's empirical metrics as strict acceptance criteria:

### A. Localization & Radius Error Limits (Paper Table 5)
Evaluated across 36 stereo pairs in greenhouse conditions at distances $280\text{ mm} \le Z \le 480\text{ mm}$:

| Test Condition | Occlusion | Lighting | Maximum Allowable Centroid Error | Maximum Allowable Radius Error |
| :--- | :--- | :--- | :--- | :--- |
| **Optimal** | None | Shading | $\le 2.8\text{ mm}$ | $\le 2.1\text{ mm}$ |
| **Direct Sunlight** | None | Sunlight | $\le 4.2\text{ mm}$ | $\le 3.3\text{ mm}$ |
| **Occluded Shading** | Present | Shading | $\le 3.5\text{ mm}$ | $\le 2.8\text{ mm}$ |
| **Severe Stress** | Present | Sunlight | $\le 5.0\text{ mm}$ | $\le 3.9\text{ mm}$ |

### B. Instance Segmentation Accuracy Benchmarks (Paper Table 4)
* Precision: $\ge 96.0\%$
* Recall: $\ge 95.0\%$
* Average Precision (AP): $\ge 98.0\%$

---

## 3. Test Suites & Verification Protocols

```
tests/
├── unit/
│   ├── test_calibration.py         # Validates epipolar rectification & Q matrix calculation
│   ├── test_stereo_sgbm.py         # Tests disparity computation on synthetic and sample stereo pairs
│   ├── test_mask_sanitation.py     # Verifies morphological erosion and contour checks
│   ├── test_pointcloud_filters.py  # Tests voxel grid downsampling and LOF outlier detection
│   └── test_sphere_fitter.py       # Mathematical tests on synthetic sphere point clouds
├── integration/
│   ├── test_pipeline_synthetic.py  # End-to-end pipeline run on generated stereo scene
│   └── test_benchmark_runner.py    # Regression check against Paper Table 5 accuracy bounds
└── fixtures/
    ├── sample_calib.yaml           # Ground truth calibration parameters (Paper Table 1)
    └── synthetic_sphere.npy        # Standardized 3D point cloud fixture
```

### A. Synthetic Sphere Fitter Verification Algorithm
```python
def generate_synthetic_sphere_points(
    center=(0.0, 0.0, 400.0), 
    radius=40.0, 
    num_points=1000, 
    noise_std=0.5,
    hemisphere_only=True
) -> np.ndarray:
    """
    Generates a synthetic noisy point cloud of a front-facing hemisphere.
    Guarantees mathematically verifiable ground truth for sphere_fit tests.
    """
    ...
```
* **Acceptance Criterion:** The estimated center must satisfy $\|\mathbf{c}_{\text{est}} - \mathbf{c}_{\text{true}}\| < 1.0\text{ mm}$ and $|r_{\text{est}} - r_{\text{true}}| < 1.0\text{ mm}$ under $\sigma_{\text{noise}} = 0.5\text{ mm}$.

### B. Epipolar Line Rectification Sanity Check
* Compute matched SIFT keypoints $(p_l, p_r)$ between rectified left and right images.
* Assert: $\text{mean}(|y_{l,i} - y_{r,i}|) < 0.5\text{ pixel}$ and $\max(|y_{l,i} - y_{r,i}|) < 1.5\text{ pixels}$.

---

## 4. Input / Output Contracts

* **Inputs:**
  * Pipeline output objects (`LocalizedTomato`, `DisparityResult`, `RectificationResult`).
  * Ground truth physical measurements (caliper diameter, laser rangefinder distance).
* **Outputs:**
  * `VerificationReport`:
    * `passed`: Boolean indicator.
    * `centroid_error_mm`: Absolute Euclidean distance to ground truth.
    * `radius_error_mm`: Absolute difference in radius.
    * `residual_rmse_mm`: Sphere surface residual root mean square error.
    * `violations`: List of triggered boundary violations.

---

## 5. Automated CI/CD Regression Guard

Every code change must execute the test suite:
```bash
pytest tests/ -v --tb=short
```
If a refactoring or parameter modification degrades centroid accuracy beyond $\pm 5.0\text{ mm}$ on the test fixtures, the build is rejected.
