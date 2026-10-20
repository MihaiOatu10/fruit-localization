# fruit-localization

Binocular 3D fruit localization for robotic harvesting. Each stereo pair goes through four steps:

1. **YOLOv8-seg** instance masks on the rectified left frame
2. **RAFT-Stereo** disparity (official pretrained weights, no fine-tuning; StereoSGBM + WLS as fallback)
3. 3D reprojection, surface depth-band gating, voxel downsampling and LOF outlier removal
4. A ray-constrained least-squares **sphere fit** per fruit, giving centroid (X, Y, Z) and diameter in **mm**

The method follows Zheng et al., *Agronomy* 2023, 13, 2339 (tomato). It is evaluated on the kiwifruit stereo dataset from Jing et al. 2024, captured with a ZED 2i at 1280×720, already rectified.

## Layout

```
main.py                 CLI entry point (inference + JSON payload + visualizations)
pipeline.py             End-to-end orchestration
core/                   calibration, stereo_matcher, segmentation, pointcloud, sphere_fit
configs/                camera_calib.yaml (ZED 2i), stereo_sgbm.yaml, geometry_config.yaml, model_config.yaml
third_party/raft_stereo Vendored official RAFT-Stereo (MIT, princeton-vl @ 6e93ed2)
scripts/                Dataset preparation and YOLO training
tests/                  Pytest suite
utils/visualizer.py     Overlay and disparity rendering
agents/, AGENTS.md      Multi-agent architecture notes and payload schemas
```

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for RAFT-Stereo. `.devcontainer/` contains a ready-made CUDA 12.1 / PyTorch 2.2 image.

### Weights (included)

| File | Source |
|---|---|
| `weights/raftstereo-middlebury.pth` | Official RAFT-Stereo checkpoint, [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) (MIT) |
| `weights/best.pt` | YOLOv8-seg fine-tuned on the kiwifruit set (`scripts/train_yolo.py` to retrain) |

Both ship with the repo, so inference is plug and play. If one is missing, segmentation falls back to COCO `yolov8n-seg.pt`, which has no kiwifruit class, and stereo matching falls back to StereoSGBM.

### Data (not tracked in git)

- Stereo pairs: [Smart-Ag/Kiwifruit_stereo_camera_image](https://github.com/Smart-Ag/Kiwifruit_stereo_camera_image), placed in `stereo-original-image/left/` and `stereo-original-image/right/`
- YOLO segmentation set: Roboflow export in `data/{train,valid,test}/`, described by `data/data.yaml`

> **Note:** the source stereo images come **unlabeled**. To train or retrain the segmenter, label the fruit masks by hand first (e.g. in Roboflow or LabelMe) and export them in YOLOv8-seg format. You don't need labels for inference with the included `weights/best.pt`.

## Usage

```bash
python main.py --left stereo-original-image/left/00400.png \
               --right stereo-original-image/right/00400.png \
               --device cuda:0 --json out/00400.json
```

This writes `output_result.png` (masks and fitted spheres) and `output_result_disparity.png`. The JSON payload holds each fruit's position, radius, fit RMSE and its status: `VALID_PICK_TARGET` or `FIT_REJECTED`.

Train the segmenter:

```bash
python scripts/train_yolo.py --data dataset.yaml --model yolov8s-seg.pt --epochs 50 --device 0
```

## Key configuration

- `configs/camera_calib.yaml`: ZED 2i intrinsics, B = 119.671 mm, `already_rectified: true` (passthrough, no remap)
- `configs/stereo_sgbm.yaml`: `engine: raft|sgbm|auto`, RAFT `iters: 20`, fp16
- `configs/geometry_config.yaml`: depth envelope, surface band, and sphere validity checks (radius 17.5–45 mm, centre close to the camera ray, RMSE limit that grows with Z² for stereo depth noise)

## Tests

```bash
pytest -q
```

RAFT tests are skipped when CUDA or the checkpoint is unavailable. The end-to-end test is skipped when the stereo images are missing.

## License / credits

RAFT-Stereo code is © Princeton Vision & Learning Lab, MIT license (`third_party/raft_stereo/LICENSE`). The reference papers and the dataset images belong to their authors and are not redistributed here.
