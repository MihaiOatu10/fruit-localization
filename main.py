"""
Command-line entry point for the binocular 3D fruit localization pipeline.

Usage:
    python main.py --left L.png --right R.png --weights weights/best.pt --conf 0.35 --device cuda:0
    python main.py infer --left L.png --right R.png   # equivalent, "infer" is the default command
"""

# torch must load its DLLs before cv2/scipy/sklearn: on Windows the reverse order corrupts the
# process and segfaults later (access violation 0xC0000005) while ultralytics is imported.
import torch  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pipeline import TomatoLocalizationPipeline
from utils.visualizer import Visualizer


def read_image(path: str) -> np.ndarray:
    """Reads a BGR image; np.fromfile + imdecode also handles non-ASCII Windows paths."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def resolve_project_path(path: str) -> Path:
    """Relative paths resolve against the project root (like configs/weights), not the CWD."""
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def build_payload(result: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
    """Serializes pipeline output into the orchestrator's robotics JSON schema."""
    fruits: List[Dict[str, Any]] = []
    for fruit in result["localized_fruits"]:
        c = fruit["centroid_mm"]
        fruits.append({
            "id": int(fruit["instance_id"]),
            "class_name": fruit["class_name"],
            "confidence": round(float(fruit["confidence"]), 4),
            "position_camera_frame_mm": {"x": round(float(c[0]), 2), "y": round(float(c[1]), 2), "z": round(float(c[2]), 2)},
            "radius_mm": round(float(fruit["radius_mm"]), 2),
            "diameter_mm": round(float(fruit["diameter_mm"]), 2),
            "fit_rmse_mm": round(float(fruit["fit_rmse_mm"]), 3),
            "points_count": int(fruit["point_count"]),
            "status": "VALID_PICK_TARGET" if fruit["is_valid"] else "FIT_REJECTED",
            "bbox_2d": [int(round(v)) for v in fruit["bbox"]],
        })

    return {
        "frame_id": frame_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "segmented_instances": len(result["instances"]),
        "detections_count": len(fruits),
        "tomatoes": fruits,
        "telemetry": {k: round(float(v), 2) for k, v in result["telemetry"].items()},
    }


def print_summary(payload: Dict[str, Any]) -> None:
    print(
        f"\n[Main] {payload['segmented_instances']} instances segmented -> "
        f"{payload['detections_count']} fruits localized.",
        flush=True,
    )
    if payload["tomatoes"]:
        print(f"{'ID':>4} {'conf':>5} {'X':>9} {'Y':>9} {'Z':>9} {'r':>7} {'rmse':>7} {'pts':>6}  status")
        for t in payload["tomatoes"]:
            pos = t["position_camera_frame_mm"]
            print(
                f"{t['id']:>4} {t['confidence']:>5.2f} {pos['x']:>9.1f} {pos['y']:>9.1f} {pos['z']:>9.1f} "
                f"{t['radius_mm']:>7.1f} {t['fit_rmse_mm']:>7.2f} {t['points_count']:>6}  {t['status']}"
            )
    tel = payload["telemetry"]
    print("[Main] Telemetry (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in tel.items()), flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binocular 3D fruit localization (YOLO-seg + RAFT-Stereo + sphere fitting)")
    parser.add_argument("command", nargs="?", default="infer", choices=["infer"], help="Pipeline command (default: infer)")
    parser.add_argument("--left", required=True, help="Path to the raw left stereo image")
    parser.add_argument("--right", required=True, help="Path to the raw right stereo image")
    parser.add_argument("--weights", default=None, help="YOLO-seg weights (default: resolved from configs/model_config.yaml)")
    parser.add_argument("--conf", type=float, default=0.10, help="Segmentation confidence threshold")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--frame-id", type=int, default=0, help="Frame identifier written to the JSON payload")
    parser.add_argument("--output", default="output_result.png",
                        help="Annotated overlay path; the disparity map is saved next to it as <stem>_disparity<ext>")
    parser.add_argument("--json", default=None, help="Optional path to write the robotics JSON payload")
    parser.add_argument("--no-vis", action="store_true", help="Skip writing visualization images")
    parser.add_argument("--calib-config", default="configs/camera_calib.yaml")
    parser.add_argument("--stereo-config", default="configs/stereo_sgbm.yaml")
    parser.add_argument("--geometry-config", default="configs/geometry_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        image_left = read_image(args.left)
        image_right = read_image(args.right)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Main] Error: {e}", file=sys.stderr, flush=True)
        return 1

    pipeline = TomatoLocalizationPipeline(
        calib_config_path=args.calib_config,
        stereo_config_path=args.stereo_config,
        geometry_config_path=args.geometry_config,
        model_config_path=args.model_config,
        model_weights=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )
    result = pipeline.process_pair(image_left, image_right)

    payload = build_payload(result, args.frame_id)
    print_summary(payload)

    if args.json:
        json_path = resolve_project_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[Main] JSON payload written to {json_path}", flush=True)

    if not args.no_vis:
        out_path = resolve_project_path(args.output)
        overlay = Visualizer.draw_detections_2d(result["rectified_left"], result["localized_fruits"], result["instances"])
        disp_path = out_path.with_name(f"{out_path.stem}_disparity{out_path.suffix or '.png'}")
        write_image(out_path, overlay)
        write_image(disp_path, Visualizer.colorize_disparity(result["disparity_map"]))
        print(f"[Main] Visualizations written to {out_path} and {disp_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
