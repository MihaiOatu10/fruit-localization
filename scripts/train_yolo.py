"""
YOLO Instance Segmentation Training and Fine-Tuning Script.

Fine-tunes Ultralytics YOLO-seg (YOLOv8-seg / YOLOv11-seg) on the fruit
dataset using configuration defined in dataset.yaml or data/data.yaml.
Supports automatic dataset verification, flexible split discovery (train/valid/test),
hardware acceleration auto-detection, and post-training validation metrics reporting.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import yaml
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_yolo")

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_split_path(base_dir: Path, rel_path: str) -> Path:
    """
    Resolves relative dataset directory paths against a base directory
    while handling both relative and absolute paths.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def verify_dataset(data_yaml_path: Path) -> Dict[str, Any]:
    """
    Validates dataset configuration, ensures image and label directories exist,
    counts images and polygon instances per split, and logs summary statistics.
    """
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = data_yaml_path.parent
    base_prefix = cfg.get("path", "")
    if base_prefix:
        prefix_p = Path(base_prefix)
        base_dir = prefix_p if prefix_p.is_absolute() else (yaml_dir / prefix_p).resolve()
    else:
        base_dir = yaml_dir

    # Support train, val/valid, and test keys
    splits = {
        "train": cfg.get("train"),
        "val": cfg.get("val") or cfg.get("valid"),
        "test": cfg.get("test"),
    }

    stats = {}
    print("\n" + "=" * 65)
    print("DATASET VERIFICATION & INTEGRITY CHECK")
    print("=" * 65)
    print(f"Config File: {data_yaml_path.resolve()}")
    print(f"Base Directory: {base_dir}")
    print(f"Class Names: {cfg.get('names', {})}")
    print("-" * 65)

    for split_name, split_rel in splits.items():
        if not split_rel:
            continue

        img_dir = resolve_split_path(base_dir, split_rel)

        # Fallback check if 'val' vs 'valid' naming differs
        if not img_dir.exists():
            if split_name == "val":
                alt_dir = img_dir.parent / "valid" / "images"
                if alt_dir.exists():
                    img_dir = alt_dir
            elif split_name == "valid":
                alt_dir = img_dir.parent / "val" / "images"
                if alt_dir.exists():
                    img_dir = alt_dir

        if not img_dir.exists():
            print(f" [!] Warning: Split '{split_name}' directory not found: {img_dir}")
            stats[split_name] = {"images": 0, "labels": 0, "dir": str(img_dir), "exists": False}
            continue

        img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]

        # Determine label directory (standard YOLO: images/ -> labels/ at same hierarchy)
        if img_dir.name == "images":
            lbl_dir = img_dir.parent / "labels"
        else:
            lbl_dir = img_dir / "labels"

        lbl_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        # Count total instances
        total_instances = 0
        for lf in lbl_files:
            try:
                with open(lf, "r", encoding="utf-8") as f_lbl:
                    lines = [ln.strip() for ln in f_lbl if ln.strip()]
                    total_instances += len(lines)
            except Exception:
                pass

        stats[split_name] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "instances": total_instances,
            "img_dir": str(img_dir),
            "lbl_dir": str(lbl_dir) if lbl_dir.exists() else "None",
            "exists": True,
        }

        print(
            f" Split: {split_name:<6} | Images: {len(img_files):>4} | "
            f"Label Files: {len(lbl_files):>4} | Polygon Instances: {total_instances:>5} | "
            f"Dir: {img_dir.name}/"
        )

    print("=" * 65 + "\n")
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO-seg on Fruit Dataset")
    parser.add_argument(
        "--data",
        default="dataset.yaml",
        help="Path to dataset YAML (defaults to dataset.yaml or data/data.yaml)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="Pre-trained base weights (e.g. yolov8n-seg.pt, yolov8s-seg.pt, yolov11n-seg.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device ('cpu', '0', 'cuda:0', 'mps'). Auto-selects CUDA if available.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--project",
        default="runs/train-seg",
        help="Project output directory (default: runs/train-seg)",
    )
    parser.add_argument(
        "--name",
        default="fruit_segmentation",
        help="Experiment name (default: fruit_segmentation)",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=True,
        help="Evaluate on test split after training completes",
    )
    return parser.parse_args()


def select_device(requested_device: Optional[str] = None) -> str:
    """
    Selects optimal execution device (CUDA GPU, Apple MPS, or CPU).
    """
    if requested_device is not None:
        req = str(requested_device).strip()
        if req.lower() == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return req
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and req == "mps":
                return "mps"
            else:
                logger.warning(f"Requested device '{req}' not available. Falling back to CPU.")
                return "cpu"
        except ImportError:
            return "cpu"

    # Auto-detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}. Using device '0'.")
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected. Using device 'mps'.")
            return "mps"
    except ImportError:
        pass

    logger.info("No GPU acceleration detected. Defaulting to device 'cpu'.")
    return "cpu"


def find_dataset_yaml(specified_path: str) -> Path:
    """
    Finds and resolves the dataset YAML configuration path.
    """
    p = Path(specified_path)
    if p.is_file():
        return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / p,
        repo_root / "dataset.yaml",
        repo_root / "data" / "data.yaml",
        repo_root / "data.yaml",
    ]

    for cand in candidates:
        if cand.is_file():
            logger.info(f"Resolved dataset YAML to: {cand.resolve()}")
            return cand

    raise FileNotFoundError(
        f"Could not locate dataset configuration at '{specified_path}'. "
        "Checked dataset.yaml and data/data.yaml."
    )


def train():
    args = parse_args()
    data_path = find_dataset_yaml(args.data)
    device = select_device(args.device)

    # 1. Dataset verification
    stats = verify_dataset(data_path)
    if stats.get("train", {}).get("images", 0) == 0:
        logger.error("No training images found! Please verify dataset splits.")
        sys.exit(1)

    print("==================================================")
    print("Starting YOLO Instance Segmentation Fine-Tuning")
    print(f" Dataset YAML : {data_path.resolve()}")
    print(f" Base Model   : {args.model}")
    print(f" Epochs       : {args.epochs}")
    print(f" Image Size   : {args.imgsz}")
    print(f" Batch Size   : {args.batch}")
    print(f" Device       : {device}")
    print(f" Output Dir   : {args.project}/{args.name}")
    print("==================================================")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Install via: pip install ultralytics")
        sys.exit(1)

    # 2. Initialize model with pre-trained weights
    model = YOLO(args.model)

    # 3. Execute training
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        exist_ok=True,
        save=True,
        plots=True,
    )

    print("\nTraining completed successfully!")

    # 4. Evaluate on validation and test sets
    try:
        print("\nEvaluating trained model on validation set...")
        val_metrics = model.val(data=str(data_path.resolve()), split="val")
        if hasattr(val_metrics, "seg"):
            print(f" Validation Mask mAP50    : {val_metrics.seg.map50:.4f}")
            print(f" Validation Mask mAP50-95 : {val_metrics.seg.map:.4f}")
        if hasattr(val_metrics, "box"):
            print(f" Validation Box mAP50     : {val_metrics.box.map50:.4f}")
            print(f" Validation Box mAP50-95  : {val_metrics.box.map:.4f}")

        if args.evaluate_test and stats.get("test", {}).get("images", 0) > 0:
            print("\nEvaluating trained model on test set...")
            test_metrics = model.val(data=str(data_path.resolve()), split="test")
            if hasattr(test_metrics, "seg"):
                print(f" Test Mask mAP50          : {test_metrics.seg.map50:.4f}")
                print(f" Test Mask mAP50-95       : {test_metrics.seg.map:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute final validation metrics: {e}")

    # 5. Save best weights to weights/best.pt for pipeline integration
    save_dir = Path(args.project) / args.name / "weights"
    best_weights = save_dir / "best.pt"
    if best_weights.is_file():
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        dest_best = weights_dir / "best.pt"
        shutil.copy2(best_weights, dest_best)
        print(f"\n[Artifact] Best weights saved to: {best_weights.resolve()}")
        print(f"[Artifact] Staged for inference pipeline at: {dest_best.resolve()}")

    print("\n" + "=" * 65)
    print("HOW TO RUN INFERENCE WITH THE TRAINED MODEL:")
    print("=" * 65)
    print("Single-pair 3D localization:")
    print("  python main.py infer --left <path_to_left> --right <path_to_right> --weights weights/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    train()
