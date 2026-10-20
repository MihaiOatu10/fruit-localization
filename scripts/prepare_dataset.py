"""
Dataset Preparation & Test Holdout Partitioning Script.

Inspects 'stereo original image' directory, matches left/right stereo image pairs
and LabelMe JSON annotations, strictly reserves the last 50 stereo pairs for
evaluation, and partitions remaining pairs into YOLOv5-seg / YOLOv8-seg training
and validation sets.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG"}


def find_image_pairs(
    left_dir: Path, 
    right_dir: Path
) -> List[Dict[str, Optional[Path]]]:
    """
    Finds and pairs stereo images between left and right directories.
    Also locates any matching LabelMe JSON annotations.
    """
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Missing stereo subfolders: {left_dir} or {right_dir}")

    left_images = {}
    for f in left_dir.iterdir():
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTS:
            left_images[f.stem] = f

    right_images = {}
    for f in right_dir.iterdir():
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTS:
            right_images[f.stem] = f

    common_stems = sorted(list(set(left_images.keys()) & set(right_images.keys())))
    if not common_stems:
        # Check if right images have differing prefix/suffix (e.g., 'left_01' vs 'right_01')
        logger.warning("No identical stems found. Checking prefix-stripped matching...")
        # Fallback sorting matching
        sorted_left = sorted(list(left_images.values()))
        sorted_right = sorted(list(right_images.values()))
        min_len = min(len(sorted_left), len(sorted_right))
        pairs = []
        for i in range(min_len):
            lf = sorted_left[i]
            rf = sorted_right[i]
            l_json = left_dir / f"{lf.stem}.json"
            r_json = right_dir / f"{rf.stem}.json"
            pairs.append({
                "stem": f"pair_{i:05d}_{lf.stem}",
                "left_img": lf,
                "right_img": rf,
                "left_json": l_json if l_json.exists() else None,
                "right_json": r_json if r_json.exists() else None,
            })
        return pairs

    pairs = []
    for stem in common_stems:
        lf = left_images[stem]
        rf = right_images[stem]
        l_json = left_dir / f"{stem}.json"
        r_json = right_dir / f"{stem}.json"
        pairs.append({
            "stem": stem,
            "left_img": lf,
            "right_img": rf,
            "left_json": l_json if l_json.exists() else None,
            "right_json": r_json if r_json.exists() else None,
        })
    return pairs


def labelme_to_yolo_seg(json_path: Path, class_mapping: Dict[str, int] = None) -> List[str]:
    """
    Converts LabelMe JSON polygon annotations into YOLO-seg format:
    <class_id> x1 y1 x2 y2 ... xn yn (normalized coordinates)
    """
    if class_mapping is None:
        class_mapping = {
            "kiwifruit": 0,
            "tomato": 0,
            "fruit": 0,
            "kiwi-segmentation": 0,
            "kiwi": 0,
        }

    yolo_lines = []
    if not json_path or not json_path.exists():
        return yolo_lines

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            return yolo_lines

        for shape in data.get("shapes", []):
            label = shape.get("label", "fruit").lower()
            cls_id = class_mapping.get(label, 0)
            points = shape.get("points", [])
            if len(points) < 3:
                continue

            normalized_pts = []
            for pt in points:
                nx = max(0.0, min(1.0, float(pt[0]) / img_w))
                ny = max(0.0, min(1.0, float(pt[1]) / img_h))
                normalized_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])

            line = f"{cls_id} " + " ".join(normalized_pts)
            yolo_lines.append(line)
    except Exception as e:
        logger.error(f"Error parsing LabelMe annotation {json_path}: {e}")

    return yolo_lines


def stage_dataset(
    source_dir: str = "stereo original image",
    target_dir: str = "data",
    holdout_count: int = 50,
    val_ratio: float = 0.20
) -> Dict[str, int]:
    """
    Executes complete dataset splitting:
    1. Holds out last 50 stereo pairs into target_dir/test_holdout.
    2. Splits remainder into target_dir/train and target_dir/val.
    """
    src = Path(source_dir)
    dst = Path(target_dir)

    left_dir = src / "left"
    right_dir = src / "right"

    logger.info(f"Scanning stereo pairs in '{left_dir}' and '{right_dir}'...")
    pairs = find_image_pairs(left_dir, right_dir)
    total_pairs = len(pairs)
    logger.info(f"Discovered {total_pairs} synchronized stereo pairs.")

    if total_pairs <= holdout_count:
        logger.warning(
            f"Total pairs ({total_pairs}) is <= holdout count ({holdout_count}). "
            "All available pairs will be assigned to test_holdout!"
        )
        train_pairs = []
        val_pairs = []
        test_pairs = pairs
    else:
        train_val_pool = pairs[:-holdout_count]
        test_pairs = pairs[-holdout_count:]

        n_val = max(1, int(len(train_val_pool) * val_ratio))
        val_pairs = train_val_pool[-n_val:]
        train_pairs = train_val_pool[:-n_val]

    logger.info(
        f"Partition layout: Train={len(train_pairs)}, Val={len(val_pairs)}, "
        f"Strict Test Holdout={len(test_pairs)}"
    )

    # Directory structures
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test_holdout": test_pairs
    }

    manifest = {"train": [], "val": [], "test_holdout": []}

    for split_name, split_list in splits.items():
        if split_name == "test_holdout":
            # For test holdout, preserve both left and right stereo streams
            split_left_img_dir = dst / split_name / "left" / "images"
            split_left_lbl_dir = dst / split_name / "left" / "labels"
            split_right_img_dir = dst / split_name / "right" / "images"
            split_right_lbl_dir = dst / split_name / "right" / "labels"

            for d in [split_left_img_dir, split_left_lbl_dir, split_right_img_dir, split_right_lbl_dir]:
                d.mkdir(parents=True, exist_ok=True)

            for item in split_list:
                stem = item["stem"]
                # Copy left
                lf = item["left_img"]
                dst_lf = split_left_img_dir / lf.name
                shutil.copy2(lf, dst_lf)
                if item["left_json"]:
                    shutil.copy2(item["left_json"], split_left_lbl_dir / item["left_json"].name)
                    yolo_lines = labelme_to_yolo_seg(item["left_json"])
                    with open(split_left_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as out:
                        out.write("\n".join(yolo_lines))

                # Copy right
                rf = item["right_img"]
                dst_rf = split_right_img_dir / rf.name
                shutil.copy2(rf, dst_rf)
                if item["right_json"]:
                    shutil.copy2(item["right_json"], split_right_lbl_dir / item["right_json"].name)
                    yolo_lines = labelme_to_yolo_seg(item["right_json"])
                    with open(split_right_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as out:
                        out.write("\n".join(yolo_lines))

                manifest[split_name].append({
                    "stem": stem,
                    "left_img": str(dst_lf),
                    "right_img": str(dst_rf),
                })
        else:
            # For YOLO train/val, primary training is on left camera images
            split_img_dir = dst / split_name / "images"
            split_lbl_dir = dst / split_name / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for item in split_list:
                stem = item["stem"]
                lf = item["left_img"]
                dst_lf = split_img_dir / lf.name
                shutil.copy2(lf, dst_lf)

                if item["left_json"]:
                    yolo_lines = labelme_to_yolo_seg(item["left_json"])
                    with open(split_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as out:
                        out.write("\n".join(yolo_lines))
                    # Also keep json backup
                    shutil.copy2(item["left_json"], split_lbl_dir / item["left_json"].name)

                manifest[split_name].append({
                    "stem": stem,
                    "left_img": str(dst_lf),
                })

    # Save partition manifest
    manifest_path = dst / "dataset_partition_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset successfully staged at '{dst}'. Manifest written to '{manifest_path}'.")
    return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "stereo original image"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    stage_dataset(source_dir=src_dir, target_dir=out_dir, holdout_count=50)
