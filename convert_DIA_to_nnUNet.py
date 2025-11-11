#!/usr/bin/env python3
"""
Convert DIA dataset to nnUNet v2 format for medical image segmentation training.

Expected DIA dataset structure at repository root:
- DIA/Images/*.png
- DIA/Masks/*.png

Output (default):
- data/nnUNet_raw/Dataset804_DIA/
  - imagesTr/{case_id}_0000.png
  - labelsTr/{case_id}.png
  - dataset.json

Notes:
- Masks with values 0/255 are converted to 0/1 as required by nnUNet.
- By default, all images are placed into the Training split (no imagesTs).
  If you need a held-out test split, use --train-split <0.0-1.0>.
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def setup_dataset_structure(dataset_id: int, dataset_name: str, out_base: Path):
    dataset_dir = out_base / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"
    for p in (images_dir, labels_dir, images_ts_dir):
        p.mkdir(parents=True, exist_ok=True)
    return dataset_dir, images_dir, labels_dir, images_ts_dir


def to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """Convert a grayscale mask to binary 0/1. Handles {0,255} and {0,1}."""
    unique = np.unique(arr)
    if unique.dtype == np.bool_:
        return arr.astype(np.uint8)
    # Normalize to {0,1}
    out = np.zeros_like(arr, dtype=np.uint8)
    if 255 in unique:
        out[arr >= 127] = 1
    else:
        out[arr > 0] = 1
    return out


def convert_images_and_labels(source_dir: Path,
                              images_dir: Path,
                              labels_dir: Path,
                              case_prefix: str,
                              train_split: float,
                              images_ts_dir: Path):
    images_path = source_dir / "Images"
    masks_path = source_dir / "Masks"

    image_files = sorted(images_path.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No .png images found in {images_path}")

    converted_cases = []
    skipped = 0

    for idx, img_file in enumerate(image_files):
        mask_file = masks_path / img_file.name
        if not mask_file.exists():
            print(f"[WARN] Mask missing for {img_file.name}; skipping")
            skipped += 1
            continue

        # Load image & mask to check size and binarize mask
        try:
            img = Image.open(img_file).convert('L')
            msk = Image.open(mask_file).convert('L')
        except Exception as e:
            print(f"[WARN] Failed to open {img_file.name}: {e}; skipping")
            skipped += 1
            continue

        if img.size != msk.size:
            print(f"[WARN] Size mismatch {img_file.name}: img {img.size} vs msk {msk.size}; skipping")
            skipped += 1
            continue

        case_id = f"{case_prefix}{idx:05d}"

        # Save image (uint8) and mask (binary 0/1)
        target_img = images_dir / f"{case_id}_0000.png"
        target_msk = labels_dir / f"{case_id}.png"

        img.save(target_img)
        m_arr = np.array(msk)
        m_bin = to_binary_mask(m_arr)
        Image.fromarray(m_bin, mode='L').save(target_msk)

        converted_cases.append(case_id)
        if len(converted_cases) % 500 == 0:
            print(f"Processed {len(converted_cases)} cases...")

    print(f"Converted {len(converted_cases)} cases. Skipped {skipped}.")
    # train/test split indices
    n_total = len(converted_cases)
    n_train = int(round(train_split * n_total))
    train_cases = converted_cases[:n_train]
    test_cases = converted_cases[n_train:]

    # If test cases requested, copy images into imagesTs (labels not required)
    if test_cases:
        for cid in test_cases:
            src_img = images_dir / f"{cid}_0000.png"
            dst_img = images_ts_dir / f"{cid}_0000.png"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
    return train_cases, test_cases


def write_dataset_json(dataset_dir: Path,
                       train_cases: list[str],
                       test_cases: list[str],
                       dataset_name: str):
    dataset_json = {
        "channel_names": {"0": "Image"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "dataset_name": f"{dataset_name}_Segmentation",
        "description": f"{dataset_name} dataset converted to nnUNet v2 format",
        "reference": "",
        "licence": "Research Use",
        "training": [
            {"image": f"./imagesTr/{cid}_0000.png", "label": f"./labelsTr/{cid}.png"}
            for cid in train_cases
        ],
        "test": [f"./imagesTs/{cid}_0000.png" for cid in test_cases],
    }

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote dataset.json at {dataset_dir}")


def main():
    ap = argparse.ArgumentParser(description="Convert DIA to nnUNet v2 format")
    ap.add_argument("--src", type=str, default="DIA", help="Source DIA folder (containing Images/, Masks/)")
    ap.add_argument("--out", type=str, default="data/nnUNet_raw", help="Output base for nnUNet_raw")
    ap.add_argument("--dataset-id", type=int, default=804, help="nnUNet Dataset ID (e.g., 804)")
    ap.add_argument("--dataset-name", type=str, default="DIA", help="Dataset name tag (e.g., DIA)")
    ap.add_argument("--train-split", type=float, default=1.0, help="Proportion for Training [0,1]. 1.0 = all train")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    if not (src_dir / "Images").exists() or not (src_dir / "Masks").exists():
        raise FileNotFoundError(f"{src_dir} must contain 'Images' and 'Masks' subfolders")

    dataset_dir, images_dir, labels_dir, images_ts_dir = setup_dataset_structure(
        args.dataset_id, args.dataset_name, out_base
    )

    print(f"Converting from: {src_dir}")
    print(f"Writing to:      {dataset_dir}")
    train_cases, test_cases = convert_images_and_labels(
        src_dir, images_dir, labels_dir, case_prefix="dia_", train_split=args.train_split, images_ts_dir=images_ts_dir
    )

    write_dataset_json(dataset_dir, train_cases, test_cases, args.dataset_name)

    # Quick verification printout
    samp = list(sorted(images_dir.glob("*.png")))[:5]
    print("\nVerification samples:")
    for p in samp:
        cid = p.stem.replace("_0000", "")
        l = labels_dir / f"{cid}.png"
        if l.exists():
            img = Image.open(p)
            lab = Image.open(l)
            print(f"  {cid}: img {img.size}, label {lab.size}")
    print("\n✅ DIA dataset conversion complete.")
    print("Next:")
    print("  1) export nnUNet_raw/nnUNet_preprocessed/nnUNet_results paths")
    print("  2) run: nnUNetv2_plan_and_preprocess -d {id} --verify_dataset_integrity".format(id=args.dataset_id))
    print("  3) train: nnUNetv2_train {id} 2d all -tr nnUNetTrainerLightMUNet".format(id=args.dataset_id))


if __name__ == "__main__":
    main()

