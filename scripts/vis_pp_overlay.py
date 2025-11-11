#!/usr/bin/env python3
"""
Quick overlay visualization for raw vs post-processed segmentations.

Inputs:
  --images_dir: directory of original images used for prediction (subset OK)
  --raw_pred_dir: directory with raw predictions (e.g., tta_on or tta_on_subsetN)
  --pp_pred_dir: directory with post-processed predictions (e.g., *_pp)
  --out_dir: output directory for comparison images
  --area_threshold: connected component area threshold for 'small islands' count

Output:
  For each case present in pp_pred_dir, writes an overlay PNG with:
   - Original image
   - Raw prediction overlay + small CC count
   - Post-processed overlay + small CC count

Notes:
  - Designed for 2D PNG/JPG/TIFF datasets (SimpleITKIO with .png etc.)
  - Handles stems with or without suffix "_0000"
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def read_gray_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def read_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (m > 0).astype(np.uint8)


def find_matching_image(images_dir: Path, stem: str):
    # Try exact stem with common extensions
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Try toggling _0000 suffix
    if stem.endswith("_0000"):
        base = stem[:-5]
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = images_dir / f"{base}{ext}"
            if p.exists():
                return p
    else:
        alt = stem + "_0000"
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = images_dir / f"{alt}{ext}"
            if p.exists():
                return p
    return None


def count_small_components(mask: np.ndarray, area_th: int = 100) -> int:
    # Connected components via OpenCV
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    small = 0
    for i in range(1, num):
        area = int((labels == i).sum())
        if area < area_th:
            small += 1
    return small


def color_overlay(gray: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    if gray.ndim == 2:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        rgb = gray.copy()
    color_arr = np.zeros_like(rgb)
    color_arr[mask > 0] = color
    out = cv2.addWeighted(rgb, 1.0, color_arr, alpha, 0)
    return out


def make_canvas(img: np.ndarray, raw_vis: np.ndarray, pp_vis: np.ndarray,
                raw_cc: int, pp_cc: int) -> np.ndarray:
    # Resize to same size
    h, w = img.shape[:2]
    raw_vis = cv2.resize(raw_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    pp_vis = cv2.resize(pp_vis, (w, h), interpolation=cv2.INTER_NEAREST)
    # Concatenate horizontally: Original | Raw | Post
    canvas = np.concatenate([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), raw_vis, pp_vis], axis=1)
    # Add titles and CC counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Original', (10, 30), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Raw  smallCC<{"?"}: {raw_cc}', (w + 10, 30), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Post smallCC<{"?"}: {pp_cc}', (2 * w + 10, 30), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True, help='Subset images directory used for prediction')
    ap.add_argument('--raw_pred_dir', required=False, help='Directory containing raw predictions')
    ap.add_argument('--pp_pred_dir', required=True, help='Directory containing post-processed predictions')
    ap.add_argument('--gt_dir', required=False, help='Directory containing ground-truth masks (to compare PP vs GT)')
    ap.add_argument('--out_dir', required=False, default=None, help='Output directory for overlays')
    ap.add_argument('--area_threshold', type=int, default=100, help='CC area threshold for small islands count')
    ap.add_argument('--pp_only', type=int, default=0, help='If 1, write only PP overlays (no RAW panel)')
    ap.add_argument('--pp_vs_gt', type=int, default=0, help='If 1, overlay Post-Processed vs GT on original (ignores RAW)')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    raw_dir = Path(args.raw_pred_dir) if args.raw_pred_dir else None
    pp_dir = Path(args.pp_pred_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else (pp_dir.parent / f"{pp_dir.name}_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If pp_vs_gt mode, require gt_dir
    if args.pp_vs_gt:
        if gt_dir is None:
            raise SystemExit("pp_vs_gt=1 requires --gt_dir")

    # Iterate over files present in pp_dir (authoritative set)
    pp_files = sorted([p for p in pp_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}])
    if not pp_files:
        print(f"No images found in {pp_dir}")
        return

    total = 0
    for pp_path in pp_files:
        stem = pp_path.stem
        raw_path = None
        if not args.pp_vs_gt:
            if raw_dir is None:
                print("[skip] raw_pred_dir not provided and pp_vs_gt=0")
                continue
            raw_path = raw_dir / pp_path.name
            if not raw_path.exists():
                print(f"[skip] Raw prediction missing for {pp_path.name}")
                continue

        img_path = find_matching_image(images_dir, stem)
        if img_path is None:
            print(f"[skip] Cannot locate original image for stem {stem}")
            continue

        try:
            img = read_gray_image(img_path)
            pp_mask = read_mask(pp_path)
            raw_mask = read_mask(raw_path) if raw_path is not None else None
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        # Resize masks to image size if needed
        h, w = img.shape[:2]
        if raw_mask is not None and raw_mask.shape[:2] != (h, w):
            raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if pp_mask.shape[:2] != (h, w):
            pp_mask = cv2.resize(pp_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if args.pp_vs_gt:
            # Read GT and overlay PP (blue) vs GT (green)
            gt_path = gt_dir / pp_path.name
            if not gt_path.exists():
                print(f"[skip] GT missing for {pp_path.name}")
                continue
            gt_mask = read_mask(gt_path)
            if gt_mask.shape[:2] != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Overlay GT then PP
            base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis1 = color_overlay(img, gt_mask, color=(0, 255, 0), alpha=0.5)
            vis2 = color_overlay(vis1, pp_mask, color=(255, 0, 0), alpha=0.5)
            out_path = out_dir / f"{stem}_pp_vs_gt.png"
            cv2.imwrite(str(out_path), vis2)
            total += 1
            continue

        if args.pp_only:
            pp_vis = color_overlay(img, pp_mask, color=(0, 255, 0), alpha=0.5)
            out_path = out_dir / f"{stem}_pp_overlay.png"
            cv2.imwrite(str(out_path), pp_vis)
        else:
            raw_cc = count_small_components(raw_mask, args.area_threshold)
            pp_cc = count_small_components(pp_mask, args.area_threshold)
            raw_vis = color_overlay(img, raw_mask, color=(0, 0, 255), alpha=0.5)
            pp_vis = color_overlay(img, pp_mask, color=(0, 255, 0), alpha=0.5)
            canvas = make_canvas(img, raw_vis, pp_vis, raw_cc, pp_cc)
            out_path = out_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(out_path), canvas)
        total += 1
    print(f"Done. Wrote {total} overlays -> {out_dir}")


if __name__ == '__main__':
    main()
