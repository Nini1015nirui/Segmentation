#!/usr/bin/env python3
"""
Simple postprocessing for binary segmentation masks:
 - Option 1 (default for single-object): keep only the largest connected component (LCC)
 - Option 2: remove components smaller than a given area threshold
 - Optional morphological closing to fill small holes

This avoids multiprocessing and nnU-Net's PP pipeline for constrained environments.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, area_th: int) -> np.ndarray:
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        comp = (labels == i)
        if int(comp.sum()) >= area_th:
            out[comp] = 1
    return out


def keep_largest_component(mask: np.ndarray, pre_erode_k: int = 0, post_dilate_k: int = 0) -> np.ndarray:
    m = mask.astype(np.uint8)
    if pre_erode_k and pre_erode_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre_erode_k, pre_erode_k))
        m = cv2.erode(m, k)
    # Find LCC on (optionally) eroded mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:  # 0=background, 1=foreground or none
        lcc = (labels > 0).astype(np.uint8)
    else:
        areas = stats[1:, 4]
        keep_idx = int(np.argmax(areas)) + 1
        lcc = (labels == keep_idx).astype(np.uint8)
    if post_dilate_k and post_dilate_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_dilate_k, post_dilate_k))
        lcc = cv2.dilate(lcc, k)
    # Ensure we don't add new islands beyond original mask
    lcc = (lcc & (mask.astype(np.uint8) > 0)).astype(np.uint8)
    return lcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in_dir', required=True, help='Input dir with raw predictions (binary masks)')
    ap.add_argument('-o', '--out_dir', required=True, help='Output dir for postprocessed predictions')
    ap.add_argument('--area_th', type=int, default=200, help='Area threshold for CC removal (used when --keep_lcc=0)')
    ap.add_argument('--closing', type=int, default=0, help='Kernel size for morphological closing (0 disables)')
    ap.add_argument('--keep_lcc', type=int, default=1, help='Keep only the largest connected component (1=yes, 0=no)')
    ap.add_argument('--lcc_pre_erode', type=int, default=0, help='Pre-erode kernel size before LCC (break thin links)')
    ap.add_argument('--lcc_post_dilate', type=int, default=0, help='Post-dilate kernel size after LCC (recover shape)')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        print(f"No prediction files found in {in_dir}")
        return 0

    k = None
    if args.closing and args.closing > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.closing, args.closing))

    n = 0
    for p in files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"[skip] cannot read {p}")
            continue
        m_bin = (m > 0).astype(np.uint8)
        if k is not None:
            m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, k)
        if args.keep_lcc:
            m_pp = keep_largest_component(m_bin, args.lcc_pre_erode, args.lcc_post_dilate)
        else:
            m_pp = remove_small_components(m_bin, args.area_th)
        cv2.imwrite(str(out_dir / p.name), (m_pp * 255).astype(np.uint8))
        n += 1
    print(f"Postprocessed {n} masks -> {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
