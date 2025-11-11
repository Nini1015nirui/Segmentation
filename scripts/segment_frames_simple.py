#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Ensure project root is on sys.path to import sibling script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse functions from segment_dicom_video
from scripts.segment_dicom_video import segment_frame, save_mask, make_overlay


def main():
    ap = argparse.ArgumentParser(description="Segment a range of frames (PNG) and write masks + overlays")
    ap.add_argument('--frames_dir', required=True, help='Directory with extracted frames (frame_XXXXXX.png)')
    ap.add_argument('--masks_dir', required=True, help='Directory to write binary masks')
    ap.add_argument('--overlays_dir', required=True, help='Directory to write overlay images')
    ap.add_argument('--start', type=int, default=1, help='Start frame index (1-based)')
    ap.add_argument('--end', type=int, default=0, help='End frame index (inclusive). 0=auto to last frame')
    ap.add_argument('--roi_top', type=float, default=0.35, help='ROI top fraction (0-1)')
    ap.add_argument('--lcc_pre_erode', type=int, default=2, help='Pre-erosion radius before LCC')
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    overlays_dir = Path(args.overlays_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return 0

    def idx_of(p: Path) -> int:
        # frame_000123.png -> 123
        stem = p.stem
        num = stem.split('_')[-1]
        try:
            return int(num)
        except Exception:
            return 0

    frames = [f for f in frames if idx_of(f) >= args.start and (args.end == 0 or idx_of(f) <= args.end)]
    total = len(frames)
    if total == 0:
        print("No frames in requested range")
        return 0

    for i, f in enumerate(frames, 1):
        mask = segment_frame(f, roi_top_frac=args.roi_top, lcc_pre_erode=args.lcc_pre_erode)
        out_mask = masks_dir / f.name
        save_mask(mask, out_mask)
        out_overlay = overlays_dir / f.name
        make_overlay(f, mask, out_overlay)
        if i % 25 == 0 or i == total:
            print(f"Processed {i}/{total} frames in batch [{args.start}, {args.end or 'last'}]")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
