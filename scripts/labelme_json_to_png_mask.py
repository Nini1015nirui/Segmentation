#!/usr/bin/env python3
"""
Convert LabelMe-style JSON annotations (polygons) to binary PNG masks (0/255).

Usage:
  python scripts/labelme_json_to_png_mask.py \
      --images_dir 可视化测试/Images \
      --json_dir 可视化测试/Masks \
      --out_dir 可视化测试/Masks

Notes:
  - Expects LabelMe JSON with keys: shapes (with polygon points), imageHeight, imageWidth.
  - Draws all polygons as foreground (255) onto a single-channel mask.
  - Output filename matches the JSON base name with .png extension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


def load_labelme_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def to_png_mask(json_path: Path, out_path: Path) -> None:
    data = load_labelme_json(json_path)
    h = int(data.get('imageHeight'))
    w = int(data.get('imageWidth'))
    if not (h and w):
        raise ValueError(f"{json_path} missing imageHeight/imageWidth")

    mask = Image.new('L', (w, h), 0)  # background 0
    draw = ImageDraw.Draw(mask)

    shapes = data.get('shapes', [])
    for shp in shapes:
        pts: List[Tuple[float, float]] = shp.get('points', [])
        if not pts:
            continue
        # Draw polygon filled as foreground (255)
        try:
            draw.polygon([tuple(p) for p in pts], outline=255, fill=255)
        except Exception as e:
            raise RuntimeError(f"Failed drawing polygon in {json_path}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', type=Path, required=False, help='Directory with source images (optional)')
    ap.add_argument('--json_dir', type=Path, required=True, help='Directory with LabelMe JSON files')
    ap.add_argument('--out_dir', type=Path, required=True, help='Directory to write PNG masks')
    args = ap.parse_args()

    json_dir: Path = args.json_dir
    out_dir: Path = args.out_dir

    json_files = sorted(json_dir.glob('*.json'))
    if not json_files:
        raise SystemExit(f"No JSON files found in {json_dir}")

    count = 0
    for jp in json_files:
        outp = out_dir / (jp.stem + '.png')
        to_png_mask(jp, outp)
        count += 1
    print(f"Converted {count} JSON files to PNG masks in {out_dir}")


if __name__ == '__main__':
    main()

