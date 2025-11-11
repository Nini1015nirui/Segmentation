#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import subprocess


def run(cmd):
    print('$', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def overlay_dir(images_dir: Path, masks_dir: Path, out_dir: Path, color=(255, 0, 0), alpha=0.35):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}])
    if not images:
        print(f"No images in {images_dir}")
        return 0
    n = 0
    for img_path in images:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"[skip] mask not found for {img_path.name}")
            continue
        base = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        base_arr = np.array(base).astype(np.float32)
        mask01 = (np.array(mask) > 0).astype(np.float32)
        a3 = (mask01 * alpha)[:, :, None]
        color_arr = np.zeros_like(base_arr)
        color_arr[..., 0] = color[0]
        color_arr[..., 1] = color[1]
        color_arr[..., 2] = color[2]
        blended = (base_arr * (1.0 - a3) + color_arr * a3).astype(np.uint8)
        Image.fromarray(blended).save(out_dir / img_path.name)
        n += 1
        if n % 25 == 0:
            print(f"overlay {n}")
    print(f"overlay done: {n}")
    return n


def main():
    ap = argparse.ArgumentParser(description='Overlay binary masks onto frames and optionally build MP4')
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--out_frames_dir', required=True)
    ap.add_argument('--out_mp4', default=None)
    ap.add_argument('--fps', type=float, default=25)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    out_frames_dir = Path(args.out_frames_dir)

    overlay_dir(frames_dir, masks_dir, out_frames_dir)

    if args.out_mp4:
        pat = str(out_frames_dir / 'frame_%06d.png')
        run(['ffmpeg', '-y', '-framerate', str(args.fps), '-i', pat, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.out_mp4])
        print('wrote', args.out_mp4)


if __name__ == '__main__':
    main()

