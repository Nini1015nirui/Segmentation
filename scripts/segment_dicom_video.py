#!/usr/bin/env python3
"""
Segment diaphragm from a DICOM cine video without NN weights.

Pipeline:
1) Extract frames from the input DICOM using ffmpeg.
2) For each frame, run a lightweight heuristic segmentation:
   - grayscale -> contrast equalize -> gaussian blur
   - ROI focus (lower 65% of the image by default)
   - Otsu threshold -> binary mask
   - Keep largest connected component (optional erosion+dilation)
3) Save binary masks (0/255) and overlay frames.
4) Assemble an MP4 video from the overlays and export an mp4 of the original video as well.

Notes:
- This is a classical image processing baseline intended for environments
  without pre-trained NN weights. It can be replaced by nnUNet inference
  when checkpoint weights are available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffmpeg_extract_frames(dicom_path: Path, out_dir: Path, fps: float | None = None) -> Tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Probe video properties
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(dicom_path)],
        capture_output=True, text=True
    )
    width = height = 0
    fps_num = fps_den = 0
    if probe.returncode == 0:
        try:
            lines = [l.strip() for l in probe.stdout.splitlines() if l.strip()]
            if len(lines) >= 3:
                width = int(lines[0]); height = int(lines[1])
                fr = lines[2]
                if "/" in fr:
                    n, d = fr.split("/")
                    fps_num, fps_den = int(n), int(d)
        except Exception:
            pass
    # Extract frames as PNGs
    # Use exact index with %06d to ensure proper ordering
    pattern = str(out_dir / "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-i", str(dicom_path)]
    if fps is not None:
        cmd += ["-r", str(fps)]
    cmd += [pattern]
    run(cmd)
    # Count frames
    n_frames = len(list(out_dir.glob("frame_*.png")))
    print(f"Extracted {n_frames} frames to {out_dir}")
    return width, height, n_frames


def otsu_threshold(img: np.ndarray) -> int:
    # img assumed uint8 0..255
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0
    max_var, threshold = 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def connected_components_keep_largest(mask: np.ndarray, pre_erode: int = 0, post_dilate: int = 0) -> np.ndarray:
    """Label connected components (4-connectivity) and keep the largest area.
    mask: binary np.uint8 array with values {0, 255} or {0,1}
    """
    m = (mask > 0).astype(np.uint8)
    if pre_erode > 0:
        m = binary_erode(m, pre_erode)
    h, w = m.shape
    visited = np.zeros_like(m, dtype=bool)
    best_area = 0
    best_coords = None
    # Simple BFS
    from collections import deque
    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            coords = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and m[ny, nx] == 1:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        coords.append((ny, nx))
            if len(coords) > best_area:
                best_area = len(coords)
                best_coords = coords
    out = np.zeros_like(m, dtype=np.uint8)
    if best_coords:
        for (yy, xx) in best_coords:
            out[yy, xx] = 1
    if post_dilate > 0:
        out = binary_dilate(out, post_dilate)
    return (out * 255).astype(np.uint8)


def binary_erode(mask01: np.ndarray, radius: int) -> np.ndarray:
    s = max(1, radius)
    h, w = mask01.shape
    out = np.zeros_like(mask01)
    # Using a square structuring element (2s+1)x(2s+1)
    for y in range(h):
        y0 = max(0, y - s); y1 = min(h, y + s + 1)
        for x in range(w):
            x0 = max(0, x - s); x1 = min(w, x + s + 1)
            if np.all(mask01[y0:y1, x0:x1] == 1):
                out[y, x] = 1
    return out


def binary_dilate(mask01: np.ndarray, radius: int) -> np.ndarray:
    s = max(1, radius)
    h, w = mask01.shape
    out = np.zeros_like(mask01)
    # Using a square structuring element (2s+1)x(2s+1)
    yy, xx = np.where(mask01 > 0)
    for (y, x) in zip(yy, xx):
        y0 = max(0, y - s); y1 = min(h, y + s + 1)
        x0 = max(0, x - s); x1 = min(w, x + s + 1)
        out[y0:y1, x0:x1] = 1
    return out


def segment_frame(img_path: Path, roi_top_frac: float = 0.35, lcc_pre_erode: int = 2) -> np.ndarray:
    """Return binary mask (uint8 0/255) for one frame.

    Heuristic tailored for ultrasound-like diaphragm appearance:
    - Enhance contrast, smooth, focus on lower ROI, Otsu threshold.
    - Keep largest connected component and project back to full image size.
    """
    im = Image.open(img_path).convert('L')
    # Contrast enhancement and smoothing
    im_eq = ImageOps.equalize(im)
    im_blur = im_eq.filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.array(im_blur, dtype=np.uint8)
    h, w = arr.shape
    y0 = int(h * roi_top_frac)
    roi = arr[y0:, :]
    # Otsu threshold on ROI
    th = otsu_threshold(roi)
    bin_roi = (roi >= th).astype(np.uint8) * 255
    # Largest connected component cleanup
    bin_roi = connected_components_keep_largest(bin_roi, pre_erode=lcc_pre_erode, post_dilate=lcc_pre_erode)
    # Expand back to full image
    mask_full = np.zeros_like(arr, dtype=np.uint8)
    mask_full[y0:, :] = bin_roi
    return mask_full


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out = Image.fromarray(mask)
    out.save(out_path)


def make_overlay(img_path: Path, mask: np.ndarray, out_path: Path, color=(255, 0, 0), alpha: float = 0.35) -> None:
    base = Image.open(img_path).convert('RGB')
    color_img = Image.new('RGB', base.size, color)
    # Build alpha mask from binary
    a = (mask > 0).astype(np.float32) * alpha
    a = np.clip(a, 0.0, 1.0)
    # Blend per pixel: out = base*(1-a) + color*a
    b = np.array(base).astype(np.float32)
    c = np.array(color_img).astype(np.float32)
    a3 = a[:, :, None]
    blended = (b * (1.0 - a3) + c * a3).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def assemble_video(frames_dir: Path, out_mp4: Path, fps: float | None = None) -> None:
    pattern = str(frames_dir / "frame_%06d.png")
    cmd = ["ffmpeg", "-y"]
    if fps is not None:
        cmd += ["-framerate", str(fps)]
    cmd += ["-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Segment diaphragm in DICOM cine video using classical image processing")
    ap.add_argument("dicom", type=str, help="Input DICOM cine video path")
    ap.add_argument("--workdir", type=str, default="data/video_output", help="Working directory for outputs")
    ap.add_argument("--name", type=str, default=None, help="Run name (defaults to DICOM filename stem)")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS for output video assembly")
    ap.add_argument("--roi_top", type=float, default=0.35, help="ROI top fraction (0-1), focus lower part of image")
    ap.add_argument("--lcc_pre_erode", type=int, default=2, help="Pre-erosion radius before LCC; also used for post-dilation")
    args = ap.parse_args()

    dicom_path = Path(args.dicom)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dicom_path}")

    run_name = args.name or dicom_path.name
    workdir = Path(args.workdir) / Path(run_name)
    frames_dir = workdir / "frames"
    masks_dir = workdir / "masks_raw"
    overlays_dir = workdir / "overlays"
    for d in (frames_dir, masks_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Extract frames
    w, h, n = ffmpeg_extract_frames(dicom_path, frames_dir, fps=None)

    # 2) Segment each frame
    frames = sorted(frames_dir.glob("frame_*.png"))
    for i, f in enumerate(frames, 1):
        mask = segment_frame(f, roi_top_frac=args.roi_top, lcc_pre_erode=args.lcc_pre_erode)
        out_mask = masks_dir / f.name
        save_mask(mask, out_mask)
        # 3) Overlay
        out_overlay = overlays_dir / f.name
        make_overlay(f, mask, out_overlay)
        if i % 25 == 0 or i == len(frames):
            print(f"Processed {i}/{len(frames)} frames")

    # 4) Assemble videos
    # Original video (re-encode for convenience)
    orig_mp4 = workdir / "original.mp4"
    run(["ffmpeg", "-y", "-i", str(dicom_path), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(orig_mp4)])
    # Overlays mp4
    overlays_mp4 = workdir / "overlays.mp4"
    assemble_video(overlays_dir, overlays_mp4, fps=args.fps)

    print("\n✅ Done.")
    print(f"Frames:      {frames_dir}")
    print(f"Masks (raw): {masks_dir}")
    print(f"Overlays:    {overlays_dir}")
    print(f"Original MP4: {orig_mp4}")
    print(f"Overlay MP4:  {overlays_mp4}")


if __name__ == "__main__":
    main()

