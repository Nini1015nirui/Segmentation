#!/usr/bin/env python3
"""
Analyze diaphragm thickness changes over time from binary segmentation masks.

Inputs
  - mask_dir: directory containing per-frame binary masks (0 background, 255 foreground)
  - Optional original frames dir (for visualization), not required

Outputs
  - CSV with per-frame thickness metrics
  - Optional plots (PNG) with time series and a few example overlays

Usage
  # 基本像素厚度
  python scripts/analyze_diaphragm_thickness.py \
      --mask-dir data/video_output/ceshi_vedio/nnunet_raw \
      --out-dir  data/video_output/ceshi_vedio/analysis \
      --frames-dir data/video_output/ceshi_vedio/overlays_nnunet

  # 加入US Region标定（优先从DICOM读取；否则手动提供radial间距）
  python scripts/analyze_diaphragm_thickness.py \
      --mask-dir data/.../nnunet_raw \
      --out-dir  data/.../analysis_mm \
      --dicom-path path/to/ultrasound.dcm  # 若环境已安装pydicom
  或
  python scripts/analyze_diaphragm_thickness.py \
      --mask-dir data/.../nnunet_raw \
      --out-dir  data/.../analysis_mm \
      --physical-delta-y-mm 0.2             # 仅提供径向mm/px

Notes
  - Thickness is reported in pixels (no physical calibration available).
  - We compute two robust metrics per frame:
      1) Column range thickness: for each image column, thickness = max_y - min_y
         of the foreground in that column; frame value = median across central ROI.
      2) Distance transform thickness: 2 * distance at skeleton points; frame value
         = median across skeleton.
  - If the mask is sparse or noisy, a small morphological closing is applied.
"""

import argparse
import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_mask(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    # Accept either 0/255 or 0/1
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # Normalize to bool mask
    mask = arr > 0
    return mask


def preprocess_mask(mask: np.ndarray, close_radius: int = 1) -> np.ndarray:
    """Optionally apply a small morphological closing to reduce holes/gaps."""
    if close_radius and close_radius > 0:
        structure = ndi.generate_binary_structure(2, 1)
        if close_radius > 1:
            # Create a larger structuring element by iterating dilation of structure
            se = structure
            for _ in range(close_radius - 1):
                se = ndi.binary_dilation(se, structure=structure)
            structure = se
        mask = ndi.binary_closing(mask, structure=structure)
    return mask


def column_thickness(mask: np.ndarray, roi_frac: float = 0.6) -> tuple[float, float, float]:
    """
    Compute per-column vertical thickness and return robust stats within a central ROI.
    Returns (median, p10, p90) in pixels. NaNs are ignored.
    """
    h, w = mask.shape
    x0 = int((1 - roi_frac) / 2 * w)
    x1 = int((1 + roi_frac) / 2 * w)
    xs = range(max(0, x0), min(w, x1))
    col_thick = []
    for x in xs:
        col = mask[:, x]
        if not col.any():
            continue
        ys = np.where(col)[0]
        t = ys.max() - ys.min() + 1
        col_thick.append(t)
    if len(col_thick) == 0:
        return (np.nan, np.nan, np.nan)
    col_thick = np.array(col_thick, dtype=float)
    return (np.nanmedian(col_thick), np.nanpercentile(col_thick, 10), np.nanpercentile(col_thick, 90))


def distance_skeleton_thickness(mask: np.ndarray) -> tuple[float, float, float]:
    """
    Compute thickness via EDT and skeletonization: local thickness = 2 * distance to boundary at skeleton.
    Returns (median, mean, max) in pixels. NaNs are ignored.
    """
    if not mask.any():
        return (np.nan, np.nan, np.nan)
    # Ensure mask is a single simply-connected band by keeping the largest component
    labeled, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        mask = labeled == keep
    # Distance inside mask to background
    dist = ndi.distance_transform_edt(mask)
    # Skeleton
    skel = skeletonize(mask)
    vals = (dist[skel] * 2.0).astype(float)
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    return (np.nanmedian(vals), float(np.nanmean(vals)), float(np.nanmax(vals)))


def smooth_series(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return y
    # Simple moving median for robustness
    pad = win // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = np.nanmedian(y_pad[i:i+win])
    return out


def analyze_series(values: np.ndarray) -> dict:
    """Basic trend analysis: slope via linear fit, range, and relative change."""
    idx = np.arange(len(values))
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return {"slope": np.nan, "min": np.nan, "max": np.nan, "range": np.nan, "rel_change": np.nan}
    x = idx[mask].astype(float)
    y = values[mask].astype(float)
    # Linear regression (least squares)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    yrange = ymax - ymin
    rel = yrange / (y.mean() + 1e-6)
    return {"slope": float(slope), "min": ymin, "max": ymax, "range": float(yrange), "rel_change": float(rel)}


def plot_time_series(out_dir: Path, series_dict: dict[str, np.ndarray], title: str = "Diaphragm Thickness vs Frame"):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for name, arr in series_dict.items():
        plt.plot(arr, label=name, linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Thickness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "thickness_timeseries.png", dpi=160)
    plt.close()


def detect_cycles_mm(y_mm: np.ndarray, smooth_win: int = 5, prominence_mm: float = 0.05, min_distance: int = 8):
    """
    Detect respiratory cycles from a mm series using smoothed curve and peak/trough detection.
    Cycle = interval between consecutive minima with highest maximum in between.
    Returns dict with 'mins', 'maxs', 'cycles' list.
    """
    if y_mm is None or len(y_mm) == 0:
        return {"mins": np.array([], dtype=int), "maxs": np.array([], dtype=int), "cycles": []}
    y = np.array(y_mm, dtype=float)
    ys = smooth_series(y, win=max(1, smooth_win))
    # Prominence and distance safeguards
    prom = max(1e-6, float(prominence_mm))
    dist = max(1, int(min_distance))
    max_idx, _ = find_peaks(ys, prominence=prom, distance=dist)
    min_idx, _ = find_peaks(-ys, prominence=prom/2.0, distance=dist)

    cycles = []
    if len(min_idx) >= 2:
        for a, b in zip(min_idx[:-1], min_idx[1:]):
            mids = max_idx[(max_idx > a) & (max_idx < b)]
            if len(mids) == 0:
                continue
            k = int(mids[np.argmax(ys[mids])])
            tmin = float(y[a])
            tmax = float(y[k])
            tf = np.nan if tmin <= 0 else (tmax - tmin) / tmin
            cycles.append({
                "start_min_frame": int(a + 1),
                "peak_frame": int(k + 1),
                "end_min_frame": int(b + 1),
                "Tmin_mm": tmin,
                "Tmax_mm": tmax,
                "TFdi": float(tf),
                "duration_frames": int(b - a),
            })

    return {"mins": min_idx, "maxs": max_idx, "cycles": cycles}


def plot_cycles(out_dir: Path, y_mm: np.ndarray, mins: np.ndarray, maxs: np.ndarray, cycles: list):
    if y_mm is None or len(y_mm) == 0:
        return
    x = np.arange(len(y_mm))
    plt.figure(figsize=(10, 4))
    plt.plot(x, y_mm, label="col_median_mm", linewidth=1.5)
    if len(mins) > 0:
        plt.scatter(mins, y_mm[mins], c='tab:blue', s=15, label='minima')
    if len(maxs) > 0:
        plt.scatter(maxs, y_mm[maxs], c='tab:orange', s=15, label='maxima')
    for c in cycles:
        a = c["start_min_frame"] - 1
        b = c["end_min_frame"] - 1
        plt.axvspan(a, b, color='gray', alpha=0.08)
    plt.xlabel("Frame")
    plt.ylabel("Thickness (mm)")
    plt.title("Cycles on Thickness (mm)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "thickness_cycles.png", dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", required=True, help="Directory with binary masks (PNG)")
    ap.add_argument("--frames-dir", default=None, help="Optional frames or overlays dir for sampling overlays")
    ap.add_argument("--out-dir", required=True, help="Output directory for CSV/plots")
    ap.add_argument("--close-radius", type=int, default=1, help="Binary closing radius (pixels)")
    ap.add_argument("--roi-frac", type=float, default=0.6, help="Horizontal ROI fraction for column metric (0-1)")
    ap.add_argument("--smooth", type=int, default=5, help="Smoothing window for plots (frames)")
    # Physical calibration options
    ap.add_argument("--dicom-path", default=None, help="Optional DICOM to read US Region calibration (requires pydicom)")
    ap.add_argument("--region-index", type=int, default=0, help="US Region Sequence item index (default 0)")
    ap.add_argument("--physical-delta-y-mm", type=float, default=None, help="Manual radial spacing (mm/pixel) if DICOM not provided")
    # Cycle detection options (operate on mm series if available)
    ap.add_argument("--cycle-prom-mm", type=float, default=0.05, help="Peak prominence in mm for cycle detection")
    ap.add_argument("--cycle-min-distance", type=int, default=8, help="Minimum frames between peaks/troughs")
    args = ap.parse_args()

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort frames by index
    paths = sorted([p for p in mask_dir.glob("*.png")])
    if len(paths) == 0:
        raise SystemExit(f"No PNG masks found in {mask_dir}")

    # Try to get radial mm spacing either from DICOM or manual arg
    radial_mm_per_px = None
    dicom_info = {}
    if args.dicom_path:
        try:
            import pydicom  # type: ignore
            ds = pydicom.dcmread(args.dicom_path, stop_before_pixels=True)
            de = ds.get((0x0018, 0x6011))  # US Region Sequence (SQ)
            seq = getattr(de, 'value', None)
            if seq is not None and len(seq) > args.region_index:
                it = seq[args.region_index]
                # PhysicalDeltaY is radial spacing in mm/px when RegionSpatialFormat=1 (POLAR)
                delta_y = float(getattr(it, 'PhysicalDeltaY', float('nan')))
                units_y = int(getattr(it, 'PhysicalUnitsYDirection', -1)) if hasattr(it, 'PhysicalUnitsYDirection') else None
                # Convert to mm/px according to units
                radial_mm_per_px = None
                if np.isfinite(delta_y):
                    if units_y == 3:  # centimeters
                        radial_mm_per_px = delta_y * 10.0
                    elif units_y == 6:  # millimeters
                        radial_mm_per_px = delta_y
                    else:
                        # Assume value already in mm if units unspecified/unexpected
                        radial_mm_per_px = delta_y
                dicom_info = {
                    'RegionSpatialFormat': int(getattr(it, 'RegionSpatialFormat', -1)) if hasattr(it, 'RegionSpatialFormat') else None,
                    'PhysicalDeltaX': float(getattr(it, 'PhysicalDeltaX', float('nan'))),
                    'PhysicalDeltaY': float(getattr(it, 'PhysicalDeltaY', float('nan'))),
                    'ReferencePixelX0': int(getattr(it, 'ReferencePixelX0', -1)) if hasattr(it, 'ReferencePixelX0') else None,
                    'ReferencePixelY0': int(getattr(it, 'ReferencePixelY0', -1)) if hasattr(it, 'ReferencePixelY0') else None,
                    'PhysicalUnitsXDirection': int(getattr(it, 'PhysicalUnitsXDirection', -1)) if hasattr(it, 'PhysicalUnitsXDirection') else None,
                    'PhysicalUnitsYDirection': int(getattr(it, 'PhysicalUnitsYDirection', -1)) if hasattr(it, 'PhysicalUnitsYDirection') else None,
                }
        except Exception as e:
            print(f"Warning: failed to read DICOM calibration: {e}")
    if radial_mm_per_px is None and args.physical_delta_y_mm is not None:
        radial_mm_per_px = float(args.physical_delta_y_mm)

    rows = []
    col_series = []
    col_series_mm = []
    dt_series = []
    for i, p in enumerate(paths, start=1):
        mask = read_mask(p)
        mask = preprocess_mask(mask, close_radius=args.close_radius)
        col_med, col_p10, col_p90 = column_thickness(mask, roi_frac=args.roi_frac)
        dt_med, dt_mean, dt_max = distance_skeleton_thickness(mask)
        rowd = {
            "frame_index": i,
            "frame_name": p.name,
            "col_median": col_med,
            "col_p10": col_p10,
            "col_p90": col_p90,
            "dt_median": dt_med,
            "dt_mean": dt_mean,
            "dt_max": dt_max,
        }
        if radial_mm_per_px is not None and not np.isnan(radial_mm_per_px):
            col_med_mm = col_med * radial_mm_per_px if not np.isnan(col_med) else np.nan
            col_p10_mm = col_p10 * radial_mm_per_px if not np.isnan(col_p10) else np.nan
            col_p90_mm = col_p90 * radial_mm_per_px if not np.isnan(col_p90) else np.nan
            rowd.update({
                "col_median_mm": col_med_mm,
                "col_p10_mm": col_p10_mm,
                "col_p90_mm": col_p90_mm,
            })
            col_series_mm.append(col_med_mm)
        rows.append(rowd)
        col_series.append(col_med)
        dt_series.append(dt_med)

    col_series = np.array(col_series, dtype=float)
    dt_series = np.array(dt_series, dtype=float)

    # Save CSV
    csv_path = out_dir / "thickness_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()})

    # Analyze trends
    col_stats = analyze_series(col_series)
    dt_stats = analyze_series(dt_series)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        if dicom_info:
            f.write("US Region Calibration (from DICOM)\n")
            for k,v in dicom_info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        if radial_mm_per_px is not None:
            f.write(f"Radial spacing (mm/px): {radial_mm_per_px}\n\n")

        f.write("Column thickness median trend (pixels)\n")
        for k, v in col_stats.items():
            f.write(f"  {k}: {v}\n")
        if radial_mm_per_px is not None and len(col_series_mm) > 0:
            col_mm = np.array(col_series_mm, dtype=float)
            col_mm_stats = analyze_series(col_mm)
            f.write("\nColumn thickness median trend (mm)\n")
            for k, v in col_mm_stats.items():
                f.write(f"  {k}: {v}\n")

        f.write("\nDistance-skeleton median trend (pixels)\n")
        for k, v in dt_stats.items():
            f.write(f"  {k}: {v}\n")

    # Plot time series (smoothed)
    col_s = smooth_series(col_series, win=max(1, args.smooth))
    dt_s = smooth_series(dt_series, win=max(1, args.smooth))
    series = {"col_median_px": col_s, "dt_median_px": dt_s}
    if radial_mm_per_px is not None and len(col_series_mm) > 0:
        col_mm = np.array(col_series_mm, dtype=float)
        series["col_median_mm"] = smooth_series(col_mm, win=max(1, args.smooth))
    plot_time_series(out_dir, series)

    # Cycle detection (mm) and export
    if radial_mm_per_px is not None and len(col_series_mm) > 0:
        det = detect_cycles_mm(np.array(col_series_mm, dtype=float), smooth_win=max(1, args.smooth), prominence_mm=args.cycle_prom_mm, min_distance=max(1, args.cycle_min_distance))
        cycles = det["cycles"]
        if len(cycles) > 0:
            cyc_path = out_dir / "cycle_metrics_mm.csv"
            with open(cyc_path, "w", newline="", encoding="utf-8") as f:
                fn = ["cycle_index","start_min_frame","peak_frame","end_min_frame","Tmin_mm","Tmax_mm","TFdi","duration_frames"]
                w = csv.DictWriter(f, fieldnames=fn)
                w.writeheader()
                for i, c in enumerate(cycles, start=1):
                    w.writerow({"cycle_index": i, **c})
            tfs = np.array([c["TFdi"] for c in cycles], dtype=float)
            tf_mean = float(np.nanmean(tfs)) if tfs.size else np.nan
            tf_std = float(np.nanstd(tfs)) if tfs.size else np.nan
            with open(out_dir / "summary.txt", "a", encoding="utf-8") as f:
                f.write("\nCycle-level metrics (mm)\n")
                f.write(f"  cycles: {len(cycles)}\n")
                f.write(f"  TFdi mean: {tf_mean}\n")
                f.write(f"  TFdi std: {tf_std}\n")
            plot_cycles(out_dir, np.array(col_series_mm, dtype=float), det["mins"], det["maxs"], cycles)

    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'summary.txt'}")
    print(f"Saved: {out_dir / 'thickness_timeseries.png'}")


if __name__ == "__main__":
    main()
