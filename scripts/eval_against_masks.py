#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import csv
import cv2
import numpy as np


def as_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def metrics_binary(pred: np.ndarray, gt: np.ndarray) -> dict:
    # pred, gt are uint8 {0,1}
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    eps = 1e-8
    dice = (2 * tp) / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    acc = (tp + tn) / max(tp + tn + fp + fn, eps)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, dice=dice, iou=iou, precision=precision, recall=recall, accuracy=acc)


def draw_overlays(img_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, pp: np.ndarray | None = None) -> np.ndarray:
    # Make a copy and dim it slightly
    overlay = (img_bgr * 0.8).astype(np.uint8)
    # Draw contours
    def draw_mask_contour(dst, mask, color, thickness=2):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dst, cnts, -1, color, thickness=thickness)

    # GT = green, RAW = red, PP = blue
    draw_mask_contour(overlay, gt, (0, 255, 0), 2)
    draw_mask_contour(overlay, pred, (0, 0, 255), 2)
    if pp is not None:
        draw_mask_contour(overlay, pp, (255, 0, 0), 2)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions against GT masks and create overlays")
    ap.add_argument('--images_dir', required=True, help='Directory with original images (for overlays)')
    ap.add_argument('--gt_dir', required=True, help='Directory with ground-truth binary masks')
    ap.add_argument('--raw_pred_dir', required=True, help='Directory with raw prediction masks')
    ap.add_argument('--pp_pred_dir', required=False, default=None, help='Directory with postprocessed masks (optional)')
    ap.add_argument('--out_dir', required=True, help='Output directory for metrics and overlays')
    ap.add_argument('--pattern', default='*.png', help='Filename glob to filter cases (default: *.png)')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    raw_dir = Path(args.raw_pred_dir)
    pp_dir = Path(args.pp_pred_dir) if args.pp_pred_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / 'overlays'
    ov_dir.mkdir(exist_ok=True)

    gt_files = sorted(gt_dir.glob(args.pattern))
    if not gt_files:
        print(f"No GT files found in {gt_dir} with pattern {args.pattern}")
        return 1

    rows = []
    agg_raw = []
    agg_pp = []

    for gt_path in gt_files:
        stem = gt_path.name
        raw_path = raw_dir / stem
        pp_path = (pp_dir / stem) if pp_dir else None

        if not raw_path.exists():
            print(f"[skip] missing raw prediction for {stem}")
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or raw is None:
            print(f"[skip] read error: {gt_path} or {raw_path}")
            continue

        # Ensure same shape
        if raw.shape != gt.shape:
            raw = cv2.resize(raw, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_b = as_bool(gt)
        raw_b = as_bool(raw)

        m_raw = metrics_binary(raw_b, gt_b)
        agg_raw.append(m_raw)

        pp_b = None
        m_pp = None
        if pp_path and pp_path.exists():
            pp = cv2.imread(str(pp_path), cv2.IMREAD_GRAYSCALE)
            if pp is not None:
                if pp.shape != gt.shape:
                    pp = cv2.resize(pp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                pp_b = as_bool(pp)
                m_pp = metrics_binary(pp_b, gt_b)
                agg_pp.append(m_pp)

        # Try to read original image for overlay
        img_path = images_dir / stem
        img_bgr = None
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            # create a gray canvas if missing
            img_bgr = np.stack([gt]*3, axis=-1)
        ov = draw_overlays(img_bgr, gt_b, raw_b, pp_b)
        cv2.imwrite(str(ov_dir / stem), ov)

        rows.append({
            'case': stem,
            **{f'raw_{k}': v for k, v in m_raw.items()},
            **({f'pp_{k}': v for k, v in m_pp.items()} if m_pp else {}),
        })

    # Write CSV
    csv_path = out_dir / 'metrics.csv'
    if rows:
        # Collect all keys
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Write JSON summary
    def mean_of(metric_list: list[dict], key: str) -> float:
        arr = [float(m[key]) for m in metric_list if key in m]
        return float(np.mean(arr)) if arr else 0.0

    summary = {
        'num_cases': len(rows),
        'raw': {
            'dice': mean_of(agg_raw, 'dice'),
            'iou': mean_of(agg_raw, 'iou'),
            'precision': mean_of(agg_raw, 'precision'),
            'recall': mean_of(agg_raw, 'recall'),
            'accuracy': mean_of(agg_raw, 'accuracy'),
        },
        'pp': {
            'dice': mean_of(agg_pp, 'dice'),
            'iou': mean_of(agg_pp, 'iou'),
            'precision': mean_of(agg_pp, 'precision'),
            'recall': mean_of(agg_pp, 'recall'),
            'accuracy': mean_of(agg_pp, 'accuracy'),
        } if agg_pp else None,
    }
    with open(out_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Per-case metrics CSV: {csv_path}")
    print(f"Overlays: {ov_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

