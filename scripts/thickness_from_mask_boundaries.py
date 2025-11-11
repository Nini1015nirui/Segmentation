#!/usr/bin/env python3
"""
Thickness from segmentation boundaries (mask-only, no image gradients):
  - Build centerline from mask (column midline), select ROI by target arclength
  - At each centerline point, cast a normal ray and intersect with mask boundaries on both sides
  - Thickness = distance along normal between the two boundary intersections (in mm)
  - Aggregate per-frame via median; build time-series and cycle metrics (T_EE, T_EI, DTF)

Inputs
  --mask-dir: directory with binary masks (0/255)
  --dicom-path or (--mm-x & --mm-y): calibration to convert px to mm
Optional
  --frames-dir: frames for debug overlay only (not used for measurement)

Outputs
  - thickness_maskb_metrics.csv: per-frame median thickness (mm) and QC
  - thickness_maskb_timeseries.png: time series plot
  - cycle_metrics_maskb_mm.csv: per-cycle Tmin/Tmax/TFdi
  - summary_maskb.txt: calibration and stats
  - debug_*.png: optional overlays for a few frames
"""

import argparse
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dicom_mm(dicom_path: Path):
    mmx = mmy = None
    info = {}
    try:
        import pydicom
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        de = ds.get((0x0018, 0x6011))
        seq = getattr(de, 'value', None)
        if seq and len(seq) > 0:
            it0 = seq[0]
            def to_mm(val, units):
                if val is None:
                    return None
                v = float(val)
                if units == 3:  # cm
                    return v * 10.0
                return v
            ux = int(getattr(it0, 'PhysicalUnitsXDirection', -1)) if hasattr(it0, 'PhysicalUnitsXDirection') else None
            uy = int(getattr(it0, 'PhysicalUnitsYDirection', -1)) if hasattr(it0, 'PhysicalUnitsYDirection') else None
            mmx = to_mm(getattr(it0, 'PhysicalDeltaX', None), ux)
            mmy = to_mm(getattr(it0, 'PhysicalDeltaY', None), uy)
            info = {
                'RegionSpatialFormat': int(getattr(it0, 'RegionSpatialFormat', -1)) if hasattr(it0, 'RegionSpatialFormat') else None,
                'PhysicalDeltaX_raw': getattr(it0, 'PhysicalDeltaX', None),
                'PhysicalDeltaY_raw': getattr(it0, 'PhysicalDeltaY', None),
                'PhysicalUnitsXDirection': ux,
                'PhysicalUnitsYDirection': uy,
            }
        if mmx is None or mmy is None:
            ps = ds.get((0x0028, 0x0030))
            if ps is not None:
                try:
                    mmy, mmx = [float(x) for x in ps.value]
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: DICOM read/calibration failed: {e}")
    return mmx, mmy, info


def to_gray(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)


def extract_centerline_from_mask(mask: np.ndarray):
    h, w = mask.shape
    xs = np.arange(w)
    y_center = np.full(w, np.nan, dtype=np.float32)
    valid = np.zeros(w, dtype=bool)
    cols = np.where(mask.sum(axis=0) > 0)[0]
    for x in cols:
        ys = np.where(mask[:, x])[0]
        y_center[x] = 0.5 * (ys.min() + ys.max())
        valid[x] = True
    # smooth centerline where valid
    idx = np.where(valid)[0]
    if idx.size >= 7:
        yc = y_center[valid]
        win = 7 if yc.size >= 7 else (yc.size // 2 * 2 + 1)
        if win >= 5:
            y_center[valid] = savgol_filter(yc, window_length=win, polyorder=2, mode='interp')
    return xs.astype(np.float32), y_center, valid


def select_roi_by_length(xs, y_center, valid, mmx, mmy, roi_len_mm=15.0):
    idx = np.where(valid)[0]
    if idx.size < 20:
        return None
    mid = idx[idx.size // 2]
    def arc_len(i0, i1):
        s = 0.0
        step = 1 if i1 >= i0 else -1
        prev = i0
        for i in range(i0+step, i1+step, step):
            dy = (y_center[i] - y_center[prev]) * mmy
            dx = (xs[i] - xs[prev]) * mmx
            s += math.hypot(dx, dy)
            prev = i
        return s
    L = 0.0
    l = r = mid
    while L < roi_len_mm and (l > idx[0] or r < idx[-1]):
        ls = arc_len(l, l-1) if l > idx[0] else float('inf')
        rs = arc_len(r, r+1) if r < idx[-1] else float('inf')
        if ls <= rs:
            if l > idx[0]: l -= 1
        else:
            if r < idx[-1]: r += 1
        L = arc_len(l, r)
        if r - l > 2048:
            break
    sel = np.zeros_like(valid)
    sel[l:r+1] = True
    return sel


def normals_from_centerline(xs, y_center, sel, mmx, mmy):
    idx = np.where(sel)[0]
    if idx.size < 5:
        return None
    pts_px = np.stack([xs[idx], y_center[idx]], axis=1)
    tangents_mm = []
    for i in range(len(idx)):
        i0 = max(0, i-1)
        i1 = min(len(idx)-1, i+1)
        dx_px = xs[idx[i1]] - xs[idx[i0]]
        dy_px = y_center[idx[i1]] - y_center[idx[i0]]
        v = np.array([dx_px*mmx, dy_px*mmy], dtype=np.float32)
        nrm = np.linalg.norm(v) + 1e-8
        t = v / nrm
        tangents_mm.append(t)
    tangents_mm = np.vstack(tangents_mm)
    normals_mm = np.stack([-tangents_mm[:,1], tangents_mm[:,0]], axis=1)
    flip = np.sign(normals_mm[:,1] + 1e-8)
    flip[flip == 0] = 1
    normals_mm = normals_mm * flip[:, None]
    return idx, pts_px, normals_mm


def bilinear(img, yx):
    h, w = img.shape
    y = yx[:,0]; x = yx[:,1]
    x0 = np.floor(x).astype(int); x1 = x0 + 1
    y0 = np.floor(y).astype(int); y1 = y0 + 1
    x0c = np.clip(x0, 0, w-1); x1c = np.clip(x1, 0, w-1)
    y0c = np.clip(y0, 0, h-1); y1c = np.clip(y1, 0, h-1)
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return (Ia*wa + Ib*wb + Ic*wc + Id*wd).astype(np.float32)


def sample_along_normal_mask(mask_bin, pt_px, normal_mm, mmx, mmy, u_min_mm, u_max_mm, du_mm):
    n = int(round((u_max_mm - u_min_mm) / du_mm)) + 1
    U = np.linspace(u_min_mm, u_max_mm, n, dtype=np.float32)
    dx_per_mm = normal_mm[0] / mmx
    dy_per_mm = normal_mm[1] / mmy
    xs = pt_px[0] + U * dx_per_mm
    ys = pt_px[1] + U * dy_per_mm
    # bilinear on float mask [0,1]
    I = bilinear(mask_bin.astype(np.float32), np.stack([ys, xs], axis=1))
    return U, I


def boundary_crossings_from_center(U, I, center_index):
    """Find boundary crossings on both sides starting near center_index.
    Returns (u_minus, u_plus) where I crosses 0.5 from inside(≈1) to outside(≈0).
    """
    # ensure near center is inside; search small window for an inside point
    n = len(U)
    c = center_index
    inside = I >= 0.5
    if c < 0 or c >= n:
        return None, None
    if not inside[c]:
        # look around
        found = False
        for off in range(1, min(5, n//2)):
            for cc in (c-off, c+off):
                if 0 <= cc < n and inside[cc]:
                    c = cc
                    found = True
                    break
            if found: break
        if not inside[c]:
            return None, None
    # forward (+u)
    up = None
    for k in range(c+1, n):
        if inside[k-1] and not inside[k]:
            # linear interpolate crossing where I=0.5 between k-1 and k
            u0,u1 = U[k-1], U[k]
            i0,i1 = I[k-1], I[k]
            t = (0.5 - i0) / (i1 - i0 + 1e-6)
            up = u0 + t*(u1-u0)
            break
    # backward (-u)
    um = None
    for k in range(c-1, -1, -1):
        if inside[k+1] and not inside[k]:
            u0,u1 = U[k], U[k+1]
            i0,i1 = I[k], I[k+1]
            t = (0.5 - i1) / (i0 - i1 + 1e-6)
            um = u1 + t*(u0-u1)
            break
    return um, up


def detect_cycles(y_mm, prom_mm=0.02, min_dist_frames=10):
    y = np.asarray(y_mm, dtype=float)
    if y.size == 0 or np.all(~np.isfinite(y)):
        return []
    ys = y.copy()
    if y.size >= 5:
        win = min(9, y.size - (y.size+1)%2)
        if win >= 5 and win % 2 == 1:
            ys = savgol_filter(y, window_length=win, polyorder=2, mode='interp')
    mins, _ = find_peaks(-ys, prominence=max(1e-6, prom_mm/2.0), distance=max(1, min_dist_frames))
    maxs, _ = find_peaks(ys, prominence=max(1e-6, prom_mm), distance=max(1, min_dist_frames))
    cycles = []
    for a, b in zip(mins[:-1], mins[1:]):
        mids = maxs[(maxs > a) & (maxs < b)]
        if mids.size == 0:
            continue
        k = int(mids[np.argmax(ys[mids])])
        tmin = float(y[a]); tmax = float(y[k])
        tf = (tmax - tmin) / tmin if tmin > 0 else float('nan')
        cycles.append({
            'start_min_frame': int(a+1),
            'peak_frame': int(k+1),
            'end_min_frame': int(b+1),
            'Tmin_mm': tmin,
            'Tmax_mm': tmax,
            'TFdi': tf,
            'duration_frames': int(b - a),
        })
    return cycles


def overlay_debug(out_dir: Path, frame_name: str, frame_img, pts_px, normals_mm, mmx, mmy, u1s, u2s):
    if frame_img is None:
        return
    img = to_gray(frame_img)
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img, cmap='gray')
    step = max(1, len(pts_px)//60)
    for i in range(0, len(pts_px), step):
        pt = pts_px[i]
        nmm = normals_mm[i]
        dx = nmm[0]/mmx; dy = nmm[1]/mmy
        ax.plot([pt[0]-2*dx, pt[0]+2*dx], [pt[1]-2*dy, pt[1]+2*dy], 'c-', lw=0.5)
        if np.isfinite(u1s[i]) and np.isfinite(u2s[i]):
            x1 = pt[0] + u1s[i]*dx; y1 = pt[1] + u1s[i]*dy
            x2 = pt[0] + u2s[i]*dx; y2 = pt[1] + u2s[i]*dy
            ax.plot([x1],[y1],'r.', ms=2)
            ax.plot([x2],[y2],'g.', ms=2)
    ax.set_title(f"Boundary-normal intersections: {frame_name}")
    ax.set_xlim([0, w]); ax.set_ylim([h, 0])
    fig.tight_layout(); fig.savefig(out_dir / f"debug_{frame_name}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask-dir', required=True)
    ap.add_argument('--dicom-path', default=None)
    ap.add_argument('--mm-x', type=float, default=None)
    ap.add_argument('--mm-y', type=float, default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--frames-dir', default=None, help='Optional frames dir for debug overlays')
    # geometry
    ap.add_argument('--roi-len-mm', type=float, default=15.0)
    ap.add_argument('--pre-mm', type=float, default=4.0)
    ap.add_argument('--post-mm', type=float, default=6.0)
    ap.add_argument('--du-mm', type=float, default=0.075)
    # cycles
    ap.add_argument('--cycle-prom-mm', type=float, default=0.03)
    ap.add_argument('--cycle-min-distance', type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mmx = args.mm_x; mmy = args.mm_y
    info = {}
    if args.dicom_path:
        mmx_d, mmy_d, info = read_dicom_mm(Path(args.dicom_path))
        mmx = mmx or mmx_d; mmy = mmy or mmy_d
    if mmx is None or mmy is None:
        raise SystemExit('Calibration missing: provide --dicom-path or both --mm-x and --mm-y')

    mask_paths = sorted(Path(args.mask_dir).glob('frame_*.png'))
    if not mask_paths:
        raise SystemExit('No masks found (frame_*.png)')
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    rows = []
    T_series = []
    for mpath in mask_paths:
        mask = np.array(Image.open(mpath))
        mask_bin = (mask > 0).astype(np.uint8)
        xs, y_center, valid = extract_centerline_from_mask(mask_bin)
        sel = select_roi_by_length(xs, y_center, valid, mmx, mmy, roi_len_mm=args.roi_len_mm)
        if sel is None:
            rows.append({'frame_name': mpath.name, 'T_median_mm': '', 'IQR_mm': '', 'valid_ratio': 0.0})
            T_series.append(np.nan)
            continue
        idx, pts_px, normals_mm = normals_from_centerline(xs, y_center, sel, mmx, mmy)
        if idx is None:
            rows.append({'frame_name': mpath.name, 'T_median_mm': '', 'IQR_mm': '', 'valid_ratio': 0.0})
            T_series.append(np.nan)
            continue
        n_sel = int(np.sum(sel))
        n_ok = int(np.sum(valid & sel))
        valid_ratio = (n_ok / n_sel) if n_sel > 0 else 0.0
        # per-location thickness from mask boundary intersections
        Tloc = []; u1s=[]; u2s=[]
        # precompute center index in sampled vector U: nearest to 0
        for pt, nmm in zip(pts_px, normals_mm):
            U, I = sample_along_normal_mask(mask_bin, pt, nmm, mmx, mmy, -args.pre_mm, args.post_mm, args.du_mm)
            # denoise slightly
            if I.size >= 5:
                k = 3 if I.size >= 3 else 1
                I = median_filter(I, size=k)
            # center index
            c = int(np.argmin(np.abs(U)))
            um, up = boundary_crossings_from_center(U, I, c)
            if um is None or up is None:
                Tloc.append(np.nan); u1s.append(np.nan); u2s.append(np.nan)
            else:
                Tloc.append(up - um); u1s.append(um); u2s.append(up)
        Tloc = np.array(Tloc, dtype=float)
        # fill gaps before smoothing
        if np.sum(~np.isnan(Tloc)) >= 5:
            Tfill = Tloc.copy()
            # nearest fill
            last = None
            for i in range(len(Tfill)):
                if np.isfinite(Tfill[i]):
                    last = Tfill[i]
                elif last is not None:
                    Tfill[i] = last
            if np.any(np.isnan(Tfill)):
                # fill remaining from right
                nxt = None
                for i in range(len(Tfill)-1,-1,-1):
                    if np.isfinite(Tfill[i]):
                        nxt = Tfill[i]
                    elif nxt is not None:
                        Tfill[i] = nxt
            win = min(len(Tfill) - (len(Tfill)+1)%2,  nine := 9) if len(Tfill)>=9 else max(5, len(Tfill)//2*2+1)
            if win % 2 == 0: win += 1
            if win >= 5:
                T_s = savgol_filter(Tfill, window_length=win, polyorder=2, mode='interp')
            else:
                T_s = Tfill
        else:
            T_s = Tloc
        # per-frame stats
        med = float(np.nanmedian(T_s)) if np.any(np.isfinite(T_s)) else float('nan')
        p25 = float(np.nanpercentile(T_s,25)) if np.any(np.isfinite(T_s)) else float('nan')
        p75 = float(np.nanpercentile(T_s,75)) if np.any(np.isfinite(T_s)) else float('nan')
        rows.append({'frame_name': mpath.name, 'T_median_mm': med, 'IQR_mm': (p75-p25) if (not math.isnan(p75) and not math.isnan(p25)) else '' , 'valid_ratio': valid_ratio})
        T_series.append(med)
        # debug overlay
        if frames_dir is not None and mpath.name in {mask_paths[0].name, mask_paths[len(mask_paths)//2].name, mask_paths[-1].name}:
            fpath = frames_dir / mpath.name
            if fpath.exists():
                frame_img = np.array(Image.open(fpath))
            else:
                frame_img = None
            overlay_debug(out_dir, mpath.name, frame_img, pts_px, normals_mm, mmx, mmy, np.array(u1s), np.array(u2s))

    # save CSV
    csv_path = out_dir / 'thickness_maskb_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # time series + cycles
    y = np.array(T_series, dtype=float)
    fig = plt.figure(figsize=(10,4))
    plt.plot(y, 'b-', lw=1.5, label='T_median_mm (mask-boundary)')
    plt.xlabel('Frame'); plt.ylabel('Thickness (mm)'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'thickness_maskb_timeseries.png', dpi=160)
    plt.close(fig)

    cycles = detect_cycles(y, prom_mm=args.cycle_prom_mm, min_dist_frames=args.cycle_min_distance)
    if cycles:
        cyc_csv = out_dir / 'cycle_metrics_maskb_mm.csv'
        with open(cyc_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['cycle_index','start_min_frame','peak_frame','end_min_frame','Tmin_mm','Tmax_mm','TFdi','duration_frames'])
            writer.writeheader()
            for i, c in enumerate(cycles, start=1):
                writer.writerow({'cycle_index': i, **c})
        tfs = np.array([c['TFdi'] for c in cycles if np.isfinite(c['TFdi'])], dtype=float)
        tf_mean = float(np.mean(tfs)) if tfs.size else float('nan')
        tf_std = float(np.std(tfs)) if tfs.size else float('nan')
        with open(out_dir / 'summary_maskb.txt', 'w', encoding='utf-8') as f:
            if info:
                f.write('Calibration from DICOM (US Region / PixelSpacing)\n')
                for k,v in info.items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
            f.write(f'mm_per_px: X={mmx}, Y={mmy}\n')
            f.write(f'Frames processed: {len(mask_paths)}\n')
            f.write(f'Cycles: {len(cycles)}\n')
            f.write(f'TFdi mean: {tf_mean}\n')
            f.write(f'TFdi std: {tf_std}\n')
    else:
        with open(out_dir / 'summary_maskb.txt', 'w', encoding='utf-8') as f:
            if info:
                f.write('Calibration from DICOM (US Region / PixelSpacing)\n')
                for k,v in info.items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
            f.write(f'mm_per_px: X={mmx}, Y={mmy}\n')
            f.write(f'Frames processed: {len(mask_paths)}\n')
            f.write('Cycles: 0\n')

    print(f'Saved: {csv_path}')
    print(f'Saved: {out_dir / "thickness_maskb_timeseries.png"}')
    print(f'Saved: {out_dir / "summary_maskb.txt"}')


if __name__ == '__main__':
    main()

