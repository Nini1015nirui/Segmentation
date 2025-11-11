#!/usr/bin/env python3
"""
Plan A implementation: Normal-profile double-boundary search with global smoothing (DP),
from frames + segmentation + DICOM calibration to per-frame thickness (mm), cycles, and QC.

Minimal viable engineering version tuned for current dataset layout:
  - frames:   data/video_output/ceshi_vedio/frames/frame_*.png (RGB)
  - masks:    data/video_output/ceshi_vedio/nnunet_raw/frame_*.png (0/255)
  - DICOM:    测试vedio (single file with US Region Sequence)

Key steps
  1) Read calibration (mm/px) from DICOM US Region or fallback to manual args
  2) For each frame: build a centerline from mask, select ROI (~15 mm)
  3) Along the centerline, sample 1D intensity profiles along the (metric-correct) normal
  4) On each profile, find K boundary candidate pairs (opposite-signed gradient peaks), filter by thickness prior
  5) Use DP across arclength to choose a globally smooth, non-crossing boundary sequence with skip support
  6) Compute local thickness T(s) and robust per-frame statistic (median mm), plus QC
  7) Build time series and detect cycles to report T_EE/T_EI/DTF

Notes
  - We avoid heavy resampling by sampling along normals in pixel space with metric-correct steps.
  - This is a robust baseline and can be tuned further (valley/SNR penalties, curvature-aware weights, etc.).
"""

import argparse
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dicom_mm(dicom_path: Path):
    mmx = mmy = None
    info = {}
    try:
        import pydicom
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        # Prefer US Region Sequence
        de = ds.get((0x0018, 0x6011))
        seq = getattr(de, 'value', None)
        if seq and len(seq) > 0:
            it0 = seq[0]
            # Units: 3=cm, 6=mm (DICOM units code for distance)
            def get_mm(val, units_code):
                if val is None or (isinstance(val, float) and (not np.isfinite(val))):
                    return None
                if units_code == 3:  # cm
                    return float(val) * 10.0
                return float(val)
            units_x = int(getattr(it0, 'PhysicalUnitsXDirection', -1)) if hasattr(it0, 'PhysicalUnitsXDirection') else None
            units_y = int(getattr(it0, 'PhysicalUnitsYDirection', -1)) if hasattr(it0, 'PhysicalUnitsYDirection') else None
            mmx = get_mm(getattr(it0, 'PhysicalDeltaX', None), units_x)
            mmy = get_mm(getattr(it0, 'PhysicalDeltaY', None), units_y)
            info = {
                'RegionSpatialFormat': int(getattr(it0, 'RegionSpatialFormat', -1)) if hasattr(it0, 'RegionSpatialFormat') else None,
                'PhysicalDeltaX_raw': getattr(it0, 'PhysicalDeltaX', None),
                'PhysicalDeltaY_raw': getattr(it0, 'PhysicalDeltaY', None),
                'PhysicalUnitsXDirection': units_x,
                'PhysicalUnitsYDirection': units_y,
            }
        # Fallback: PixelSpacing
        if mmx is None or mmy is None:
            ps = ds.get((0x0028, 0x0030))
            if ps is not None:
                try:
                    mmy, mmx = [float(x) for x in ps.value]  # DICOM PixelSpacing = [row(mm), col(mm)]
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: DICOM read/calibration failed: {e}")
    return mmx, mmy, info


def to_gray(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)
    # RGB to gray
    return (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]).astype(np.float32)


def extract_centerline_from_mask(mask: np.ndarray):
    """Centerline from mask via column-wise midlines: y_c(x) = (ymin+ymax)/2 where mask is present.
    Returns arrays x_idx (pixel) and y_center (pixel), and valid boolean mask.
    """
    h, w = mask.shape
    xs = np.arange(w)
    y_center = np.full(w, np.nan, dtype=np.float32)
    valid = np.zeros(w, dtype=bool)
    cols = np.where(mask.sum(axis=0) > 0)[0]
    for x in cols:
        ys = np.where(mask[:, x])[0]
        y_center[x] = 0.5 * (ys.min() + ys.max())
        valid[x] = True
    # Smooth the centerline where valid
    idx = np.where(valid)[0]
    if idx.size >= 7:
        yc = y_center[valid]
        # light smoothing (Savitzky–Golay)
        win = 7 if yc.size >= 7 else (yc.size // 2 * 2 + 1)
        if win >= 5:
            y_center[valid] = savgol_filter(yc, window_length=win, polyorder=2, mode='interp')
    return xs.astype(np.float32), y_center, valid


def select_roi_by_length(xs, y_center, valid, mmx, mmy, roi_len_mm=15.0):
    """Select a contiguous arclength segment around the central valid region with target length (mm)."""
    idx = np.where(valid)[0]
    if idx.size < 20:
        return None
    # Choose central segment
    mid = idx[idx.size // 2]
    # Expand to left/right until reach arclength ~ roi_len_mm
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
        # try grow the shorter side next
        ls = arc_len(l, l-1) if l > idx[0] else float('inf')
        rs = arc_len(r, r+1) if r < idx[-1] else float('inf')
        if ls <= rs:
            if l > idx[0]: l -= 1
        else:
            if r < idx[-1]: r += 1
        L = arc_len(l, r)
        if r - l > 1024:  # safety
            break
    sel = np.zeros_like(valid)
    sel[l:r+1] = True
    return sel


def normals_from_centerline(xs, y_center, sel, mmx, mmy):
    idx = np.where(sel)[0]
    if idx.size < 5:
        return None
    pts_px = np.stack([xs[idx], y_center[idx]], axis=1)
    # compute tangent in mm units
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
    # normal is rotated left: n = [-t_y, t_x], enforce pointing to +y (deeper)
    normals_mm = np.stack([-tangents_mm[:,1], tangents_mm[:,0]], axis=1)
    # Flip sign so that normal tends to increasing image y
    flip = np.sign(normals_mm[:,1] + 1e-8)
    flip[flip == 0] = 1
    normals_mm = normals_mm * flip[:, None]
    return idx, pts_px, normals_mm


def bilinear(img, yx):
    """Bilinear interpolate image at points yx (N,2) with order (y,x)."""
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


def sample_profile(img_gray, pt_px, normal_mm, mmx, mmy, u_min_mm, u_max_mm, du_mm):
    """Sample intensity along normal in mm coordinates from u_min to u_max.
    Return u (mm) and I (float32)[n].
    """
    n = int(round((u_max_mm - u_min_mm) / du_mm)) + 1
    U = np.linspace(u_min_mm, u_max_mm, n, dtype=np.float32)
    # Convert mm displacements to pixel coordinates
    # delta_px = [ (n_x * du) / mmx, (n_y * du) / mmy ]
    dx_per_mm = normal_mm[0] / mmx
    dy_per_mm = normal_mm[1] / mmy
    xs = pt_px[0] + U * dx_per_mm
    ys = pt_px[1] + U * dy_per_mm
    I = bilinear(img_gray, np.stack([ys, xs], axis=1))
    return U, I


def gradient_profile(I, du_mm, sigma_mm=0.25):
    """Simple smoothed derivative: median filter + Savitzky–Golay derivative approximation."""
    I2 = I.copy()
    # light median to suppress speckle
    k = max(1, int(round(0.3 / max(1e-3, du_mm))))
    if k % 2 == 0: k += 1
    if k >= 3:
        I2 = median_filter(I2, size=k, mode='nearest')
    # Savitzky–Golay smoothing + numerical derivative
    win = max(5, int(round(sigma_mm / du_mm)) * 4 + 1)
    if win % 2 == 0: win += 1
    if win >= 5 and win < I2.size:
        Is = savgol_filter(I2, window_length=win, polyorder=2, deriv=0, delta=du_mm, mode='nearest')
        G = savgol_filter(I2, window_length=win, polyorder=2, deriv=1, delta=du_mm, mode='nearest')
        return Is.astype(np.float32), G.astype(np.float32)
    # fallback
    G = np.gradient(I2, du_mm)
    return I2.astype(np.float32), G.astype(np.float32)


def profile_candidates(U, I, G, K=6, dmin=0.8, dmax=6.0, require_cross_center=True):
    """Generate up to K boundary candidate pairs (u1,u2) with opposite-signed G peaks and spacing prior.
    Simple scoring phi: -(|G(u1)|+|G(u2)|) + lambda_valley * valley_penalty.
    """
    # Normalize intensity to [0,1]
    Imin, Imax = float(np.min(I)), float(np.max(I))
    Ir = (I - Imin) / (max(1e-6, Imax - Imin))
    # Find peaks in G and -G
    # thresholds as percentiles
    thr = np.percentile(np.abs(G), 70)
    pos_idx, _ = find_peaks(G, height=max(1e-6, thr*0.5), distance=2)
    neg_idx, _ = find_peaks(-G, height=max(1e-6, thr*0.5), distance=2)
    cands = []
    # two polarities: (neg, pos) and (pos, neg)
    pairs = []
    for i in neg_idx:
        for j in pos_idx:
            if j <= i: continue
            pairs.append((i,j))
    for i in pos_idx:
        for j in neg_idx:
            if j <= i: continue
            pairs.append((i,j))
    for i,j in pairs:
        u1, u2 = U[i], U[j]
        if require_cross_center and not (u1 < 0.0 < u2):
            continue
        d = abs(u2 - u1)
        if d < dmin or d > dmax:
            continue
        # valley penalty
        a, b = min(i,j), max(i,j)
        if b - a >= 2:
            valley = float(np.min(Ir[a:b+1]))
        else:
            valley = float(Ir[a])
        # edges approximate intensity values
        i1 = float(Ir[i]); i2 = float(Ir[j])
        edge_max = max(i1, i2); edge_min = min(i1, i2)
        if edge_max - edge_min < 1e-3:
            valley_pen = 1.0  # weak contrast
        else:
            # lower valley is better -> lower penalty
            valley_pen = (valley - edge_min) / (edge_max - edge_min + 1e-6)
        phi_local = - (abs(G[i]) + abs(G[j])) + 1.0 * valley_pen
        cands.append((phi_local, u1, u2, float(G[i]), float(G[j]), float(valley)))
    if not cands:
        return []
    cands.sort(key=lambda x: x[0])
    return cands[:K]


def dp_solve(cand_list_per_i, w_d=1.5, w_u=0.2, skip_pen=5.0):
    """Dynamic programming across arclength selecting one cand per index (or skip).
    cand_list_per_i: list length N, each is a list of tuples (phi, u1,u2,g1,g2,valley) in mm domain.
    Returns: chosen indices per i (k or -1 for skip).
    """
    N = len(cand_list_per_i)
    Kmax = max(len(c) for c in cand_list_per_i)
    if Kmax == 0:
        return [-1]*N
    # costs[i][k] for k in 0..Ki-1 ; plus skip as index Ki
    costs = [None]*N
    prev = [None]*N
    # init
    Ki = len(cand_list_per_i[0])
    costs0 = np.full(Ki+1, np.inf, dtype=np.float32)
    prev0 = np.full(Ki+1, -1, dtype=int)
    for k,(phi, u1,u2, *_rest) in enumerate(cand_list_per_i[0]):
        costs0[k] = phi
    costs0[Ki] = skip_pen
    costs[0] = costs0; prev[0] = prev0
    # iterate
    for i in range(1, N):
        Ki = len(cand_list_per_i[i])
        Kj = len(cand_list_per_i[i-1])
        c_new = np.full(Ki+1, np.inf, dtype=np.float32)
        p_new = np.full(Ki+1, -1, dtype=int)
        # transitions from real candidates j
        for j,(phi_j, u1j,u2j,*_) in enumerate(cand_list_per_i[i-1]):
            base = costs[i-1][j]
            if not np.isfinite(base):
                continue
            for k,(phi_i, u1i,u2i,*__) in enumerate(cand_list_per_i[i]):
                trans = w_d*abs((u2i-u1i) - (u2j-u1j)) + w_u*(abs(u1i-u1j)+abs(u2i-u2j))
                c = base + phi_i + trans
                if c < c_new[k]:
                    c_new[k] = c; p_new[k] = j
            # to skip
            c = base + skip_pen
            if c < c_new[Ki]:
                c_new[Ki] = c; p_new[Ki] = j
        # transitions from skip state
        base = costs[i-1][Kj] if costs[i-1].shape[0] > Kj else np.inf
        if np.isfinite(base):
            for k,(phi_i, u1i,u2i,*__) in enumerate(cand_list_per_i[i]):
                c = base + phi_i + skip_pen
                if c < c_new[k]:
                    c_new[k] = c; p_new[k] = Kj
            # skip->skip
            c = base + skip_pen
            if c < c_new[Ki]:
                c_new[Ki] = c; p_new[Ki] = Kj
        costs[i] = c_new; prev[i] = p_new
    # backtrack best
    last = costs[-1]
    k = int(np.argmin(last))
    sel = [k]
    for i in range(N-1, 0, -1):
        k = int(prev[i][k])
        sel.append(k)
    sel = sel[::-1]
    return sel


def frame_thickness_dp(img_gray, mask_bin, mmx, mmy, params):
    xs, y_center, valid = extract_centerline_from_mask(mask_bin)
    sel = select_roi_by_length(xs, y_center, valid, mmx, mmy, roi_len_mm=params['roi_len_mm'])
    if sel is None:
        return None
    # ratio of usable centerline points inside ROI
    n_sel = int(np.sum(sel))
    n_ok = int(np.sum(valid & sel))
    roi_valid_ratio = (n_ok / n_sel) if n_sel > 0 else 0.0
    out = {
        'roi_valid_ratio': float(roi_valid_ratio),
        'n_samples': n_sel,
    }
    idx, pts_px, normals_mm = normals_from_centerline(xs, y_center, sel, mmx, mmy)
    if idx is None:
        return None
    # profiles and candidates per arclength index
    cand_list = []
    chosen_debug = []
    for ii, (ix, pt, nmm) in enumerate(zip(idx, pts_px, normals_mm)):
        U, I = sample_profile(img_gray, pt, nmm, mmx, mmy,
                              u_min_mm=-params['pre_mm'], u_max_mm=params['post_mm'], du_mm=params['du_mm'])
        Is, G = gradient_profile(I, params['du_mm'], sigma_mm=params['grad_sigma_mm'])
        cands = profile_candidates(U, Is, G, K=params['K'], dmin=params['dmin_mm'], dmax=params['dmax_mm'], require_cross_center=True)
        cand_list.append(cands)
        chosen_debug.append((U, Is, G, cands))
    sel_idx = dp_solve(cand_list, w_d=params['w_d'], w_u=params['w_u'], skip_pen=params['skip_pen'])
    # build local thickness list and QC
    T_mm = []
    qgrad = []
    U1 = []; U2 = []
    for s, cands in zip(sel_idx, cand_list):
        if s == -1 or s >= len(cands):
            T_mm.append(np.nan); qgrad.append(np.nan); U1.append(np.nan); U2.append(np.nan)
        else:
            _, u1,u2,g1,g2,_ = cands[s]
            T_mm.append(abs(u2-u1))
            qgrad.append(abs(g1)+abs(g2))
            U1.append(u1); U2.append(u2)
    T_mm = np.array(T_mm, dtype=float)
    qgrad = np.array(qgrad, dtype=float)
    # robust smoothing along arclength
    if np.sum(~np.isnan(T_mm)) >= 7:
        T_mm_f = T_mm.copy()
        # fill NaNs by nearest before smoothing
        nn = np.where(~np.isnan(T_mm))[0]
        if nn.size > 0:
            last = nn[0]
            for i in range(len(T_mm)):
                if np.isnan(T_mm_f[i]):
                    T_mm_f[i] = T_mm_f[last]
                else:
                    last = i
            win = min(len(T_mm_f) - (len(T_mm_f)+1)%2, max(5, (len(T_mm_f)//5)*2+1))
            win = max(win, 5)
            if win % 2 == 0: win += 1
            T_mm_s = savgol_filter(T_mm_f, window_length=win, polyorder=2, mode='interp')
        else:
            T_mm_s = T_mm
    else:
        T_mm_s = T_mm
    # per-frame summary
    valid_mask = ~np.isnan(T_mm_s)
    if np.any(valid_mask):
        T_med = float(np.nanmedian(T_mm_s))
        T_p25 = float(np.nanpercentile(T_mm_s, 25))
        T_p75 = float(np.nanpercentile(T_mm_s, 75))
        grad_q = float(np.nanpercentile(qgrad[valid_mask], 25))
    else:
        T_med = T_p25 = T_p75 = grad_q = float('nan')
    out.update({
        'T_profile_mm': T_mm_s,
        'u1_mm': np.array(U1, dtype=float),
        'u2_mm': np.array(U2, dtype=float),
        'T_median_mm': T_med,
        'T_IQR_mm': T_p75 - T_p25 if (not math.isnan(T_p25) and not math.isnan(T_p75)) else float('nan'),
        'grad_q25': grad_q,
        'sel_idx': sel_idx,
        'cand_debug': chosen_debug,
        'center_idx': idx,
        'center_pts_px': pts_px,
    })
    return out


def detect_cycles(y_mm, fps=None, prom_mm=0.02, min_dist_frames=10):
    y = np.asarray(y_mm, dtype=float)
    if y.size == 0 or np.all(~np.isfinite(y)):
        return []
    ys = y.copy()
    # light smoothing
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


def overlay_debug_plot(out_dir: Path, frame_name: str, img_gray, center_pts_px, normals_mm, mmx, mmy, per_frame_out, max_samples=50):
    h, w = img_gray.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_gray, cmap='gray', vmin=np.percentile(img_gray, 1), vmax=np.percentile(img_gray, 99))
    # centerline
    if center_pts_px is not None and len(center_pts_px) > 0:
        ax.plot(center_pts_px[:,0], center_pts_px[:,1], 'y-', lw=1, alpha=0.8, label='centerline')
    # normals subset
    if center_pts_px is not None and normals_mm is not None:
        step = max(1, len(center_pts_px)//max_samples)
        for i in range(0, len(center_pts_px), step):
            pt = center_pts_px[i]
            nmm = normals_mm[i]
            # draw short normal segment (±2 mm)
            du = np.linspace(-2, 2, 5)
            dx_per_mm = nmm[0]/mmx; dy_per_mm = nmm[1]/mmy
            xs = pt[0] + du*dx_per_mm
            ys = pt[1] + du*dy_per_mm
            ax.plot(xs, ys, 'c-', lw=0.5, alpha=0.6)
    # chosen boundaries markers
    if 'u1_mm' in per_frame_out:
        u1 = per_frame_out['u1_mm']; u2 = per_frame_out['u2_mm']
        idx = per_frame_out['center_idx']
        if idx is not None and normals_mm is not None:
            for i, (ix, u1i, u2i) in enumerate(zip(idx, u1, u2)):
                if not np.isfinite(u1i) or not np.isfinite(u2i):
                    continue
                pt = center_pts_px[i]
                nmm = normals_mm[i]
                dx_per_mm = nmm[0]/mmx; dy_per_mm = nmm[1]/mmy
                x1 = pt[0] + u1i*dx_per_mm; y1 = pt[1] + u1i*dy_per_mm
                x2 = pt[0] + u2i*dx_per_mm; y2 = pt[1] + u2i*dy_per_mm
                ax.plot([x1],[y1],'r.', ms=2)
                ax.plot([x2],[y2],'g.', ms=2)
    ax.set_title(f"Debug overlay: {frame_name}")
    ax.set_xlim([0, w]); ax.set_ylim([h, 0])
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"debug_{frame_name}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-dir', required=True, help='Directory with frame_*.png (B-mode frames)')
    ap.add_argument('--mask-dir', required=True, help='Directory with frame_*.png masks (0/255)')
    ap.add_argument('--dicom-path', default=None, help='DICOM path to read calibration (mm/px)')
    ap.add_argument('--mm-x', type=float, default=None, help='Manual mm/px in X (if DICOM missing)')
    ap.add_argument('--mm-y', type=float, default=None, help='Manual mm/px in Y (if DICOM missing)')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    # ROI and sampling params
    ap.add_argument('--roi-len-mm', type=float, default=15.0)
    ap.add_argument('--pre-mm', type=float, default=4.0, help='Profile extends this far to negative u (towards one side)')
    ap.add_argument('--post-mm', type=float, default=6.0, help='Profile extends this far to positive u')
    ap.add_argument('--du-mm', type=float, default=0.075)
    ap.add_argument('--grad-sigma-mm', type=float, default=0.25)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--dmin-mm', type=float, default=0.8)
    ap.add_argument('--dmax-mm', type=float, default=6.0)
    ap.add_argument('--w-d', type=float, default=1.5)
    ap.add_argument('--w-u', type=float, default=0.2)
    ap.add_argument('--skip-pen', type=float, default=5.0)
    # Time series / cycles
    ap.add_argument('--cycle-prom-mm', type=float, default=0.03)
    ap.add_argument('--cycle-min-distance', type=int, default=6)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mmx = args.mm_x; mmy = args.mm_y; info = {}
    if args.dicom_path:
        mmx_d, mmy_d, info = read_dicom_mm(Path(args.dicom_path))
        mmx = mmx or mmx_d; mmy = mmy or mmy_d
    if mmx is None or mmy is None:
        raise SystemExit('Calibration missing: provide --dicom-path or both --mm-x/--mm-y')

    params = {
        'roi_len_mm': args.roi_len_mm,
        'pre_mm': args.pre_mm,
        'post_mm': args.post_mm,
        'du_mm': args.du_mm,
        'grad_sigma_mm': args.grad_sigma_mm,
        'K': args.K,
        'dmin_mm': args.dmin_mm,
        'dmax_mm': args.dmax_mm,
        'w_d': args.w_d,
        'w_u': args.w_u,
        'skip_pen': args.skip_pen,
    }

    # match frames and masks by filename
    frame_paths = sorted(frames_dir.glob('frame_*.png'))
    mask_paths = sorted(masks_dir.glob('frame_*.png'))
    names = sorted(set(p.name for p in frame_paths) & set(p.name for p in mask_paths))
    if not names:
        raise SystemExit('No matching frame_*.png found in frames-dir and mask-dir')

    rows = []
    per_frame_T = []
    for nm in names:
        fpath = frames_dir / nm
        mpath = masks_dir / nm
        img = np.array(Image.open(fpath))
        img_gray = to_gray(img)
        mask = np.array(Image.open(mpath))
        mask_bin = mask > 0
        out = frame_thickness_dp(img_gray, mask_bin, mmx, mmy, params)
        if out is None:
            rows.append({'frame_name': nm, 'T_median_mm': '', 'T_IQR_mm': '', 'roi_valid_ratio': 0.0, 'grad_q25': ''})
            per_frame_T.append(np.nan)
            continue
        rows.append({
            'frame_name': nm,
            'T_median_mm': out['T_median_mm'],
            'T_IQR_mm': out['T_IQR_mm'],
            'roi_valid_ratio': out['roi_valid_ratio'],
            'grad_q25': out['grad_q25'],
        })
        per_frame_T.append(out['T_median_mm'])
        # Sparse debug overlay for first, middle, last
        if nm in {names[0], names[len(names)//2], names[-1]}:
            # recompute center/normals for plotting
            xs, y_center, valid = extract_centerline_from_mask(mask_bin)
            sel = select_roi_by_length(xs, y_center, valid, mmx, mmy, roi_len_mm=params['roi_len_mm'])
            if sel is not None:
                idx, pts_px, normals_mm = normals_from_centerline(xs, y_center, sel, mmx, mmy)
                overlay_debug_plot(out_dir, nm, img_gray, pts_px, normals_mm, mmx, mmy, out)

    # save CSV
    csv_path = out_dir / 'thickness_dp_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # time series and cycles
    y = np.array(per_frame_T, dtype=float)
    fig = plt.figure(figsize=(10,4))
    plt.plot(y, 'b-', lw=1.5, label='T_median_mm (DP)')
    plt.xlabel('Frame'); plt.ylabel('Thickness (mm)'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'thickness_dp_timeseries.png', dpi=160)
    plt.close(fig)

    cycles = detect_cycles(y, prom_mm=args.cycle_prom_mm, min_dist_frames=args.cycle_min_distance)
    if cycles:
        cyc_csv = out_dir / 'cycle_metrics_dp_mm.csv'
        with open(cyc_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['cycle_index','start_min_frame','peak_frame','end_min_frame','Tmin_mm','Tmax_mm','TFdi','duration_frames'])
            writer.writeheader()
            for i, c in enumerate(cycles, start=1):
                writer.writerow({'cycle_index': i, **c})
        tfs = np.array([c['TFdi'] for c in cycles if np.isfinite(c['TFdi'])], dtype=float)
        tf_mean = float(np.mean(tfs)) if tfs.size else float('nan')
        tf_std = float(np.std(tfs)) if tfs.size else float('nan')
        with open(out_dir / 'summary_dp.txt', 'w', encoding='utf-8') as f:
            if info:
                f.write('Calibration from DICOM (US Region / PixelSpacing)\n')
                for k,v in info.items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
            f.write(f'mm_per_px: X={mmx}, Y={mmy}\n')
            f.write(f'Frames processed: {len(names)}\n')
            f.write(f'Cycles: {len(cycles)}\n')
            f.write(f'TFdi mean: {tf_mean}\n')
            f.write(f'TFdi std: {tf_std}\n')
    else:
        with open(out_dir / 'summary_dp.txt', 'w', encoding='utf-8') as f:
            if info:
                f.write('Calibration from DICOM (US Region / PixelSpacing)\n')
                for k,v in info.items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
            f.write(f'mm_per_px: X={mmx}, Y={mmy}\n')
            f.write(f'Frames processed: {len(names)}\n')
            f.write('Cycles: 0\n')

    print(f'Saved: {csv_path}')
    print(f'Saved: {out_dir / "thickness_dp_timeseries.png"}')
    print(f'Saved: {out_dir / "summary_dp.txt"}')


if __name__ == '__main__':
    main()
