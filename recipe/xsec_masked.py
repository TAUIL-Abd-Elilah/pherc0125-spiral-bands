"""E5 proof 1: cross-sections of the joined band 6-7-8 snapped windings over full-res CT, 30 sites drawn (seed 1118) before
any rendering. Per site: CT crop with every winding's polyline at that z, coloured by winding number. Measures, real vs control
(the same polylines moved 10 px radially = half a sheet spacing): share of polyline points on a predicted sheet (m7 > 0),
and share of points with another winding within 3 px (doubled sheets)."""
import os, sys, json, numpy as np, tifffile, cv2
os.environ['ROUTE24_CACHE'] = 'D:/Unroll0125/e5/ctcache'; os.environ['ROUTE24_CAP_GB'] = '6'
sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/_fl/route24'); sys.path.insert(0, 'D:/Unroll0125/gates')
import zfetch, zarr
from lib import slice_at_z, umb
from scipy.spatial import cKDTree
U = 'D:/Unroll0125'; O = f'{U}/e5/proof_xsec_masked'
VOL = 'PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr'
BANDS = {6: ('s6m20cw', 10500, 11550), 7: ('s7ancm', 11500, 12550), 8: ('b08cww125a', 12500, 13550)}
M7 = zarr.open(f'{U}/m7.zarr', mode='r')['0']
rng = np.random.default_rng(1118)
zs = list(rng.uniform(10560, 13490, 22).astype(int)) + list(rng.uniform(11500, 11550, 4).astype(int)) + list(rng.uniform(12500, 12550, 4).astype(int))
sites = [dict(i=i, z=int(z), pick=float(rng.uniform())) for i, z in enumerate(zs)]
json.dump(sites, open(f'{O}/sites.json', 'w'), indent=1)
def mesh(tag, w):
    d = f'{U}/e5/release/meshes/{tag}/w{w:03d}_{tag}'
    if not os.path.exists(f'{d}/x.tif'): return None
    P = np.stack([tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'], -1)
    P[~((P > -0.5).all(-1))] = np.nan; return P
cache = {}
def polys(z):
    out = {}
    for b, (tag, z0, z1) in BANDS.items():
        if not (z0 <= z < z1): continue
        for w in range(10, 150):
            k = (tag, w)
            if k not in cache: cache[k] = mesh(tag, w)
            P = cache[k]
            if P is None: continue
            L = slice_at_z(P, z); ok = np.isfinite(L[:, 0])
            if ok.sum() >= 2: out[(b, w)] = L[ok][:, :2]
    return out
rows = []; thumbs = []
cmap = lambda w: tuple(int(c) for c in cv2.applyColorMap(np.uint8([[ (w * 37) % 255 ]]), cv2.COLORMAP_HSV)[0, 0])
for s in sites:
    z = s['z']; PL = polys(z)
    allp = np.concatenate([L for L in PL.values()]); c = allp[int(s['pick'] * len(allp))]
    H = 512; x0 = int(c[0]) - H // 2; y0 = int(c[1]) - H // 2
    ct = zfetch.read_box(VOL, 0, (z, y0, x0), (z + 1, y0 + H, x0 + H))[0]
    m7 = np.asarray(M7[z, y0:y0 + H, x0:x0 + H]) > 0
    ux, uy = umb(z)
    img = cv2.cvtColor(cv2.equalizeHist(ct), cv2.COLOR_GRAY2BGR); raw = img.copy()
    pts_all = []; lab_all = []
    for (b, w), L in PL.items():
        m = (L[:, 0] >= x0) & (L[:, 0] < x0 + H) & (L[:, 1] >= y0) & (L[:, 1] < y0 + H)
        if m.sum() < 2: continue
        # densify
        Q = L[m]; seg = np.hypot(*np.diff(Q, axis=0).T); keep = np.r_[True, seg < 60]
        dense = []
        for j in range(len(Q) - 1):
            if seg[j] >= 60: continue
            n = max(1, int(seg[j] / 1.5)); t = np.arange(n)[:, None] / n; dense.append(Q[j] * (1 - t) + Q[j + 1] * t)
        if not dense: continue
        D = np.concatenate(dense); pts_all.append(D); lab_all.append(np.full(len(D), b * 1000 + w))
        cv2.polylines(img, [np.int32(D - [x0, y0])], False, cmap(w), 1)
    if not pts_all: continue
    P = np.concatenate(pts_all); Lb = np.concatenate(lab_all)
    def onsheet(Q):
        ix = np.clip(np.int32(Q[:, 0] - x0), 0, H - 1); iy = np.clip(np.int32(Q[:, 1] - y0), 0, H - 1); return m7[iy, ix]
    rad = np.stack([P[:, 0] - ux, P[:, 1] - uy], 1); rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
    real = onsheet(P).mean(); ctrl = 0.5 * (onsheet(P + 10 * rad).mean() + onsheet(P - 10 * rad).mean())
    tr = cKDTree(P); dd, ii = tr.query(P, k=8, distance_upper_bound=3.0)
    dbl = np.mean([(np.isfinite(dd[j]) & (Lb[np.minimum(ii[j], len(P) - 1)] % 1000 != Lb[j] % 1000)).any() for j in range(len(P))])
    seam = any(z0 <= z < z1 for _, z0, z1 in [(0, 11500, 11550), (0, 12500, 12550)])
    r = dict(i=s['i'], z=z, x0=x0, y0=y0, windings=len(set(Lb % 1000)), on_sheet=round(float(real), 3), on_sheet_control=round(float(ctrl), 3), doubled=round(float(dbl), 3), seam=seam)
    rows.append(r); print(r, flush=True)
    both = np.concatenate([raw, img], 1)
    cv2.putText(both, f"site {s['i']} z {z} on-sheet {real:.2f} (ctrl {ctrl:.2f}) doubled {dbl:.2f}{' SEAM' if seam else ''}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(f"{O}/site_{s['i']:02d}_z{z}.png", both); thumbs.append(cv2.resize(img, (256, 256)))
json.dump(rows, open(f'{O}/xsec_results.json', 'w'), indent=1)
g = [np.concatenate(thumbs[k:k + 6] + [np.zeros_like(thumbs[0])] * (6 - len(thumbs[k:k + 6])), 1) for k in range(0, len(thumbs), 6)]
cv2.imwrite(f'{O}/contact_sheet.png', np.concatenate(g, 0))
a = np.array([[r['on_sheet'], r['on_sheet_control'], r['doubled']] for r in rows])
print('MEAN on_sheet %.3f control %.3f doubled %.3f over %d sites' % (*a.mean(0), len(rows)))
