"""E5 proof 2: flattened CT on the snapped surface (no ink model). 30 sites (seed 2118, 10 per band) drawn before rendering:
random winding, random grid cell with valid nodes. Crop = 14x14 grid cells (20 vx each) upsampled to 1 px per voxel (~280 px),
CT sampled at the surface (mean of offsets -1..+1 along the normal) and, as control, 9 vx off the surface (gap side).
Statistic: structure-tensor coherence (directional texture), real vs control, plus mean CT brightness (on papyrus vs air)."""
import os, sys, json, numpy as np, tifffile, cv2
os.environ['ROUTE24_CACHE'] = 'D:/Unroll0125/e5/ctcache'; os.environ['ROUTE24_CAP_GB'] = '6'
sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/_fl/route24'); import zfetch
from scipy.ndimage import map_coordinates
U = 'D:/Unroll0125'; O = f'{U}/e5/proof_fibres'
VOL = 'PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr'
BANDS = {6: 's6m20cw', 7: 's7ancm', 8: 'b08cww125a'}
rng = np.random.default_rng(2118); G = 14
def mesh(tag, w):
    d = f'{U}/run_{tag}/snap/w{w:03d}_{tag}'
    if not os.path.exists(f'{d}/x.tif'): return None
    P = np.stack([tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'], -1); P[~((P > -0.5).all(-1))] = np.nan; return P
sites = []
for b, tag in BANDS.items():
    ws = [w for w in range(10, 150) if os.path.exists(f'{U}/run_{tag}/snap/w{w:03d}_{tag}/x.tif')]
    while sum(s['band'] == b for s in sites) < 10:
        w = int(rng.choice(ws)); P = mesh(tag, w); H, W, _ = P.shape
        if H <= G or W <= G: continue
        r, c = int(rng.integers(0, H - G)), int(rng.integers(0, W - G))
        if np.isfinite(P[r:r + G, c:c + G]).all(): sites.append(dict(band=b, tag=tag, w=w, r=r, c=c))
json.dump(sites, open(f'{O}/sites.json', 'w'), indent=1)
def coh(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    J = [cv2.GaussianBlur(a, (0, 0), 4) for a in (gx * gx, gy * gy, gx * gy)]
    l = np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2); return float(np.median(l / (J[0] + J[1] + 1e-6)))
rows = []; thumbs = []
for i, s in enumerate(sites):
    P = mesh(s['tag'], s['w'])[s['r']:s['r'] + G, s['c']:s['c'] + G]
    S = (G - 1) * 20; up = cv2.resize(P.astype(np.float32), (S, S), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    tu = np.gradient(up, axis=1); tv = np.gradient(up, axis=0); n = np.cross(tu, tv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    lo = np.floor(np.nanmin(up.reshape(-1, 3), 0) - 12).astype(int); hi = np.ceil(np.nanmax(up.reshape(-1, 3), 0) + 12).astype(int)
    box = zfetch.read_box(VOL, 0, (lo[2], lo[1], lo[0]), (hi[2], hi[1], hi[0])).astype(np.float32)
    def samp(off):
        acc = 0
        for d in (-1, 0, 1):
            q = up + (off + d) * n
            acc = acc + map_coordinates(box, [q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]], order=1)
        return acc / 3
    real = samp(0); ctrl = samp(9)
    r8 = np.clip(real, 0, 255).astype(np.uint8); c8 = np.clip(ctrl, 0, 255).astype(np.uint8)
    rec = dict(i=i, band=s['band'], w=s['w'], r=s['r'], c=s['c'], bright=round(float(real.mean()), 1), bright_ctrl=round(float(ctrl.mean()), 1),
               coherence=round(coh(r8), 3), coherence_ctrl=round(coh(c8), 3))
    rows.append(rec); print(rec, flush=True)
    vis = np.concatenate([cv2.equalizeHist(r8), cv2.equalizeHist(c8)], 1); vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis, f"{i} b{s['band']} w{s['w']}  surface | 9 vx off", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(f'{O}/crop_{i:02d}_b{s["band"]}_w{s["w"]:03d}.png', vis); thumbs.append(cv2.resize(cv2.equalizeHist(r8), (200, 200)))
json.dump(rows, open(f'{O}/fibre_results.json', 'w'), indent=1)
g = [np.concatenate(thumbs[k:k + 6], 1) for k in range(0, 30, 6)]; cv2.imwrite(f'{O}/contact_sheet.png', np.concatenate(g, 0))
a = np.array([[r['bright'], r['bright_ctrl'], r['coherence'], r['coherence_ctrl']] for r in rows])
print('MEAN bright %.1f ctrl %.1f | coherence %.3f ctrl %.3f' % tuple(a.mean(0)))
for b in (6, 7, 8):
    m = np.array([r['band'] == b for r in rows]); print(b, 'bright %.1f/%.1f coh %.3f/%.3f' % tuple(a[m].mean(0)))
