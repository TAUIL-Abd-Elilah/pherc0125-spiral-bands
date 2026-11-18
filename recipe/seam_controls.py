"""Controls for the join/chain seam scores (read-only on fits). A = earlier band's windings (odd = held out), B = later band's
fitted windings (cloud). Control: A's nodes moved along A's own surface normal by d voxels before scoring. If the seam score is
real geometry, same-index-and-on-surface should collapse when A is moved off its sheet by ~half a sheet spacing (10 vx at 20 px)."""
import sys, glob, json, numpy as np, collections
sys.path.insert(0, 'D:/Unroll0125/gates'); from seam_lib import load, cloud, empties
from scipy.spatial import cKDTree
U = 'D:/Unroll0125'
def md(tag): return glob.glob(f'{U}/out/*_{tag}/meshes/fitted_{tag}')[0]
def nodes(tag, run, z0, z1, only):
    e = empties(run); out = []
    for w in range(10, 150):
        if w % 2 != only: continue
        n = f'w{w:03d}_{tag}'; d = f'{md(tag)}/{n}'
        if n in e or not glob.os.path.exists(f'{d}/x.tif'): continue
        P, v = load(d)
        tu = np.gradient(P, axis=1); tv = np.gradient(P, axis=0); nn = np.cross(tu, tv)
        ln = np.linalg.norm(nn, axis=-1, keepdims=True); nn = nn / np.where(ln > 1e-6, ln, 1)
        m = v & (P[..., 2] >= z0) & (P[..., 2] < z1) & (ln[..., 0] > 1e-6)
        out.append((w, P[m], nn[m]))
    return out
res = {}
for A, B, z0, z1 in [('s6m20cw', 's7ancm', 11500, 11550), ('s7ancm', 'b08cww125a', 12500, 12550), ('s6m20cw', 'b05cww125a', 10500, 10550)]:
    pts, nrm, lab = cloud(md(B), B, empties(f'run_{B}'), z0, z1); tree = cKDTree(pts)
    for only in (1, 0):
        NA = nodes(A, f'run_{A}', z0, z1, only)
        for d in (0, 3, -3, 6, -6, 10, -10):
            same = []; perp = []
            for w, q, n in NA:
                qq = q + d * n; dd, i = tree.query(qq, k=1, workers=4)
                same.append(lab[i] == w); perp.append(np.abs(((qq - pts[i]) * nrm[i]).sum(1)))
            s = np.concatenate(same); p = np.concatenate(perp)
            r = dict(same=round(float(s.mean()), 3), perp_med=round(float(np.median(p)), 2), same_and_perp2=round(float((s & (p <= 2)).mean()), 3), n=int(len(s)))
            res[f'{A}->{B} {"odd" if only else "even"} shift {d:+d}'] = r; print(f'{A}->{B} {"odd " if only else "even"} shift {d:+3d}: {r}', flush=True)
json.dump(res, open(f'{U}/e5/validity/seam_controls.json', 'w'), indent=1)
