"""Sheets-on-a-ray count in the style of lattice-unroll's README (same idea, our own code; not their script):
at a slice z, 72 rays from the scroll axis (umbilicus.json); sheets = runs of m7 > 0 (>= 2 px) along the ray; each winding
polyline crossing the ray is assigned to the sheet run within 2 px. Per ray: sheets reached (between the innermost and the
outermost crossing), sheets with exactly one surface, with two or more, with none. Mean over rays. Raw snapped and masked sets."""
import os, sys, json, numpy as np, tifffile, zarr
sys.path.insert(0, 'D:/Unroll0125/gates'); from lib import slice_at_z, umb
U = 'D:/Unroll0125'; M7 = zarr.open(f'{U}/m7.zarr', mode='r')['0']
SETS = {'snapped': lambda tag: f'{U}/run_{tag}/snap', 'masked': lambda tag: f'{U}/e5/release/meshes/{tag}'}
SL = [(10800, 's6m20cw'), (11525, 's6m20cw'), (11525, 's7ancm'), (11800, 's7ancm'), (12525, 's7ancm'), (12525, 'b08cww125a'), (12800, 'b08cww125a'), (13300, 'b08cww125a')]
def polys(base, tag, z):
    out = []
    for w in range(10, 150):
        d = f'{base}/w{w:03d}_{tag}'
        if not os.path.exists(f'{d}/x.tif'): continue
        P = np.stack([tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'], -1); P[~((P > -0.5).all(-1))] = np.nan
        L = slice_at_z(P, z)[:, :2]; out.append((w, L))
    return out
res = {}
for z, tag in SL:
    m7 = np.asarray(M7[z]) > 0; ux, uy = umb(z)
    for name, base in SETS.items():
        PL = polys(base(tag), tag, z); stats = []
        for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            d = np.array([np.cos(a), np.sin(a)]); t = np.arange(0, 4200.0)
            xs = ux + t * d[0]; ys = uy + t * d[1]; ok = (xs >= 0) & (ys >= 0) & (xs < m7.shape[1]) & (ys < m7.shape[0])
            prof = np.zeros(len(t), bool); prof[ok] = m7[ys[ok].astype(int), xs[ok].astype(int)]
            e = np.diff(np.r_[0, prof.astype(int), 0]); st = np.nonzero(e == 1)[0]; en = np.nonzero(e == -1)[0]
            runs = [(s, f) for s, f in zip(st, en) if f - s >= 2]
            hits = []
            for w, L in PL:
                A = L[:-1]; B = L[1:]; good = np.isfinite(A).all(1) & np.isfinite(B).all(1) & (np.hypot(*(B - A).T) < 60)
                A = A[good] - [ux, uy]; B = B[good] - [ux, uy]
                # intersection of segment A->B with ray s*d, s >= 0
                den = (B - A) @ np.array([-d[1], d[0]]); num = A @ np.array([-d[1], d[0]])
                with np.errstate(divide='ignore', invalid='ignore'): u = -num / den
                m = (u >= 0) & (u <= 1) & (den != 0)
                pt = A[m] + u[m, None] * (B - A)[m]; s = pt @ d; s = s[s > 0]
                hits += [(si, w) for si in s]
            if not hits: continue
            lo = min(h[0] for h in hits); hi = max(h[0] for h in hits)
            cnt = {}
            for si, w in hits:
                for k, (rs, rf) in enumerate(runs):
                    if rs - 2 <= si <= rf + 2: cnt.setdefault(k, set()).add(w); break
            reached = [k for k, (rs, rf) in enumerate(runs) if rf >= lo - 2 and rs <= hi + 2]
            one = sum(1 for k in reached if len(cnt.get(k, ())) == 1); two = sum(1 for k in reached if len(cnt.get(k, ())) >= 2)
            stats.append((len(reached), one, two, len(reached) - one - two))
        a = np.array(stats).mean(0); res[f'{name} {tag} z{z}'] = dict(reached=round(a[0], 1), one=round(a[1], 1), two_plus=round(a[2], 1), none=round(a[3], 1), rays=len(stats))
        print(f'{name:8s} {tag:11s} z{z}: reached {a[0]:.1f}, exactly one {a[1]:.1f}, two+ {a[2]:.1f}, none {a[3]:.1f} ({len(stats)} rays)', flush=True)
json.dump(res, open(f'{U}/e5/rays.json', 'w'), indent=1)
