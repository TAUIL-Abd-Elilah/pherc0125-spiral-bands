"""Mask rule (written 18 Nov before any masked score): node kept iff
 (a) inside the scroll: low-res CT (ct.zarr level 4, 16 vx voxels) > 0 at all 27 L4 voxels around the node, and
 (b) on a sheet: m7 > 0 at the node or at +-1, +-2, +-3 vx along the mesh normal, and
 (c) it belongs to a 4-connected kept component of >= 100 nodes in its winding's grid.
Output: masked tifxyz copies (dropped nodes = -1) in e5/release/meshes/<tag>/, mask_report.json."""
import os, json, shutil, numpy as np, tifffile, zarr, collections
from scipy.ndimage import label, minimum_filter
U = 'D:/Unroll0125'; O = f'{U}/e5/release/meshes'
BANDS = {6: 's6m20cw', 7: 's7ancm', 8: 'b08cww125a'}
L4 = np.asarray(zarr.open(f'{U}/ct.zarr/4', mode='r')[:]); IN4 = minimum_filter(L4, size=3) > 0
M7 = zarr.open(f'{U}/m7.zarr', mode='r')['0']; CH = 192
rep = {}
for b, tag in BANDS.items():
    ws = [w for w in range(10, 150) if os.path.exists(f'{U}/run_{tag}/snap/w{w:03d}_{tag}/x.tif')]
    meshes = {}; req = collections.defaultdict(list)
    for w in ws:
        d = f'{U}/run_{tag}/snap/w{w:03d}_{tag}'
        P = np.stack([tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'], -1)
        v = (P > -0.5).all(-1)
        tu = np.gradient(P, axis=1); tv = np.gradient(P, axis=0); n = np.cross(tu, tv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        iz = np.clip((P[..., 2] / 16).astype(int), 0, L4.shape[0] - 1); iy = np.clip((P[..., 1] / 16).astype(int), 0, L4.shape[1] - 1); ix = np.clip((P[..., 0] / 16).astype(int), 0, L4.shape[2] - 1)
        ins = v & IN4[iz, iy, ix]
        meshes[w] = (P, v, ins, np.zeros(v.shape, bool))
        rr, cc = np.nonzero(ins)
        for k in (0, 1, -1, 2, -2, 3, -3):
            q = np.rint(P[rr, cc] + k * n[rr, cc]).astype(int)
            key = q // CH
            for j, (kz, ky, kx) in enumerate(map(tuple, key[:, ::-1])):
                req[(kz, ky, kx)].append((w, rr[j], cc[j], q[j, 2] % CH, q[j, 1] % CH, q[j, 0] % CH))
    print(f'band {b}: {len(req)} m7 chunks', flush=True)
    for (kz, ky, kx), lst in req.items():
        blk = np.asarray(M7[kz * CH:(kz + 1) * CH, ky * CH:(ky + 1) * CH, kx * CH:(kx + 1) * CH])
        a = np.array([t[3:] for t in lst]); a = np.minimum(a, np.array(blk.shape) - 1)
        hit = blk[a[:, 0], a[:, 1], a[:, 2]] > 0
        for t, h in zip(lst, hit):
            if h: meshes[t[0]][3][t[1], t[2]] = True
    tot = dict(nodes=0, inside=0, onsheet=0, kept=0, windings_kept=0)
    for w, (P, v, ins, on) in meshes.items():
        k = ins & on; lab, nl = label(k)
        if nl:
            sz = np.bincount(lab.ravel()); good = sz >= 100; good[0] = False; k = good[lab]
        tot['nodes'] += int(v.sum()); tot['inside'] += int(ins.sum()); tot['onsheet'] += int((ins & on).sum()); tot['kept'] += int(k.sum())
        if k.sum() == 0: continue
        tot['windings_kept'] += 1
        dst = f'{O}/{tag}/w{w:03d}_{tag}'; os.makedirs(dst, exist_ok=True)
        for i, c in enumerate('xyz'):
            a = P[..., i].astype(np.float32).copy(); a[~k] = -1; tifffile.imwrite(f'{dst}/{c}.tif', a)
        m = json.load(open(f'{U}/run_{tag}/snap/w{w:03d}_{tag}/meta.json')); m['e5_mask'] = 'e5/mask.py rule (a)(b)(c), 18 Nov 2026'; m['area_cm2_masked'] = round(float(k.sum()) * 20 * 20 * (9.362e-4) ** 2, 4)
        json.dump(m, open(f'{dst}/meta.json', 'w'), indent=1)
    tot['area_cm2_kept'] = round(tot['kept'] * 400 * (9.362e-4) ** 2, 1); rep[tag] = tot; print(b, tag, tot, flush=True)
json.dump(rep, open(f'{U}/e5/mask_report.json', 'w'), indent=1)
