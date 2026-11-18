# PHerc0125 bands 6-8: spiral-fitted sheet surfaces with one winding numbering

Surfaces of PHerc0125 (volume `20250821151825`, 9.362 µm, 113 keV) from z 10500 to 13550, fitted in three
overlapping bands that share one winding numbering. They are masked to papyrus by a rule fixed before scoring.
This is **not** a full unroll and has **no ink**. The spiral **sense is unverified** (see Limits).

| Band | z range | Run tag | Windings kept | Nodes kept | Area kept |
|---|---|---|---|---|---|
| 6 | 10500-11550 | `s6m20cw` | 99 | 1,125,349 of 2,504,043 | 394.5 cm² |
| 7 | 11500-12550 | `s7ancm` | 93 | 1,103,270 of 2,466,515 | 386.8 cm² |
| 8 | 12500-13550 | `b08cww125a` | 93 | 1,122,403 of 2,462,606 | 393.5 cm² |

The total is about 1,175 cm², counting the two 50-slice overlaps twice. About 45 % of the fitted surface is kept.

## What is in this repository

- `meshes/<tag>/w###_<tag>/{x,y,z}.tif` + `meta.json`: one tifxyz per winding, with full-resolution voxel coordinates
  (scale 0.05). Dropped nodes are `-1`. Winding `w###` has the same index across the three bands.
- `proofs/`: contact sheets and per-site JSON for the checks below.
- `results/`: `seam_controls.json`, `rays.json`, `mask_report.json`.
- `recipe/`: the scripts as run. They contain local paths: edit `U` and the cache paths before you run them.
  - `villa_0fb38c45_local.diff`: the fitter changes, about 20 lines.
  - `mask.py`, `seam_controls.py`, `xsec.py`, `xsec_masked.py`, `fibres.py`, `rays.py`.

## How it was made

1. **Fitter.** villa `volume-cartographer/scripts/spiral/fit_spiral.py` at commit `0fb38c45`, plus
   `recipe/villa_0fb38c45_local.diff`. The diff makes the following changes, and none of them alters the model:
   - the outer shell term is optional;
   - a periodic checkpoint is written (`FIT_SPIRAL_CKPT_EVERY`);
   - a GPU memory fraction can be set;
   - the DBM signature uses name and size;
   - a `TrackGraph` csr `isinstance` fix.
2. **Inputs.** All are public Vesuvius Challenge data for PHerc0125:
   - tracks;
   - the lasagna normals `20260419180421`;
   - the surface prediction `surface-20260413222639-surface-m7-L0-th0.2` ("m7");
   - its signed distance transform, built with villa's `make_surf_sdt.py`.
3. **Config.** The overrides on top of the villa defaults are:
   - shell loss weights 0;
   - `dense_spacing_mode: phase`;
   - 20,000 steps;
   - `model_initial_dr_per_winding: 20.0`;
   - sense CW.
4. **Pitch.** 20.0 px was chosen from three arms (16.0, 20.0 and 23.4) by a rule set before the runs. Fitted windings per
   m7 sheet came out at about 1.00 in all planes, against 1.20 at 16 px and 0.84-0.88 at 23.4. The pitch that matches m7
   falls with radius: about 23.5 px inside, 20 px in the middle and 17 px outside. One global value therefore cannot match
   everywhere.
5. **Joins.** Band 7 starts from band 6's checkpoint and is pinned to band 6's **even** windings in the overlap. The
   **odd** windings are held out and used to score the join. Band 8 is joined to band 7 in the same way.
6. **Snap.** Each node is moved along its normal to the nearest m7 sheet: at most 15 vx, regularised, with median offset
   about 3 vx.
7. **Mask.** The rule was fixed before any masked scoring. A node is kept if and only if all three of these hold:
   - (a) the CT level-4 value is > 0 at all 27 neighbours, so the node is inside the scroll;
   - (b) m7 > 0 within ±3 vx along the normal, so the node is on a predicted sheet;
   - (c) its 4-connected component has at least 100 nodes.

## Checks, each with a control

**Seams (held-out windings).** Each held-out odd winding of the new band is compared with the same-index winding of
the previous band.

| Join | Same index | Perp. median (vx) | Pinned even | Satisfied tracks |
|---|---|---|---|---|
| 6→7 (`s7ancm`) | 0.913 | 1.43 | 0.929 | 55.19 % |
| 7→8 (`b08cww125a`) | 0.908 | 1.63 | 0.926 | 55.78 % |

The pass bar was fixed before the runs: odd ≥ 0.80 with median < 2 vx, even ≥ 0.90, and satisfied ≥ 55 %.

Control: the same score with the surface shifted along its normal (`results/seam_controls.json`).

| Normal shift | Same index within 2 vx |
|---|---|
| none (real) | 0.55-0.69 |
| ±6 vx | 0.05-0.10 |
| ±10 vx | 0.02-0.04 |

So the joins follow real geometry. This is agreement between two fits, not proof against the CT.

**Cross-sections against CT (30 seeded sites, `proofs/xsec_*`).** Each site is a full-resolution CT slice with every
winding drawn on it.
- **On-sheet** is the share of polyline points on m7 papyrus.
- The **control** is the mean of the same polyline shifted 10 px radially outward and 10 px inward.
- **Doubled** is the share of points that have another winding within 3 px.

| Set | On-sheet | Control | Doubled |
|---|---|---|---|
| Raw snapped (27 sites with m7) | 0.403 | 0.195 | 0.134 |
| Masked (30 sites) | **0.732** | 0.147 | 0.202 |

Before masking, about 9 of 30 sites drew windings through empty space. After masking, every site shows windings over
papyrus only. The doubled figure for the masked set counts both bands at seam sites, which raises it.

**Fibres (30 seeded flattened crops, `proofs/fibres/`).** Each crop is a 2-D render of the surface beside the same patch
9 vx off.
- 20/30 crops show the papyrus fibre grid clearly. 6 are outside the scroll (raw set) and 4 are noisy.
- The surface is brighter than the 9-vx-off render in 22 of 24.
- Orientation coherence is 0.449 on the surface, against 0.432 off it.

**Sheets per ray (`results/rays.json`).** Each z has 72 rays from the umbilicus, and the number of distinct windings
crossing each papyrus layer is counted. This is the measure used by lattice-unroll.

| Set | Reached | Exactly one | Two or more | None |
|---|---|---|---|---|
| Snapped | 71-78 | 39-44 | 10-12 | 20-23 |
| Masked | 67-74 | 31-37 | 6.8-8.5 | 26-30 |
| lattice-unroll README, PHerc0175A (other scroll) | 114/117 | 96/93 | 3.4/3.3 | – |

lattice-unroll has not been run on PHerc0125 here, so this is **not a same-scroll comparison**. On this measure, our
masked surfaces still show about twice the doubling that lattice-unroll reports on another scroll.

## Limits

- **Sense unverified.** CW was used. The CW/ACW test was undecided: the excess was −0.004 to −0.012, against a bar of ±0.04.
- **Doubled sheets remain**: 6.8-8.5 per ray after masking.
- **55 % of the fitted surface is masked away.** The kept surface is patchy, not a continuous sheet per winding.
- **Band 5 failed its join**, on satisfied tracks only (54.04 % against the 55 % bar), so it is not included. Band 4 was not run.
- The seam score uses the overlap only. With anchors on, the fitter skips track points within 16 vx of pinned vertices.
  The cross-section sites cover z 10500-13550, including the bands' transition zones, but no separate transition test
  was run.
- The fibre and cross-section checks are visual plus simple statistics. They are not ink results.

## Credits

- The Vesuvius Challenge team built the villa spiral fitter, the tracks, the lasagna normals and the m7 surface
  predictions.
- Earlier PHerc0125 surfaces: pscamillo (stock-fitter meshes) and others in the community.
- lattice-unroll's sheets-per-ray measure was the model for `rays.py`. Our code is our own.

AI assistance: the code, runs and write-up were done with Claude (Anthropic) under the author's direction.

Licence: MIT (see `LICENSE`).
