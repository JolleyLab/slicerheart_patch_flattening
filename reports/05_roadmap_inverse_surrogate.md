# Report 5: training on dataset v1 and roadmap to an inverse (pressurised) template surrogate

For whoever trains on a GPU. The data are described in
[`neural/DATASET.md`](../slicerheart_patch_flattening/neural/DATASET.md) (record format, families,
physics grid, limitations); the code entry points are `neural/dataset_io.py` (loader, deterministic
split), `neural/model.py`, `neural/train.py`, `physics_ml/inverse.py`. Report 4 has the CPU
feasibility numbers this builds on.

## 1. What the data are and what they are for

Dataset v1 = 100 000 synthetic baffles (ten shards of 10 000, release `dataset-v1`) with
geometry, the ARAP and LSCM templates, per-vertex stretch and curvature, metrics and dart
proposals, plus physics shards (about 5 000 samples) carrying the inflated 3D state of the ARAP
template sewn on the suture line for 2 pressures x 2 materials. A 100-sample slice is in the repo.

Two surrogates can be trained from it:

| surrogate | input | output | labels | why |
|---|---|---|---|---|
| A. flattening (level 1) | 3D surface | template outline / per-vertex UV | `arap_uv`, `arap_outline` | instant preview while the surgeon edits; uncertainty by ensemble. Cannot beat ARAP (its label), which already runs in 0.1 s, so A is a stepping stone, not the goal |
| B. inverse (level 2) | loaded 3D surface + pressure + material | unloaded template | `inflated_xyz_*` -> `arap_uv` (forward pairs read backwards), later adjoint-inverse labels | the slow problem (minutes per baffle with the adjoint) and the clinically relevant one (15-25 % template change between low- and high-pressure designs, Kizilski et al. 2025) |

## 2. Baselines and acceptance criteria (same split, same metric)

Outline error after rigid alignment, per shape, on the hash-defined test split:

| baseline | outline error vs label |
|---|---|
| orthographic projection | 2.5 mm RMS |
| LSCM vs ARAP (two valid methods) | 0.22 mm RMS, 2.4 mm max |
| CPU surrogate, Report 4 (2 200 samples, 2 min) | 0.16 mm median, 0.5 mm p95, 13 mm max |
| **acceptance for a usable template surrogate** | **< 0.2 mm RMS and < 0.5 mm worst case** on test; no fold-overs in the polished output |

Report median, p95, p99 and max per shape, by family, plus inference time. The worst case is the
number that matters: a template is cut once.

## 3. Suggested path on the generated data (surrogate A, 1-2 weeks of GPU work)

1. **Reproduce Report 4 on GPU** with the existing `OutlineSurrogate`:
   `python -m slicerheart_patch_flattening.neural.train --shards shard_00.zip ... --device cuda --amp --sizes 1000 10000 100000 --epochs 200`.
   This gives the real scaling curve (the CPU one was flat because of the 2-minute budget).
2. **Fix the start-point ambiguity** (`--shift-invariant`, or Fourier descriptors of the outline):
   the 13 mm outliers in Report 4 are symmetric baffles whose canonical start point flips.
3. **Per-vertex model**: predict `arap_uv` on the mesh (DiffusionNet or a message-passing GNN on
   `faces`, inputs: positions in the canonical frame, normals, `vertex_angle_defect`). Judge it
   with `metrics.distortion_metrics` exactly like the solvers (edge error p95, suture mismatch,
   fold-overs), not only outline RMS.
4. **Physics-informed loss**: replace or complement the label loss with the ARAP / membrane energy
   of the prediction (`geometric/common.py` has the per-face Jacobians; the energy is a few lines
   of torch) and add 1-2 local/global polish iterations at inference. This removes the label
   ceiling and the outliers, and is the architecture to carry into B.
5. **Ensemble of 5** for a per-point uncertainty band drawn on the template.

Deliverable: a model that meets the acceptance criteria on test, with the scaling curve and the
by-family table, saved as `results/neural_*.json` and weights under `results/`.

## 4. Roadmap to the inverse surrogate (surrogate B)

**Stage 0, labels as they are (physics shards).** Train B as (inflated_xyz, pressure, material)
-> `arap_uv`. The pairs are exact for the forward model (the ARAP template *did* produce that
inflated state), so this already learns the pressure/material dependence. Caveat: the ARAP
template is not the least-strain template for that loaded shape (Report 3), so the label is one
valid answer among several; expect a floor of a few tenths of a millimetre on the outline.

**Stage 1, adjoint-inverse labels.** Generate labels with `physics_ml.inverse.InverseTemplate`
(strain-energy criterion) for a subset: 2 000-5 000 baffles x the pressure/material grid.
Each takes 1-5 min on CPU today; on GPU with Warp on `cuda` and the CG on the Hessian this should
drop well below a minute, and the Report-4-style surrogate from Stage 0 is the warm start that
cuts the iteration count. Retrain B on these labels: now the output is the designed (least-strain)
template, i.e. the thing the surgeon should cut.

**Stage 2, amortised physics (the recommended end state).** Make B physics-informed: the loss is
the same objective the adjoint minimises (shape mismatch after a forward solve + strain energy),
evaluated with the differentiable Warp forward model inside the training loop (Warp <-> torch via
`wp.to_torch` / `wp.from_torch`, zero copy on GPU). The network then learns the design rule
without labels, and the adjoint solver becomes the optional polish step that certifies the result
(residual < 0.05 mm). This is the "faster and more accurate than pure physics" case: faster
because inference is milliseconds and the polish starts from the answer; as accurate as the
forward model, which is the right ceiling.

**Stage 3, learning from measurements.** When the physical validation library exists (forms,
sewn patches, CT/DIC at 0/15/30 mmHg; Report 3), use measured (template, loaded shape) pairs to
fit the material model and suture uptake first (few parameters, Warp adjoint), then fine-tune B
on the corrected forward model. Only at this stage can a learned model be more accurate than the
physics it was distilled from, because the labels are then better than the model.

**Stage 4, deployment in the Baffle Planner.** `slicer_api.flatten_polydata(method="inverse_surrogate", pressure_mmhg=..., material=...)`
returning the template plus the uncertainty band and the predicted strain map, with the adjoint
polish run in the background when time allows. Same `vtkPolyData` contract as today.

## 5. Decisions the GPU user should make early

* Mesh-aware (per-vertex) vs outline model: per-vertex is needed for Stages 1-2 (strain enters the
  loss); start there if GPU time allows.
* Material parameterisation in the input: (E, nu, t) now; replace by fitted constitutive
  parameters (HGO fibre terms for pericardium / fabric anisotropy) when measured.
* Pose handling: the canonical frame (PCA of the suture line) is enough for outline models; for
  per-vertex models prefer invariant features (edge lengths, angles, curvature) over coordinates.
* Validation protocol before any clinical use: ASME V&V40-style credibility, as in Kizilski et al.
  2025; the synthetic test split is not a validation of clinical performance.

## 6. Compute and data budget

| item | size / time |
|---|---|
| dataset v1 geometry, 100 k | 5 GB, 7.7 core-hours to generate |
| physics shards, 5 k x 4 states | 1.5 GB, ~11 core-hours (CPU Warp) |
| Stage 1 labels, 5 k x 4 | 1-5 min each on CPU; target < 1 min on GPU |
| surrogate A training, 100 k, 200 epochs | hours on one GPU |
| Stage 2 training | days on one GPU (forward solve in the loop), fewer samples needed |
