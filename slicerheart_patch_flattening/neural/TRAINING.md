# Training notes for whoever picks this up on a GPU

The step-by-step plan for the inverse surrogate is `reports/05_inverse_surrogate_roadmap.md`; this file
holds the baselines, metrics and practicalities it refers to.

What exists: `data.py` (featurisation), `dataset_io.py` (shard loader, deterministic split,
torch dataset), `model.py` (`OutlineSurrogate`, 0.4 M params), `train.py` (training, baselines,
scaling study). Report 4 has the CPU feasibility numbers: median 0.16 mm outline error vs ARAP,
p95 0.5 mm, no improvement from 250 to 2200 samples at a 2-minute budget, 4 ms inference.

## Baselines to beat (same test split, same metric)

| baseline | outline error vs ARAP |
|---|---|
| orthographic projection | 2.5 mm RMS |
| LSCM (today's Baffle Planner) | 0.22 mm RMS, 2.4 mm max |
| target for a usable surrogate | < 0.2 mm RMS **and** < 0.5 mm worst case |

The worst case matters more than the mean: a template is cut once.

## Suggested first experiments (in order)

1. **Scaling curve with proper training**: `OutlineSurrogate` as is, 10 k / 30 k / 100 k samples,
   200 epochs, AdamW + one-cycle, batch 128, bf16 autocast. If the curve is still flat, the
   model is the limit; if it drops, data was.
2. **Shift-invariant loss**: the canonical start point (farthest boundary point from the centroid)
   is ambiguous on symmetric baffles and produced the outliers. Use the minimum over cyclic
   shifts of the MSE (128 shifts, cheap on GPU) or Fourier descriptors of the outline.
3. **Per-vertex model**: predict `arap_uv` on the mesh (DiffusionNet or a message-passing GNN on
   `faces`) instead of the 128-point outline; gives interior strain and works for any mesh.
4. **Physics-informed training**: replace the label loss by the ARAP energy of the prediction
   (`geometric/common.py` has the Jacobians; the energy is a few lines in torch) plus one or two
   local/global polish iterations at inference. No label ceiling, no outliers.
5. **Inverse surrogate** on the physics shards: (inflated_xyz, pressure, material) -> template,
   then seed `physics_ml.inverse.InverseTemplate` with the prediction and report how many adjoint
   iterations it saves.
6. **Ensemble of 5** for per-point uncertainty bands on the template.

## Metrics to report

Per-shape outline RMS and max (mm) after rigid alignment: median, p95, p99, max; by family;
versus the LSCM baseline; inference time per patch; and, for per-vertex models, the same
`metrics.distortion_metrics` the solvers are judged on (edge error p95, suture mismatch).

## Practicalities

* `train.py --device cuda --amp` for the existing model; `--shift-invariant` for the cyclic loss.
* Featurisation is lazy, so keep the shards as they are; `BaffleDataset` caches tensors in RAM
  (about 1 GB per 100 k samples at M=256 surface points).
* Set `M_SURFACE` higher (512-1024) for strongly curved tunnels; the cost is linear.
* Sample ids and the hash split are global: never re-split by position.
