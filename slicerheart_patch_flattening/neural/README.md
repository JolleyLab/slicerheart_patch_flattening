# neural

Learned surrogate for flattening: canonicalised boundary curve + surface samples -> template
outline residual, trained on synthetic baffles labelled by the geometric solvers (ARAP).
Feasibility study and scaling results: `reports/04_neural_surrogate_feasibility.md`.

- `data.py`: featurisation (canonical frame, arclength resampling, surface sampling, residual target)
- `model.py`: `OutlineSurrogate` (circular 1D CNN boundary encoder + PointNet surface encoder)
- `train.py`: training, baselines, scaling study (`python -m slicerheart_patch_flattening.neural.train --data ...`)
