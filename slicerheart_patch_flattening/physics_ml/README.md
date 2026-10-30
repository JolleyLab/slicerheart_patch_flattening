# physics_ml

Differentiable membrane / thin-shell model in NVIDIA Warp with pressure and material models,
the forward (inflation) problem and the adjoint inverse (template from a loaded target).
See `reports/03_physics_ml_warp_membrane_and_inverse.md` for the design, validation and findings.

- `warp_membrane.py`: `WarpMembrane`, `Material` (stvk, neo_hookean, hgo); modes `flatten` and `inflate`
- `inverse.py`: `inflate()`, `InverseTemplate`, `uniform_unload()`
- `synthetic_dataset.py`: `make_dataset()` random baffles with ARAP labels, darts, optional inflated states
