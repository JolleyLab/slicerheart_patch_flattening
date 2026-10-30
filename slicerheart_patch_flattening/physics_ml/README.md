# physics_ml: physics-based (forward / inverse) and learning-based flattening

Separate track from the geometric solvers in `algorithms/`. Nothing here changes the
traditional methods; this folder reads them only for shape generation and labels.

## What is here

| file | content |
|---|---|
| `warp_membrane.py` | Differentiable membrane / thin shell in NVIDIA Warp (CPU or CUDA). Materials: St. Venant-Kirchhoff, incompressible neo-Hookean, HGO fibre term (placeholder parameters). Pressure as a follower load (potential = -p x swept volume), discrete-shell hinge bending (Grinspun et al. 2003) as regulariser. Two configurations: level 1 flatten (3D rest -> 2D), level 2 inflate (2D template + suture line + pressure -> 3D). Energy and gradients from `wp.Tape`; quasi-static equilibrium by L-BFGS. |
| `inverse.py` | `inflate()` forward solve; `InverseTemplate`: template from a loaded target by the adjoint method through the equilibrium (implicit function theorem; Hessian-vector products by differencing the Warp gradient, so first-order AD only). Regularisers: strain energy of the inflated state (default) and a Laplacian smoothness prior. `uniform_unload()` is the Kizilski-style isotropic analytical baseline. |
| `synthetic_dataset.py` | Random baffles from the library's families with ARAP templates, metrics, dart proposals and optional Warp inflated states as labels, for a learned surrogate. |

## Validation so far (CPU, 572-vertex PAPVR baffle, StVK E = 10 MPa, nu = 0.3, t = 0.3 mm)

* Level-1 Warp flattening reproduces the numpy `membrane_stvk` solver exactly (energy 0.342354 in both, same stretch metrics); gradient vs finite differences 3e-9.
* Forward inflation of the ARAP template sewn on the suture line: 0.4 s; at 0 mmHg the membrane settles 0.21 mm RMS from the planned surface (the non-developable residual), at 30 mmHg 0.29 mm RMS.
* Adjoint gradient of the inverse vs finite differences: 0.5 % (with and without the strain term).
* Inverse from the ARAP template: target surface matched to 0.005-0.017 mm RMS in 40-60 L-BFGS iterations (3-5 min on 2 CPU cores; the cost is the forward re-solves and the CG on the Hessian).
* **The membrane inverse is ill-posed on its own.** With the suture line pinned in 3D, a template change alters the rim *stress* but not the rim position, so many templates (0.3 mm apart on the outline) give the same inflated shape to 5 um. A synthetic "true" template is therefore not recoverable from the shape alone, and that is physics, not a solver defect. With the strain-energy regulariser the inverse returns the template that reproduces the shape with the least stored strain (1.12 N mm vs 2.85 N mm for the synthetic truth), i.e. the patch in tension without wrinkles. That is the template to cut; the inverse is a design problem with a criterion, not an identification problem.

## Relation to FEBio-Warp / PatchWarp (Seda Aslan)

`prototypes/PatchWarp` in JolleyLab/FEBio-Warp is a 4-node director shell (full 3D StVK,
2x2x2 Gauss, follower pressure, FEBio-compatible chords), Newton + GPU CG with block
preconditioning, validated against FEBio (pressure-only max displacement to 1e-6 %, four-chord
case to 0.13 %), with implicit tangents and adjoints for E, nu, pressure and chord
parameters. CUDA only, SI units.

This folder is the lighter counterpart: triangle membrane + hinge bending on the meshes the
Baffle Planner produces, mm/N/MPa units, CPU or GPU, first-order AD, and an adjoint with
respect to the *rest shape* (the template), which PatchWarp does not have yet. The natural
merge: add a `residual_partial('rest')` to PatchWarp's `StaticSensitivity` and use its
quad shell as the forward model for the same `InverseTemplate` loop once a quad remesh of
the baffle is available; keep this membrane as the fast CPU reference and as the label
generator for the surrogate.

## Toward the learned surrogate (next)

1. `make_dataset(n=3000)` with ARAP labels and dart proposals (about 0.3 s per sample), then
   with inflated states at 2-3 pressures and 2-3 materials (about 1 s per sample on CPU; GPU
   batches later).
2. Surrogate: boundary curve (128 x 3, canonical frame) + 256 surface samples with normals ->
   template outline (128 x 2) + suture mismatch + dart need. A small 1D-CNN/PointNet in
   PyTorch; the residual to the projected boundary is the target. Loss on the outline after
   rigid alignment. The conditioning vector (pressure, material) is part of the input from the
   start so the pressurised labels drop in without changing the model.
3. Use: instant template preview while the surgeon edits the baffle; initial guess for the
   adjoint inverse; uncertainty from an ensemble.

## Physical validation library (ideas, for discussion)

Goal: a set of physical baffles with known geometry to test flattening and pressurisation
end to end, as Kizilski et al. did for straight tubes.

* **Shapes**: 8-12 of the synthetic library (two developable controls, tunnels with increasing
  bend/twist, two TPS baffles, one trouser), printed as rigid positive forms with the suture
  line engraved, at 1:1 and 2:1 scale (2:1 makes marks and strain measurement easier).
* **Forms**: SLA resin (Form 3+) or PolyJet for the rigid forms; a matching negative so a
  patch can be clamped along the suture line. For pressurisation, a sealed cavity under the
  form with a port, like the BPA fixture in Kizilski et al.
* **Patch materials**: CardioCel/bovine pericardium (0.3-0.5 mm), 0.4 mm ePTFE (Gore-Tex
  cardiovascular patch), knitted Dacron, and as a well-characterised engineering surrogate
  EcoFlex 00-30 / Dragon Skin 10 silicone sheets (0.5 and 1 mm) whose biaxial response is easy
  to measure. Silicone first (repeatable, printable grids), tissue later.
* **Measurement**: a printed/laser-marked grid or speckle on the patch, photographed before and
  after suturing and under pressure, for 2D DIC strain fields (open-source DIC tools suffice);
  micro-CT or structured-light scan of the sewn, pressurised patch for the 3D shape; suture
  uptake measured as in Kizilski et al. (marker pairs).
* **Readouts**: outline fit (gap/overlap along the suture line), strain field vs predicted
  principal stretches, 3D shape vs target at 0 / 15 / 30 mmHg, darted vs undarted versions of
  the same shape.

## Dependencies

`pip install warp-lang` (CPU works; CUDA used automatically if available), `torch` only for
the surrogate and `developabilize`. See `pyproject.toml` extras `physics` and `ml`.
