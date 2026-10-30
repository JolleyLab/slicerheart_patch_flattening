# Clinical baffles (placeholder)

Real patient-derived patch / baffle surfaces go here, one sub-folder per case:

```
clinical_baffles/
  README.md
  manifest.json              # optional: list of cases with family + notes (same schema as synthetic manifest)
  case_001/
    baffle.vtp | baffle.stl | baffle.obj   # open thin surface exported from the SlicerHeart Baffle Planner
    fixed_points.mrk.json                  # optional: Baffle Planner "fixed points" markups (2+ points)
    boundary_curve.mrk.json                # optional: input closed curve markups
    notes.md                               # anatomy, procedure (DORV / Mustard / PAPVR), age, weight
```

Requirements for a surface to be usable by the flattening library:
- a single open, manifold triangle surface (disk topology, one boundary loop). Export the thin
  (non-thickened) baffle model from the Baffle Planner, not the extruded one.
- units in mm, patient coordinate frame is irrelevant (methods are rigid-motion invariant).

Loading: `slicerheart_patch_flattening.clinical.load_clinical_cases("clinical_baffles")`
returns the same `TriMesh` objects as the synthetic library, so `evaluate.run_benchmark` can be
pointed at this folder without changes. No clinical data is committed yet.
