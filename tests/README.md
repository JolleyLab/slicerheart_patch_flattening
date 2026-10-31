# Tests

    pip install -e ".[test]"
    python -m pytest                                   # everything (about 30 s)
    python -m pytest -m "not physics"                  # without the Warp membrane track
    python -m pytest -m "not dataset and not benchmark" # only the flattening algorithms and metrics

All tests run on synthetic shapes: analytic surfaces of known curvature (a half cylinder and a
cone, which are developable; a sphere cap and a saddle, which are not; a flat disk) and the
30-shape synthetic baffle library. No data is downloaded. Tests that need an optional package
are skipped when it is missing: `warp-lang` (marker `physics`), `libigl` (the LSCM cross-check;
no wheels for Python 3.13 yet), `torch` (developabilization and the dataset loader).

What is covered, by file:

| File | Checks |
|---|---|
| `test_mesh.py` | `TriMesh`: disk topology, Gauss-Bonnet, curvature of the analytic shapes, VTK round trip, STL/PLY/OBJ/VTP I/O, exact and Dijkstra geodesics (straight line on a plane, great circle on a sphere) |
| `test_library.py` | the synthetic library: every shape a valid disk of pediatric size, the `developable` flag against the integrated curvature, reproducibility by seed, manifest write/load, the generators |
| `test_geometric.py` | every free-boundary method is isometric on developable shapes; the half cylinder recovers its exact template; fixed-boundary methods map the boundary onto a circle without flips; LSCM matches libigl; LSCM wins on angles and ARAP on lengths; CETM/BFF keep the suture-line lengths; rescale modes; every method on every library shape |
| `test_invariance.py` | a flattening does not change with the world coordinate system (LPS/RAS, rotation, translation, reflection), the face order, the vertex numbering or the face orientation; scaling the input scales the template |
| `test_metrics.py` | the distortion metrics on flattenings with known distortion: identity, uniform scale, anisotropic stretch, fold-overs, the size tolerance, Feret length and width |
| `test_darts.py` | dart recommendation (none on developable shapes, darts on a cap, gussets on a saddle), mesh cutting (still a disk, area kept, perimeter grows by twice the cut), re-flattening reduces the suture-line mismatch |
| `test_uncertainty.py` | rigid alignment, method spread (zero for rigidly moved copies), subdivision, refinement sensitivity |
| `test_slicer_api.py` | the `vtkPolyData` entry point of the Baffle Planner: arrays, fixed points as pins and template orientation, non-triangle input |
| `test_export.py` | SVG/DXF/PDF templates in true millimetres, seam allowance offset, the recommended-template pipeline |
| `test_physics_ml.py` | the Warp membrane track: StVK, neo-Hookean and HGO energies against closed forms, gradients against finite differences, the Warp flattening against the numpy membrane, zero-pressure and bending sanity, the pressurised circular membrane against Hencky's solution (centre deflection within 3 %), forward inflation and the adjoint gradient |
| `test_dataset.py` | the synthetic dataset generator and sharded script, the shard loader and splits (the neural surrogate itself is work in progress and has no tests yet) |
| `test_benchmark.py` | the benchmark harness and its Markdown/PDF report on three shapes |

## Regression metrics

Tests record the numbers behind their assertions (max stretch, edge errors and suture-line
mismatch per shape and method, the per-family distortion over the whole library, dart
improvement, refinement sensitivity, the Hencky error) through the `metrics` fixture in
`conftest.py`. Every recorded value is compared with `tests/metrics/baseline.json`: an error
metric may not get worse than its baseline by more than 5 % (relative, plus a small absolute
floor for near-zero values), so a change that stays inside a test's hard limit but degrades a
method still fails. The values of each run are written to `.test-reports/latest.json` at the
repository root (ignored by git; the `SLICERHEART_PATCH_FLATTENING_TEST_REPORT_DIR` environment
variable sets another folder).

    python -m pytest                       # compare with the baseline
    python -m pytest --update-baseline     # accept the current values as the new baseline
    python -m pytest --metrics-tolerance 0.02

When a change improves a metric the comparison passes; run with `--update-baseline` and commit
the new baseline so the improvement is protected too. Metrics of skipped tests are left
unchanged in the baseline. The CI workflow uploads `latest.json` of every platform as an
artifact, so platform differences can be compared by hand.

## Continuous integration

`.github/workflows/ci.yml` runs the tests on every push to `main`, every pull request and on
demand: Python 3.10, 3.12 and 3.13 on Ubuntu, 3.12 on Windows and macOS, and 3.12 with the VTK
version pinned for the current 3D Slicer release. A separate job runs the quick benchmark
(6 shapes x 4 methods) and attaches its report. The full benchmark and the dataset generation
have their own on-demand workflows.
