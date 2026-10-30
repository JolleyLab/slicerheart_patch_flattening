# slicerheart_patch_flattening

Flattening of 3D surgical patch and baffle surfaces into 2D cut templates, with error metrics,
darting and template export, for [SlicerHeart](https://github.com/SlicerHeart/SlicerHeart)
virtual surgical planning (DORV baffles, Mustard/Senning baffles, PAPVR baffles, vascular patches).

Three approaches live in one package:

| folder | approach | status |
|---|---|---|
| `geometric/` | mesh parameterisation (ARAP, LSCM, conformal, membrane, ...), darting, developabilization | benchmarked, ARAP is the default |
| `physics_ml/` | NVIDIA Warp membrane/shell with pressure and material models; forward inflation and adjoint inverse (template from a loaded target) | validated prototype, CPU/GPU |
| `neural/` | learned surrogate (PyTorch) from a synthetic library: boundary curve + surface samples to template outline | feasibility study (Report 4): not recommended for unloaded flattening; candidate for amortising the physics inverse |

Study results are in [`reports/`](reports/); the summary comparing the three is
[`reports/00_summary_geometric_vs_physics_vs_neural.md`](reports/00_summary_geometric_vs_physics_vs_neural.md).

## Features

- 17 flattening methods behind one interface, `flatten(mesh, method)`; `flatten_recommended(mesh)` runs the default (ARAP), computes the metrics and proposes darts/gussets when needed
- Metrics that matter for a cut template: principal stretches, edge and suture-line length errors, cumulative suture-line mismatch (mm), template length/width vs exact geodesics, fold-overs, developability
- Uncertainty: inter-method outline spread, mesh-refinement sensitivity
- Darting: curvature-driven dart/gusset proposal, mesh slit, re-flatten
- Developabilization: nearest developable baffle with the suture line fixed
- Template export at 1:1 mm as SVG, DXF (laser cutter) and PDF, with suture-bite offset
- Synthetic library of 30 pediatric baffle shapes, a random generator, and dataset v1 (100 k samples with ARAP/LSCM labels and physics-labelled shards, published as release assets; `scripts/generate_dataset.py`)
- `slicer_api.flatten_polydata()`: the Baffle Planner interface (open `vtkPolyData` in, flattened `vtkPolyData` with stretch arrays out)
- Benchmark script with Markdown/PDF report; tests run on every pull request

## Installation

```bash
pip install git+https://github.com/JolleyLab/slicerheart_patch_flattening
# or, from a clone:
pip install -e ".[ref,dev,physics,ml]"
```

Dependencies: numpy, scipy, vtk, matplotlib, pandas, shapely, pygeodesic, tabulate. Optional:
`libigl` (cross-checks), `warp-lang` (physics_ml), `torch` (neural, developabilize).

In 3D Slicer: `slicer.util.pip_install("git+https://github.com/JolleyLab/slicerheart_patch_flattening")`.

## Usage

```python
from slicerheart_patch_flattening import TriMesh, flatten, flatten_recommended
from slicerheart_patch_flattening import export

mesh = TriMesh.load("baffle.vtp")                 # open, single-boundary triangle surface, mm
r = flatten_recommended(mesh)                      # ARAP template + metrics (+ darts if needed)
print(r.summary())
outline = export.template_outline(mesh, r.result.uv)
export.write_svg("template.svg", outline, "baffle", suture_offset_mm=1.0)   # also write_dxf, write_pdf

uv = flatten(mesh, "cetm").uv                      # any other method: lscm, membrane_stvk, cetm, ...
```

From the Baffle Planner (or any VTK pipeline):

```python
from slicerheart_patch_flattening.slicer_api import flatten_polydata
flat_polydata, result, metrics = flatten_polydata(baffle_polydata, "arap", fixed_points_world=pts)
```

Command line:

```bash
python scripts/build_library.py output/synthetic_library          # 30-shape synthetic library
python -m slicerheart_patch_flattening.evaluate --library output/synthetic_library --out output/eval
python scripts/run_benchmark.py --out output/benchmark             # everything + report.md / report.pdf
python scripts/generate_dataset.py --out output/dataset --shard 0 --n 10000 --workers 4 --zip   # dataset shard
```

## Modules

| module | content |
|---|---|
| `mesh.py` | `TriMesh`: numpy V/F, boundary loop, angle defects, geodesics (pygeodesic), VTK I/O |
| `geometric/methods.py`, `methods_extra.py` | pca_projection, tutte, harmonic, mean_value, lscm (+edge/perimeter scaling), asap, arap, arap_hybrid, spring, membrane_stvk, symmetric_dirichlet, membrane_suture, bff_lite, cetm, exp_map |
| `geometric/darts.py` | dart/gusset proposal, mesh cut, `flatten_with_darts` |
| `geometric/developabilize.py` | nearest developable surface (PyTorch L-BFGS) |
| `metrics.py`, `uncertainty.py` | distortion, suture-line and size metrics; method spread, refinement sensitivity |
| `pipeline.py` | `flatten_recommended` |
| `export.py`, `slicer_api.py`, `clinical.py` | SVG/DXF/PDF export; Baffle Planner interface; loader for `clinical_baffles/` |
| `library/` | synthetic shape generators and the 30-shape library |
| `physics_ml/` | Warp membrane, forward inflation, adjoint inverse, random dataset generator |
| `neural/` | surrogate data pipeline, model, training and scaling study |
| `evaluate.py`, `report.py`, `viz.py` | benchmark harness, Markdown/PDF report, figures |

## Reports

- [00 Summary: geometric vs physics vs neural](reports/00_summary_geometric_vs_physics_vs_neural.md)
- [01 Geometric methods, first benchmark](reports/01_geometric_methods_benchmark.md)
- [02 Geometric methods, wider exploration, darting, developabilization](reports/02_geometric_methods_wider_exploration.md)
- [03 Physics track: Warp membrane, pressurised forward model, adjoint inverse](reports/03_physics_ml_warp_membrane_and_inverse.md)
- [04 Neural surrogate feasibility](reports/04_neural_surrogate_feasibility.md)
- [05 Training on dataset v1 and roadmap for the inverse surrogate](reports/05_inverse_surrogate_roadmap.md)
- Dataset v1 record format and provenance: [neural/DATASET.md](slicerheart_patch_flattening/neural/DATASET.md)

`results/` holds the CSVs and `docs/figures/` the figures of the committed benchmark run.

## Testing

```bash
python -m pytest tests
```

GitHub Actions runs the tests on every pull request and push to `main` (Python 3.10 and 3.12)
and attaches a quick benchmark report as an artifact; the `benchmark` workflow runs the full
benchmark on demand and on every release and attaches `benchmark_report_<tag>.pdf` to the release.

## Clinical cases

`clinical_baffles/` is the place for real baffles exported from the Baffle Planner (see its
README); `evaluate.run_benchmark` runs on them unchanged.

## References

See [REFERENCES.md](REFERENCES.md); every entry was verified against OpenAlex. Main ones: Vigil et al.
2021 (Baffle Planner), Kizilski et al. 2025 (patch-planning validation, tolerances, darting),
Levy et al. 2002 (LSCM), Liu et al. 2008 (ARAP), Springborn et al. 2008 (CETM), Sawhney & Crane 2017 (BFF).

## License

MIT. Copyright (c) 2026 JolleyLab, Children's Hospital of Philadelphia.
