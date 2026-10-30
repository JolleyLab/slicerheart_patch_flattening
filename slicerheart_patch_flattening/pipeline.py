"""Recommended end-to-end flattening pipeline (the library default).

Outcome of the two benchmark passes (README, "Results" and "Wider exploration"):

  * template:  ARAP (Liu et al. 2008), the isometric local/global method. Lowest
               length error of the fold-free methods (edge p95 5.6 % vs 9.1 % for the
               LSCM used by the Baffle Planner today), 0.1 s per patch, identical to
               symmetric Dirichlet / membrane results to within 0.3 %.
  * scale:     none needed (ARAP is in true units); the Baffle Planner's sqrt(area)
               rescale is only required for conformal maps.
  * check:     distortion_metrics -> suture-line mismatch, max/min stretch, size check.
  * darts:     if the cumulative suture-line mismatch exceeds `dart_threshold_mm`
               (default 1 mm, i.e. clearly above cutting precision) propose darts /
               gussets and return the darted ARAP template as well, with its metrics.
  * optional:  a "suture-exact" alternative (CETM) for the surgeon who wants the
               boundary to match exactly and accepts the interior stretch map.

`flatten_recommended(mesh)` returns everything needed by a planner UI or the
Slicer API in one call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import TriMesh
from .geometric.methods import flatten
from .geometric.common import FlatResult
from .metrics import distortion_metrics
from .geometric.darts import recommend_darts, flatten_with_darts

DEFAULT_METHOD = "arap"
DART_THRESHOLD_MM = 1.0


@dataclass
class RecommendedTemplate:
    method: str
    result: FlatResult
    metrics: dict
    darts: list = field(default_factory=list)              # proposals (empty if not needed)
    darted_result: FlatResult | None = None
    darted_mesh: TriMesh | None = None
    darted_metrics: dict | None = None
    suture_exact_result: FlatResult | None = None
    suture_exact_metrics: dict | None = None

    @property
    def needs_darts(self) -> bool:
        return bool(self.darts)

    def summary(self) -> str:
        M = self.metrics
        lines = [f"{self.method}: max stretch {M['stretch_Linf']:.3f}, min stretch {M['stretch_min']:.3f}, "
                 f"edge error p95 {M['edge_rel_err_p95_abs_pct']:.1f} %, suture-line mismatch "
                 f"{M['boundary_cumulative_mismatch_mm']:.2f} mm, size check "
                 f"{'pass' if M['passes_size_tolerance'] else 'fail'}"]
        if self.darts:
            d = ", ".join(f"{x['kind']} {abs(x['angle_deg']):.0f} deg" for x in self.darts)
            Mc = self.darted_metrics
            lines.append(f"darts proposed ({d}): suture-line mismatch {Mc['boundary_cumulative_mismatch_mm']:.2f} mm, "
                         f"edge error p95 {Mc['edge_rel_err_p95_abs_pct']:.1f} % after cutting")
        return "\n".join(lines)


def flatten_recommended(mesh: TriMesh, method: str = DEFAULT_METHOD, dart_threshold_mm: float = DART_THRESHOLD_MM,
                        max_darts: int = 2, suture_exact: bool = False, **kw) -> RecommendedTemplate:
    res = flatten(mesh, method, **kw)
    M = distortion_metrics(mesh, res.uv)
    out = RecommendedTemplate(method=method, result=res, metrics=M)
    if M["boundary_cumulative_mismatch_mm"] > dart_threshold_mm:
        darts = recommend_darts(mesh, max_darts=max_darts)
        if darts:
            rc, cut, applied = flatten_with_darts(mesh, method, darts, **kw)
            if applied:
                out.darts = applied
                out.darted_result, out.darted_mesh = rc, cut
                out.darted_metrics = distortion_metrics(cut, rc.uv)
    if suture_exact:
        rs = flatten(mesh, "cetm")
        out.suture_exact_result = rs
        out.suture_exact_metrics = distortion_metrics(mesh, rs.uv)
    return out
