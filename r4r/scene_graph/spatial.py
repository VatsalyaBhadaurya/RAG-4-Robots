"""Deterministic spatial-relation inference (Phase 2 baseline).

Given physical nodes with pose + dimensions, derive geometric relations:
on_top_of / supports, inside / contains, near, next_to, aligned_with.

ASSUMPTION (MVP simplification, flagged in the decision log): boxes are treated
as axis-aligned bounding boxes in the world frame; orientation (quaternion) is
ignored for relation inference. This is adequate for a tabletop MVP and avoids a
full collision/geometry engine. It is replaced later by MuJoCo/geometry queries
where fidelity matters (e.g. grasp/collision scoring in Phase 4/6).

These are heuristics, not ground truth — every edge is emitted with
SourceType.DERIVED and a confidence reflecting the geometric margin, so the
retriever/validator can discount them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from r4r.schemas.edges import RelationType, SceneEdge
from r4r.schemas.nodes import NodeKind, SceneNode
from r4r.schemas.provenance import Provenance


class SpatialConfig(BaseModel):
    """Tunable thresholds (metres). Defaults target a tabletop in metres."""

    contact_tol: float = Field(0.02, description="max vertical gap counted as 'on top of'")
    footprint_overlap_min: float = Field(0.1, description="min xy IoU-ish overlap ratio for support")
    near_dist: float = Field(0.25, description="centre distance under which objects are 'near'")
    next_to_height_tol: float = Field(0.05, description="centre-z difference to count as same level")
    aligned_tol: float = Field(0.02, description="centre offset on an axis to count as aligned")


# (axis-aligned) box as (min_xyz, max_xyz)
_Box = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def _aabb(node: SceneNode) -> Optional[_Box]:
    if node.pose is None or node.dims is None:
        return None
    cx, cy, cz = node.pose.x, node.pose.y, node.pose.z
    hx, hy, hz = node.dims.half()
    return ((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))


def _xy_overlap_ratio(a: _Box, b: _Box) -> float:
    """Overlap area / smaller footprint area (asymmetric 'containment-ish')."""
    (ax0, ay0, _), (ax1, ay1, _) = a
    (bx0, by0, _), (bx1, by1, _) = b
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ox * oy
    if inter <= 0.0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _centre_dist_xy(a: SceneNode, b: SceneNode) -> float:
    dx = a.pose.x - b.pose.x
    dy = a.pose.y - b.pose.y
    return (dx * dx + dy * dy) ** 0.5


def _derived(conf: float) -> Provenance:
    return Provenance.derived(source_id="spatial_heuristic", confidence=conf)


def _xy_contained(inner: _Box, outer: _Box) -> bool:
    (ix0, iy0, _), (ix1, iy1, _) = inner
    (ox0, oy0, _), (ox1, oy1, _) = outer
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


def infer_spatial_edges(nodes: List[SceneNode], cfg: Optional[SpatialConfig] = None) -> List[SceneEdge]:
    """Return derived spatial edges for the given physical nodes.

    Only emits edges between nodes that have both pose and dims. Pure function:
    does not mutate the graph (caller decides what to add)."""
    cfg = cfg or SpatialConfig()
    phys = [n for n in nodes if _aabb(n) is not None]
    boxes = {n.object_id: _aabb(n) for n in phys}
    edges: List[SceneEdge] = []

    for a in phys:
        for b in phys:
            if a.object_id == b.object_id:
                continue
            ba, bb = boxes[a.object_id], boxes[b.object_id]

            # --- on_top_of / supports: a sits on b ---
            a_bottom = ba[0][2]
            b_top = bb[1][2]
            gap = a_bottom - b_top
            overlap = _xy_overlap_ratio(ba, bb)
            if abs(gap) <= cfg.contact_tol and overlap >= cfg.footprint_overlap_min and a_bottom >= bb[0][2]:
                conf = min(1.0, overlap)
                edges.append(SceneEdge(src=a.object_id, rel=RelationType.ON_TOP_OF,
                                       dst=b.object_id, confidence=conf,
                                       attributes={"footprint_overlap": round(overlap, 3)},
                                       provenance=_derived(conf)))
                edges.append(SceneEdge(src=b.object_id, rel=RelationType.SUPPORTS,
                                       dst=a.object_id, confidence=conf,
                                       provenance=_derived(conf)))

            # --- inside / contains: a inside container b ---
            if b.kind == NodeKind.CONTAINER and a.object_id != b.object_id:
                if _xy_contained(ba, bb) and ba[0][2] >= bb[0][2] - cfg.contact_tol and ba[1][2] <= bb[1][2] + cfg.contact_tol:
                    edges.append(SceneEdge(src=a.object_id, rel=RelationType.INSIDE,
                                           dst=b.object_id, confidence=0.9,
                                           provenance=_derived(0.9)))
                    edges.append(SceneEdge(src=b.object_id, rel=RelationType.CONTAINS,
                                           dst=a.object_id, confidence=0.9,
                                           provenance=_derived(0.9)))

    # --- symmetric relations: near / next_to / aligned_with (unordered pairs) ---
    for i in range(len(phys)):
        for j in range(i + 1, len(phys)):
            a, b = phys[i], phys[j]
            dist = _centre_dist_xy(a, b)
            if dist <= cfg.near_dist:
                conf = max(0.0, 1.0 - dist / cfg.near_dist)
                edges.append(SceneEdge(src=a.object_id, rel=RelationType.NEAR, dst=b.object_id,
                                       confidence=conf, attributes={"dist_xy": round(dist, 3)},
                                       provenance=_derived(conf)))
                # next_to = near AND roughly same height AND not stacked
                same_level = abs(a.pose.z - b.pose.z) <= cfg.next_to_height_tol
                if same_level:
                    edges.append(SceneEdge(src=a.object_id, rel=RelationType.NEXT_TO, dst=b.object_id,
                                           confidence=conf, provenance=_derived(conf)))
            # aligned_with: centres share an x or y line
            if abs(a.pose.x - b.pose.x) <= cfg.aligned_tol or abs(a.pose.y - b.pose.y) <= cfg.aligned_tol:
                edges.append(SceneEdge(src=a.object_id, rel=RelationType.ALIGNED_WITH, dst=b.object_id,
                                       confidence=0.8, provenance=_derived(0.8)))

    return edges
