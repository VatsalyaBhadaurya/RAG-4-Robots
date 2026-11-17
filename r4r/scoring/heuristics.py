"""Deterministic heuristic scorers (the baseline a GNN must beat).

Each scorer is a transparent geometric/kinematic rule with an explanation:
- reachability: object within the robot's workspace sphere (robot base + radius).
- graspability: object's smallest cross-section fits the gripper opening AND the
  object carries a graspable affordance.
- collision_risk: small clearance to neighbouring objects, or supporting a stack.
- support_stability: whether grasping the object would disturb objects resting
  on it.

These are evidence-grounded approximations, NOT learned and NOT ground truth;
they are replaced/augmented by MuJoCo geometry (Phase 6) and possibly a GNN once
labeled data exists. Every number comes with an explanation and an `uncertain`
flag when inputs are missing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.edges import RelationType
from r4r.schemas.nodes import NodeKind, SceneNode
from r4r.scoring.report import ObjectScores, ScoreReport

_Box = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class HeuristicConfig(BaseModel):
    reach_radius_default: float = Field(0.85, description="fallback workspace radius (m)")
    gripper_width_default: float = Field(0.09, description="fallback gripper opening (m)")
    clearance_ref: float = Field(0.10, description="clearance (m) at/above which collision risk ~0")
    reach_threshold: float = Field(0.2, description="min reachability score to be graspable")
    grasp_threshold: float = Field(0.5, description="min graspability score to be graspable")
    risk_threshold: float = Field(0.8, description="max collision risk to be graspable")


def _aabb(n: SceneNode) -> Optional[_Box]:
    if n.pose is None or n.dims is None:
        return None
    cx, cy, cz = n.pose.x, n.pose.y, n.pose.z
    hx, hy, hz = n.dims.half()
    return ((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))


def _dist3(a: SceneNode, b: SceneNode) -> float:
    return math.sqrt((a.pose.x - b.pose.x) ** 2 + (a.pose.y - b.pose.y) ** 2
                     + (a.pose.z - b.pose.z) ** 2)


def _gap(a: _Box, b: _Box) -> float:
    """Separation between two AABBs (0 if touching, negative if overlapping)."""
    dx = max(a[0][0] - b[1][0], b[0][0] - a[1][0])
    dy = max(a[0][1] - b[1][1], b[0][1] - a[1][1])
    dz = max(a[0][2] - b[1][2], b[0][2] - a[1][2])
    # if separated on any axis, distance is along the positive gaps
    pos = [v for v in (dx, dy, dz) if v > 0]
    if pos:
        return math.sqrt(sum(v * v for v in pos))
    return max(dx, dy, dz)  # all <=0 -> overlapping; return least-negative


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _find_one(graph: SceneGraph, kind: NodeKind) -> Optional[SceneNode]:
    nodes = graph.nodes_by_kind(kind)
    return nodes[0] if nodes else None


def score_scene(graph: SceneGraph, cfg: Optional[HeuristicConfig] = None) -> ScoreReport:
    cfg = cfg or HeuristicConfig()
    robot = _find_one(graph, NodeKind.ROBOT)
    ee = _find_one(graph, NodeKind.END_EFFECTOR)

    gripper_w = cfg.gripper_width_default
    if ee is not None and isinstance(ee.attributes.get("max_width"), (int, float)):
        gripper_w = float(ee.attributes["max_width"])
    reach_r = cfg.reach_radius_default
    if robot is not None and isinstance(robot.attributes.get("workspace_radius"), (int, float)):
        reach_r = float(robot.attributes["workspace_radius"])

    # all physical objects with geometry are collision candidates
    phys = [n for n in graph.nodes()
            if n.kind in (NodeKind.OBJECT, NodeKind.CONTAINER, NodeKind.SURFACE)
            and _aabb(n) is not None]
    boxes = {n.object_id: _aabb(n) for n in phys}

    report = ScoreReport(method="heuristic_v1")
    for obj in graph.nodes_by_kind(NodeKind.OBJECT):
        why: List[str] = []
        uncertain = False

        # --- reachability ---
        if robot is not None and obj.pose is not None:
            d = _dist3(obj, robot)
            reach = _clamp(1.0 - d / reach_r) if reach_r > 0 else 0.0
            why.append(f"reach: dist {d:.2f}m vs radius {reach_r:.2f}m -> {reach:.2f}")
        else:
            reach, uncertain = 0.5, True
            why.append("reach: robot pose or object pose missing -> uncertain")

        # --- graspability ---
        if obj.dims is not None:
            min_dim = min(obj.dims.x, obj.dims.y, obj.dims.z)
            width_fit = 1.0 if min_dim <= gripper_w else 0.0
            has_aff = any(a.startswith("graspable") for a in obj.affordances)
            aff_factor = 1.0 if has_aff else 0.5
            grasp = _clamp(width_fit * aff_factor)
            why.append(f"grasp: min_dim {min_dim:.3f}m vs gripper {gripper_w:.3f}m "
                       f"(fit={width_fit:.0f}), affordance={'yes' if has_aff else 'no'} -> {grasp:.2f}")
        else:
            grasp, uncertain = 0.0, True
            why.append("grasp: object dimensions missing -> uncertain")

        # --- collision risk (nearest-neighbour clearance) ---
        risk = 0.0
        if obj.object_id in boxes:
            nearest = None
            min_gap = math.inf
            for other in phys:
                if other.object_id == obj.object_id or other.kind == NodeKind.SURFACE:
                    continue
                g = _gap(boxes[obj.object_id], boxes[other.object_id])
                if g < min_gap:
                    min_gap, nearest = g, other.object_id
            if nearest is not None:
                risk = _clamp(1.0 - min_gap / cfg.clearance_ref)
                why.append(f"collision: clearance {min_gap:.3f}m to {nearest} -> risk {risk:.2f}")
        # stacking: grasping an object that supports others is risky
        supported = [e.dst for e in graph.out_edges(obj.object_id) if e.rel == RelationType.SUPPORTS]
        if supported:
            risk = max(risk, 0.7)
            why.append(f"collision: supports {supported} (stack) -> risk >= 0.70")

        # --- support stability (is it safe to remove?) ---
        stability = 1.0 - 0.5 * min(1.0, len(supported))
        if supported:
            why.append(f"support: carrying {len(supported)} object(s) -> stability {stability:.2f}")

        graspable = (reach >= cfg.reach_threshold
                     and grasp >= cfg.grasp_threshold
                     and risk <= cfg.risk_threshold
                     and not uncertain)

        report.scores[obj.object_id] = ObjectScores(
            object_id=obj.object_id,
            reachability=round(reach, 3),
            graspability=round(grasp, 3),
            collision_risk=round(risk, 3),
            support_stability=round(stability, 3),
            graspable=graspable,
            uncertain=uncertain,
            explanations=why,
        )

    report.ranking = sorted(
        report.scores,
        key=lambda oid: report.scores[oid].composite(),
        reverse=True,
    )
    return report
