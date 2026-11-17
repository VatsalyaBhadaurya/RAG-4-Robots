"""Deterministic plan validator + safety guard.

Checks an ActionPlan against the live scene graph, robot constraints, and the
heuristic score report. Produces ERROR violations (block execution) or WARNING
violations (surface + log). This runs AFTER the planner and BEFORE any execution
— the LLM is never trusted to self-certify safety.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional

from pydantic import BaseModel, Field

from r4r.planner.context import RobotConstraints
from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import NodeKind
from r4r.schemas.plan import ActionPlan, ActionType
from r4r.scoring.report import ScoreReport
from r4r.validation.result import Severity, ValidationResult, Violation

# Rough solid densities (kg/m^3). Used for a CONSERVATIVE (over-)estimate of
# mass for payload checks — real objects are often hollow, so over-estimating is
# the safe direction. Unknown material -> uncertain (warning, not error).
_DENSITY: Dict[str, float] = {
    "ceramic": 2300.0, "glass": 2500.0, "aluminium": 2700.0, "metal": 7800.0,
    "metal_plastic": 3000.0, "wood": 650.0, "cardboard": 700.0, "plastic": 950.0,
}
_FRAGILE = {"ceramic", "glass"}
_SYNTHETIC = ("pregrasp", "approach", "_slot", "_rack", "pose_", "home")


class ValidationConfig(BaseModel):
    min_visibility: float = 0.3
    min_confidence: float = 0.3
    reach_threshold: float = Field(0.2, description="min reachability score")
    risk_threshold: float = Field(0.8, description="max collision risk")


def _looks_synthetic(t: str) -> bool:
    tl = t.lower()
    return tl.startswith(("pregrasp_", "approach_", "pose_", "home")) or any(s in tl for s in _SYNTHETIC)


def _dist3(ax, ay, az, bx, by, bz) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


class PlanValidator:
    def __init__(self, cfg: Optional[ValidationConfig] = None) -> None:
        self.cfg = cfg or ValidationConfig()

    def validate(
        self,
        plan: ActionPlan,
        graph: SceneGraph,
        constraints: Optional[RobotConstraints] = None,
        score_report: Optional[ScoreReport] = None,
    ) -> ValidationResult:
        res = ValidationResult()
        constraints = constraints or RobotConstraints.from_graph(graph)
        robot = next(iter(graph.nodes_by_kind(NodeKind.ROBOT)), None)
        reach_r = constraints.workspace_radius or 0.85

        # 0. refusal / empty plan: not executable (not a safety violation)
        res.checked.append("non_empty")
        if not plan.steps:
            res.add(Violation(code="no_steps", severity=Severity.ERROR,
                              message="plan has no executable steps (refusal/fallback)"))

        # 1. id grounding + existence
        res.checked.append("grounding")
        for label, oid in (("selected_target", plan.selected_target),
                           ("destination", plan.destination)):
            if oid is None:
                continue
            if not graph.has_node(oid):
                res.add(Violation(code="unknown_object", severity=Severity.ERROR,
                                  message=f"{label} '{oid}' does not exist in the scene graph",
                                  subject=oid))
        for i, step in enumerate(plan.steps):
            t = step.target
            if graph.has_node(t) or _looks_synthetic(t) or (plan.selected_target and plan.selected_target in t) \
                    or (plan.destination and plan.destination in t):
                continue
            res.add(Violation(code="ungrounded_step_target", severity=Severity.ERROR,
                              message=f"step target '{t}' is neither a known id nor a recognised pose",
                              step_index=i, subject=t))

        # 2. visibility / confidence of the real objects involved
        res.checked.append("perceptible")
        for oid in [x for x in (plan.selected_target, plan.destination) if x and graph.has_node(x)]:
            n = graph.get_node(oid)
            if n.visibility < self.cfg.min_visibility:
                res.add(Violation(code="low_visibility", severity=Severity.ERROR,
                                  message=f"'{oid}' visibility {n.visibility:.2f} < {self.cfg.min_visibility}",
                                  subject=oid))
            if n.confidence < self.cfg.min_confidence:
                res.add(Violation(code="low_confidence", severity=Severity.ERROR,
                                  message=f"'{oid}' confidence {n.confidence:.2f} < {self.cfg.min_confidence}",
                                  subject=oid))

        # 3. reachability of target + destination
        res.checked.append("reachability")
        if robot is not None:
            for oid in [x for x in (plan.selected_target, plan.destination) if x and graph.has_node(x)]:
                n = graph.get_node(oid)
                if n.pose is None:
                    continue
                d = _dist3(n.pose.x, n.pose.y, n.pose.z, robot.pose.x, robot.pose.y, robot.pose.z)
                reach = max(0.0, 1.0 - d / reach_r) if reach_r > 0 else 0.0
                if reach < self.cfg.reach_threshold:
                    res.add(Violation(code="not_reachable", severity=Severity.ERROR,
                                      message=f"'{oid}' reachability {reach:.2f} < {self.cfg.reach_threshold} "
                                              f"(dist {d:.2f}m > radius {reach_r:.2f}m)",
                                      subject=oid))

        # 4. grasp feasibility (gripper fit, collision risk, fragile) for the target
        res.checked.append("grasp_feasibility")
        tgt = plan.selected_target
        if tgt and graph.has_node(tgt):
            n = graph.get_node(tgt)
            if n.dims is not None and constraints.gripper_max_width is not None:
                min_dim = min(n.dims.x, n.dims.y, n.dims.z)
                if min_dim > constraints.gripper_max_width:
                    res.add(Violation(code="gripper_too_small", severity=Severity.ERROR,
                                      message=f"'{tgt}' min dim {min_dim:.3f}m > gripper {constraints.gripper_max_width:.3f}m",
                                      subject=tgt))
            if score_report is not None and tgt in score_report.scores:
                s = score_report.for_object(tgt)
                if s.collision_risk > self.cfg.risk_threshold:
                    res.add(Violation(code="collision_risk", severity=Severity.ERROR,
                                      message=f"'{tgt}' collision_risk {s.collision_risk:.2f} > {self.cfg.risk_threshold}",
                                      subject=tgt))
                if s.uncertain:
                    res.add(Violation(code="uncertain_scores", severity=Severity.WARNING,
                                      message=f"'{tgt}' scored with missing features (uncertain)", subject=tgt))
            # fragile handling: grasp step should use low force
            if (n.material or "").lower() in _FRAGILE:
                grasp_steps = [st for st in plan.steps if st.action == ActionType.GRASP and tgt in st.target]
                if grasp_steps and not any((st.max_force or "").lower() == "low" for st in grasp_steps):
                    res.add(Violation(code="fragile_force", severity=Severity.WARNING,
                                      message=f"'{tgt}' is fragile ({n.material}); grasp should use low force",
                                      subject=tgt))

        # 5. payload (conservative mass estimate)
        res.checked.append("payload")
        if tgt and graph.has_node(tgt) and constraints.payload_kg is not None:
            n = graph.get_node(tgt)
            if n.dims is not None:
                dens = _DENSITY.get((n.material or "").lower())
                if dens is None:
                    res.add(Violation(code="unknown_mass", severity=Severity.WARNING,
                                      message=f"'{tgt}' material unknown; cannot verify payload", subject=tgt))
                else:
                    mass = n.dims.x * n.dims.y * n.dims.z * dens
                    if mass > constraints.payload_kg:
                        res.add(Violation(code="overweight", severity=Severity.ERROR,
                                          message=f"'{tgt}' est. mass {mass:.2f}kg > payload {constraints.payload_kg:.2f}kg "
                                                  f"(conservative solid estimate)", subject=tgt))

        # 6. human proximity geofence
        res.checked.append("human_proximity")
        if robot is not None:
            for n in graph.nodes():
                is_human = n.label.lower() == "human" or bool(n.attributes.get("is_human"))
                if is_human and n.pose is not None:
                    d = _dist3(n.pose.x, n.pose.y, n.pose.z, robot.pose.x, robot.pose.y, robot.pose.z)
                    if d <= reach_r:
                        res.add(Violation(code="human_in_workspace", severity=Severity.ERROR,
                                          message=f"human '{n.object_id}' within workspace ({d:.2f}m) — stop and confirm",
                                          subject=n.object_id))

        # 7. action-sequence sanity
        res.checked.append("sequence")
        actions = [s.action for s in plan.steps]
        holding = False
        for i, a in enumerate(actions):
            if a == ActionType.GRASP:
                holding = True
            elif a in (ActionType.PLACE, ActionType.RELEASE):
                if not holding:
                    res.add(Violation(code="place_without_grasp", severity=Severity.ERROR,
                                      message=f"step {i} '{a.value}' with nothing grasped", step_index=i))
                holding = False
        if plan.destination and ActionType.GRASP not in actions:
            res.add(Violation(code="missing_grasp", severity=Severity.ERROR,
                              message="placement task has no grasp step"))

        return res
