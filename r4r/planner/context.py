"""Planner context assembly.

Collects everything the LLM is allowed to see into one structured object, and
computes the `allowed_ids` whitelist used to reject fabricated references.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import NodeKind
from r4r.scoring.report import ScoreReport


class RobotConstraints(BaseModel):
    """Capability/safety envelope extracted from the robot + end-effector nodes."""

    workspace_radius: Optional[float] = None
    payload_kg: Optional[float] = None
    gripper_max_width: Optional[float] = None
    gripper_max_force_n: Optional[float] = None
    allowed_actions: List[str] = Field(default_factory=list)

    @staticmethod
    def from_graph(graph: SceneGraph) -> "RobotConstraints":
        robot = next(iter(graph.nodes_by_kind(NodeKind.ROBOT)), None)
        ee = next(iter(graph.nodes_by_kind(NodeKind.END_EFFECTOR)), None)
        rc = RobotConstraints(allowed_actions=[
            "move_to", "grasp", "place", "release", "open", "close",
            "push", "inspect", "wait",
        ])
        if robot is not None:
            wr = robot.attributes.get("workspace_radius")
            pl = robot.attributes.get("payload_kg")
            rc.workspace_radius = float(wr) if isinstance(wr, (int, float)) else None
            rc.payload_kg = float(pl) if isinstance(pl, (int, float)) else None
        if ee is not None:
            mw = ee.attributes.get("max_width")
            mf = ee.attributes.get("max_force_n")
            rc.gripper_max_width = float(mw) if isinstance(mw, (int, float)) else None
            rc.gripper_max_force_n = float(mf) if isinstance(mf, (int, float)) else None
        return rc


class PlannerContext(BaseModel):
    """Everything handed to the LLM planner (compact + auditable)."""

    instruction: str
    scene_subgraph: Dict[str, object]
    knowledge: List[Dict[str, object]] = Field(default_factory=list)
    scores: Dict[str, object] = Field(default_factory=dict)
    constraints: RobotConstraints = Field(default_factory=RobotConstraints)
    allowed_ids: List[str] = Field(default_factory=list)


def build_context(
    instruction: str,
    serialized_subgraph: Dict[str, object],
    knowledge_chunks: List[Dict[str, object]],
    score_report: Optional[ScoreReport],
    graph: SceneGraph,
    allowed_ids: Optional[List[str]] = None,
) -> PlannerContext:
    """Assemble a PlannerContext. `allowed_ids` defaults to the ids present in
    the serialized subgraph (the only entities the plan may reference)."""
    if allowed_ids is None:
        allowed_ids = [n["id"] for n in serialized_subgraph.get("nodes", [])]
    return PlannerContext(
        instruction=instruction,
        scene_subgraph=serialized_subgraph,
        knowledge=knowledge_chunks,
        scores=(score_report.compact() if score_report else {}),
        constraints=RobotConstraints.from_graph(graph),
        allowed_ids=list(allowed_ids),
    )
