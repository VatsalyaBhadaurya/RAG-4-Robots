"""Scene-graph node schema.

A node is any entity in the heterogeneous graph: physical objects, surfaces,
containers, the robot and its end effector, rooms, and task goals. Knowledge
nodes (procedures, APIs, safety rules) reuse the same container via NodeKind so
the GraphRAG retriever can traverse scene and knowledge uniformly.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from r4r.schemas.provenance import Provenance


class NodeKind(str, Enum):
    """Heterogeneous node types. Drives GNN node-type embeddings later."""

    OBJECT = "object"
    SURFACE = "surface"          # table, shelf, floor
    CONTAINER = "container"      # bin, box, dishwasher, drawer
    ROBOT = "robot"
    END_EFFECTOR = "end_effector"
    ROOM = "room"
    TASK_GOAL = "task_goal"
    # knowledge-side kinds (static RAG)
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    TOOL = "tool"
    API = "api"
    SAFETY_RULE = "safety_rule"
    DEMONSTRATION = "demonstration"
    TASK_TEMPLATE = "task_template"


# Kinds that represent live physical entities (have a pose in the world frame).
PHYSICAL_KINDS = {
    NodeKind.OBJECT,
    NodeKind.SURFACE,
    NodeKind.CONTAINER,
    NodeKind.ROBOT,
    NodeKind.END_EFFECTOR,
}

# Kinds that represent static knowledge (retrieved as RAG chunks, no pose).
KNOWLEDGE_KINDS = {
    NodeKind.PROCEDURE,
    NodeKind.CONSTRAINT,
    NodeKind.TOOL,
    NodeKind.API,
    NodeKind.SAFETY_RULE,
    NodeKind.DEMONSTRATION,
    NodeKind.TASK_TEMPLATE,
}


class Pose(BaseModel):
    """6-DoF pose in a named reference frame. Quaternion is (w, x, y, z)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    frame: str = Field("world", description="reference frame id")

    def position(self) -> List[float]:
        return [self.x, self.y, self.z]


class Dimensions(BaseModel):
    """Axis-aligned bounding-box extents (metres), full side lengths."""

    x: float = Field(..., gt=0.0)
    y: float = Field(..., gt=0.0)
    z: float = Field(..., gt=0.0)

    def half(self) -> List[float]:
        return [self.x / 2.0, self.y / 2.0, self.z / 2.0]


class SceneNode(BaseModel):
    """A single node. `attributes` is an open bag for kind-specific metadata
    (e.g. color, grasp_templates, api_signature) that we don't want to hardcode
    yet. `provenance` is required so every node is auditable."""

    object_id: str = Field(..., min_length=1)
    kind: NodeKind
    label: str = Field(..., description="class/label, e.g. 'mug', 'dishwasher'")
    pose: Optional[Pose] = None
    dims: Optional[Dimensions] = None
    material: Optional[str] = None
    visibility: float = Field(1.0, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    affordances: List[str] = Field(default_factory=list)
    attributes: Dict[str, object] = Field(default_factory=dict)
    provenance: Provenance

    def is_physical(self) -> bool:
        return self.kind in PHYSICAL_KINDS

    def requires_pose(self) -> bool:
        """Physical nodes should carry a pose; knowledge nodes need not."""
        return self.is_physical()
