"""Data contracts for the r4r system (pydantic v2).

These schemas are the single source of truth passed between modules:
perception -> scene graph -> retrieval -> scoring -> planner -> validator.
Keeping them strict (and serializable) is what makes every retrieved fact
auditable and every plan machine-checkable.
"""

from __future__ import annotations

from r4r.schemas.provenance import Provenance, SourceType
from r4r.schemas.nodes import Dimensions, NodeKind, Pose, SceneNode
from r4r.schemas.edges import RelationType, SceneEdge
from r4r.schemas.plan import ActionPlan, ActionStep, PlanFallback

__all__ = [
    "Provenance",
    "SourceType",
    "Dimensions",
    "NodeKind",
    "Pose",
    "SceneNode",
    "RelationType",
    "SceneEdge",
    "ActionPlan",
    "ActionStep",
    "PlanFallback",
]
