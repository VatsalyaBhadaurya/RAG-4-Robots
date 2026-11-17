"""Scene-graph edge schema.

Edges are typed relations. Spatial/containment/support relations are usually
DERIVED by the spatial module; procedural/constraint relations are usually
STATIC knowledge. Every edge carries provenance + confidence so retrieval can
filter by relation type and trust.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field

from r4r.schemas.provenance import Provenance


class RelationType(str, Enum):
    """Edge relation vocabulary.

    Spatial/physical (mostly derived from geometry):
    """

    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"
    INSIDE = "inside"
    CONTAINS = "contains"
    SUPPORTS = "supports"
    REACHABLE_FROM = "reachable_from"
    NEAR = "near"
    ALIGNED_WITH = "aligned_with"
    OCCLUDES = "occludes"
    # procedural / knowledge relations (static RAG)
    REQUIRES_TOOL = "requires_tool"
    GOVERNED_BY = "governed_by"      # object/action -> safety_rule / constraint
    STEP_OF = "step_of"              # procedure step ordering
    AFFORDS = "affords"              # object -> affordance/action template


# Relations whose (src, dst) order is semantically reversible-with-rename.
INVERSE_RELATIONS = {
    RelationType.ON_TOP_OF: RelationType.SUPPORTS,
    RelationType.SUPPORTS: RelationType.ON_TOP_OF,
    RelationType.INSIDE: RelationType.CONTAINS,
    RelationType.CONTAINS: RelationType.INSIDE,
}

# Symmetric relations (A rel B implies B rel A).
SYMMETRIC_RELATIONS = {
    RelationType.NEXT_TO,
    RelationType.NEAR,
    RelationType.ALIGNED_WITH,
}


class SceneEdge(BaseModel):
    """A directed, typed relation src --rel--> dst."""

    src: str = Field(..., min_length=1)
    rel: RelationType
    dst: str = Field(..., min_length=1)
    attributes: Dict[str, object] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    provenance: Provenance

    def key(self) -> str:
        """Stable key for a (src, rel, dst) triple, used as the MultiDiGraph key."""
        return f"{self.rel.value}"
