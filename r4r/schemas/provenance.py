"""Provenance: where every fact came from.

Non-negotiable requirement: every retrieved node/edge carries provenance so the
planner (and any audit) can tell live perception from static knowledge, and can
weigh confidence and staleness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Whether a fact is live perception or static knowledge."""

    PERCEPTION = "perception"   # live sensor-derived observation
    STATIC = "static"           # manuals, APIs, safety rules, task templates
    DERIVED = "derived"         # computed by us (e.g. a spatial edge heuristic)
    HUMAN = "human"             # human-provided correction/annotation


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Provenance(BaseModel):
    """Audit trail for a single fact (node or edge).

    confidence is in [0, 1]. is_live is a convenience derived from source_type
    but stored explicitly so serialized subgraphs are self-describing.
    """

    source_type: SourceType
    source_id: str = Field(..., description="e.g. 'cam0', 'manual:dishwasher', 'spatial_heuristic'")
    observed_at: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    @property
    def is_live(self) -> bool:
        return self.source_type == SourceType.PERCEPTION

    @staticmethod
    def derived(source_id: str, confidence: float = 1.0,
                observed_at: Optional[datetime] = None) -> "Provenance":
        """Convenience constructor for facts we compute ourselves."""
        return Provenance(
            source_type=SourceType.DERIVED,
            source_id=source_id,
            confidence=confidence,
            observed_at=observed_at or _utcnow(),
        )
