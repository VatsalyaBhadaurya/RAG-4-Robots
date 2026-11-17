"""Scoring output schema fed to the LLM planner and the Phase 6 validator.

Scores are in [0, 1]. collision_risk is a RISK (higher = worse); the others are
desirability (higher = better). `uncertain` marks scores computed with missing
features (e.g. no dimensions) so the planner/validator can treat them cautiously
rather than trusting a fabricated number. `explanations` make every score
auditable — a requirement carried over from provenance.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ObjectScores(BaseModel):
    object_id: str
    reachability: float = Field(..., ge=0.0, le=1.0)
    graspability: float = Field(..., ge=0.0, le=1.0)
    collision_risk: float = Field(..., ge=0.0, le=1.0)   # higher = riskier
    support_stability: float = Field(..., ge=0.0, le=1.0)
    graspable: bool = False          # thresholded decision
    uncertain: bool = False          # computed with missing/low-confidence features
    explanations: List[str] = Field(default_factory=list)

    def composite(self) -> float:
        """Desirability of grasping this object now (used for ranking)."""
        return (self.graspability
                * self.reachability
                * (1.0 - self.collision_risk)
                * self.support_stability)


class ScoreReport(BaseModel):
    method: str = "heuristic_v1"
    scores: Dict[str, ObjectScores] = Field(default_factory=dict)
    ranking: List[str] = Field(default_factory=list)  # object ids, best first

    def best(self) -> str:
        return self.ranking[0] if self.ranking else ""

    def for_object(self, object_id: str) -> ObjectScores:
        return self.scores[object_id]

    def compact(self) -> Dict[str, object]:
        """Compact form to include in the LLM planner context."""
        return {
            "method": self.method,
            "ranking": self.ranking,
            "scores": {
                oid: {
                    "reach": round(s.reachability, 2),
                    "grasp": round(s.graspability, 2),
                    "collision_risk": round(s.collision_risk, 2),
                    "support_stability": round(s.support_stability, 2),
                    "graspable": s.graspable,
                    "uncertain": s.uncertain,
                    "why": s.explanations,
                }
                for oid, s in self.scores.items()
            },
        }
