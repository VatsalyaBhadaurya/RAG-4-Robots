"""Strict action-plan schema (the planner's ONLY legal output).

Defined now (Phase 1/2) to anchor the whole pipeline even though the LLM
planner is built in Phase 5. The validator (Phase 6) checks a parsed ActionPlan
against the live scene graph and robot constraints BEFORE anything executes.
The LLM is never the sole safety authority — this object is what gets validated.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PlanFallback(str, Enum):
    """Safe behaviours when the plan cannot proceed."""

    ASK_HUMAN = "ask_human"
    ABORT = "abort"
    HOLD_POSITION = "hold_position"
    RETRY_PERCEPTION = "retry_perception"


class ActionType(str, Enum):
    """Whitelisted high-level actions. The validator rejects anything else.

    These are symbolic/high-level; low-level motor commands are produced only by
    the (separately validated) controller, never by the LLM.
    """

    MOVE_TO = "move_to"
    GRASP = "grasp"
    PLACE = "place"
    RELEASE = "release"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    INSPECT = "inspect"
    WAIT = "wait"


class ActionStep(BaseModel):
    """A single symbolic step. `target` must resolve to a real node id in the
    scene graph (validator enforces this — no fabricated ids)."""

    action: ActionType
    target: str = Field(..., min_length=1)
    grasp_type: Optional[str] = None            # e.g. 'side', 'top'
    max_force: Optional[str] = None             # e.g. 'low', 'medium'
    constraints: List[str] = Field(default_factory=list)  # e.g. 'collision_free'
    params: Dict[str, object] = Field(default_factory=dict)


class ActionPlan(BaseModel):
    """The machine-readable plan. Mirrors the project's reference schema."""

    task: str = Field(..., min_length=1)
    assumptions: List[str] = Field(default_factory=list)
    selected_target: Optional[str] = None
    destination: Optional[str] = None
    steps: List[ActionStep] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    required_confirmations: List[str] = Field(default_factory=list)
    fallback: PlanFallback = PlanFallback.ASK_HUMAN

    def referenced_ids(self) -> List[str]:
        """All node ids the plan touches — used by the validator to check that
        every referenced entity actually exists and is visible."""
        ids: List[str] = []
        if self.selected_target:
            ids.append(self.selected_target)
        if self.destination:
            ids.append(self.destination)
        ids.extend(s.target for s in self.steps)
        # de-dup, preserve order
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out
