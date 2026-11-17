"""Phase 5 — LLM planner.

Turns (instruction + retrieved subgraph + knowledge + GNN/heuristic scores +
robot/safety constraints) into a STRICT `ActionPlan` JSON object. The LLM is
constrained and never trusted as the sole authority:
- it may only reference node ids present in the retrieved subgraph,
- output is parsed + validated against the pydantic schema,
- a repair/retry loop fixes malformed JSON,
- on any unresolved failure the planner returns a safe `ask_human` fallback.

Deterministic safety validation (kinematics/collision) is Phase 6; this phase
guarantees schema-validity and id-grounding only.
"""

from __future__ import annotations

from r4r.planner.context import PlannerContext, RobotConstraints, build_context
from r4r.planner.backends import LLMBackend, MockBackend, OllamaChatBackend
from r4r.planner.planner import Planner, PlanResult

__all__ = [
    "PlannerContext",
    "RobotConstraints",
    "build_context",
    "LLMBackend",
    "MockBackend",
    "OllamaChatBackend",
    "Planner",
    "PlanResult",
]
