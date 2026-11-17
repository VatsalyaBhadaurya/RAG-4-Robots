"""Phase 6 — deterministic plan validation + safety guard.

This is the authority the LLM is NOT: a plan only executes if PlanValidator
returns ok (no ERROR violations). Checks are deterministic and explainable:
target existence/visibility, id grounding, reachability, collision risk,
payload, gripper fit, fragile-handling, human proximity, and action-sequence
sanity. `safety_violation_rate` must be 0 for execution.
"""

from __future__ import annotations

from r4r.validation.result import Severity, ValidationResult, Violation
from r4r.validation.validator import PlanValidator, ValidationConfig

__all__ = [
    "Severity",
    "ValidationResult",
    "Violation",
    "PlanValidator",
    "ValidationConfig",
]
