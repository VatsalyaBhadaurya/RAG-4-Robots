"""Planner: backend -> parse -> validate -> ground -> (repair) -> fallback.

Guarantees for the caller:
- `result.plan` is always a schema-valid ActionPlan (even the fallback is valid).
- `result.valid` is True only if the model produced a grounded plan within the
  retry budget; otherwise a safe `ask_human` fallback is returned with reasons.
- selected_target / destination are hard-grounded against allowed_ids (no
  fabricated object references); synthetic step targets (pregrasp_*, *_slot_*)
  are allowed but surfaced.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from r4r.planner.backends import LLMBackend
from r4r.planner.context import PlannerContext
from r4r.planner.prompt import build_messages, repair_message
from r4r.schemas.plan import ActionPlan, PlanFallback

_SYNTHETIC_PREFIXES = ("pregrasp_", "pre_grasp_", "pose_", "home", "approach_")
_SYNTHETIC_SUBSTR = ("_slot", "_rack", "pregrasp", "approach")


class PlanResult(BaseModel):
    plan: ActionPlan
    valid: bool
    used_fallback: bool
    attempts: int
    errors: List[str] = []
    raw: str = ""
    backend: str = ""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:48] or "task"


def _extract_json(content: str) -> str:
    """Strip markdown fences / prose and isolate the outermost JSON object."""
    c = content.strip()
    if c.startswith("```"):
        c = re.sub(r"^```[a-zA-Z]*\n?", "", c)
        c = re.sub(r"\n?```$", "", c).strip()
    start, end = c.find("{"), c.rfind("}")
    if start != -1 and end != -1 and end > start:
        return c[start:end + 1]
    return c


def _looks_synthetic(target: str) -> bool:
    t = target.lower()
    return t.startswith(_SYNTHETIC_PREFIXES) or any(s in t for s in _SYNTHETIC_SUBSTR)


def _ground(plan: ActionPlan, allowed: set) -> List[str]:
    """Hard grounding errors (block validity). Returns list of error strings."""
    errs: List[str] = []
    if plan.selected_target and plan.selected_target not in allowed:
        errs.append(f"selected_target '{plan.selected_target}' not in allowed_ids")
    if plan.destination and plan.destination not in allowed:
        errs.append(f"destination '{plan.destination}' not in allowed_ids")
    return errs


def _ungrounded_steps(plan: ActionPlan, allowed: set) -> List[str]:
    """Soft warnings: step targets that are neither known ids nor clearly synthetic."""
    warn = []
    for s in plan.steps:
        t = s.target
        if t in allowed or _looks_synthetic(t):
            continue
        # allow references that embed a known id, e.g. 'pregrasp_red_mug_1'
        if any(a in t for a in allowed):
            continue
        warn.append(f"step target '{t}' is not a known id or a recognised pose")
    return warn


def _safe_fallback(ctx: PlannerContext, reasons: List[str]) -> ActionPlan:
    return ActionPlan(
        task=_slug(ctx.instruction),
        assumptions=[],
        selected_target=None,
        destination=None,
        steps=[],
        risks=reasons or ["planner could not produce a grounded, valid plan"],
        required_confirmations=["human review required before execution"],
        fallback=PlanFallback.ASK_HUMAN,
    )


class Planner:
    def __init__(self, backend: LLMBackend, max_retries: int = 2) -> None:
        self.backend = backend
        self.max_retries = max_retries

    def plan(self, ctx: PlannerContext) -> PlanResult:
        allowed = set(ctx.allowed_ids)
        messages = build_messages(ctx)
        schema = ActionPlan.model_json_schema()
        last_raw = ""
        all_errors: List[str] = []

        for attempt in range(1, self.max_retries + 2):  # 1 initial + max_retries
            content = self.backend.chat(messages, schema=schema)
            last_raw = content
            try:
                obj = json.loads(_extract_json(content))
                plan = ActionPlan.model_validate(obj)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                err = f"attempt {attempt}: invalid JSON/schema: {exc}"
                all_errors.append(err)
                messages = messages + [repair_message(err)]
                continue

            ground_errs = _ground(plan, allowed)
            if ground_errs:
                err = f"attempt {attempt}: grounding: " + "; ".join(ground_errs)
                all_errors.append(err)
                messages = messages + [repair_message(err)]
                continue

            # valid + grounded. Attach soft step warnings as risks (non-blocking).
            warns = _ungrounded_steps(plan, allowed)
            if warns:
                plan = plan.model_copy(update={"risks": list(plan.risks) + warns})
            return PlanResult(plan=plan, valid=True, used_fallback=False,
                              attempts=attempt, errors=all_errors, raw=last_raw,
                              backend=self.backend.name)

        # exhausted retries -> safe fallback
        return PlanResult(
            plan=_safe_fallback(ctx, all_errors[-1:] if all_errors else []),
            valid=False, used_fallback=True,
            attempts=self.max_retries + 1, errors=all_errors, raw=last_raw,
            backend=self.backend.name,
        )
