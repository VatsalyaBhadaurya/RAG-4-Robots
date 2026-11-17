"""Prompt construction for the planner.

The system prompt pins the safety/grounding rules; the user prompt carries the
compact context as JSON. We also pass the ActionPlan JSON schema to the backend
for structured decoding, but repeat the key constraints in text because local
models follow explicit instructions more reliably than schema alone.
"""

from __future__ import annotations

import json
from typing import List

from r4r.planner.context import PlannerContext

SYSTEM_PROMPT = """You are a robot task planner. You convert a natural-language \
instruction into a STRICT JSON action plan for a single-arm manipulator.

HARD RULES (violating any of these makes the plan invalid):
1. Output ONLY a single JSON object matching the ActionPlan schema. No prose, no \
markdown fences.
2. You may ONLY reference object ids that appear in `allowed_ids`. NEVER invent \
object ids, poses, or capabilities. `selected_target` and `destination` MUST be \
ids from `allowed_ids` (or null if not applicable).
3. Use ONLY these actions: move_to, grasp, place, release, open, close, push, \
inspect, wait.
4. Respect the robot constraints (reachability, payload, gripper width/force) and \
the safety rules in `knowledge`. Use the provided `scores` (reachability, \
graspability, collision_risk, support_stability) to choose safe targets.
5. If the task is ambiguous, unsafe, references an object not in `allowed_ids`, or \
cannot be done within constraints: do NOT guess. Return a plan with empty `steps`, \
explain in `risks`, list what you need in `required_confirmations`, and set \
`fallback` to "ask_human".
6. Record any premise you relied on in `assumptions`.

ActionPlan schema (fields): task (str), assumptions (str[]), selected_target \
(str|null), destination (str|null), steps (list of {action, target, grasp_type?, \
max_force?, constraints?[]}), risks (str[]), required_confirmations (str[]), \
fallback (one of: ask_human, abort, hold_position, retry_perception)."""


def build_messages(ctx: PlannerContext) -> List[dict]:
    user = {
        "instruction": ctx.instruction,
        "allowed_ids": ctx.allowed_ids,
        "robot_constraints": ctx.constraints.model_dump(exclude_none=True),
        "scores": ctx.scores,
        "scene_subgraph": ctx.scene_subgraph,
        "knowledge": ctx.knowledge,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"), default=str)},
    ]


def repair_message(errors: str) -> dict:
    return {
        "role": "user",
        "content": (
            "Your previous output was invalid:\n" + errors +
            "\nReturn ONLY a corrected JSON object matching the ActionPlan schema. "
            "Reference only ids from allowed_ids. No prose."
        ),
    }
