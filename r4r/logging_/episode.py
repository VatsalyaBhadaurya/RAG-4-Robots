"""EpisodeLog schema — the auditable record of one instruction end to end.

Captures every required field: instruction, retrieved-subgraph hash/summary,
GNN/heuristic inputs+outputs, LLM plan, validator result, executed actions,
sensor observations (post-exec poses), success/failure + failure category, and
human corrections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeLog(BaseModel):
    episode_id: str
    created_at: str = Field(default_factory=_now)
    instruction: str

    # retrieval
    subgraph_hash: str = ""
    subgraph_summary: Dict[str, int] = Field(default_factory=dict)  # nodes/edges counts
    retrieval_seeds: List[str] = Field(default_factory=list)
    knowledge_ids: List[str] = Field(default_factory=list)
    token_estimate: int = 0
    full_graph_token_estimate: int = 0

    # scoring (heuristic/GNN in+out)
    scores: Dict[str, object] = Field(default_factory=dict)

    # planning
    plan: Dict[str, object] = Field(default_factory=dict)
    plan_valid: bool = False
    plan_used_fallback: bool = False
    plan_attempts: int = 0
    planner_backend: str = ""

    # validation
    validation_ok: bool = False
    validation_violations: List[Dict[str, object]] = Field(default_factory=list)

    # execution (sensor observations = read-back poses)
    executed: bool = False
    execution_success: bool = False
    execution_notes: List[str] = Field(default_factory=list)
    final_poses: Dict[str, List[float]] = Field(default_factory=dict)

    # outcome
    success: bool = False
    failure_category: Optional[str] = None  # None iff success
    human_corrections: List[str] = Field(default_factory=list)
