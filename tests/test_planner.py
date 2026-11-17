"""Planner tests: parse/validate/repair/ground/fallback (offline MockBackend)."""

from __future__ import annotations

import json

from r4r.embeddings import HashEmbedder
from r4r.eval.benchmark import build_eval_graph
from r4r.planner import MockBackend, Planner, build_context
from r4r.retrieval import GraphRAGRetriever
from r4r.schemas.plan import PlanFallback
from r4r.scoring import score_scene


def _ctx(instruction="Put the red mug in the bin."):
    graph = build_eval_graph()
    retriever = GraphRAGRetriever(graph, HashEmbedder(dim=512)).build()
    res = retriever.retrieve(instruction, k_entities=4, depth=1)
    report = score_scene(graph)
    return build_context(instruction, res.serialized_subgraph,
                         res.knowledge_chunks, report, graph)


_VALID_PLAN = json.dumps({
    "task": "put_red_mug_in_bin",
    "assumptions": [],
    "selected_target": "red_mug_1",
    "destination": "bin_1",
    "steps": [
        {"action": "move_to", "target": "pregrasp_red_mug_1", "constraints": ["collision_free"]},
        {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
        {"action": "place", "target": "bin_1", "constraints": ["gentle_place"]},
    ],
    "risks": [], "required_confirmations": [], "fallback": "ask_human",
})


def test_context_has_allowed_ids_and_constraints():
    ctx = _ctx()
    assert "red_mug_1" in ctx.allowed_ids
    assert "bin_1" in ctx.allowed_ids
    assert ctx.constraints.gripper_max_width == 0.09
    assert ctx.constraints.payload_kg == 2.0


def test_valid_plan_accepted():
    res = Planner(MockBackend(responses=[_VALID_PLAN])).plan(_ctx())
    assert res.valid is True
    assert res.used_fallback is False
    assert res.plan.selected_target == "red_mug_1"
    assert res.plan.destination == "bin_1"
    assert res.attempts == 1


def test_markdown_fenced_json_is_parsed():
    fenced = "```json\n" + _VALID_PLAN + "\n```"
    res = Planner(MockBackend(responses=[fenced])).plan(_ctx())
    assert res.valid is True
    assert res.plan.selected_target == "red_mug_1"


def test_repair_loop_recovers_from_malformed_then_valid():
    backend = MockBackend(responses=["not json at all", _VALID_PLAN])
    res = Planner(backend, max_retries=2).plan(_ctx())
    assert res.valid is True
    assert res.attempts == 2
    assert len(res.errors) >= 1


def test_fabricated_id_triggers_safe_fallback():
    bad = json.dumps({
        "task": "put", "selected_target": "purple_mug_9", "destination": "bin_1",
        "steps": [], "fallback": "ask_human",
    })
    res = Planner(MockBackend(responses=[bad]), max_retries=1).plan(_ctx())
    assert res.valid is False
    assert res.used_fallback is True
    assert res.plan.fallback == PlanFallback.ASK_HUMAN
    assert res.plan.steps == []
    assert any("allowed_ids" in e for e in res.errors)


def test_synthetic_step_target_is_allowed():
    res = Planner(MockBackend(responses=[_VALID_PLAN])).plan(_ctx())
    # pregrasp_red_mug_1 should not be flagged as a fabricated-id risk
    assert not any("not a known id" in r for r in res.plan.risks)


def test_fallback_plan_is_always_schema_valid():
    # even garbage every attempt yields a valid ActionPlan object
    res = Planner(MockBackend(responses=["garbage"]), max_retries=1).plan(_ctx())
    assert res.plan is not None
    assert res.plan.fallback == PlanFallback.ASK_HUMAN
    assert res.plan.task  # non-empty
