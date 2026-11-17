"""End-to-end pipeline tests: retrieve->score->plan->validate->execute->update->log.

Uses MockBackend for deterministic planning; the execute path needs mujoco
(skipped otherwise). Verifies the safety guard: a blocked plan never executes
and never mutates the world.
"""

from __future__ import annotations

import json

import pytest

from r4r.embeddings import HashEmbedder
from r4r.pipeline import run_episode
from r4r.planner import MockBackend
from r4r.scenes import build_pick_place_scene, build_tabletop_scene
from r4r.scene_graph import infer_spatial_edges

_PLAN = json.dumps({
    "task": "put_red_mug_in_bin", "assumptions": [],
    "selected_target": "red_mug_1", "destination": "bin_1",
    "steps": [
        {"action": "move_to", "target": "pregrasp_red_mug_1"},
        {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
        {"action": "place", "target": "bin_1"},
    ],
    "risks": [], "required_confirmations": [], "fallback": "ask_human",
})

_FABRICATED = json.dumps({
    "task": "x", "selected_target": "unicorn_7", "destination": "bin_1",
    "steps": [{"action": "grasp", "target": "unicorn_7"}], "fallback": "ask_human",
})


def _with_edges(scene):
    for e in infer_spatial_edges(scene.nodes()):
        scene.add_edge(e)
    return scene


def test_fabricated_id_is_refused_and_not_executed():
    g = _with_edges(build_pick_place_scene())
    log, _ = run_episode(g, "Put the red mug in the bin.", HashEmbedder(512),
                         MockBackend(responses=[_FABRICATED]), execute=False)
    assert log.plan_used_fallback is True
    assert log.failure_category == "plan_refused"
    assert log.executed is False


def test_unreachable_destination_blocks_execution_and_preserves_world():
    # tabletop scene: the bin sits outside the robot workspace -> must be blocked
    g = _with_edges(build_tabletop_scene())
    before = g.get_node("red_mug_1").pose.position()
    log, g2 = run_episode(g, "Put the red mug in the bin.", HashEmbedder(512),
                          MockBackend(responses=[_PLAN]), execute=True)
    assert log.validation_ok is False
    assert log.failure_category == "validation_blocked"
    assert log.executed is False
    assert g2.get_node("red_mug_1").pose.position() == before  # world untouched


# ---- execution path requires mujoco ----
pytest.importorskip("mujoco")


def test_full_success_executes_and_updates_world(tmp_path):
    from r4r.logging_ import EpisodeLogger

    g = _with_edges(build_pick_place_scene())
    logger = EpisodeLogger(str(tmp_path / "ep.jsonl"))
    log, g2 = run_episode(g, "Put the red mug in the bin.", HashEmbedder(512),
                          MockBackend(responses=[_PLAN]), logger=logger, execute=True)
    assert log.validation_ok is True
    assert log.executed is True
    assert log.success is True
    assert log.failure_category is None
    # the mug actually moved into the bin region (x shifted toward +x)
    assert g2.get_node("red_mug_1").pose.x > 0.0
    # persisted
    assert logger.read_all()[0].success is True
