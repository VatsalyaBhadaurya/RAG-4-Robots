"""run_episode driving the REAL Piper arm (skipped if mujoco/model absent)."""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("mujoco")

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.knowledge import add_knowledge  # noqa: E402
from r4r.pipeline import run_episode  # noqa: E402
from r4r.planner import MockBackend  # noqa: E402
from r4r.scene_graph import infer_spatial_edges  # noqa: E402
from r4r.scenes import build_piper_scene_graph  # noqa: E402
from r4r.sim.piper import PIPER_SCENE  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(PIPER_SCENE),
                                reason="agilex_piper model not vendored")

_PLAN = json.dumps({
    "task": "put_red_mug_in_bin", "selected_target": "red_mug_1", "destination": "bin_1",
    "steps": [{"action": "move_to", "target": "pregrasp_red_mug_1"},
              {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
              {"action": "place", "target": "bin_1"},
              {"action": "release", "target": "red_mug_1"}],
    "fallback": "ask_human"})


def _graph():
    g = build_piper_scene_graph()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    add_knowledge(g)
    return g


def test_validator_passes_on_piper_scene():
    # sanity: the mirrored graph is reachable/graspable so a good plan validates
    from r4r.planner.context import RobotConstraints
    from r4r.scoring import score_scene
    from r4r.validation import PlanValidator
    from r4r.schemas.plan import ActionPlan
    g = _graph()
    plan = ActionPlan.model_validate_json(_PLAN)
    res = PlanValidator().validate(plan, g, RobotConstraints.from_graph(g), score_scene(g))
    assert res.ok is True, [v.code for v in res.errors()]


def test_pipeline_drives_piper_arm_to_success():
    g = _graph()
    log, g2 = run_episode(g, "Put the red mug in the bin.", HashEmbedder(512),
                          MockBackend(responses=[_PLAN]), execute=True,
                          execution_backend="piper")
    assert log.validation_ok is True
    assert log.executed is True
    assert log.success is True
    assert log.failure_category is None
    # world-state updated from the arm sim: cube now in the bin footprint
    mug = g2.get_node("red_mug_1").pose
    assert 0.216 <= mug.x <= 0.384 and -0.324 <= mug.y <= -0.156
