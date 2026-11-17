"""MuJoCo execution tests (skipped if mujoco missing)."""

from __future__ import annotations

import pytest

from r4r.scenes import build_pick_place_scene
from r4r.schemas.plan import ActionPlan, ActionStep
from r4r.sim.mjcf import build_mjcf

pytest.importorskip("mujoco")

from r4r.sim import MujocoExecutor, update_graph_from_execution  # noqa: E402


def _put_plan():
    return ActionPlan(task="put", selected_target="red_mug_1", destination="bin_1",
                      steps=[ActionStep(action="move_to", target="pregrasp_red_mug_1"),
                             ActionStep(action="grasp", target="red_mug_1", grasp_type="side", max_force="low"),
                             ActionStep(action="place", target="bin_1")])


def test_mjcf_loads_in_mujoco():
    import mujoco
    xml, meta = build_mjcf(build_pick_place_scene())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody >= 2            # world + mug (+ container geoms are static)
    assert "red_mug_1" in meta.free_objects
    assert "bin_1" in meta.containers


def test_put_in_bin_succeeds_and_updates_graph():
    g = build_pick_place_scene()
    ex = MujocoExecutor(g)
    res = ex.execute(_put_plan())
    assert res.success is True
    xmin, xmax, ymin, ymax, floor_z, rim_z = ex.meta.containers["bin_1"]
    fp = res.final_poses["red_mug_1"]
    assert xmin <= fp.x <= xmax and ymin <= fp.y <= ymax   # ended inside the bin
    update_graph_from_execution(g, res)
    assert abs(g.get_node("red_mug_1").pose.x - fp.x) < 1e-6  # world-state updated


def test_pick_only_reports_lift():
    g = build_pick_place_scene()
    plan = ActionPlan(task="pick", selected_target="red_mug_1",
                      steps=[ActionStep(action="grasp", target="red_mug_1")])
    res = MujocoExecutor(g).execute(plan)
    assert res.success is True   # lifted clear of the table


def test_non_movable_target_fails_gracefully():
    g = build_pick_place_scene()
    plan = ActionPlan(task="x", selected_target="table_1",
                      steps=[ActionStep(action="grasp", target="table_1")])
    res = MujocoExecutor(g).execute(plan)
    assert res.success is False
    assert res.notes
