"""Piper arm executor test (skipped if mujoco or the vendored model is absent)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

from r4r.schemas.plan import ActionPlan, ActionStep  # noqa: E402
from r4r.sim.piper import PIPER_SCENE, PiperArmExecutor  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(PIPER_SCENE), reason="agilex_piper model not vendored")


def _plan():
    return ActionPlan(task="put", selected_target="red_mug_1", destination="bin_1",
                      steps=[ActionStep(action="move_to", target="pregrasp_red_mug_1"),
                             ActionStep(action="grasp", target="red_mug_1", grasp_type="side", max_force="low"),
                             ActionStep(action="place", target="bin_1"),
                             ActionStep(action="release", target="red_mug_1")])


def test_piper_scene_loads():
    ex = PiperArmExecutor()
    assert ex.model.nu == 7            # 6 arm joints + gripper
    assert ex._bid("red_mug_1") >= 0
    assert ex._gid("bin_1_floor") >= 0


def test_piper_picks_and_places_cube_in_bin():
    ex = PiperArmExecutor()
    result = ex.execute(_plan())
    assert result.success is True, result.notes
    fp = result.final_poses["red_mug_1"]
    # cube ended inside the bin footprint and below the rim
    assert 0.216 <= fp.x <= 0.384
    assert -0.324 <= fp.y <= -0.156
    assert fp.z < 0.13
