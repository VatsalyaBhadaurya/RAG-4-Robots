"""Deterministic validator / safety-guard tests."""

from __future__ import annotations

from r4r.planner.context import RobotConstraints
from r4r.scene_graph import SceneGraph
from r4r.schemas import Dimensions, NodeKind, Pose, SceneNode
from r4r.schemas.plan import ActionPlan, ActionStep
from r4r.schemas.provenance import Provenance, SourceType
from r4r.scoring import score_scene
from r4r.validation import PlanValidator, Severity


def _node(oid, kind, label, x, y, z, d=0.08, material=None, affs=None, vis=1.0, conf=0.95, attrs=None):
    return SceneNode(object_id=oid, kind=kind, label=label,
                     pose=Pose(x=x, y=y, z=z), dims=Dimensions(x=d, y=d, z=d),
                     material=material, visibility=vis, confidence=conf,
                     affordances=affs or [], attributes=attrs or {},
                     provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="t"))


def _reachable_scene() -> SceneGraph:
    """Robot at origin; mug and bin both inside the 0.85 m workspace."""
    g = SceneGraph()
    g.add_node(SceneNode(object_id="robot_1", kind=NodeKind.ROBOT, label="arm",
                         pose=Pose(x=0, y=0, z=0.75),
                         attributes={"workspace_radius": 0.85, "payload_kg": 2.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="spec")))
    g.add_node(SceneNode(object_id="ee_1", kind=NodeKind.END_EFFECTOR, label="gripper",
                         pose=Pose(x=0, y=0, z=0.95), attributes={"max_width": 0.09, "max_force_n": 40.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="spec")))
    g.add_node(_node("mug_1", NodeKind.OBJECT, "mug", 0.25, 0.0, 0.8, d=0.08,
                     material="ceramic", affs=["graspable_side"]))
    g.add_node(_node("bin_1", NodeKind.CONTAINER, "bin", 0.30, 0.20, 0.8, d=0.25,
                     affs=["container"]))
    return g


def _put_plan():
    return ActionPlan(
        task="put_mug_in_bin", selected_target="mug_1", destination="bin_1",
        steps=[
            ActionStep(action="move_to", target="pregrasp_mug_1"),
            ActionStep(action="grasp", target="mug_1", grasp_type="side", max_force="low"),
            ActionStep(action="place", target="bin_1"),
        ],
    )


def _validate(g, plan):
    return PlanValidator().validate(plan, g, RobotConstraints.from_graph(g), score_scene(g))


def test_clean_plan_passes():
    res = _validate(_reachable_scene(), _put_plan())
    assert res.ok is True, res.summary() + " :: " + str([v.code for v in res.errors()])


def test_fabricated_object_blocked():
    plan = _put_plan()
    plan.selected_target = "ghost_9"
    res = _validate(_reachable_scene(), plan)
    assert res.ok is False
    assert any(v.code == "unknown_object" for v in res.errors())


def test_low_visibility_blocked():
    g = _reachable_scene()
    g.update_node("mug_1", visibility=0.1)
    res = _validate(g, _put_plan())
    assert any(v.code == "low_visibility" for v in res.errors())


def test_unreachable_target_blocked():
    g = _reachable_scene()
    g.update_node("mug_1", pose=Pose(x=5.0, y=0.0, z=0.8))  # far away
    res = _validate(g, _put_plan())
    assert any(v.code == "not_reachable" for v in res.errors())


def test_overweight_blocked():
    g = _reachable_scene()
    # a dense, large metal block well over the 2 kg payload
    g.add_node(_node("anvil_1", NodeKind.OBJECT, "anvil", 0.25, 0.0, 0.8, d=0.2,
                     material="metal", affs=["graspable_side"]))
    plan = _put_plan()
    plan.selected_target = "anvil_1"
    res = _validate(g, plan)
    assert any(v.code == "overweight" for v in res.errors())


def test_place_without_grasp_blocked():
    plan = ActionPlan(task="bad", selected_target="mug_1", destination="bin_1",
                      steps=[ActionStep(action="place", target="bin_1")])
    res = _validate(_reachable_scene(), plan)
    assert res.ok is False
    assert any(v.code in ("place_without_grasp", "missing_grasp") for v in res.errors())


def test_empty_plan_is_not_executable():
    plan = ActionPlan(task="refusal", steps=[])
    res = _validate(_reachable_scene(), plan)
    assert res.ok is False
    assert any(v.code == "no_steps" for v in res.errors())


def test_fragile_without_low_force_warns():
    plan = _put_plan()
    plan.steps[1] = ActionStep(action="grasp", target="mug_1", grasp_type="side")  # no low force
    res = _validate(_reachable_scene(), plan)
    assert any(v.code == "fragile_force" and v.severity == Severity.WARNING for v in res.violations)


def test_human_in_workspace_blocked():
    g = _reachable_scene()
    g.add_node(SceneNode(object_id="human_1", kind=NodeKind.OBJECT, label="human",
                         pose=Pose(x=0.3, y=0.0, z=0.9), attributes={"is_human": True},
                         provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="cam")))
    res = _validate(g, _put_plan())
    assert any(v.code == "human_in_workspace" for v in res.errors())
