"""Schema contract tests: validation, provenance, plan id extraction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from r4r.schemas import (
    ActionPlan,
    ActionStep,
    Dimensions,
    NodeKind,
    Pose,
    Provenance,
    RelationType,
    SceneEdge,
    SceneNode,
    SourceType,
)


def _prov():
    return Provenance(source_type=SourceType.PERCEPTION, source_id="cam0", confidence=0.9)


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        Provenance(source_type=SourceType.PERCEPTION, source_id="cam0", confidence=1.5)


def test_scene_node_physical_flag():
    n = SceneNode(object_id="mug_1", kind=NodeKind.OBJECT, label="mug",
                  pose=Pose(x=1.0, y=0, z=0), dims=Dimensions(x=0.1, y=0.1, z=0.1),
                  provenance=_prov())
    assert n.is_physical() is True
    assert n.requires_pose() is True

    k = SceneNode(object_id="rule_1", kind=NodeKind.SAFETY_RULE, label="no_sharp_near_human",
                  provenance=Provenance(source_type=SourceType.STATIC, source_id="safety_manual"))
    assert k.is_physical() is False


def test_dimensions_must_be_positive():
    with pytest.raises(ValidationError):
        Dimensions(x=0.0, y=0.1, z=0.1)


def test_provenance_is_live():
    assert _prov().is_live is True
    assert Provenance.derived("spatial_heuristic").is_live is False


def test_edge_requires_valid_relation():
    e = SceneEdge(src="a", rel=RelationType.ON_TOP_OF, dst="b", provenance=_prov())
    assert e.key() == "on_top_of"


def test_action_plan_referenced_ids_dedup_and_order():
    plan = ActionPlan(
        task="put_red_mug_in_bin",
        selected_target="red_mug_1",
        destination="bin_1",
        steps=[
            ActionStep(action="move_to", target="red_mug_1"),
            ActionStep(action="grasp", target="red_mug_1", grasp_type="side"),
            ActionStep(action="place", target="bin_1"),
        ],
    )
    assert plan.referenced_ids() == ["red_mug_1", "bin_1"]
    assert plan.fallback.value == "ask_human"


def test_action_plan_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ActionStep(action="teleport", target="red_mug_1")
