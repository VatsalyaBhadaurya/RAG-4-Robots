"""Heuristic scorer tests on the tabletop scene."""

from __future__ import annotations

from r4r.scene_graph import infer_spatial_edges
from r4r.scenes import build_tabletop_scene
from r4r.schemas import Dimensions, NodeKind, Pose, SceneNode
from r4r.schemas.provenance import Provenance, SourceType
from r4r.scoring import score_scene
from r4r.scoring.heuristics import HeuristicConfig


def _scene():
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    return g


def test_scores_every_object():
    report = score_scene(_scene())
    objs = {"red_mug_1", "blue_bowl_1", "cereal_box_1", "screwdriver_1",
            "soda_can_1", "block_1"}
    assert objs.issubset(set(report.scores))


def test_small_reachable_object_is_graspable():
    report = score_scene(_scene())
    mug = report.for_object("red_mug_1")
    assert mug.graspability >= 0.5      # 0.08m min-dim fits 0.09m gripper + affordance
    assert mug.reachability > 0.0
    assert mug.graspable is True


def test_supporting_object_has_collision_risk_and_lower_stability():
    report = score_scene(_scene())
    box = report.for_object("cereal_box_1")  # the block rests on it
    assert box.collision_risk >= 0.7
    assert box.support_stability < 1.0


def test_free_object_on_top_is_stable():
    report = score_scene(_scene())
    block = report.for_object("block_1")  # nothing on top of the block
    assert block.support_stability == 1.0


def test_ranking_prefers_graspable_objects():
    report = score_scene(_scene())
    assert report.ranking
    assert report.for_object(report.best()).graspable is True


def test_missing_dimensions_flagged_uncertain():
    g = _scene()
    g.upsert_node(SceneNode(object_id="mystery_1", kind=NodeKind.OBJECT, label="mystery",
                            pose=Pose(x=-0.3, y=0.0, z=0.8),
                            provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="t")))
    report = score_scene(g)
    m = report.for_object("mystery_1")
    assert m.uncertain is True
    assert m.graspable is False


def test_out_of_reach_object_scores_low_reach():
    g = _scene()
    # object far outside the 0.85 m workspace
    g.upsert_node(SceneNode(object_id="far_1", kind=NodeKind.OBJECT, label="far",
                            pose=Pose(x=5.0, y=5.0, z=0.8),
                            dims=Dimensions(x=0.05, y=0.05, z=0.05),
                            affordances=["graspable_top"],
                            provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="t")))
    report = score_scene(g)
    far = report.for_object("far_1")
    assert far.reachability == 0.0
    assert far.graspable is False
