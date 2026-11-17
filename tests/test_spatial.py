"""Spatial heuristic tests on controlled geometry."""

from __future__ import annotations

from r4r.scene_graph.spatial import SpatialConfig, infer_spatial_edges
from r4r.schemas import Dimensions, NodeKind, Pose, SceneNode
from r4r.schemas.edges import RelationType
from r4r.schemas.provenance import Provenance, SourceType


def _n(oid, x, y, z, dx, dy, dz, kind=NodeKind.OBJECT):
    return SceneNode(object_id=oid, kind=kind, label=oid,
                     pose=Pose(x=x, y=y, z=z), dims=Dimensions(x=dx, y=dy, z=dz),
                     provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="t"))


def _rels(edges):
    return {(e.src, e.rel, e.dst) for e in edges}


def test_on_top_of_and_supports_detected():
    # box bottom at 0.0..0.2 ; block sits exactly on top (bottom at 0.2)
    box = _n("box", 0, 0, 0.1, 0.2, 0.2, 0.2)
    block = _n("block", 0, 0, 0.225, 0.05, 0.05, 0.05)
    rels = _rels(infer_spatial_edges([box, block]))
    assert ("block", RelationType.ON_TOP_OF, "box") in rels
    assert ("box", RelationType.SUPPORTS, "block") in rels


def test_floating_object_not_on_top():
    box = _n("box", 0, 0, 0.1, 0.2, 0.2, 0.2)
    floating = _n("floating", 0, 0, 0.5, 0.05, 0.05, 0.05)  # big gap
    rels = _rels(infer_spatial_edges([box, floating]))
    assert ("floating", RelationType.ON_TOP_OF, "box") not in rels


def test_inside_container_detected():
    bin_ = _n("bin", 0, 0, 0.15, 0.3, 0.3, 0.3, kind=NodeKind.CONTAINER)
    item = _n("item", 0, 0, 0.1, 0.05, 0.05, 0.05)
    rels = _rels(infer_spatial_edges([bin_, item]))
    assert ("item", RelationType.INSIDE, "bin") in rels
    assert ("bin", RelationType.CONTAINS, "item") in rels


def test_near_and_next_to_for_adjacent_same_level():
    a = _n("a", 0.0, 0.0, 0.1, 0.05, 0.05, 0.1)
    b = _n("b", 0.1, 0.0, 0.1, 0.05, 0.05, 0.1)  # 0.1 m apart, same z
    rels = _rels(infer_spatial_edges([a, b]))
    assert ("a", RelationType.NEAR, "b") in rels
    assert ("a", RelationType.NEXT_TO, "b") in rels


def test_far_objects_not_near():
    a = _n("a", 0.0, 0.0, 0.1, 0.05, 0.05, 0.1)
    b = _n("b", 2.0, 0.0, 0.1, 0.05, 0.05, 0.1)
    rels = _rels(infer_spatial_edges([a, b]))
    assert ("a", RelationType.NEAR, "b") not in rels


def test_derived_provenance_and_confidence():
    a = _n("a", 0.0, 0.0, 0.1, 0.05, 0.05, 0.1)
    b = _n("b", 0.05, 0.0, 0.1, 0.05, 0.05, 0.1)
    for e in infer_spatial_edges([a, b]):
        assert e.provenance.source_type == SourceType.DERIVED
        assert 0.0 <= e.confidence <= 1.0


def test_nodes_without_geometry_are_skipped():
    a = _n("a", 0.0, 0.0, 0.1, 0.05, 0.05, 0.1)
    no_geo = SceneNode(object_id="ghost", kind=NodeKind.TASK_GOAL, label="goal",
                       provenance=Provenance(source_type=SourceType.STATIC, source_id="t"))
    # should not raise and should not reference the geometry-less node
    edges = infer_spatial_edges([a, no_geo])
    assert all("ghost" not in (e.src, e.dst) for e in edges)
