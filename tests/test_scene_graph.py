"""SceneGraph: CRUD, queries, persistence round-trip, induced subgraph."""

from __future__ import annotations

import pytest

from r4r.scene_graph import SceneGraph, infer_spatial_edges
from r4r.scenes import build_tabletop_scene
from r4r.schemas import NodeKind, RelationType, SceneEdge, SceneNode
from r4r.schemas.provenance import Provenance, SourceType


def _built_scene() -> SceneGraph:
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    return g


def test_scene_builds_expected_counts():
    g = build_tabletop_scene()
    n, _ = g.stats()
    assert n == 10  # table, 5 objects, block, bin, robot, ee
    assert len(g.nodes_by_kind(NodeKind.CONTAINER)) == 1


def test_duplicate_node_rejected_unless_overwrite():
    g = build_tabletop_scene()
    dup = g.get_node("red_mug_1")
    with pytest.raises(ValueError):
        g.add_node(dup)
    g.add_node(dup, overwrite=True)  # should not raise


def test_edge_requires_existing_endpoints():
    g = build_tabletop_scene()
    bad = SceneEdge(src="ghost", rel=RelationType.NEAR, dst="red_mug_1",
                    provenance=Provenance.derived("x"))
    with pytest.raises(KeyError):
        g.add_edge(bad)


def test_update_node_is_validated():
    g = build_tabletop_scene()
    updated = g.update_node("red_mug_1", visibility=0.5)
    assert updated.visibility == 0.5
    assert g.get_node("red_mug_1").visibility == 0.5


def test_persistence_round_trip(tmp_path):
    g = _built_scene()
    p = tmp_path / "scene.json"
    g.save_json(str(p))
    g2 = SceneGraph.load_json(str(p))
    assert g.stats() == g2.stats()
    assert g.content_hash() == g2.content_hash()
    # typed objects survive the round trip
    assert isinstance(g2.get_node("red_mug_1"), SceneNode)
    assert g2.get_node("red_mug_1").attributes["color"] == "red"


def test_objects_on_table_query():
    g = _built_scene()
    on_table = {n.object_id for n in g.objects_on("table_1")}
    # the five objects placed directly on the table (not the stacked block)
    assert {"red_mug_1", "blue_bowl_1", "cereal_box_1", "soda_can_1"}.issubset(on_table)
    assert "block_1" not in on_table  # block sits on the cereal box, not the table


def test_induced_subgraph_depth_limits_scope():
    g = _built_scene()
    # depth 1 around the mug should include the mug and its direct neighbours
    sub1 = g.induced_subgraph(["red_mug_1"], depth=1)
    assert "red_mug_1" in sub1.node_ids()
    assert len(sub1) < len(g)  # strictly smaller than the full graph
    # depth 0 returns just the seed
    sub0 = g.induced_subgraph(["red_mug_1"], depth=0)
    assert sub0.node_ids() == ["red_mug_1"]


def test_induced_subgraph_relation_filter():
    g = _built_scene()
    # expanding only along ON_TOP_OF from the block should reach its support
    sub = g.induced_subgraph(["block_1"], depth=1, relations={RelationType.ON_TOP_OF})
    assert "cereal_box_1" in sub.node_ids()
