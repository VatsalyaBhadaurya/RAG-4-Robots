"""Subgraph -> tensor schema tests (NumPy core + PyG adapters)."""

from __future__ import annotations

import numpy as np
import pytest

from r4r.knowledge import add_knowledge
from r4r.scene_graph import infer_spatial_edges
from r4r.scenes import build_tabletop_scene
from r4r.schemas.edges import RelationType
from r4r.scoring.tensors import NODE_FEATURE_DIM, subgraph_to_tensors


def _scene():
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    return g


def test_tensor_shapes_consistent():
    t = subgraph_to_tensors(_scene())
    n = t.num_nodes
    assert t.node_features.shape == (n, NODE_FEATURE_DIM)
    assert t.node_feature_mask.shape == (n, NODE_FEATURE_DIM)
    assert t.node_type.shape == (n,)
    assert t.edge_index.shape == (2, t.num_edges)
    assert t.edge_attr.shape == (t.num_edges, len(RelationType) + 1)


def test_geometry_mask_set_for_objects_and_unset_for_knowledge():
    g = _scene()
    add_knowledge(g)  # knowledge nodes have no pose/dims
    t = subgraph_to_tensors(g)
    mug = t.index_of("red_mug_1")
    assert t.node_feature_mask[mug, 0:8].sum() == 8  # dims+pose present
    rule = t.index_of("safety_sharp_tools")
    assert t.node_feature_mask[rule, 0:8].sum() == 0  # no geometry


def test_edge_attr_is_onehot_plus_confidence():
    t = subgraph_to_tensors(_scene())
    if t.num_edges == 0:
        pytest.skip("no edges")
    row = t.edge_attr[0]
    onehot = row[:-1]
    assert np.isclose(onehot.sum(), 1.0)       # exactly one relation type
    assert 0.0 <= row[-1] <= 1.0               # confidence in last column


def test_edge_index_references_valid_nodes():
    t = subgraph_to_tensors(_scene())
    assert t.edge_index.max() < t.num_nodes
    assert t.edge_index.min() >= 0


# ---- PyG adapters (skip if torch_geometric missing) ----
pyg = pytest.importorskip("torch_geometric")


def test_to_pyg_data_roundtrip():
    t = subgraph_to_tensors(_scene())
    data = t.to_pyg_data()
    assert data.x.shape[0] == t.num_nodes
    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_index.shape[1] == t.num_edges


def test_to_hetero_data_has_types_and_edges():
    g = _scene()
    add_knowledge(g)
    t = subgraph_to_tensors(g)
    data = t.to_hetero_data()
    # object and surface node types should be present
    assert "object" in data.node_types
    assert "surface" in data.node_types
    # at least one relation edge type exists (e.g. object on_top_of surface)
    assert len(data.edge_types) >= 1
    total_nodes = sum(data[nt].x.shape[0] for nt in data.node_types)
    assert total_nodes == t.num_nodes


def test_graphnet_bundle_contains_core_arrays():
    t = subgraph_to_tensors(_scene())
    b = t.to_graphnet_bundle()
    assert set(b) == {"x", "edge_index", "edge_attr", "node_type"}
    assert b["x"].shape[0] == t.num_nodes
