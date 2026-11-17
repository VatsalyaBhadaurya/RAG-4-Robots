"""GraphRAG retrieval tests (offline HashEmbedder)."""

from __future__ import annotations

from r4r.embeddings import HashEmbedder
from r4r.knowledge import add_knowledge
from r4r.retrieval import GraphRAGRetriever, abstract_task
from r4r.scene_graph import infer_spatial_edges
from r4r.scenes import build_tabletop_scene
from r4r.schemas.nodes import KNOWLEDGE_KINDS


def _graph():
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    add_knowledge(g)
    return g


def test_abstract_task_put_in_container():
    a = abstract_task("Put the red mug in the bin.")
    assert a.target_phrase == "red mug"
    assert a.destination_phrase == "bin"
    assert "red mug" in a.query_texts()


def test_abstract_task_pick_up():
    a = abstract_task("Pick up the screwdriver next to the box.")
    assert a.target_phrase and "screwdriver" in a.target_phrase


def test_abstract_task_query_form():
    a = abstract_task("Which objects on the table can the robot safely grasp?")
    assert a.is_query is True


def test_retriever_returns_task_relevant_seeds():
    r = GraphRAGRetriever(_graph(), HashEmbedder(dim=512)).build()
    res = r.retrieve("Put the red mug in the bin.", k_entities=4, depth=1)
    assert "red_mug_1" in res.seed_ids
    assert "bin_1" in res.seed_ids


def test_retrieved_subgraph_excludes_knowledge_nodes():
    r = GraphRAGRetriever(_graph(), HashEmbedder(dim=512)).build()
    res = r.retrieve("Put the red mug in the bin.", k_entities=4, depth=1)
    for nid in res.subgraph.node_ids():
        assert res.subgraph.get_node(nid).kind not in KNOWLEDGE_KINDS


def test_retrieval_saves_tokens_vs_full_graph():
    r = GraphRAGRetriever(_graph(), HashEmbedder(dim=512)).build()
    res = r.retrieve("Grasp the blue bowl.", k_entities=2, depth=1)
    # focused retrieval should be smaller than dumping the whole scene
    assert res.token_estimate < res.full_graph_token_estimate
    assert res.token_savings_ratio > 1.0


def test_serialized_subgraph_carries_provenance():
    r = GraphRAGRetriever(_graph(), HashEmbedder(dim=512)).build()
    res = r.retrieve("Put the red mug in the bin.", k_entities=3, depth=1)
    nodes = res.serialized_subgraph["nodes"]
    assert nodes, "expected at least one node in the subgraph"
    for n in nodes:
        assert "provenance" in n
        assert "confidence" in n
        assert "observed_at" in n["provenance"]


def test_knowledge_chunks_retrieved_for_fragile_task():
    r = GraphRAGRetriever(_graph(), HashEmbedder(dim=512)).build()
    res = r.retrieve("Put the ceramic mug in the bin gently.", k_entities=3, k_knowledge=3)
    ids = {c["id"] for c in res.knowledge_chunks}
    # fragile-force or side-grasp knowledge should surface for a ceramic mug task
    assert ids, "expected some knowledge chunks"
