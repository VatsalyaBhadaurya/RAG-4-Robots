"""GraphRAG retrieval (Phase 3).

Two-stage retrieval, following EmbodiedRAG / G-Retriever:
  1. Semantic entity retrieval: embed node/knowledge descriptions + the task,
     fetch candidate seeds from the vector index.
  2. Graph expansion: take the induced subgraph around those seeds (configurable
     depth + relation filters) via SceneGraph.induced_subgraph().
Then serialize a COMPACT subgraph (+ knowledge chunks) with provenance,
confidence and timestamps — never the whole graph.
"""

from __future__ import annotations

from r4r.retrieval.serialize import (
    estimate_tokens,
    node_to_text,
    serialize_subgraph,
)
from r4r.retrieval.abstraction import TaskAbstraction, abstract_task
from r4r.retrieval.retriever import GraphRAGRetriever, RetrievalResult

__all__ = [
    "GraphRAGRetriever",
    "RetrievalResult",
    "TaskAbstraction",
    "abstract_task",
    "node_to_text",
    "serialize_subgraph",
    "estimate_tokens",
]
