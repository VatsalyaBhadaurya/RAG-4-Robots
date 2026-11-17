"""GraphRAGRetriever: the two-stage retriever.

Stage 1 (semantic): embed each node/knowledge description once into a vector
index; embed the task's abstraction phrases; fetch candidate ids.
Stage 2 (graph): expand the induced subgraph around the scene seeds.

Output is a compact, auditable RetrievalResult with provenance + a token
comparison against full-graph prompting (an evaluation requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from r4r.embeddings.base import Embedder
from r4r.index.base import VectorIndex
from r4r.index.numpy_index import NumpyVectorIndex
from r4r.retrieval.abstraction import TaskAbstraction, abstract_task
from r4r.retrieval.serialize import (
    estimate_tokens,
    node_to_compact,
    node_to_text,
    serialize_subgraph,
)
from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import KNOWLEDGE_KINDS, NodeKind


@dataclass
class RetrievalResult:
    instruction: str
    abstraction: TaskAbstraction
    seed_ids: List[str]
    subgraph: SceneGraph
    serialized_subgraph: Dict[str, object]
    knowledge_chunks: List[Dict[str, object]]
    token_estimate: int             # compact subgraph + knowledge
    full_graph_token_estimate: int  # serializing the whole scene graph instead
    seed_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def token_savings_ratio(self) -> float:
        if self.token_estimate == 0:
            return 0.0
        return self.full_graph_token_estimate / self.token_estimate

    def context_payload(self) -> Dict[str, object]:
        """Exactly what gets handed to the LLM planner (Phase 5)."""
        return {
            "instruction": self.instruction,
            "scene_subgraph": self.serialized_subgraph,
            "knowledge": self.knowledge_chunks,
        }


class GraphRAGRetriever:
    def __init__(
        self,
        graph: SceneGraph,
        embedder: Embedder,
        index: Optional[VectorIndex] = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.index = index if index is not None else NumpyVectorIndex()
        self._built = False

    def build(self) -> "GraphRAGRetriever":
        """Embed every node once and populate the vector index."""
        nodes = self.graph.nodes()
        if not nodes:
            self._built = True
            return self
        texts = [node_to_text(n) for n in nodes]
        vectors = self.embedder.embed(texts)
        ids = [n.object_id for n in nodes]
        metas = [{"kind": n.kind.value,
                  "is_knowledge": n.kind in KNOWLEDGE_KINDS}
                 for n in nodes]
        self.index.add(ids, vectors, metas)
        self._built = True
        return self

    def retrieve(
        self,
        instruction: str,
        k_entities: int = 5,
        depth: int = 1,
        relations: Optional[Set] = None,
        k_knowledge: int = 3,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        if not self._built:
            self.build()
        abstraction = abstract_task(instruction)

        # --- stage 1: semantic retrieval over abstraction phrases ---
        phrases = abstraction.query_texts()
        qvecs = self.embedder.embed(phrases)
        best: Dict[str, float] = {}
        # query generously, we separate scene vs knowledge afterwards
        pool = max(k_entities + k_knowledge + 4, 8)
        for qv in qvecs:
            for hit in self.index.query(qv, k=pool):
                if hit.score < min_score:
                    continue
                prev = best.get(hit.id)
                if prev is None or hit.score > prev:
                    best[hit.id] = hit.score

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        scene_seeds: List[str] = []
        knowledge_ids: List[str] = []
        for nid, score in ranked:
            kind = self.graph.get_node(nid).kind
            if kind in KNOWLEDGE_KINDS:
                if len(knowledge_ids) < k_knowledge:
                    knowledge_ids.append(nid)
            else:
                if len(scene_seeds) < k_entities:
                    scene_seeds.append(nid)

        # --- stage 2: graph expansion around scene seeds ---
        subgraph = self.graph.induced_subgraph(scene_seeds, depth=depth, relations=relations)
        serialized = serialize_subgraph(subgraph)
        knowledge_chunks = [node_to_compact(self.graph.get_node(i)) for i in knowledge_ids]

        token_estimate = estimate_tokens(serialized) + estimate_tokens(knowledge_chunks)
        full_tokens = self._full_graph_tokens()

        return RetrievalResult(
            instruction=instruction,
            abstraction=abstraction,
            seed_ids=scene_seeds,
            subgraph=subgraph,
            serialized_subgraph=serialized,
            knowledge_chunks=knowledge_chunks,
            token_estimate=token_estimate,
            full_graph_token_estimate=full_tokens,
            seed_scores={nid: best[nid] for nid in scene_seeds},
        )

    def _full_graph_tokens(self) -> int:
        """Token estimate for the naive 'full-graph prompting' baseline: serialize
        the ENTIRE heterogeneous graph (whole scene + all knowledge docs), which
        is what sending everything to the LLM would cost. GraphRAG is compared
        against this."""
        all_ids = [n.object_id for n in self.graph.nodes()]
        full = self.graph.induced_subgraph(all_ids, depth=0)
        return estimate_tokens(serialize_subgraph(full))
