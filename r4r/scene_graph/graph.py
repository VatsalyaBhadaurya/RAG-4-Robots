"""SceneGraph: typed wrapper over networkx.MultiDiGraph.

Design choices:
- MultiDiGraph: relations are directed and typed; two nodes may be connected by
  several relation types at once (e.g. NEAR and ALIGNED_WITH), so we key
  parallel edges by relation value.
- SceneNode/SceneEdge pydantic objects live under the 'data' attribute, so the
  graph stays fully typed and serializable.
- induced_subgraph() is the retrieval primitive for Phase 3 GraphRAG: given seed
  node ids, expand outward up to `depth` hops, optionally filtering relations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from r4r.schemas.edges import RelationType, SceneEdge
from r4r.schemas.nodes import NodeKind, SceneNode


class SceneGraph:
    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ---- nodes ---------------------------------------------------------
    def add_node(self, node: SceneNode, overwrite: bool = False) -> None:
        if self._g.has_node(node.object_id) and not overwrite:
            raise ValueError(
                f"node '{node.object_id}' already exists (use overwrite=True)"
            )
        self._g.add_node(node.object_id, data=node)

    def upsert_node(self, node: SceneNode) -> None:
        """Add or replace a node — the normal path for live perception updates."""
        self._g.add_node(node.object_id, data=node)

    def update_node(self, object_id: str, **fields: object) -> SceneNode:
        """Patch selected fields of an existing node (pydantic-validated)."""
        node = self.get_node(object_id)
        updated = node.model_copy(update=fields)
        self._g.nodes[object_id]["data"] = updated
        return updated

    def remove_node(self, object_id: str) -> None:
        self._g.remove_node(object_id)

    def has_node(self, object_id: str) -> bool:
        return self._g.has_node(object_id)

    def get_node(self, object_id: str) -> SceneNode:
        if not self._g.has_node(object_id):
            raise KeyError(f"no such node: '{object_id}'")
        return self._g.nodes[object_id]["data"]

    def nodes(self) -> List[SceneNode]:
        return [self._g.nodes[n]["data"] for n in self._g.nodes]

    def node_ids(self) -> List[str]:
        return list(self._g.nodes)

    # ---- edges ---------------------------------------------------------
    def add_edge(self, edge: SceneEdge) -> None:
        if not self._g.has_node(edge.src):
            raise KeyError(f"edge src missing: '{edge.src}'")
        if not self._g.has_node(edge.dst):
            raise KeyError(f"edge dst missing: '{edge.dst}'")
        # key by relation so (src, rel, dst) is unique; re-adding updates it.
        self._g.add_edge(edge.src, edge.dst, key=edge.rel.value, data=edge)

    def remove_edge(self, src: str, rel: RelationType, dst: str) -> None:
        self._g.remove_edge(src, dst, key=rel.value)

    def has_edge(self, src: str, rel: RelationType, dst: str) -> bool:
        return self._g.has_edge(src, dst, key=rel.value)

    def edges(self) -> List[SceneEdge]:
        return [d["data"] for _, _, d in self._g.edges(data=True)]

    def out_edges(self, object_id: str) -> List[SceneEdge]:
        return [d["data"] for _, _, d in self._g.out_edges(object_id, data=True)]

    def in_edges(self, object_id: str) -> List[SceneEdge]:
        return [d["data"] for _, _, d in self._g.in_edges(object_id, data=True)]

    # ---- queries -------------------------------------------------------
    def nodes_by_kind(self, kind: NodeKind) -> List[SceneNode]:
        return [n for n in self.nodes() if n.kind == kind]

    def nodes_by_label(self, label: str) -> List[SceneNode]:
        label = label.lower()
        return [n for n in self.nodes() if n.label.lower() == label]

    def objects_on(self, surface_id: str) -> List[SceneNode]:
        """Objects X where X --on_top_of--> surface_id."""
        out = []
        for e in self.in_edges(surface_id):
            if e.rel == RelationType.ON_TOP_OF:
                out.append(self.get_node(e.src))
        return out

    # ---- retrieval primitive (used by Phase 3 GraphRAG) ----------------
    def induced_subgraph(
        self,
        seed_ids: Iterable[str],
        depth: int = 1,
        relations: Optional[Set[RelationType]] = None,
        directed: bool = False,
    ) -> "SceneGraph":
        """Return the subgraph induced by `seed_ids` expanded `depth` hops.

        relations: if given, only traverse these relation types during expansion
                   (edges of other types between selected nodes are still kept).
        directed:  if True, follow out-edges only; if False, treat edges as
                   undirected for neighbour discovery (default — robots care about
                   'what is related', not edge direction).
        """
        seeds = [s for s in seed_ids if self._g.has_node(s)]
        selected: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)

        for _ in range(max(0, depth)):
            next_frontier: Set[str] = set()
            for n in frontier:
                for e in self._iter_incident(n, directed):
                    if relations is not None and e.rel not in relations:
                        continue
                    nbr = e.dst if e.src == n else e.src
                    if nbr not in selected:
                        next_frontier.add(nbr)
            selected |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        sub = SceneGraph()
        for nid in selected:
            sub.upsert_node(self.get_node(nid))
        # keep every edge whose endpoints are both selected
        for _, _, d in self._g.edges(data=True):
            e: SceneEdge = d["data"]
            if e.src in selected and e.dst in selected:
                sub.add_edge(e)
        return sub

    def _iter_incident(self, node_id: str, directed: bool) -> Iterable[SceneEdge]:
        yield from self.out_edges(node_id)
        if not directed:
            yield from self.in_edges(node_id)

    # ---- persistence & summary ----------------------------------------
    def to_dict(self) -> Dict[str, list]:
        return {
            "nodes": [n.model_dump(mode="json") for n in self.nodes()],
            "edges": [e.model_dump(mode="json") for e in self.edges()],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, list]) -> "SceneGraph":
        g = cls()
        for nd in payload.get("nodes", []):
            g.upsert_node(SceneNode.model_validate(nd))
        for ed in payload.get("edges", []):
            g.add_edge(SceneEdge.model_validate(ed))
        return g

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str) -> "SceneGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def content_hash(self) -> str:
        """Stable hash of the graph content — logged per episode so a retrieved
        subgraph can be referenced unambiguously (an evaluation requirement)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def stats(self) -> Tuple[int, int]:
        return self._g.number_of_nodes(), self._g.number_of_edges()

    def __len__(self) -> int:
        return self._g.number_of_nodes()
