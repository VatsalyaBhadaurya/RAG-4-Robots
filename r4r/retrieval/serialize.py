"""Serialization: nodes/edges -> embeddable text and compact LLM context.

Two distinct serializations:
- *_to_text(): natural-language-ish strings fed to the embedder (retrieval).
- serialize_subgraph(): a compact, structured dict fed to the LLM planner —
  includes ids, labels, attributes, edges, edge attributes, confidence,
  provenance, and timestamps (an explicit project requirement).
"""

from __future__ import annotations

import json
from typing import Dict, List

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.edges import SceneEdge
from r4r.schemas.nodes import SceneNode


def estimate_tokens(obj) -> int:
    """Rough token estimate (~4 chars/token) for comparing prompt sizes.

    Heuristic only — used for relative GraphRAG-vs-full-graph comparisons, not
    billing. Flagged as an approximation in the eval report."""
    text = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"), default=str)
    return max(1, len(text) // 4)


def node_to_text(node: SceneNode) -> str:
    """Compact description used for embedding/semantic matching."""
    parts: List[str] = [node.kind.value, node.label]
    color = node.attributes.get("color")
    if color:
        parts.append(f"{color} {node.label}")
    if node.material:
        parts.append(node.material)
    if node.affordances:
        parts.append("affords " + ", ".join(node.affordances))
    # knowledge nodes carry their text in attributes['text']
    text = node.attributes.get("text")
    if isinstance(text, str):
        parts.append(text)
    return " | ".join(str(p) for p in parts)


def _prov_summary(provenance) -> Dict[str, object]:
    return {
        "source_type": provenance.source_type.value,
        "source_id": provenance.source_id,
        "observed_at": provenance.observed_at.isoformat(),
        "confidence": round(provenance.confidence, 3),
        "is_live": provenance.is_live,
    }


def node_to_compact(node: SceneNode) -> Dict[str, object]:
    d: Dict[str, object] = {
        "id": node.object_id,
        "kind": node.kind.value,
        "label": node.label,
        "confidence": round(node.confidence, 3),
        "visibility": round(node.visibility, 3),
        "provenance": _prov_summary(node.provenance),
    }
    if node.pose is not None:
        d["pos"] = [round(node.pose.x, 3), round(node.pose.y, 3), round(node.pose.z, 3)]
    if node.dims is not None:
        d["dims"] = [round(node.dims.x, 3), round(node.dims.y, 3), round(node.dims.z, 3)]
    if node.material:
        d["material"] = node.material
    if node.affordances:
        d["affordances"] = list(node.affordances)
    if node.attributes:
        d["attributes"] = {k: v for k, v in node.attributes.items() if k != "text"}
        if "text" in node.attributes:
            d["text"] = node.attributes["text"]
    return d


def edge_to_compact(edge: SceneEdge) -> Dict[str, object]:
    d: Dict[str, object] = {
        "src": edge.src,
        "rel": edge.rel.value,
        "dst": edge.dst,
        "confidence": round(edge.confidence, 3),
        "source_type": edge.provenance.source_type.value,
    }
    if edge.attributes:
        d["attributes"] = dict(edge.attributes)
    return d


def serialize_subgraph(subgraph: SceneGraph) -> Dict[str, object]:
    """Compact, auditable structured form of a retrieved subgraph."""
    return {
        "nodes": [node_to_compact(n) for n in subgraph.nodes()],
        "edges": [edge_to_compact(e) for e in subgraph.edges()],
        "graph_hash": subgraph.content_hash(),
    }
