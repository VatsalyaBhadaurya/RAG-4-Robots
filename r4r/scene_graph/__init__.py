"""Dynamic 3D scene graph (Phase 2).

A thin, typed wrapper over networkx.MultiDiGraph that stores SceneNode /
SceneEdge pydantic objects, supports persistence, querying, and induced-subgraph
extraction (the primitive Phase 3 GraphRAG retrieval builds on).
"""

from __future__ import annotations

from r4r.scene_graph.graph import SceneGraph
from r4r.scene_graph.spatial import SpatialConfig, infer_spatial_edges

__all__ = ["SceneGraph", "SpatialConfig", "infer_spatial_edges"]
