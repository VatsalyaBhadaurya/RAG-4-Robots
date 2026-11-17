"""r4r — RAG-for-Robotics + GraphNet.

A research-backed system that retrieves task-relevant 3D scene subgraphs
(GraphRAG), scores relational/physical properties (GNN/heuristics), and plans
safe robot actions (LLM) with deterministic validation.

Package layout (phases build incrementally):
    schemas/      pydantic data contracts (nodes, edges, provenance, plan)
    scene_graph/  heterogeneous dynamic 3D scene graph (NetworkX) + spatial edges
    (later) index/ retrieval/ scoring/ planner/ validation/ sim/ logging_/

Target runtime: Python 3.8+. All modules use `from __future__ import annotations`.
"""

__version__ = "0.1.0"
