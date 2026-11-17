"""Phase 4 — relational/physical scoring over a retrieved subgraph.

Two layers, deliberately separated:
- heuristics.py: deterministic, explainable baselines (reachability,
  graspability, collision risk, support stability). These ship first and are the
  bar a learned model must beat (project principle: heuristics before GNN).
- tensors.py: convert a subgraph into GNN-ready tensors (node_features,
  edge_index, edge_attr, masks) with an optional PyTorch Geometric HeteroData
  builder and a GraphNeT-compatible adapter stub — so the GNN can be trained
  LATER (after Phase 6 produces labeled sim episodes) without reworking the graph.

Status: heuristics + tensor schema IMPLEMENTED; GNN training DEFERRED (unproven).
"""

from __future__ import annotations

from r4r.scoring.report import ObjectScores, ScoreReport
from r4r.scoring.heuristics import HeuristicConfig, score_scene
from r4r.scoring.tensors import (
    SubgraphTensors,
    subgraph_to_tensors,
)

__all__ = [
    "ObjectScores",
    "ScoreReport",
    "HeuristicConfig",
    "score_scene",
    "SubgraphTensors",
    "subgraph_to_tensors",
]
