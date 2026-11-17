# Architecture

## Pipeline

```
Camera / RGB-D / robot state  (MVP: MuJoCo sim, perception MOCKED as GT poses)
        |
Perception & object detection            [Phase 2: mocked; later real detector]
        |
3D scene-graph builder                   [Phase 2: r4r.scene_graph]
        |
Graph store (NetworkX) + vector index (Chroma/FAISS)   [Phase 2 graph / Phase 3 index]
        |
GraphRAG retriever                       [Phase 3: entity retrieval -> induced subgraph]
        |
Relevant scene subgraph + knowledge chunks (compact, with provenance)
        |
GraphNet/GNN scoring (PyTorch Geometric) [Phase 4: heuristics first, GNN if it wins]
        |
LLM planner (Ollama, strict JSON)        [Phase 5]
        |
Plan validator + safety guard (deterministic)  [Phase 6]
        |
Robot controller / MuJoCo simulator      [Phase 6]
        |
World-state update + episode logging     [Phase 6]
```

## Dual RAG tracks
- **Static knowledge RAG:** manuals, robot APIs, safety procedures, task
  templates, demonstrations → embedded, stored as knowledge-kind nodes.
- **Dynamic embodied GraphRAG:** live scene-graph nodes/edges with
  spatial/semantic relations, timestamps, confidence.

Both live in one **heterogeneous** graph (`NodeKind` distinguishes scene vs
knowledge) so retrieval can traverse them together.

## Module map (current)
| Module | Status | Responsibility |
|--------|--------|----------------|
| `r4r/schemas` | **Implemented** | Typed contracts: nodes, edges, provenance, action plan |
| `r4r/scene_graph/graph.py` | **Implemented** | Dynamic graph: CRUD, query, persistence, induced subgraph |
| `r4r/scene_graph/spatial.py` | **Implemented** | AABB spatial-relation heuristics |
| `r4r/scenes/tabletop.py` | **Implemented (mocked perception)** | Deterministic demo/benchmark scene |
| `r4r/index` | Planned (P3) | Vector embeddings + retrieval |
| `r4r/retrieval` | Planned (P3) | Entity match -> subgraph expansion + serialization |
| `r4r/scoring` | Planned (P4) | Heuristic + GNN scores; PyG tensor conversion; `to_graphnet()` |
| `r4r/planner` | Planned (P5) | Ollama client -> strict `ActionPlan` |
| `r4r/validation` | Planned (P6) | Deterministic plan validator + safety guard |
| `r4r/sim` | Planned (P6) | MuJoCo tabletop; execution + world update |
| `r4r/logging_` | Planned (P6) | Episode logs for eval + future training |

## Retrieval primitive
`SceneGraph.induced_subgraph(seed_ids, depth, relations, directed)` is the core
GraphRAG expansion step. Phase 3 feeds it seeds from the vector index and
serializes the result (ids, labels, attributes, edges, confidence, timestamps,
provenance) into the compact LLM context.
```
