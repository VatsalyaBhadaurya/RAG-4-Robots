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
| `r4r/embeddings` | **Implemented** | Hash (offline) + Ollama (`nomic-embed-text`) embedders |
| `r4r/index` | **Implemented** | `NumpyVectorIndex` + optional `ChromaVectorIndex` (parity tested) |
| `r4r/retrieval` | **Implemented** | Task abstraction, two-stage retriever, compact serialization w/ provenance |
| `r4r/knowledge` | **Implemented** | Seed safety rules / templates / procedures as knowledge nodes |
| `r4r/eval` | **Implemented (P3 metrics)** | Retrieval precision/recall + token-savings benchmark |
| `r4r/scoring/heuristics.py` | **Implemented** | Reachability/graspability/collision/support heuristics + explanations |
| `r4r/scoring/tensors.py` | **Implemented** | Subgraph→tensors (NumPy) + PyG `HeteroData` + GraphNeT bundle stub |
| `r4r/scoring` (GNN training) | Deferred (needs labels, P6) | GNN trained only if it beats heuristics |
| `r4r/planner` | **Implemented** | Ollama (llama3.2) -> strict `ActionPlan`; parse/validate/repair/ground/fallback; MockBackend for tests |
| `r4r/validation` | **Implemented** | Deterministic validator + safety guard (reach/collision/payload/gripper/fragile/human/sequence) |
| `r4r/sim` | **Implemented** | MJCF from SceneGraph + MuJoCo execution (attach-and-carry) + world-state update |
| `r4r/sim/piper.py` | **Implemented** | Real AgileX Piper arm + position IK executes an ActionPlan pick-and-place |
| `r4r/logging_` | **Implemented** | JSONL episode logs (full trace) |
| `r4r/pipeline.py` | **Implemented** | `run_episode()` orchestration: retrieve→score→plan→validate→execute→update→log; `execution_backend="kinematic"|"piper"` (real arm) |

## Retrieval primitive
`SceneGraph.induced_subgraph(seed_ids, depth, relations, directed)` is the core
GraphRAG expansion step. Phase 3 feeds it seeds from the vector index and
serializes the result (ids, labels, attributes, edges, confidence, timestamps,
provenance) into the compact LLM context.
```
