# rag4robots-graphnet

Research-backed **RAG-for-Robotics** + **GraphNet/GNN** system: an embodied
assistant that retrieves task-relevant 3D scene subgraphs (GraphRAG), scores
relational/physical properties (GNN, heuristics first), and produces **safe,
validated** robot action plans with an LLM planner.

> Core principle: retrieve small task-relevant subgraphs, score with learned +
> heuristic reasoning, **validate every plan deterministically**, simulate
> before any physical execution, and measure everything. The LLM is never the
> sole safety authority.

## Status

| Phase | State |
|-------|-------|
| 1 - Research & requirements | done (see `docs/`) |
| 2 - Scene-graph MVP | implemented & tested |
| 3 - GraphRAG retrieval | implemented & tested |
| 4 - Heuristic scoring + GNN tensor schema | **implemented & tested** (this branch) |
| 5 - LLM planner (Ollama, strict JSON) | next |
| 6 - Safety, MuJoCo execution, logging | planned |

**Implemented:** typed schemas - dynamic heterogeneous scene graph (NetworkX) -
AABB spatial heuristics - induced-subgraph retrieval - two-stage GraphRAG
(Ollama `nomic-embed-text` + NumPy/Chroma vector index) - seed knowledge base -
retrieval benchmark - heuristic scorers (reachability/graspability/collision/
support) with explanations - subgraph→tensor schema (NumPy + PyG `HeteroData` +
GraphNeT bundle stub) - **54 passing tests**.

**Measured (Phase 3, `nomic-embed-text`, 10-node scene):** retrieval mean
recall 0.94, precision 0.46 (recall-first by design), token savings x1.4 vs
full-graph prompting.

**Mocked:** perception (ground-truth poses from the scene definition).
**Deferred:** GNN *training* — heuristics ship first; a GNN is adopted only if it
beats them on labeled MuJoCo data (Phase 6). Heuristic calibration is **unproven**
until then.

## Stack (decisions locked - see `docs/decision_log.md`)
GNN: **PyTorch Geometric** | Simulator: **MuJoCo** | Graph+vector:
**NetworkX + Chroma/FAISS** | LLM + embeddings: **local Ollama**.

## Quickstart
```bash
# Phase 3+ uses a Python 3.12 venv (see risk R-01). Create it once:
py -3.12 -m venv .venv
./.venv/Scripts/python -m pip install -e . pytest ollama chromadb

./.venv/Scripts/python -m pytest -q                 # 40 tests
./.venv/Scripts/python examples/build_tabletop_scene.py
./.venv/Scripts/python examples/graphrag_demo.py    # uses Ollama if served, else offline
```
Phase-2-only core also runs on the base Python 3.8 (`pip install -r requirements.txt`).
The GraphRAG demo uses Ollama embeddings if `ollama serve` is running with
`nomic-embed-text`; otherwise it falls back to the offline HashEmbedder.

## Key references
- EmbodiedRAG - arXiv 2410.23968 (dynamic 3D scene-subgraph retrieval for robot planning)
- G-Retriever - arXiv 2402.07630, NeurIPS 2024 (subgraph retrieval + GNN soft-prompt)
- Hydra - arXiv 2201.13360, RSS 2022 (real-time 3D scene graphs)
- GraphNeT - github.com/graphnet-team/graphnet (PyG-based; built for neutrino telescopes)

## Repo layout
```
r4r/schemas/      typed contracts (nodes, edges, provenance, plan)
r4r/scene_graph/  dynamic graph + spatial heuristics
r4r/scenes/       reusable synthetic scenes
r4r/embeddings/   hash (offline) + ollama embedders
r4r/index/        numpy + chroma vector indexes
r4r/retrieval/    task abstraction, two-stage retriever, serialization
r4r/knowledge/    seed safety rules / templates / procedures
r4r/scoring/      heuristic scorers + subgraph→tensor schema (PyG/GraphNeT)
r4r/eval/         retrieval benchmark + metrics
tests/            unit tests (offline; ollama/chroma tests auto-skip)
docs/             architecture, decision log, risk register
examples/         runnable demos
```
