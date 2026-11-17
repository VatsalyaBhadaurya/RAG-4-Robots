# Decision Log

Format per entry: Decision · Options · Evidence · Chosen · Rationale · Assumptions · Next validation step.

Labels: **Verified** (from a cited source) · **Inferred** (reasoned from evidence) · **Assumption** (needs confirmation) · **Unknown** (needs experiment).

---

## D-00 — Overall paradigm: GraphRAG-first, GNN second
- **Decision:** Build RAG-for-robotics (retrieve task-relevant scene subgraph + static knowledge) first; add GNN scoring as a second layer over the retrieved subgraph.
- **Evidence (Verified):** EmbodiedRAG (arXiv 2410.23968) retrieves query-relevant 3D scene subgraphs to augment an LLM planner, deployed on Spot. G-Retriever (arXiv 2402.07630, NeurIPS 2024) retrieves a subgraph (PCST) and uses a GNN soft-prompt with a frozen LLM.
- **Chosen:** GraphRAG core, GNN as augmentation.
- **Rationale:** Both cited systems retrieve small subgraphs rather than feeding whole graphs to the LLM; GNN complements rather than replaces retrieval.
- **Assumptions:** Our tabletop scenes are small enough that PCST-style optimization is optional for the MVP (plain induced-subgraph expansion suffices first).
- **Next validation:** Phase 3 — measure retrieval precision/recall + token savings vs full-graph prompting on the benchmark set.

## D-01 — GNN implementation layer → **PyTorch Geometric**
- **Options:** (a) PyTorch Geometric; (b) GraphNeT (graphnet-team); (c) other.
- **Evidence (Verified):** GraphNeT (arXiv 2210.12194; github.com/graphnet-team/graphnet) is a PyG/Lightning library built for neutrino-telescope (Cherenkov) reconstruction. PyG natively supports heterogeneous graphs (`HeteroData`, `to_hetero()`, `HGTConv`).
- **Chosen:** PyTorch Geometric, user-confirmed.
- **Rationale (Inferred):** GraphNeT's detector-pulse abstractions don't map to robot scene graphs; PyG's heterogeneous support fits our mixed scene/knowledge graph directly.
- **Assumptions:** We keep a thin `to_graphnet()` tensor adapter so a GraphNeT backend remains possible.
- **Next validation:** Phase 4 — define tensor schema; confirm `HeteroData` round-trips our subgraph.

## D-02 — Simulator → **MuJoCo**
- **Options:** PyBullet, MuJoCo, Isaac Sim/Lab, physical robot.
- **Evidence (Verified):** `mujoco` 3.2.3 already installed (Apache-2.0). Comparative sources note MuJoCo is now the academic default for contact-rich manipulation fidelity; Isaac Lab for GPU throughput; PyBullet lightweight.
- **Chosen:** MuJoCo, user-confirmed.
- **Rationale:** Best contact/grasp fidelity for the validation+scoring loop; permissive license; already present.
- **Assumptions:** Execution is simulation-only for the MVP; no physical hardware without an approved integration + safety plan.
- **Next validation:** Phase 6 — stand up a tabletop MJCF scene mirroring `build_tabletop_scene()`; verify pose round-trip.

## D-03 — Graph + vector store → **NetworkX + Chroma/FAISS (in-memory, local)**
- **Options:** NetworkX + Chroma/FAISS; Neo4j + native vector index.
- **Evidence (Verified):** `networkx` 3.1 installed. Neo4j provides real Cypher; NetworkX does not (we expose a typed Python query interface instead — see `SceneGraph`).
- **Chosen:** NetworkX + Chroma/FAISS, user-confirmed.
- **Rationale:** Zero-infra, fast iteration for a tabletop MVP. We do **not** claim Cypher support.
- **Assumptions:** Scene size stays small enough for in-memory graphs. Migrate to Neo4j only if scale/persistence demands.
- **Next validation:** Phase 3 — add Chroma index; confirm Python-version compatibility (R-01).

## D-04 — LLM planner → **local open model via Ollama**
- **Options:** Claude API; local open model (Ollama); other hosted.
- **Evidence (Verified):** `ollama` python client 0.6.2 installed; no server running / no model pulled yet.
- **Chosen:** Local Ollama, user-confirmed.
- **Rationale:** Offline/private, no API cost.
- **Assumptions (needs action):** User will run `ollama serve` and pull a model (e.g. a Llama/Qwen instruct variant); model id will be configurable. Local models adhere to JSON schemas less reliably → strict pydantic validation + repair/fallback required.
- **Next validation:** Phase 5 — measure JSON-schema validity rate across the benchmark; pick the smallest model that clears the bar.

## D-05 — Spatial relations (MVP) → **axis-aligned bounding-box heuristics**
- **Decision:** Infer on_top_of/supports/inside/contains/near/next_to/aligned_with from AABBs; ignore orientation.
- **Evidence:** Implemented in `r4r/scene_graph/spatial.py`; unit-tested.
- **Rationale (Assumption):** Adequate for a tabletop MVP without a geometry engine; every edge is `DERIVED` with a confidence margin so downstream can discount it.
- **Next validation:** Phase 4/6 — replace with MuJoCo geometry/collision queries where grasp/collision fidelity matters; measure relation accuracy vs sim ground truth.
