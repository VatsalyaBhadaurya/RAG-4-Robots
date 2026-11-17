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

## D-06 — Dev interpreter → **Python 3.12 venv** (mitigates R-01)
- **Decision:** Create `.venv` on Python 3.12.10 for Phase 3+ (ML deps); Phase 2 core still runs on 3.8.
- **Evidence (Verified):** `py -0p` shows 3.10/3.12/3.13/3.14 available. `.venv` built on 3.12; installed networkx 3.7, pydantic 2.14.3, chromadb 1.5.9; 40 tests pass.
- **Rationale:** `chromadb`/`torch-geometric` commonly need Python ≥3.9; 3.12 has broad wheel support.
- **Next validation:** Phase 4 — install `torch-geometric` into the same venv.

## D-07 — Embeddings → **Ollama `nomic-embed-text`** (768-d), with offline HashEmbedder for tests
- **Options:** sentence-transformers (local torch); Ollama embedding model; OpenAI-style API.
- **Evidence (Verified):** `nomic-embed-text` already pulled in local Ollama; returns 768-d vectors (measured). Avoids a second heavy torch/download path.
- **Chosen:** `OllamaEmbedder` at runtime; `HashEmbedder` (deterministic, dependency-free) for tests so CI never needs a server.
- **Rationale:** Local, consistent with the Ollama planner decision (D-04); keeps tests hermetic.
- **Assumptions:** Ollama served. **Next validation:** Phase 5 — reuse the same server for the planner model.

## D-08 — Retrieval design → **two-stage, recall-first**
- **Decision:** Stage 1 semantic entity retrieval (vector top-k over abstraction phrases) → Stage 2 induced-subgraph expansion (depth/relation filters). Knowledge retrieved as separate chunks. Compact serialization with provenance/confidence/timestamps.
- **Evidence (Verified):** Mirrors EmbodiedRAG (abstraction → entities → induced subgraph) and G-Retriever (semantic match → subgraph). Measured on the benchmark: mean recall 0.94, precision 0.46, token savings ×1.4 (10-node scene) with `nomic-embed-text`.
- **Rationale:** Recall-first is the safe default — missing a task-relevant entity is worse than retrieving extra context the validator can prune. "Gold" is the minimal entity set, so precision <1 partly reflects useful context (support surface, robot), not pure error.
- **Assumptions / next validation:** Precision improves with adaptive-k / score thresholding and PCST-style selection (G-Retriever); token savings grow with scene size (EmbodiedRAG's scalability motivation). Validate on larger scenes in a later phase.

## D-09 — GNN scoring → **heuristics first + PyG tensor schema; GNN training deferred**
- **Decision (user-confirmed):** Ship deterministic heuristic scorers (reachability, graspability, collision risk, support stability) + a subgraph→tensor schema (NumPy core, PyG `HeteroData` adapter, GraphNeT bundle stub). Defer GNN *training* until Phase 6 produces labeled sim episodes.
- **Evidence (Verified):** PyG `HeteroData`/`to_hetero` support heterogeneous graphs (docs). Conversion validated: 17-node scene → 9 node types + 10 typed edge relations in a real `HeteroData` (torch 2.15.0+cpu, PyG 2.8.1). 54 tests pass.
- **Rationale:** Project principle — heuristics before learning; adopt a GNN only on measured advantage. The tensor schema (with missing-feature masks) lets the GNN drop in later without reworking the graph.
- **Assumptions / Unknown:** Heuristics are rule-based, not calibrated — FP/FN and calibration are **Unknown** until measured against MuJoCo ground truth (Phase 6). Reachability uses a workspace-sphere approximation (robot base + radius), so objects beyond the radius score 0 (realistic for a fixed-base arm; a mobile base/IK check is the upgrade).
- **Next validation:** Phase 6 — label grasp/collision outcomes in MuJoCo; compute heuristic calibration + FP/FN; only then decide whether a trained GNN beats them.

## D-10 — LLM planner → **llama3.2 via Ollama, strict JSON + repair + grounding + fallback**
- **Decision (user-confirmed model):** `llama3.2:latest` default (configurable). Context = instruction + retrieved subgraph + knowledge chunks + heuristic ScoreReport + robot/safety constraints + `allowed_ids`. Output constrained to the `ActionPlan` schema via Ollama structured decoding (`format=<json schema>`).
- **Safety design (project requirement):** pydantic validation → id grounding (`selected_target`/`destination` must be in `allowed_ids`; no fabricated ids) → repair/retry loop → safe `ask_human` fallback. The planner is never the sole authority; kinematic/collision safety is Phase 6.
- **Evidence (Verified, measured):** end-to-end run produced a valid, grounded plan in 1 attempt with llama3.2 — selected_target `red_mug_1`, destination `bin_1`, side grasp consistent with the retrieved `proc_side_grasp_mug` chunk; token use 1716 vs 2521 full-graph (×1.47). 62 tests pass incl. a live-model integration test. MockBackend covers repair/fallback paths offline.
- **Assumptions / Unknown:** JSON-validity and plan-validity *rates* across the full benchmark (and across models) not yet aggregated — measure in Phase 6 with repeated trials.
- **Next validation:** Phase 6 — wire the deterministic safety validator + MuJoCo execution; report plan-validity rate, safety-violation rate (must be 0), and task success.

## D-11 — MuJoCo execution → **kinematic attach-and-carry (no arm/IK) for the MVP**
- **Decision:** Build an MJCF world from the SceneGraph (table as static box, objects as free rigid bodies, containers as open-top boxes). Execute by carrying the grasped object along a scripted waypoint path (kinematic), then releasing it to pure physics (drop + settle + collide). No articulated arm, IK, or motion planner yet.
- **Evidence (Verified):** mujoco 4.0.1 in venv. End-to-end: "Put the red mug in the bin." → mug carried and dropped, settles inside the bin interior (final z≈0.81 within [0.76, 0.97]); graph updated from read-back poses; `test_sim`/`test_pipeline` pass.
- **Rationale:** Real physics for the parts that matter (placement, collision, settling, stacking) without the heavy, failure-prone arm-IK-grasp stack — avoids premature complexity. Carry is kinematic, honestly labeled.
- **Assumptions / Risk (R-11):** grasp is assumed to succeed (no force-closure/contact grasp sim); reachability is approximated; so sim success ≠ guaranteed real-world success. Arm + IK + contact grasping + motion planning is the upgrade path.
- **Observed (safety guard working):** a live llama3.2 plan was schema-valid and grounded but had an unsafe step order; the deterministic validator blocked it (`place_without_grasp`) and execution never ran — the LLM is not the safety authority (R-03/R-06).
- **Next validation:** build the ablation harness (LLM-only → +RAG → +GraphRAG → +heuristics → +GNN) and aggregate plan-validity / task-success / safety-violation (must be 0) over repeated trials.

## D-12 — Real robot arm → **AgileX Piper (MuJoCo Menagerie) with position IK**
- **Decision (user request):** Add a real articulated arm visible in simulation. Vendor the AgileX **Piper** 6-DoF model from MuJoCo Menagerie (`third_party/mujoco_menagerie/agilex_piper`), build a combined arm+table+cube+bin scene, and drive it with damped-least-squares **position IK** on the 6 arm joints through pick→lift→transit→release waypoints.
- **Evidence (Verified, measured):** `r4r/sim/piper.py` `PiperArmExecutor.execute(ActionPlan)` runs headless and in video — the arm reaches the cube, grasps, lifts, carries, and releases it into the bin (final ≈ (0.30,-0.25,0.03), `in_bin=True`). 2 gated tests pass; `examples/arm_pick_place_demo.py` renders `runs/arm_execution.mp4`.
- **Rationale:** satisfies "actual MuJoCo sim with a real arm" and makes the manipulator visible (previous executor had no arm).
- **Honesty / R-11 (updated):** straight-down grasping is outside this small arm's dexterous workspace (verified: 6-DoF orientation IK cannot reach the table-height grasp pose), so the grasp BOND stays **kinematic** (cube follows the gripper while held) — the ARM itself is full physics + IK. Contact/force-closure grasping + a larger/mobile workspace is the next upgrade.
- **Assumptions:** cube sized < the ~0.07 m gripper opening; object + bin placed within the Piper's reach (arm base at origin, objects on its table). Model is BSD-3 (menagerie LICENSE retained).
- **Next validation:** generate the Piper scene from an arbitrary SceneGraph and gate arm execution on the validator's reachability check (unify with `run_episode`).
- **Update (integrated):** `run_episode(..., execution_backend="piper")` now drives the real arm, gated by the validator. `build_piper_scene_graph()` mirrors the MuJoCo scene coordinates so retrieval/scoring/validation reason about the executed world. Verified live: llama3.2 plan → validation ok → Piper arm places the cube in the bin → world updated (runs/arm_full_loop.mp4). Remaining: auto-generate the arm MJCF from an arbitrary graph (currently a fixed mirrored scene).
