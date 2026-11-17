# Risk Register

Severity: H(igh) / M(edium) / L(ow). Status: Open / Mitigated / Accepted.

| ID | Risk | Sev | Status | Mitigation |
|----|------|-----|--------|------------|
| R-01 | **Python 3.8.10 is EOL.** Recent `chromadb` / `torch-geometric` releases often require Python ≥3.9, blocking Phase 3/4 installs. | H | **Mitigated** | `.venv` on Python 3.12.10 created; chromadb 1.5.9 installed; 40 tests pass. Phase 3+ uses `.venv`. (D-06) |
| R-09 | **Retrieval precision is modest (~0.46)** — over-retrieval at fixed k. Could enlarge the planner's candidate set. | M | Open | Recall-first is intentional (R high ~0.94); validator prunes. Add adaptive-k / score threshold / PCST selection; measure on larger scenes. (D-08) |
| R-02 | **Ollama not running / no model pulled.** Phase 5 planner has no backend yet. | M | Open | User action: `ollama serve` + pull an instruct model. Planner interface will be built model-agnostic with a mock backend for tests. |
| R-03 | **Local LLM weak JSON-schema adherence.** Plans may be malformed/hallucinate ids. | H | **Mitigated (P5)** | Structured decoding (`format=schema`) + pydantic validation + id grounding vs `allowed_ids` + repair/retry + `ask_human` fallback. llama3.2 produced a valid grounded plan in 1 attempt; MockBackend tests cover repair/fallback. Validity *rate* across models still to aggregate (P6). |
| R-04 | **AABB spatial heuristics are approximate** (orientation ignored). Wrong on/inside relations possible for tilted/odd shapes. | M | Accepted (MVP) | Confidence-weighted DERIVED edges; replace with MuJoCo geometry queries in Phase 4/6; measure relation accuracy vs sim GT. |
| R-05 | **Perception is mocked** (sim ground-truth poses). Real detector error/occlusion not yet modeled. | M | Open | Clearly labeled MOCKED; Phase 3+ adds occlusion/ambiguity benchmark cases; integrate a real detector later with precision/recall metrics. |
| R-06 | **Physical execution risk.** Any real-robot step is high risk. | H | Controlled | Simulation-only MVP; no low-level motor commands from the LLM; deterministic safety guard + validator gate; `safety_violation_rate` must be 0 before any hardware. |
| R-07 | **GNN may not beat heuristics.** Unproven value, risk of wasted effort. | M | Open | Heuristic baseline + PyG tensor schema shipped (P4); GNN training deferred until P6 labels exist; adopt only on measured advantage (ablation ladder). |
| R-10 | **Heuristic scores are uncalibrated** (rule-based, no ground truth). Over/under-confident grasp/collision calls. | M | Open | `uncertain` flag + explanations exposed; calibration + FP/FN to be measured vs MuJoCo ground truth (P6) before any physical use. |
| R-08 | **Scene changes between planning and execution.** Stale subgraph → unsafe plan. | H | Open (design) | Timestamps + confidence on every fact; re-validate against live graph immediately before execution; benchmark includes scene-change cases. |
