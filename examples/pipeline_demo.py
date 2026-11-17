"""End-to-end demo (Phases 2-5): instruction -> GraphRAG -> scoring -> LLM plan.

Uses a real local model (llama3.2) if Ollama is served, else a canned MockBackend
so the pipeline still runs offline. Prints the retrieved context summary and the
strict ActionPlan JSON. (Deterministic safety validation is Phase 6.)

Run:  python examples/pipeline_demo.py "Put the red mug in the bin."
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.eval.benchmark import build_eval_graph  # noqa: E402
from r4r.planner import MockBackend, Planner, build_context  # noqa: E402
from r4r.retrieval import GraphRAGRetriever  # noqa: E402
from r4r.scoring import score_scene  # noqa: E402

_CANNED = json.dumps({
    "task": "put_red_mug_in_bin", "assumptions": ["bin is open and reachable"],
    "selected_target": "red_mug_1", "destination": "bin_1",
    "steps": [
        {"action": "move_to", "target": "pregrasp_red_mug_1", "constraints": ["collision_free"]},
        {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
        {"action": "place", "target": "bin_1", "constraints": ["gentle_place", "collision_free"]},
    ],
    "risks": [], "required_confirmations": [], "fallback": "ask_human",
})


def _backend():
    try:
        import ollama
        ollama.chat(model="llama3.2:latest",
                    messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
        from r4r.planner import OllamaChatBackend
        print("Planner backend: OllamaChatBackend(llama3.2:latest)\n")
        return OllamaChatBackend("llama3.2:latest")
    except Exception as exc:  # noqa: BLE001
        print(f"Planner backend: MockBackend (Ollama unavailable: {exc})\n")
        return MockBackend(responses=[_CANNED])


def main() -> None:
    instruction = sys.argv[1] if len(sys.argv) > 1 else "Put the red mug in the bin."
    graph = build_eval_graph()

    emb = HashEmbedder(dim=512)
    retriever = GraphRAGRetriever(graph, emb).build()
    res = retriever.retrieve(instruction, k_entities=4, depth=1)
    report = score_scene(graph)

    print("Instruction:", instruction)
    print("Retrieved seeds:", res.seed_ids)
    print("Knowledge:", [c["id"] for c in res.knowledge_chunks])
    print("Scorer best candidate:", report.best())
    print(f"Tokens: subgraph+knowledge={res.token_estimate} vs full={res.full_graph_token_estimate}")

    ctx = build_context(instruction, res.serialized_subgraph, res.knowledge_chunks, report, graph)
    result = Planner(_backend(), max_retries=2).plan(ctx)

    print(f"\nPlan valid={result.valid}  used_fallback={result.used_fallback}  "
          f"attempts={result.attempts}  backend={result.backend}")
    print(json.dumps(result.plan.model_dump(), indent=2))
    if result.errors:
        print("\nValidation/repair notes:")
        for e in result.errors:
            print("  -", e)


if __name__ == "__main__":
    main()
