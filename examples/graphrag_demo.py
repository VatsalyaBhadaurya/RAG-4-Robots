"""Phase 3 demo: two-stage GraphRAG retrieval + benchmark.

Uses OllamaEmbedder (nomic-embed-text) if a server is reachable, else falls back
to the offline HashEmbedder. Prints one retrieval payload and the benchmark
table (retrieval precision/recall + token savings vs full-graph prompting).

Run:  python examples/graphrag_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.eval import evaluate_retrieval  # noqa: E402
from r4r.eval.benchmark import build_eval_graph  # noqa: E402
from r4r.retrieval import GraphRAGRetriever  # noqa: E402


def _pick_embedder():
    try:
        import ollama
        ollama.embeddings(model="nomic-embed-text", prompt="ping")
        from r4r.embeddings import OllamaEmbedder
        print("Embedder: OllamaEmbedder(nomic-embed-text)\n")
        return OllamaEmbedder("nomic-embed-text")
    except Exception as exc:  # noqa: BLE001
        print(f"Embedder: HashEmbedder (Ollama unavailable: {exc})\n")
        return HashEmbedder(dim=512)


def main() -> None:
    emb = _pick_embedder()

    # one concrete retrieval
    retriever = GraphRAGRetriever(build_eval_graph(), emb).build()
    res = retriever.retrieve("Put the red mug in the bin.", k_entities=4, depth=1)
    print("Instruction:", res.instruction)
    print("Abstraction:", res.abstraction.model_dump(exclude_none=True))
    print("Seeds:", res.seed_ids)
    print("Subgraph nodes:", sorted(res.subgraph.node_ids()))
    print("Knowledge chunks:", [c["id"] for c in res.knowledge_chunks])
    print(f"Tokens: subgraph+knowledge={res.token_estimate} vs "
          f"full-graph={res.full_graph_token_estimate} "
          f"(x{res.token_savings_ratio:.1f} smaller)")
    print("\nContext payload handed to the LLM planner (Phase 5):")
    print(json.dumps(res.context_payload(), indent=2)[:1200] + "  ...\n")

    # benchmark table
    report = evaluate_retrieval(emb)
    print(f"{'case':20s} {'cat':20s} {'P':>5s} {'R':>5s} {'F1':>5s} {'sub':>4s} {'save':>5s}")
    for c in report.cases:
        print(f"{c.name:20s} {c.category:20s} {c.precision:5.2f} {c.recall:5.2f} "
              f"{c.f1:5.2f} {c.subgraph_nodes:4d} {c.token_savings_ratio:5.1f}")
    print(f"\nMEAN  precision={report.mean_precision}  recall={report.mean_recall}  "
          f"f1={report.mean_f1}  token_savings=x{report.mean_token_savings}  "
          f"[{report.embedder}]")


if __name__ == "__main__":
    main()
