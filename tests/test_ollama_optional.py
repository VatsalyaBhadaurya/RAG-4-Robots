"""Optional tests that require a running Ollama server with nomic-embed-text.

Skipped automatically when the server/model is unavailable so CI stays green
without external dependencies. Run locally with Ollama served to validate the
real semantic embedder path and that it improves the harder (non-lexical) case.
"""

from __future__ import annotations

import pytest


def _ollama_ready(model: str = "nomic-embed-text") -> bool:
    try:
        import ollama
        ollama.embeddings(model=model, prompt="ping")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_ready(),
                                reason="Ollama server/model not available")


def test_ollama_embedder_dim_and_retrieval():
    from r4r.embeddings import OllamaEmbedder
    from r4r.retrieval import GraphRAGRetriever
    from r4r.eval.benchmark import build_eval_graph

    emb = OllamaEmbedder(model="nomic-embed-text")
    assert emb.dim == 768
    r = GraphRAGRetriever(build_eval_graph(), emb).build()
    res = r.retrieve("Put the red mug in the bin.", k_entities=4, depth=1)
    assert "red_mug_1" in res.seed_ids
    assert "bin_1" in res.seed_ids


def test_ollama_semantic_helps_nonlexical_query():
    """'safely grasp' should retrieve graspable objects by meaning, not keywords."""
    from r4r.embeddings import OllamaEmbedder
    from r4r.eval import evaluate_retrieval
    from r4r.eval.benchmark import default_benchmark

    cases = [c for c in default_benchmark() if c.name == "which_graspable"]
    report = evaluate_retrieval(OllamaEmbedder("nomic-embed-text"), cases=cases, k_entities=5)
    # semantic recall on the non-lexical case should be non-trivial
    assert report.cases[0].recall >= 0.4
