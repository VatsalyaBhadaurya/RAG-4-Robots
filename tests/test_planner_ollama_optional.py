"""Optional end-to-end planner test against a real local model (llama3.2).

Skipped unless Ollama is served and the model is available. Validates that the
real structured-decoding path yields a schema-valid, grounded ActionPlan.
"""

from __future__ import annotations

import pytest

_MODEL = "llama3.2:latest"


def _ready() -> bool:
    try:
        import ollama
        ollama.chat(model=_MODEL, messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 1})
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="Ollama/llama3.2 not available")


def test_real_planner_produces_grounded_plan():
    from r4r.embeddings import HashEmbedder
    from r4r.eval.benchmark import build_eval_graph
    from r4r.planner import OllamaChatBackend, Planner, build_context
    from r4r.retrieval import GraphRAGRetriever
    from r4r.scoring import score_scene

    graph = build_eval_graph()
    retriever = GraphRAGRetriever(graph, HashEmbedder(dim=512)).build()
    res = retriever.retrieve("Put the red mug in the bin.", k_entities=4, depth=1)
    ctx = build_context("Put the red mug in the bin.", res.serialized_subgraph,
                        res.knowledge_chunks, score_scene(graph), graph)

    result = Planner(OllamaChatBackend(_MODEL), max_retries=2).plan(ctx)
    # must be a schema-valid plan; ids (if any) must be grounded
    assert result.plan is not None
    if result.plan.selected_target:
        assert result.plan.selected_target in ctx.allowed_ids
    if result.plan.destination:
        assert result.plan.destination in ctx.allowed_ids
