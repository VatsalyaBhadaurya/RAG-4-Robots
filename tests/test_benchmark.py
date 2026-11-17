"""Phase 3 retrieval benchmark smoke + sanity metrics (offline HashEmbedder)."""

from __future__ import annotations

from r4r.embeddings import HashEmbedder
from r4r.eval import default_benchmark, evaluate_retrieval


def test_benchmark_runs_and_reports_all_cases():
    report = evaluate_retrieval(HashEmbedder(dim=512))
    assert len(report.cases) == len(default_benchmark())
    assert 0.0 <= report.mean_precision <= 1.0
    assert 0.0 <= report.mean_recall <= 1.0
    assert report.embedder == "HashEmbedder"


def test_benchmark_lexical_cases_have_full_recall():
    report = evaluate_retrieval(HashEmbedder(dim=512))
    by_name = {c.name: c for c in report.cases}
    # clear lexical matches should be fully recalled even by the hash embedder
    assert by_name["mug_in_bin"].recall == 1.0
    assert by_name["mug_in_bin"].precision >= 0.5


def test_benchmark_token_savings_on_focused_cases():
    report = evaluate_retrieval(HashEmbedder(dim=512))
    by_name = {c.name: c for c in report.cases}
    assert by_name["grasp_bowl"].token_savings_ratio > 1.0
