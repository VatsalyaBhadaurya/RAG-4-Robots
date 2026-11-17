"""Evaluation harness (grows per phase). Phase 3: GraphRAG retrieval metrics."""

from __future__ import annotations

from r4r.eval.benchmark import (
    BenchmarkCase,
    CaseResult,
    RetrievalReport,
    default_benchmark,
    evaluate_retrieval,
)

__all__ = [
    "BenchmarkCase",
    "CaseResult",
    "RetrievalReport",
    "default_benchmark",
    "evaluate_retrieval",
]
