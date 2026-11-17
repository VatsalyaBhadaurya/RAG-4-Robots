"""Embedder + vector index unit tests (offline, deterministic)."""

from __future__ import annotations

import numpy as np

from r4r.embeddings import HashEmbedder, cosine_sim
from r4r.index import NumpyVectorIndex


def test_hash_embedder_normalized_and_deterministic():
    e = HashEmbedder(dim=128)
    v1 = e.embed(["red ceramic mug"])
    v2 = e.embed(["red ceramic mug"])
    assert v1.shape == (1, 128)
    np.testing.assert_allclose(v1, v2)  # deterministic
    assert abs(np.linalg.norm(v1[0]) - 1.0) < 1e-5  # L2 normalized


def test_hash_embedder_shared_tokens_more_similar():
    e = HashEmbedder(dim=512)
    mug = e.embed_one("red ceramic mug")
    mug2 = e.embed_one("red mug on the table")
    bin_ = e.embed_one("plastic bin container")
    assert cosine_sim(mug, mug2) > cosine_sim(mug, bin_)


def test_numpy_index_topk_ordering():
    e = HashEmbedder(dim=256)
    idx = NumpyVectorIndex()
    texts = {"mug": "red ceramic mug", "bin": "plastic bin", "box": "cardboard box"}
    ids = list(texts)
    idx.add(ids, e.embed([texts[i] for i in ids]), [{"kind": "object"} for _ in ids])

    hits = idx.query(e.embed_one("red mug"), k=2)
    assert hits[0].id == "mug"
    assert hits[0].score >= hits[1].score  # sorted descending
    assert len(idx) == 3


def test_numpy_index_empty_returns_nothing():
    idx = NumpyVectorIndex()
    assert idx.query(np.zeros(8, dtype=np.float32), k=5) == []
