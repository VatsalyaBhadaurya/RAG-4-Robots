"""Optional parity test for the Chroma backend (skipped if chromadb absent).

Confirms ChromaVectorIndex returns the same top hit as NumpyVectorIndex for the
same embeddings, so the two backends are interchangeable behind VectorIndex.
"""

from __future__ import annotations

import pytest

from r4r.embeddings import HashEmbedder
from r4r.index import NumpyVectorIndex

chromadb = pytest.importorskip("chromadb")


def test_chroma_matches_numpy_top_hit():
    from r4r.index.chroma_index import ChromaVectorIndex

    e = HashEmbedder(dim=256)
    docs = {
        "mug": "red ceramic mug",
        "bin": "plastic bin container",
        "box": "cardboard cereal box",
        "driver": "metal screwdriver tool",
    }
    ids = list(docs)
    vecs = e.embed([docs[i] for i in ids])
    metas = [{"kind": "object"} for _ in ids]

    npi = NumpyVectorIndex()
    npi.add(ids, vecs, metas)
    chi = ChromaVectorIndex(collection="r4r_test")
    chi.add(ids, vecs, metas)

    q = e.embed_one("red mug")
    assert npi.query(q, k=1)[0].id == chi.query(q, k=1)[0].id == "mug"
