"""In-memory exact cosine-similarity index.

Vectors are stored L2-normalized, so cosine similarity is a single matmul.
Exact (no ANN approximation) — correct and fast at tabletop scale, and keeps
tests deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from r4r.embeddings.base import l2_normalize
from r4r.index.base import ScoredHit, VectorIndex


class NumpyVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._ids: List[str] = []
        self._meta: List[Dict[str, object]] = []
        self._mat: Optional[np.ndarray] = None  # (n, dim), normalized

    def add(self, ids: List[str], vectors: np.ndarray,
            metadatas: List[Dict[str, object]]) -> None:
        if len(ids) != len(vectors) or len(ids) != len(metadatas):
            raise ValueError("ids, vectors, metadatas must be the same length")
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._ids.extend(ids)
        self._meta.extend(metadatas)
        self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])

    def query(self, vector: np.ndarray, k: int = 5) -> List[ScoredHit]:
        if self._mat is None or len(self._ids) == 0:
            return []
        q = l2_normalize(np.asarray(vector, dtype=np.float32))
        scores = self._mat @ q  # cosine similarity
        k = min(k, len(self._ids))
        # argpartition for top-k, then sort those k by score desc
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredHit(id=self._ids[i], score=float(scores[i]), metadata=self._meta[i])
                for i in top]

    def __len__(self) -> int:
        return len(self._ids)
