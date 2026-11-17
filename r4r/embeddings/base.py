"""Embedder interface + vector helpers."""

from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """Base class. Implementations return a (n, dim) float32 array of
    L2-normalized row vectors."""

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = np.linalg.norm(mat)
        return mat / (n + eps)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(np.asarray(a, dtype=np.float32))
    b = l2_normalize(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))
