"""Deterministic, offline embedder (hashed bag-of-words).

Not semantically strong, but stable and dependency-free: texts that share tokens
get similar vectors. This keeps unit tests fast and independent of any model
server. Do NOT use it for production retrieval quality — use OllamaEmbedder.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from r4r.embeddings.base import Embedder, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                out[i, self._bucket(tok)] += 1.0
        return l2_normalize(out)
