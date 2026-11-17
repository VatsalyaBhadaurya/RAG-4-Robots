"""Embedder backed by a local Ollama embedding model.

Default model `nomic-embed-text` (768-d) — verified present on the dev machine.
Requires `ollama serve` to be running. Embeddings are cached per-text to avoid
re-querying identical node/knowledge descriptions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from r4r.embeddings.base import Embedder, l2_normalize

# Known output dims for common local models (avoids a probe call when provided).
_KNOWN_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None) -> None:
        import ollama  # imported lazily so the package imports without a server

        self._client = ollama.Client(host=host) if host else ollama
        self._model = model
        self._dim: Optional[int] = _KNOWN_DIMS.get(model)
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed_text("probe"))
        return self._dim

    def _embed_text(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        resp = self._client.embeddings(model=self._model, prompt=text)
        vec = np.asarray(resp["embedding"], dtype=np.float32)
        self._cache[text] = vec
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = [self._embed_text(t) for t in texts]
        return l2_normalize(np.vstack(vecs))
