"""Text embedding backends for GraphRAG entity/knowledge retrieval.

Two backends:
- HashEmbedder: deterministic, dependency-free, OFFLINE. Used in tests so CI
  never depends on a running model server.
- OllamaEmbedder: real semantic embeddings via a local Ollama model
  (default `nomic-embed-text`, 768-d). Used at runtime.

Both return L2-normalized vectors so cosine similarity == dot product.
"""

from __future__ import annotations

from r4r.embeddings.base import Embedder, cosine_sim, l2_normalize
from r4r.embeddings.hash_embedder import HashEmbedder
from r4r.embeddings.ollama_embedder import OllamaEmbedder

__all__ = ["Embedder", "HashEmbedder", "OllamaEmbedder", "cosine_sim", "l2_normalize"]
