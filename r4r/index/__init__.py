"""Vector index backends.

- NumpyVectorIndex: in-memory exact cosine search. Always available, deterministic,
  ideal for tabletop-scale graphs and tests.
- ChromaVectorIndex: optional persistent backend (the stack decision was
  NetworkX + Chroma/FAISS). Imported lazily so core stays dependency-light.
"""

from __future__ import annotations

from r4r.index.base import ScoredHit, VectorIndex
from r4r.index.numpy_index import NumpyVectorIndex

__all__ = ["ScoredHit", "VectorIndex", "NumpyVectorIndex"]
