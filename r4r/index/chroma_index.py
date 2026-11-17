"""Optional Chroma-backed vector index (the chosen persistent backend).

Imported lazily; requires `chromadb`. We pass precomputed embeddings in and out
(no Chroma embedding function), so this never pulls onnxruntime/sentence models.
We use cosine space to match NumpyVectorIndex semantics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from r4r.embeddings.base import l2_normalize
from r4r.index.base import ScoredHit, VectorIndex


class ChromaVectorIndex(VectorIndex):
    def __init__(self, collection: str = "r4r", persist_dir: Optional[str] = None) -> None:
        import chromadb  # lazy

        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir
            else chromadb.EphemeralClient()
        )
        # fresh collection each run unless persisting
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass
        self._col = self._client.create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._n = 0

    def add(self, ids: List[str], vectors: np.ndarray,
            metadatas: List[Dict[str, object]]) -> None:
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        # Chroma requires non-empty metadata dicts with scalar values
        safe_meta = [{k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))} or {"_": 1}
                     for m in metadatas]
        self._col.add(ids=list(ids), embeddings=vecs.tolist(), metadatas=safe_meta)
        self._n += len(ids)

    def query(self, vector: np.ndarray, k: int = 5) -> List[ScoredHit]:
        if self._n == 0:
            return []
        q = l2_normalize(np.asarray(vector, dtype=np.float32))
        res = self._col.query(query_embeddings=[q.tolist()], n_results=min(k, self._n))
        hits: List[ScoredHit] = []
        ids = res["ids"][0]
        dists = res["distances"][0]
        metas = res.get("metadatas", [[{}] * len(ids)])[0]
        for i, (hid, dist) in enumerate(zip(ids, dists)):
            # cosine distance -> similarity
            hits.append(ScoredHit(id=hid, score=float(1.0 - dist), metadata=metas[i] or {}))
        return hits

    def __len__(self) -> int:
        return self._n
