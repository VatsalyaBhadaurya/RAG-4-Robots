"""Vector index interface."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pydantic import BaseModel


class ScoredHit(BaseModel):
    id: str
    score: float
    metadata: Dict[str, object] = {}


class VectorIndex:
    """Base class for a similarity index over string ids."""

    def add(self, ids: List[str], vectors: np.ndarray,
            metadatas: List[Dict[str, object]]) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, k: int = 5) -> List[ScoredHit]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
