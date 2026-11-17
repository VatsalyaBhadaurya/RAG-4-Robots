"""Append-only JSONL episode logger."""

from __future__ import annotations

import json
import os
from typing import List

from r4r.logging_.episode import EpisodeLog


class EpisodeLogger:
    def __init__(self, path: str = "runs/episodes.jsonl") -> None:
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, log: EpisodeLog) -> str:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log.model_dump(), default=str) + "\n")
        return self.path

    def read_all(self) -> List[EpisodeLog]:
        if not os.path.exists(self.path):
            return []
        out: List[EpisodeLog] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(EpisodeLog.model_validate_json(line))
        return out
