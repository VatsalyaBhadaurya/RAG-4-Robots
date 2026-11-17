"""Task abstraction (EmbodiedRAG's 'abstraction step').

Extracts the entities/attributes a task refers to, so retrieval embeds focused
phrases rather than only the raw sentence. This MVP version is a deterministic,
model-free parser over common manipulation phrasings (put X in Y, pick up X,
place X on Y, which X ...). Phase 5 can swap in an LLM-based abstractor behind
the same TaskAbstraction interface.

Labeled: heuristic parser (Inferred adequacy for MVP phrasings; a learned/LLM
abstractor is the upgrade path).
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel

# destination prepositions split a "move X -> Y" instruction.
_DEST_PREPS = r"(?:\binto\b|\bin\b|\bonto\b|\bon\b|\bto\b|\binside\b)"
_PICK_VERBS = r"(?:pick up|pick|grasp|grab|take|lift|get)"
_PUT_VERBS = r"(?:put|place|move|drop|set|load)"
_STOP = {"the", "a", "an", "this", "that", "please", "could", "you", "robot"}


def _clean(phrase: str) -> str:
    toks = [t for t in re.findall(r"[a-zA-Z0-9]+", phrase.lower()) if t not in _STOP]
    return " ".join(toks).strip()


class TaskAbstraction(BaseModel):
    """Structured hints derived from a natural-language instruction."""

    instruction: str
    target_phrase: Optional[str] = None
    destination_phrase: Optional[str] = None
    is_query: bool = False  # "which objects ..." style question, not an action

    def query_texts(self) -> List[str]:
        """Phrases to embed for entity retrieval (de-duplicated, non-empty)."""
        out = [self.instruction]
        for p in (self.target_phrase, self.destination_phrase):
            if p:
                out.append(p)
        seen = set()
        uniq = []
        for t in out:
            key = t.lower().strip()
            if key and key not in seen:
                seen.add(key)
                uniq.append(t)
        return uniq


def abstract_task(instruction: str) -> TaskAbstraction:
    text = instruction.strip()
    low = text.lower()

    # question form: "which objects ... can be grasped"
    if low.startswith(("which", "what", "where", "can ", "is ", "are ")):
        return TaskAbstraction(instruction=text, target_phrase=_clean(text) or None, is_query=True)

    # put/place X <prep> Y
    m = re.search(_PUT_VERBS + r"\s+(.*?)\s+" + _DEST_PREPS + r"\s+(.*)", low)
    if m:
        return TaskAbstraction(
            instruction=text,
            target_phrase=_clean(m.group(1)) or None,
            destination_phrase=_clean(m.group(2)) or None,
        )

    # pick up X  (optionally "... next to/near Z" stays part of the target phrase)
    m = re.search(_PICK_VERBS + r"\s+(.*)", low)
    if m:
        return TaskAbstraction(instruction=text, target_phrase=_clean(m.group(1)) or None)

    # fallback: whole sentence as the target phrase
    return TaskAbstraction(instruction=text, target_phrase=_clean(text) or None)
