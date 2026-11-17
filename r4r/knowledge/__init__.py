"""Static-knowledge RAG side: manuals, safety rules, task templates, procedures.

Knowledge is represented as SceneNodes with knowledge NodeKinds and the document
text in attributes['text'], so the one heterogeneous graph + one vector index
serve both the dynamic scene and the static knowledge (EmbodiedRAG keeps them
separate; we unify the store but tag provenance as STATIC).
"""

from __future__ import annotations

from r4r.knowledge.seed import add_knowledge, build_default_knowledge

__all__ = ["add_knowledge", "build_default_knowledge"]
