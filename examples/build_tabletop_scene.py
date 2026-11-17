"""Demo: build the tabletop scene, infer spatial edges, print a summary, and
show the induced subgraph around the red mug (the Phase 3 retrieval primitive).

Run:  python examples/build_tabletop_scene.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.scene_graph import infer_spatial_edges  # noqa: E402
from r4r.scenes import build_tabletop_scene  # noqa: E402


def main() -> None:
    g = build_tabletop_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)

    n, m = g.stats()
    print(f"Scene: {n} nodes, {m} edges   content_hash={g.content_hash()}")

    print("\nObjects on the table:")
    for node in g.objects_on("table_1"):
        print(f"  - {node.object_id:16s} ({node.label})")

    print("\nDerived relations touching red_mug_1:")
    for e in g.out_edges("red_mug_1") + g.in_edges("red_mug_1"):
        print(f"  {e.src:14s} --{e.rel.value:11s}--> {e.dst:14s}  conf={e.confidence:.2f}")

    print("\nGraphRAG primitive — induced subgraph (depth=1) around red_mug_1:")
    sub = g.induced_subgraph(["red_mug_1"], depth=1)
    print(f"  nodes: {sorted(sub.node_ids())}")
    print(f"  (full scene has {len(g)} nodes; retrieved {len(sub)})")


if __name__ == "__main__":
    main()
