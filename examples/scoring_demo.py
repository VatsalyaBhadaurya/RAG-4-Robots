"""Phase 4 demo: heuristic scoring + GNN tensor schema over the tabletop scene.

Shows the deterministic scorer output (the baseline a future GNN must beat) and
the tensor conversion (NumPy + PyTorch Geometric HeteroData if installed).

Run:  python examples/scoring_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.eval.benchmark import build_eval_graph  # noqa: E402
from r4r.scoring import score_scene, subgraph_to_tensors  # noqa: E402


def main() -> None:
    g = build_eval_graph()

    report = score_scene(g)
    print("Heuristic scores (method=%s)\n" % report.method)
    print(f"{'object':16s} {'reach':>6s} {'grasp':>6s} {'risk':>6s} {'stab':>6s} {'graspable':>9s}")
    for oid in report.ranking:
        s = report.for_object(oid)
        print(f"{oid:16s} {s.reachability:6.2f} {s.graspability:6.2f} "
              f"{s.collision_risk:6.2f} {s.support_stability:6.2f} {str(s.graspable):>9s}")
    print(f"\nBest candidate to grasp: {report.best()}")
    print("Why:", "; ".join(report.for_object(report.best()).explanations))

    t = subgraph_to_tensors(g)
    print(f"\nTensor schema: nodes={t.num_nodes} (feat dim {t.node_features.shape[1]}), "
          f"edges={t.num_edges}, edge_attr dim {t.edge_attr.shape[1]}")
    try:
        data = t.to_hetero_data()
        print("PyG HeteroData node types:", list(data.node_types))
        print("PyG HeteroData edge types:", [f"{a}-{r}->{b}" for (a, r, b) in data.edge_types])
    except Exception as exc:  # noqa: BLE001
        print(f"(PyTorch Geometric not available: {exc})")


if __name__ == "__main__":
    main()
