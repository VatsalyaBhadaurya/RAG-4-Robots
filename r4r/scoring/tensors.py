"""Convert a retrieved subgraph into GNN-ready tensors.

Core is NumPy (no torch dependency), so the schema is testable anywhere. Two
adapters are provided, imported lazily:
- to_pyg_data() / to_hetero_data(): PyTorch Geometric (the chosen GNN layer).
- to_graphnet_bundle(): a neutral tensor bundle a GraphNeT backend could wrap
  (full GraphNeT `Data` integration is DEFERRED — see decision D-01).

Fixed node-feature layout (F = 13), with a parallel mask marking features that
were missing (e.g. no dimensions) so a model can learn to ignore them rather
than trust a fabricated zero:
    0 dim_x  1 dim_y  2 dim_z  3 volume  4 min_dim
    5 pos_x  6 pos_y  7 pos_z  8 visibility  9 confidence
    10 aff_graspable  11 aff_container  12 aff_supports
Geometry features (0..7) are masked out when pose/dims are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.edges import RelationType
from r4r.schemas.nodes import NodeKind, SceneNode

NODE_FEATURE_DIM = 13
_NODE_KINDS = list(NodeKind)
_REL_TYPES = list(RelationType)
_KIND_INDEX = {k: i for i, k in enumerate(_NODE_KINDS)}
_REL_INDEX = {r: i for i, r in enumerate(_REL_TYPES)}


def _node_features(n: SceneNode) -> (np.ndarray):
    f = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    m = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    if n.dims is not None:
        f[0], f[1], f[2] = n.dims.x, n.dims.y, n.dims.z
        f[3] = n.dims.x * n.dims.y * n.dims.z
        f[4] = min(n.dims.x, n.dims.y, n.dims.z)
        m[0:5] = 1.0
    if n.pose is not None:
        f[5], f[6], f[7] = n.pose.x, n.pose.y, n.pose.z
        m[5:8] = 1.0
    f[8] = n.visibility
    f[9] = n.confidence
    m[8:10] = 1.0
    affs = set(n.affordances)
    f[10] = 1.0 if any(a.startswith("graspable") for a in affs) else 0.0
    f[11] = 1.0 if ("container" in affs or "container_like" in affs) else 0.0
    f[12] = 1.0 if "supports" in affs else 0.0
    m[10:13] = 1.0
    return f, m


@dataclass
class SubgraphTensors:
    node_ids: List[str]
    node_type: np.ndarray        # (N,) int
    node_features: np.ndarray    # (N, F)
    node_feature_mask: np.ndarray  # (N, F)
    edge_index: np.ndarray       # (2, E) int
    edge_type: np.ndarray        # (E,) int
    edge_attr: np.ndarray        # (E, R+1): rel one-hot + confidence
    node_labels: Optional[np.ndarray] = None  # targets, filled later (Phase 4b/6)
    label_mask: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def index_of(self, node_id: str) -> int:
        return self.node_ids.index(node_id)

    # ---- adapters (lazy imports) ----
    def to_pyg_data(self):
        """Homogeneous torch_geometric.data.Data (node_type kept as an attr)."""
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.tensor(self.node_features),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr),
            node_type=torch.tensor(self.node_type, dtype=torch.long),
            feature_mask=torch.tensor(self.node_feature_mask),
        )

    def to_hetero_data(self):
        """PyTorch Geometric HeteroData, nodes grouped by NodeKind and edges by
        (src_kind, relation, dst_kind) — the native heterogeneous layout."""
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        # local index per type
        local: Dict[int, int] = {}
        per_type_rows: Dict[str, List[int]] = {}
        for gi, kidx in enumerate(self.node_type):
            kname = _NODE_KINDS[int(kidx)].value
            per_type_rows.setdefault(kname, [])
            local[gi] = len(per_type_rows[kname])
            per_type_rows[kname].append(gi)
        for kname, rows in per_type_rows.items():
            data[kname].x = torch.tensor(self.node_features[rows])
            data[kname].node_ids = [self.node_ids[i] for i in rows]

        # group edges by canonical (src_kind, rel, dst_kind)
        grouped: Dict[tuple, List[tuple]] = {}
        for e in range(self.num_edges):
            s, d = int(self.edge_index[0, e]), int(self.edge_index[1, e])
            rel = _REL_TYPES[int(self.edge_type[e])].value
            key = (_NODE_KINDS[int(self.node_type[s])].value, rel,
                   _NODE_KINDS[int(self.node_type[d])].value)
            grouped.setdefault(key, []).append((local[s], local[d], self.edge_attr[e, -1]))
        for key, triples in grouped.items():
            ei = torch.tensor([[t[0] for t in triples], [t[1] for t in triples]], dtype=torch.long)
            ea = torch.tensor([[t[2]] for t in triples], dtype=torch.float32)
            data[key].edge_index = ei
            data[key].edge_attr = ea
        return data

    def to_graphnet_bundle(self) -> Dict[str, np.ndarray]:
        """Neutral tensor bundle for a possible GraphNeT backend (adapter stub).

        GraphNeT wraps PyG under the hood, so these arrays are what a GraphNeT
        `Data`/`Detector` layer would consume. Full GraphNeT integration is
        deferred (D-01); this keeps the option open without a hard dependency."""
        return {
            "x": self.node_features,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "node_type": self.node_type,
        }


def subgraph_to_tensors(subgraph: SceneGraph) -> SubgraphTensors:
    nodes = subgraph.nodes()
    node_ids = [n.object_id for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}

    feats = np.zeros((len(nodes), NODE_FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((len(nodes), NODE_FEATURE_DIM), dtype=np.float32)
    ntype = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        f, m = _node_features(n)
        feats[i], masks[i] = f, m
        ntype[i] = _KIND_INDEX[n.kind]

    edges = subgraph.edges()
    E = len(edges)
    ei = np.zeros((2, E), dtype=np.int64)
    etype = np.zeros(E, dtype=np.int64)
    eattr = np.zeros((E, len(_REL_TYPES) + 1), dtype=np.float32)
    for j, e in enumerate(edges):
        ei[0, j], ei[1, j] = idx[e.src], idx[e.dst]
        r = _REL_INDEX[e.rel]
        etype[j] = r
        eattr[j, r] = 1.0
        eattr[j, -1] = e.confidence

    return SubgraphTensors(
        node_ids=node_ids,
        node_type=ntype,
        node_features=feats,
        node_feature_mask=masks,
        edge_index=ei,
        edge_type=etype,
        edge_attr=eattr,
    )
