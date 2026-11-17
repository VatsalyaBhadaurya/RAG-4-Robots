"""A deterministic tabletop scene (MVP target: 6-10 objects).

Ground-truth poses stand in for perception (labeled: MOCKED). Used by unit
tests, the Phase 3 retrieval demo, and the evaluation benchmark. Coordinates are
in metres, world frame, z-up. The table top sits at z = 0.75.

Layout (looking down, x right / y up):
    - table (surface) centred at origin, top at z=0.75
    - red_mug, blue_bowl, cereal_box, screwdriver, soda_can on the table
    - a small block stacked on the cereal_box (tests on_top_of / supports)
    - a bin (container) at the table edge (tests inside/contains once filled)
    - robot + end_effector beside the table
"""

from __future__ import annotations

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import Dimensions, NodeKind, Pose, SceneNode
from r4r.schemas.provenance import Provenance, SourceType

_TOP = 0.75  # table-top height


def _obj(oid, label, x, y, z, dx, dy, dz, kind=NodeKind.OBJECT, material=None,
         affordances=None, conf=0.95, attrs=None) -> SceneNode:
    return SceneNode(
        object_id=oid,
        kind=kind,
        label=label,
        pose=Pose(x=x, y=y, z=z),
        dims=Dimensions(x=dx, y=dy, z=dz),
        material=material,
        visibility=1.0,
        confidence=conf,
        affordances=affordances or [],
        attributes=attrs or {},
        # MOCKED perception: ground-truth from the scene definition.
        provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="sim_gt", confidence=conf),
    )


def build_tabletop_scene() -> SceneGraph:
    g = SceneGraph()

    # surface: table top centred at origin
    g.add_node(_obj("table_1", "table", 0.0, 0.0, _TOP - 0.02, 1.2, 0.8, 0.04,
                    kind=NodeKind.SURFACE, material="wood", affordances=["supports"]))

    # objects resting on the table (bottom at _TOP)
    g.add_node(_obj("red_mug_1", "mug", -0.30, 0.10, _TOP + 0.05, 0.08, 0.08, 0.10,
                    material="ceramic", affordances=["graspable_side", "containable"],
                    attrs={"color": "red"}))
    g.add_node(_obj("blue_bowl_1", "bowl", -0.10, -0.15, _TOP + 0.04, 0.14, 0.14, 0.08,
                    material="ceramic", affordances=["graspable_side", "container_like"],
                    attrs={"color": "blue"}))
    g.add_node(_obj("cereal_box_1", "box", 0.15, 0.12, _TOP + 0.10, 0.18, 0.06, 0.20,
                    material="cardboard", affordances=["graspable_side", "supports"]))
    g.add_node(_obj("screwdriver_1", "screwdriver", 0.18, 0.12, _TOP + 0.015, 0.20, 0.03, 0.03,
                    material="metal_plastic", affordances=["graspable_top"]))
    g.add_node(_obj("soda_can_1", "can", 0.35, -0.10, _TOP + 0.06, 0.07, 0.07, 0.12,
                    material="aluminium", affordances=["graspable_side"], attrs={"color": "silver"}))

    # a block stacked on top of the cereal box: bottom at _TOP + 0.20
    g.add_node(_obj("block_1", "block", 0.15, 0.12, _TOP + 0.20 + 0.025, 0.05, 0.05, 0.05,
                    material="wood", affordances=["graspable_top"], attrs={"color": "green"}))

    # a bin (container) near the table edge
    g.add_node(_obj("bin_1", "bin", 0.45, 0.30, _TOP + 0.12, 0.30, 0.30, 0.25,
                    kind=NodeKind.CONTAINER, material="plastic",
                    affordances=["container", "accepts_placement"]))

    # robot + end effector beside the table (no dims needed for reachability MVP)
    g.add_node(SceneNode(object_id="robot_1", kind=NodeKind.ROBOT, label="manipulator",
                         pose=Pose(x=-0.7, y=0.0, z=_TOP),
                         attributes={"workspace_radius": 0.85, "payload_kg": 2.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="robot_spec")))
    g.add_node(SceneNode(object_id="ee_1", kind=NodeKind.END_EFFECTOR, label="parallel_gripper",
                         pose=Pose(x=-0.7, y=0.0, z=_TOP + 0.2),
                         attributes={"max_width": 0.09, "max_force_n": 40.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="robot_spec")))

    return g
