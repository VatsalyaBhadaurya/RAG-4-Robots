"""A small seed knowledge base for tabletop manipulation.

These are illustrative STATIC documents (labeled: seed content, not an
authoritative manual). They exist so GraphRAG can retrieve task-relevant
knowledge chunks alongside the scene subgraph, and so the Phase 6 validator has
machine-readable safety rules to check against.
"""

from __future__ import annotations

from typing import List

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import NodeKind, SceneNode
from r4r.schemas.provenance import Provenance, SourceType


def _doc(oid: str, kind: NodeKind, label: str, text: str,
         attributes=None) -> SceneNode:
    attrs = {"text": text}
    if attributes:
        attrs.update(attributes)
    return SceneNode(
        object_id=oid,
        kind=kind,
        label=label,
        affordances=[],
        attributes=attrs,
        provenance=Provenance(source_type=SourceType.STATIC, source_id=f"kb:{oid}"),
    )


def build_default_knowledge() -> List[SceneNode]:
    return [
        _doc("safety_sharp_tools", NodeKind.SAFETY_RULE, "handle sharp tools safely",
             "Sharp or pointed tools such as screwdrivers and knives must be grasped "
             "by the handle, never by the tip or blade. Keep the pointed end away from "
             "any detected human.",
             attributes={"applies_to_labels": "screwdriver,knife", "severity": "high"}),
        _doc("safety_fragile_force", NodeKind.SAFETY_RULE, "low force for fragile items",
             "Ceramic and glass objects (mugs, bowls, plates) are fragile. Use a low "
             "grasp force and gentle placement to avoid breakage.",
             attributes={"applies_to_materials": "ceramic,glass", "severity": "medium"}),
        _doc("safety_human_proximity", NodeKind.SAFETY_RULE, "human proximity stop",
             "If a human is within the robot workspace geofence, stop motion and "
             "request confirmation before continuing.",
             attributes={"severity": "high"}),
        _doc("constraint_payload", NodeKind.CONSTRAINT, "payload limit",
             "The manipulator payload limit is 2 kg. Do not attempt to lift objects "
             "whose estimated mass exceeds the payload limit.",
             attributes={"payload_kg": 2.0}),
        _doc("template_put_in_container", NodeKind.TASK_TEMPLATE, "put object in container",
             "To place an object into a container: move to a pre-grasp pose above the "
             "object, grasp it, lift, move over the container opening, then gently "
             "release. Verify the container is open and reachable first.",
             attributes={"task": "put_object_in_container"}),
        _doc("proc_side_grasp_mug", NodeKind.PROCEDURE, "side grasp for mugs",
             "Mugs are best grasped from the side across the body (or by the handle if "
             "detected), using a parallel gripper opened to just wider than the mug "
             "diameter, with low force.",
             attributes={"object": "mug", "grasp_type": "side"}),
        _doc("proc_top_grasp_tool", NodeKind.PROCEDURE, "top grasp for tools",
             "Elongated tools such as screwdrivers are grasped from the top across the "
             "handle shaft, aligning the gripper perpendicular to the tool axis.",
             attributes={"object": "screwdriver", "grasp_type": "top"}),
    ]


def add_knowledge(graph: SceneGraph, nodes=None) -> SceneGraph:
    """Add knowledge documents to a scene graph (in place) and return it."""
    for n in (nodes if nodes is not None else build_default_knowledge()):
        graph.upsert_node(n)
    return graph
