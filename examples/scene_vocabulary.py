"""Print the vocabularies the system recognizes, so you know what to put in a
scene graph. Object IDs/labels are FREE-FORM (you invent them); but node kinds,
edge relations, and actions are strict enums, and specific materials/affordances/
labels trigger physics and safety behavior.

Run:  python examples/scene_vocabulary.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.knowledge import build_default_knowledge
from r4r.scenes import build_pick_place_scene, build_piper_scene_graph, build_tabletop_scene
from r4r.schemas.edges import RelationType
from r4r.schemas.nodes import KNOWLEDGE_KINDS, PHYSICAL_KINDS, NodeKind
from r4r.schemas.plan import ActionType, PlanFallback
from r4r.sim.mjcf import _DENSITY as SIM_DENSITY
from r4r.validation.validator import _DENSITY as VAL_DENSITY
from r4r.validation.validator import _FRAGILE


def line(t): print("\n" + t + "\n" + "-" * len(t))


def main() -> None:
    line("NODE KINDS (strict — the 'kind' field)")
    print("physical (need a pose):", sorted(k.value for k in PHYSICAL_KINDS))
    print("knowledge (static RAG):", sorted(k.value for k in KNOWLEDGE_KINDS))
    print("all:", [k.value for k in NodeKind])

    line("EDGE RELATIONS (strict — spatial auto-derived + knowledge)")
    print([r.value for r in RelationType])

    line("ACTIONS the planner may use (strict — validator rejects others)")
    print([a.value for a in ActionType])
    print("fallbacks:", [f.value for f in PlanFallback])

    line("MATERIALS with known density (payload + sim mass); others -> 'unknown_mass' warning")
    print("validator:", sorted(VAL_DENSITY))
    print("simulator:", sorted(SIM_DENSITY))
    print("fragile (need low grasp force):", sorted(_FRAGILE))

    line("AFFORDANCES the scorer/validator react to (free-form strings; these matter)")
    print("graspable*    -> graspability (e.g. graspable_side, graspable_top)")
    print("container / container_like / accepts_placement -> destination-like")
    print("supports      -> can hold other objects")
    print("containable   -> can be placed inside a container")

    line("LABELS that trigger KNOWLEDGE rules (from the seed knowledge base)")
    for n in build_default_knowledge():
        keys = {k: v for k, v in n.attributes.items()
                if k in ("applies_to_labels", "applies_to_materials", "object")}
        print(f"  {n.object_id:24s} [{n.kind.value:13s}] {keys}")

    line("OBJECT IDs/LABELS already defined in the built scenes (use as examples)")
    for name, builder in [("build_tabletop_scene", build_tabletop_scene),
                          ("build_pick_place_scene", build_pick_place_scene),
                          ("build_piper_scene_graph", build_piper_scene_graph)]:
        g = builder()
        print(f"\n  {name}():")
        for node in g.nodes():
            extra = f" material={node.material}" if node.material else ""
            aff = f" affordances={node.affordances}" if node.affordances else ""
            print(f"    {node.object_id:16s} kind={node.kind.value:12s} label={node.label}{extra}{aff}")


if __name__ == "__main__":
    main()
