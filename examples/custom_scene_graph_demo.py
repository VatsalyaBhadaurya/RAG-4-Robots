"""Template: build your OWN SceneGraph from scratch and run it on the real arm.

This shows every node you need. Edit the objects, then run:
    python examples/custom_scene_graph_demo.py
Output: runs/custom_graph.mp4 + the full episode trace printed.

Rules for the Piper arm executor (it's a fixed tabletop arm):
  - robot_1 at the origin (0,0,0), workspace_radius ~0.55 m
  - put objects WITHIN reach of the origin and keep the table top at z = 0
  - the grasp target must be an OBJECT, smaller than ~0.07 m (gripper opening)
  - the destination must be a CONTAINER (auto-built as an open-top bin)
If a scene breaks reach/size, the validator BLOCKS execution and says why.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.knowledge import add_knowledge  # noqa: E402
from r4r.pipeline import run_episode  # noqa: E402
from r4r.planner import MockBackend  # noqa: E402
from r4r.scene_graph import SceneGraph, infer_spatial_edges  # noqa: E402
from r4r.schemas import Dimensions, NodeKind, Pose, SceneNode  # noqa: E402
from r4r.schemas.provenance import Provenance, SourceType  # noqa: E402

INSTRUCTION = "Put the red mug in the bin."


def perceived(object_id, kind, label, pos, dims, material=None, affordances=None, attrs=None):
    """Helper to make a node as if it came from perception (ground-truth pose)."""
    return SceneNode(
        object_id=object_id, kind=kind, label=label,
        pose=Pose(x=pos[0], y=pos[1], z=pos[2]),
        dims=Dimensions(x=dims[0], y=dims[1], z=dims[2]),
        material=material, visibility=1.0, confidence=0.95,
        affordances=affordances or [], attributes=attrs or {},
        provenance=Provenance(source_type=SourceType.PERCEPTION, source_id="cam0", confidence=0.95),
    )


def build_my_scene() -> SceneGraph:
    g = SceneGraph()

    # 1) a table surface, top at z = 0 (the arm is mounted on it)
    g.add_node(perceived("table_1", NodeKind.SURFACE, "table",
                         pos=(0.30, 0.0, -0.03), dims=(1.0, 1.0, 0.06),
                         material="wood", affordances=["supports"]))

    # 2) your objects (OBJECT kind). Keep them < ~0.07 m and within reach.
    g.add_node(perceived("cereal_box_1", NodeKind.OBJECT, "box",
                         pos=(0.38, 0.05, 0.024), dims=(0.044, 0.044, 0.048),
                         material="cardboard", affordances=["graspable_side", "containable"],
                         attrs={"color": "red"}))
    # a distractor object the planner/validator must NOT pick for this task
    g.add_node(perceived("green_block_1", NodeKind.OBJECT, "block",
                         pos=(0.33, -0.12, 0.022), dims=(0.04, 0.04, 0.044),
                         material="wood", affordances=["graspable_side"],
                         attrs={"color": "green"}))

    # 3) a destination CONTAINER (auto-built as an open-top bin)
    g.add_node(perceived("bin_1", NodeKind.CONTAINER, "bin",
                         pos=(0.28, 0.26, 0.06), dims=(0.18, 0.18, 0.12),
                         material="plastic", affordances=["container", "accepts_placement"]))

    # 4) the robot + end-effector (NOT rendered as bodies — the Piper model is
    #    the arm; these drive reachability/gripper checks in the validator)
    g.add_node(SceneNode(object_id="robot_1", kind=NodeKind.ROBOT, label="piper_arm",
                         pose=Pose(x=0.0, y=0.0, z=0.0),
                         attributes={"workspace_radius": 0.55, "payload_kg": 2.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="robot_spec")))
    g.add_node(SceneNode(object_id="ee_1", kind=NodeKind.END_EFFECTOR, label="parallel_gripper",
                         pose=Pose(x=0.0, y=0.0, z=0.2),
                         attributes={"max_width": 0.07, "max_force_n": 20.0},
                         provenance=Provenance(source_type=SourceType.STATIC, source_id="robot_spec")))

    # 5) derive spatial edges + attach static knowledge (safety rules, procedures)
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    add_knowledge(g)
    return g


# A canned plan so the demo is deterministic/offline. Swap MockBackend for
# OllamaChatBackend("llama3.2:latest") to let the LLM plan instead.
_CANNED = (
    '{"task":"put_red_mug_in_bin","selected_target":"red_mug_1","destination":"bin_1",'
    '"steps":[{"action":"move_to","target":"pregrasp_red_mug_1"},'
    '{"action":"grasp","target":"red_mug_1","grasp_type":"side","max_force":"low"},'
    '{"action":"place","target":"bin_1"},{"action":"release","target":"red_mug_1"}],'
    '"fallback":"ask_human"}'
)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    g = build_my_scene()

    frames = []
    rec = {"r": None, "i": 0}

    def capture(ex):
        if rec["r"] is None:
            rec["r"] = mujoco.Renderer(ex.model, height=480, width=640)
        rec["i"] += 1
        if rec["i"] % 7 == 0:
            rec["r"].update_scene(ex.data, camera="fixed")
            frames.append(rec["r"].render())

    log, g = run_episode(
        g, INSTRUCTION, HashEmbedder(512),
        MockBackend(responses=[_CANNED]),      # <- or OllamaChatBackend("llama3.2:latest")
        execute=True, execution_backend="piper", on_exec_step=capture,
    )

    print("Instruction:", INSTRUCTION)
    print("Retrieved seeds:", log.retrieval_seeds, "| knowledge:", log.knowledge_ids)
    print("Validation ok:", log.validation_ok, "violations:", [v["code"] for v in log.validation_violations])
    print("Executed:", log.executed, "success:", log.success, "notes:", log.execution_notes)
    print("red_mug_1 final pose:", [round(v, 3) for v in g.get_node("red_mug_1").pose.position()])

    if frames:
        import imageio
        imageio.mimsave("runs/custom_graph.mp4", frames, fps=40)
        print(f"wrote runs/custom_graph.mp4 ({len(frames)} frames)")
    else:
        print("No frames: execution was blocked by the safety guard (check reach/size).")


if __name__ == "__main__":
    main()
