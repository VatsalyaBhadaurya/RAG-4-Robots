"""Run the Piper arm on a CUSTOM scene you define, recorded to video.

Shows how to test any layout: build an arm-centric SceneGraph (arm at origin,
target + bin within ~0.55 m reach, table top at z=0, object < ~0.07 m), then run
the full pipeline with the real arm. The arm MJCF is generated from your graph.

Edit TARGET_XY / BIN_XY below, then:
    python examples/arm_custom_scene_demo.py
Output: runs/arm_custom.mp4
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
from r4r.scene_graph import infer_spatial_edges  # noqa: E402
from r4r.scenes import build_piper_scene_graph  # noqa: E402

# ---- edit your layout here (metres, arm base at origin) ----
TARGET_XY = (0.40, -0.06)
BIN_XY = (0.30, 0.24)
INSTRUCTION = "Put the red mug in the bin."

_CANNED = (
    '{"task":"put","selected_target":"red_mug_1","destination":"bin_1",'
    '"steps":[{"action":"move_to","target":"pregrasp_red_mug_1"},'
    '{"action":"grasp","target":"red_mug_1","grasp_type":"side","max_force":"low"},'
    '{"action":"place","target":"bin_1"},{"action":"release","target":"red_mug_1"}],'
    '"fallback":"ask_human"}'
)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    g = build_piper_scene_graph(target_xy=TARGET_XY, bin_xy=BIN_XY)
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    add_knowledge(g)

    frames = []
    rec = {"r": None, "i": 0}

    def capture(ex):
        if rec["r"] is None:
            rec["r"] = mujoco.Renderer(ex.model, height=480, width=640)
        rec["i"] += 1
        if rec["i"] % 7 == 0:
            rec["r"].update_scene(ex.data, camera="fixed")
            frames.append(rec["r"].render())

    log, g = run_episode(g, INSTRUCTION, HashEmbedder(512),
                         MockBackend(responses=[_CANNED]), execute=True,
                         execution_backend="piper", on_exec_step=capture)

    print(f"target_xy={TARGET_XY} bin_xy={BIN_XY}")
    print(f"validation ok={log.validation_ok} violations={[v['code'] for v in log.validation_violations]}")
    print(f"executed={log.executed} success={log.success} notes={log.execution_notes}")
    if frames:
        import imageio
        imageio.mimsave("runs/arm_custom.mp4", frames, fps=40)
        print(f"wrote runs/arm_custom.mp4 ({len(frames)} frames)")
    else:
        print("blocked by safety guard (object/bin likely out of reach) — adjust TARGET_XY/BIN_XY")


if __name__ == "__main__":
    main()
