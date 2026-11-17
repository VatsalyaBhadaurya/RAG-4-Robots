"""Full embodied loop driving the REAL Piper arm, recorded to video.

retrieve -> score -> plan (llama3.2 if served, else canned) -> validate (gate)
-> AgileX Piper arm executes via IK (only if validation passes) -> world update
-> episode log. Writes runs/arm_full_loop.mp4.

Run:  python examples/arm_full_loop_demo.py "Put the red mug in the bin."
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.knowledge import add_knowledge  # noqa: E402
from r4r.logging_ import EpisodeLogger  # noqa: E402
from r4r.pipeline import run_episode  # noqa: E402
from r4r.planner import MockBackend  # noqa: E402
from r4r.scene_graph import infer_spatial_edges  # noqa: E402
from r4r.scenes import build_piper_scene_graph  # noqa: E402

_CANNED = json.dumps({
    "task": "put_cereal_in_box", "selected_target": "cereal_box_1", "destination": "bin_1",
    "steps": [{"action": "move_to", "target": "pregrasp_red_mug_1"},
              {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
              {"action": "place", "target": "bin_1"},
              {"action": "release", "target": "red_mug_1"}], "fallback": "ask_human"})


def _backend():
    try:
        import ollama
        ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 1})
        from r4r.planner import OllamaChatBackend
        print("Planner: OllamaChatBackend(llama3.2:latest)")
        return OllamaChatBackend("llama3.2:latest")
    except Exception as exc:  # noqa: BLE001
        print(f"Planner: MockBackend (Ollama unavailable: {exc})")
        return MockBackend(responses=[_CANNED])


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    instruction = sys.argv[1] if len(sys.argv) > 1 else "Put the red mug in the bin."
    g = build_piper_scene_graph()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)
    add_knowledge(g)

    frames = []
    rec = {"r": None, "i": 0}

    def capture(executor):
        if rec["r"] is None:
            rec["r"] = mujoco.Renderer(executor.model, height=480, width=640)
        rec["i"] += 1
        if rec["i"] % 7 == 0:
            rec["r"].update_scene(executor.data, camera="fixed")
            frames.append(rec["r"].render())

    log, g = run_episode(g, instruction, HashEmbedder(512), _backend(),
                         logger=EpisodeLogger("runs/episodes.jsonl"), execute=True,
                         execution_backend="piper", on_exec_step=capture)

    print(f"\nInstruction: {instruction}")
    print(f"Seeds: {log.retrieval_seeds}  knowledge: {log.knowledge_ids}")
    print(f"Plan valid={log.plan_valid} fallback={log.plan_used_fallback} backend={log.planner_backend}")
    print(f"Validation ok={log.validation_ok}  violations={[v['code'] for v in log.validation_violations]}")
    print(f"Executed={log.executed}  success={log.success}  failure_category={log.failure_category}")
    print(f"Execution notes: {log.execution_notes}")
    print(f"Mug pose after: {[round(v,3) for v in g.get_node('red_mug_1').pose.position()]}")

    if frames:
        import imageio
        imageio.mimsave("runs/arm_full_loop.mp4", frames, fps=40)
        print(f"wrote runs/arm_full_loop.mp4 ({len(frames)} frames)")
    else:
        print("(no frames — execution was blocked by the safety guard)")


if __name__ == "__main__":
    main()
