"""Full embodied loop (Phases 2-6) on the reachable pick-place scene:
retrieve -> score -> plan (llama3.2 if served, else canned) -> validate ->
MuJoCo execution (only if validation passes) -> world update -> episode log.

Run:  python examples/full_loop_demo.py "Put the red mug in the bin."
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r4r.embeddings import HashEmbedder  # noqa: E402
from r4r.logging_ import EpisodeLogger  # noqa: E402
from r4r.pipeline import run_episode  # noqa: E402
from r4r.planner import MockBackend  # noqa: E402
from r4r.scene_graph import infer_spatial_edges  # noqa: E402
from r4r.scenes import build_pick_place_scene  # noqa: E402

_CANNED = json.dumps({
    "task": "put_red_mug_in_bin", "assumptions": [], "selected_target": "red_mug_1",
    "destination": "bin_1", "steps": [
        {"action": "move_to", "target": "pregrasp_red_mug_1"},
        {"action": "grasp", "target": "red_mug_1", "grasp_type": "side", "max_force": "low"},
        {"action": "place", "target": "bin_1"}],
    "risks": [], "required_confirmations": [], "fallback": "ask_human"})


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
    instruction = sys.argv[1] if len(sys.argv) > 1 else "Put the red mug in the bin."
    g = build_pick_place_scene()
    for e in infer_spatial_edges(g.nodes()):
        g.add_edge(e)

    before = [round(v, 3) for v in g.get_node("red_mug_1").pose.position()]
    log, g = run_episode(g, instruction, HashEmbedder(512), _backend(),
                         logger=EpisodeLogger("runs/episodes.jsonl"), execute=True)

    print(f"\nInstruction: {instruction}")
    print(f"Retrieved seeds: {log.retrieval_seeds}  knowledge: {log.knowledge_ids}")
    print(f"Plan valid={log.plan_valid} fallback={log.plan_used_fallback} "
          f"attempts={log.plan_attempts} backend={log.planner_backend}")
    print(f"Validation ok={log.validation_ok}  "
          f"violations={[v['code'] for v in log.validation_violations]}")
    print(f"Executed={log.executed}  success={log.success}  "
          f"failure_category={log.failure_category}")
    print(f"Execution notes: {log.execution_notes}")
    print(f"Mug pose: before={before}  after={ [round(v,3) for v in g.get_node('red_mug_1').pose.position()] }")
    print(f"\nEpisode {log.episode_id} logged to runs/episodes.jsonl")


if __name__ == "__main__":
    main()
