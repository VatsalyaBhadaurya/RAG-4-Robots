"""Real AgileX Piper arm executing a pick-and-place plan in MuJoCo, recorded to
video. The arm is driven by inverse kinematics; the plan is a strict ActionPlan
(the same schema the LLM planner produces).

Run:  python examples/arm_pick_place_demo.py
Output: runs/arm_execution.mp4 + runs/arm_execution.gif
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from r4r.schemas.plan import ActionPlan, ActionStep  # noqa: E402
from r4r.sim import PiperArmExecutor  # noqa: E402

RENDER_EVERY = 7


def _plan() -> ActionPlan:
    return ActionPlan(task="put_red_mug_in_bin", selected_target="red_mug_1",
                      destination="bin_1", steps=[
                          ActionStep(action="move_to", target="pregrasp_red_mug_1"),
                          ActionStep(action="grasp", target="red_mug_1", grasp_type="side", max_force="low"),
                          ActionStep(action="place", target="bin_1"),
                          ActionStep(action="release", target="red_mug_1")])


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    ex = PiperArmExecutor()
    renderer = mujoco.Renderer(ex.model, height=480, width=640)
    frames = []
    state = {"i": 0}

    def capture(executor):
        state["i"] += 1
        if state["i"] % RENDER_EVERY == 0:
            renderer.update_scene(executor.data, camera="fixed")
            frames.append(renderer.render())

    ex.on_step = capture
    result = ex.execute(_plan())
    renderer.close()

    print(f"success={result.success}  notes={result.notes}")
    print(f"target final pose: {[round(v,3) for v in result.final_poses[result.target].position()]}")

    import imageio
    imageio.mimsave("runs/arm_execution.mp4", frames, fps=40)
    imageio.mimsave("runs/arm_execution.gif", frames[::2], duration=0.05)
    print(f"wrote runs/arm_execution.mp4 + .gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
