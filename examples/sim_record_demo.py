"""Record the MuJoCo execution to a video file (no GUI needed).

Runs a known-good put-in-bin plan through the physics executor, rendering the
offscreen camera every few steps, and writes runs/execution.mp4 (+ a .gif
fallback). Open the file to watch the mug get carried and dropped into the bin.

Run:  python examples/sim_record_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from r4r.scenes import build_pick_place_scene  # noqa: E402
from r4r.schemas.plan import ActionPlan, ActionStep  # noqa: E402
from r4r.sim import MujocoExecutor  # noqa: E402

RENDER_EVERY = 8          # sim steps between captured frames
HEIGHT, WIDTH = 480, 640


def _plan() -> ActionPlan:
    return ActionPlan(task="put_red_mug_in_bin", selected_target="red_mug_1",
                      destination="bin_1", steps=[
                          ActionStep(action="move_to", target="pregrasp_red_mug_1"),
                          ActionStep(action="grasp", target="red_mug_1", grasp_type="side", max_force="low"),
                          ActionStep(action="place", target="bin_1"),
                          ActionStep(action="release", target="red_mug_1")])


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    ex = MujocoExecutor(build_pick_place_scene())
    renderer = mujoco.Renderer(ex.model, height=HEIGHT, width=WIDTH)
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

    print(f"execution success={result.success}; captured {len(frames)} frames")
    print("notes:", result.notes)

    import imageio
    mp4 = os.path.join("runs", "execution.mp4")
    gif = os.path.join("runs", "execution.gif")
    try:
        imageio.mimsave(mp4, frames, fps=40)
        print("wrote", mp4)
    except Exception as exc:  # noqa: BLE001
        print("mp4 failed (%s); writing gif" % exc)
    imageio.mimsave(gif, frames[::2], duration=0.04)  # lighter gif
    print("wrote", gif)


if __name__ == "__main__":
    main()
