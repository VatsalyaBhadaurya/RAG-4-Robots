"""Watch the MuJoCo execution LIVE in an interactive window.

Opens the MuJoCo passive viewer and runs the put-in-bin plan in real time, so
you can orbit/zoom while the mug is carried and dropped into the bin. Requires a
display (run it locally, not over a headless shell).

Run:  python examples/sim_view_demo.py
Controls: drag to orbit, scroll to zoom, double-click to track a body.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from r4r.scenes import build_pick_place_scene  # noqa: E402
from r4r.schemas.plan import ActionPlan, ActionStep  # noqa: E402
from r4r.sim import MujocoExecutor  # noqa: E402


def _plan() -> ActionPlan:
    return ActionPlan(task="put_red_mug_in_bin", selected_target="red_mug_1",
                      destination="bin_1", steps=[
                          ActionStep(action="move_to", target="pregrasp_red_mug_1"),
                          ActionStep(action="grasp", target="red_mug_1", grasp_type="side", max_force="low"),
                          ActionStep(action="place", target="bin_1"),
                          ActionStep(action="release", target="red_mug_1")])


def main() -> None:
    ex = MujocoExecutor(build_pick_place_scene())
    with mujoco.viewer.launch_passive(ex.model, ex.data) as viewer:
        def sync(executor):
            viewer.sync()
            time.sleep(executor.model.opt.timestep)  # run ~real time
        ex.on_step = sync
        print("Executing in viewer... (close the window to exit)")
        result = ex.execute(_plan())
        print(f"success={result.success}; notes={result.notes}")
        # keep the window open to inspect the final state
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
