"""Real AgileX Piper arm execution in MuJoCo (plan-driven).

Loads a combined scene (Piper 6-DoF arm + table + a graspable cube + an open-top
bin, from MuJoCo Menagerie) and executes an ActionPlan's pick-and-place by:
- solving position inverse kinematics (damped least squares on the arm's 6 DoF)
  to drive the position actuators to each waypoint, and
- attaching the target to the gripper kinematically during the carry, then
  releasing it to physics over the bin.

Honesty (R-11): the ARM is fully simulated (position actuators, contacts,
gravity) and moved by real IK. Straight-down grasping is outside this small
arm's dexterous workspace, so the grasp BOND is kinematic (cube follows the
gripper while held) rather than a contact/force-closure grasp. Arm + IK +
contact grasping remains the upgrade path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import Pose
from r4r.schemas.plan import ActionPlan
from r4r.sim.executor import ExecutionResult
from r4r.sim.mjcf import SimMeta, scene_elements

PIPER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party", "mujoco_menagerie", "agilex_piper",
)
PIPER_SCENE = os.path.join(PIPER_DIR, "_r4r_arm_scene.xml")


def build_piper_mjcf(graph: SceneGraph):
    """Generate a combined MJCF (Piper arm + the graph's surfaces/objects/
    containers) so ANY scene graph can be executed by the arm. The arm base is
    at the world origin; place objects within its reach (~0.55 m) and the table
    top near z = 0. Objects must be smaller than the ~0.07 m gripper opening.

    Returns (xml_string, SimMeta)."""
    static_xml, body_xml, meta = scene_elements(graph)
    xml = f"""<mujoco model="r4r_piper_generated">
  <include file="piper.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="130" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.6 0.72" rgb2="0.1 0.12 0.16" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.3 0.32 0.35" rgb2="0.22 0.24 0.27"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.5" dir="-0.2 0.2 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="fixed" pos="0.42 -0.85 0.62" xyaxes="0.9 0.44 0 -0.22 0.45 0.87"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" pos="0 0 -0.001"/>
    {''.join(static_xml)}
    {''.join(body_xml)}
  </worldbody>
</mujoco>"""
    return xml, meta


@dataclass
class PiperConfig:
    pregrasp_z: float = 0.18
    grasp_z: float = 0.07
    lift_z: float = 0.24
    transit_z: float = 0.26
    grip_open: float = 0.035
    grip_close: float = 0.016
    move_steps: int = 220
    close_steps: int = 120
    settle_steps: int = 200
    ik_damping: float = 0.05
    ik_iters: int = 400


class PiperArmExecutor:
    def __init__(self, scene_xml: Optional[str] = None,
                 graph: Optional[SceneGraph] = None,
                 cfg: Optional[PiperConfig] = None) -> None:
        import mujoco  # lazy

        self._mj = mujoco
        self.cfg = cfg or PiperConfig()
        if graph is not None:
            # auto-generate the arm MJCF from the scene graph
            xml, _ = build_piper_mjcf(graph)
            scene_xml = os.path.join(PIPER_DIR, "_r4r_generated.xml")
            with open(scene_xml, "w", encoding="utf-8") as f:
                f.write(xml)
        elif scene_xml is None:
            scene_xml = PIPER_SCENE
        if not os.path.exists(scene_xml):
            raise FileNotFoundError(
                f"Piper scene not found: {scene_xml}. Vendor mujoco_menagerie/agilex_piper.")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.on_step = None  # optional hook(self) for viewers/recorders
        self._arm = list(range(6))
        self._l6 = self._bid("link6")
        self._l7 = self._bid("link7")
        self._l8 = self._bid("link8")

    # ---- id helpers ----
    def _bid(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)

    def _gid(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, name)

    def _tcp(self) -> np.ndarray:
        return (self.data.xpos[self._l7] + self.data.xpos[self._l8]) / 2.0

    def _free_adr(self, body_name: str) -> Tuple[int, int]:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, f"{body_name}_free")
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    # ---- IK (position only, damped least squares on the 6 arm joints) ----
    def _ik(self, target: np.ndarray) -> np.ndarray:
        mj, m, d = self._mj, self.model, self.data
        for _ in range(self.cfg.ik_iters):
            mj.mj_forward(m, d)
            p = self._tcp()
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mj.mj_jac(m, d, jp, jr, p, self._l6)
            err = target - p
            if np.linalg.norm(err) < 5e-4:
                break
            J = jp[:, self._arm]
            dq = J.T @ np.linalg.solve(J @ J.T + self.cfg.ik_damping ** 2 * np.eye(3), err)
            dq = np.clip(dq, -0.06, 0.06)
            for i, a in enumerate(self._arm):
                adr = m.jnt_qposadr[a]
                d.qpos[adr] = float(np.clip(d.qpos[adr] + dq[i], m.jnt_range[a][0], m.jnt_range[a][1]))
        return np.array([d.qpos[m.jnt_qposadr[a]] for a in self._arm])

    def _solve(self, target) -> np.ndarray:
        q0, v0 = self.data.qpos.copy(), self.data.qvel.copy()
        q = self._ik(np.asarray(target, dtype=float))
        self.data.qpos[:] = q0; self.data.qvel[:] = v0
        self._mj.mj_forward(self.model, self.data)
        return q

    def _goto(self, q_arm, grip, steps, hold_body: Optional[str] = None) -> None:
        if hold_body is not None:
            qadr, dadr = self._free_adr(hold_body)
        for _ in range(steps):
            self.data.ctrl[:6] = q_arm
            self.data.ctrl[6] = grip
            if hold_body is not None:
                self.data.qpos[qadr:qadr + 3] = self._tcp()
                self.data.qvel[dadr:dadr + 6] = 0.0
            self._mj.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self)

    # ---- container geometry ----
    def _container(self, dest: str) -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
        gid = self._gid(f"{dest}_floor")
        pos = np.array(self.model.geom_pos[gid]); size = np.array(self.model.geom_size[gid])
        cx, cy = float(pos[0]), float(pos[1])
        hx, hy = float(size[0]), float(size[1])
        rim = float(pos[2]) + 0.12
        return np.array([cx, cy]), (cx - hx, cx + hx, cy - hy, cy + hy, rim)

    def _pose_of_body(self, name: str) -> Pose:
        bid = self._bid(name)
        p = self.data.xpos[bid]; q = self.data.xquat[bid]
        return Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                   qw=float(q[0]), qx=float(q[1]), qy=float(q[2]), qz=float(q[3]))

    # ---- plan execution ----
    def execute(self, plan: ActionPlan) -> ExecutionResult:
        mj, m, d = self._mj, self.model, self.data
        c = self.cfg
        mj.mj_resetDataKeyframe(m, d, 0)
        mj.mj_forward(m, d)
        self._goto(np.array([d.qpos[m.jnt_qposadr[a]] for a in self._arm]), c.grip_open, 80)

        target = plan.selected_target
        dest = plan.destination
        notes: List[str] = []
        if target is None or self._bid(target) < 0:
            notes.append(f"target '{target}' is not a body in the arm scene")
            return ExecutionResult(False, target, dest, {}, None, 0, notes)

        start = self._pose_of_body(target)
        mx, my = start.x, start.y
        self._goto(self._solve([mx, my, c.pregrasp_z]), c.grip_open, c.move_steps)
        self._goto(self._solve([mx, my, c.grasp_z]), c.grip_open, c.move_steps)
        self._goto(self._solve([mx, my, c.grasp_z]), c.grip_close, c.close_steps, hold_body=target)
        self._goto(self._solve([mx, my, c.lift_z]), c.grip_close, c.move_steps, hold_body=target)

        success = False
        if dest and self._gid(f"{dest}_floor") >= 0:
            center, (xmin, xmax, ymin, ymax, rim) = self._container(dest)
            self._goto(self._solve([center[0], center[1], c.transit_z]), c.grip_close,
                       c.move_steps + 80, hold_body=target)
            self._goto(self._solve([center[0], center[1], c.transit_z]), c.grip_open, c.close_steps)
            self._goto(np.array([d.qpos[m.jnt_qposadr[a]] for a in self._arm]), c.grip_open, c.settle_steps)
            fp = self.data.xpos[self._bid(target)]
            success = bool(xmin <= fp[0] <= xmax and ymin <= fp[1] <= ymax and fp[2] < rim)
            notes.append(f"dropped at ({fp[0]:.2f},{fp[1]:.2f},{fp[2]:.2f}); in_bin={success}")
        else:
            self._goto(self._solve([mx, my, c.lift_z]), c.grip_close, c.settle_steps, hold_body=target)
            success = True
            notes.append("lifted (no destination container)")

        final = {target: self._pose_of_body(target)}
        return ExecutionResult(success, target, dest, final, start,
                               0, notes)
