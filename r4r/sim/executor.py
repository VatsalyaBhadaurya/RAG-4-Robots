"""MuJoCo executor: run a validated ActionPlan under real physics.

Grasp/carry is kinematic (the held object follows a scripted waypoint path);
release is pure physics (the object drops and settles, colliding with bin walls
and other bodies). Final poses are read back from the simulator to update the
scene graph — closing the perceive→plan→act→update loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import Pose
from r4r.schemas.plan import ActionPlan
from r4r.sim.mjcf import SimMeta, build_mjcf


@dataclass
class ExecConfig:
    settle_steps: int = 200
    lift_steps: int = 150
    move_steps: int = 200
    release_steps: int = 400
    carry_clearance: float = 0.25   # height above table top to carry at
    lift_success_dz: float = 0.08   # pick-only: min rise to count as lifted


@dataclass
class ExecutionResult:
    success: bool
    target: Optional[str]
    destination: Optional[str]
    final_poses: Dict[str, Pose] = field(default_factory=dict)
    initial_pose: Optional[Pose] = None
    steps: int = 0
    notes: List[str] = field(default_factory=list)


class MujocoExecutor:
    def __init__(self, graph: SceneGraph, cfg: Optional[ExecConfig] = None) -> None:
        import mujoco  # lazy

        self._mj = mujoco
        self.cfg = cfg or ExecConfig()
        self.xml, self.meta = build_mjcf(graph)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

    # ---- low-level helpers ----
    def _qadr(self, oid: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, f"{oid}_free")
        return int(self.model.jnt_qposadr[jid])

    def _dofadr(self, oid: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, f"{oid}_free")
        return int(self.model.jnt_dofadr[jid])

    def _bid(self, oid: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, oid)

    def _pos(self, oid: str) -> np.ndarray:
        return np.array(self.data.xpos[self._bid(oid)], dtype=float)

    def _hold(self, oid: str, pos: np.ndarray) -> None:
        a, d = self._qadr(oid), self._dofadr(oid)
        self.data.qpos[a:a + 3] = pos
        self.data.qvel[d:d + 6] = 0.0

    def _step(self, n: int, hold: Optional[tuple] = None) -> None:
        for _ in range(n):
            if hold is not None:
                self._hold(hold[0], hold[1])
            self._mj.mj_step(self.model, self.data)
            self._total_steps += 1

    def _pose_of(self, oid: str) -> Pose:
        bid = self._bid(oid)
        p = self.data.xpos[bid]
        q = self.data.xquat[bid]
        return Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                    qw=float(q[0]), qx=float(q[1]), qy=float(q[2]), qz=float(q[3]))

    # ---- execution ----
    def execute(self, plan: ActionPlan) -> ExecutionResult:
        self._total_steps = 0
        self._mj.mj_forward(self.model, self.data)
        self._step(self.cfg.settle_steps)  # let the scene settle

        target = plan.selected_target
        dest = plan.destination
        notes: List[str] = []

        if target is None or target not in self.meta.free_objects:
            notes.append(f"target '{target}' is not a movable free body; nothing executed")
            return ExecutionResult(False, target, dest, self._all_poses(), None,
                                   self._total_steps, notes)

        init = self._pose_of(target)
        carry_z = self.meta.table_top_z + self.cfg.carry_clearance
        start = self._pos(target)

        # lift straight up
        self._interp(target, start, np.array([start[0], start[1], carry_z]), self.cfg.lift_steps)

        success = False
        if dest and dest in self.meta.containers:
            xmin, xmax, ymin, ymax, floor_z, rim_z = self.meta.containers[dest]
            bin_xy = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, carry_z])
            cur = self._pos(target)
            self._interp(target, cur, bin_xy, self.cfg.move_steps)   # move over bin
            self._step(self.cfg.release_steps)                       # release -> drop + settle
            fp = self._pos(target)
            inside_xy = (xmin <= fp[0] <= xmax) and (ymin <= fp[1] <= ymax)
            inside_z = floor_z - 0.02 <= fp[2] <= rim_z + 0.05
            success = bool(inside_xy and inside_z)
            notes.append(f"placed at ({fp[0]:.2f},{fp[1]:.2f},{fp[2]:.2f}); "
                         f"inside_xy={inside_xy} inside_z={inside_z}")
        else:
            # pick-only: success if lifted clear of its start height
            fp = self._pos(target)
            success = bool(fp[2] - start[2] >= self.cfg.lift_success_dz)
            notes.append(f"lifted dz={fp[2] - start[2]:.2f}")

        return ExecutionResult(success, target, dest, self._all_poses(), init,
                               self._total_steps, notes)

    def _interp(self, oid: str, p0: np.ndarray, p1: np.ndarray, n: int) -> None:
        for i in range(1, n + 1):
            wp = p0 + (p1 - p0) * (i / n)
            self._step(1, hold=(oid, wp))

    def _all_poses(self) -> Dict[str, Pose]:
        return {oid: self._pose_of(oid) for oid in self.meta.free_objects}


def update_graph_from_execution(graph: SceneGraph, result: ExecutionResult) -> SceneGraph:
    """Write executed object poses back into the scene graph (world-state update)."""
    from r4r.schemas.provenance import Provenance, SourceType

    for oid, pose in result.final_poses.items():
        if graph.has_node(oid):
            node = graph.get_node(oid)
            graph.update_node(oid, pose=pose,
                              provenance=Provenance(source_type=SourceType.PERCEPTION,
                                                    source_id="sim_readback",
                                                    confidence=node.confidence))
    return graph
