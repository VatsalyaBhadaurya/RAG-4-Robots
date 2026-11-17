"""Phase 6 — MuJoCo physics execution.

Builds an MJCF world from a SceneGraph and executes a validated ActionPlan with
real physics (gravity, contacts, objects settling into the bin), then reads back
poses to update the scene graph.

SIMPLIFICATION (decision D-11): execution uses a kinematic "attach-and-carry"
abstraction for the grasped object (the object follows a scripted waypoint path
while held) rather than a full arm + IK + motion planner. Everything else —
free-body dynamics, collisions, settling, placement success — is real MuJoCo
physics. This keeps the MVP honest and testable without premature complexity;
arm/IK/motion-planning is the documented upgrade path.
"""

from __future__ import annotations

from r4r.sim.mjcf import SimMeta, build_mjcf
from r4r.sim.executor import (
    ExecConfig,
    ExecutionResult,
    MujocoExecutor,
    update_graph_from_execution,
)
from r4r.sim.piper import (
    PIPER_SCENE,
    PiperArmExecutor,
    PiperConfig,
    build_piper_mjcf,
)

__all__ = [
    "build_mjcf", "SimMeta", "MujocoExecutor", "ExecutionResult", "ExecConfig",
    "update_graph_from_execution",
    "PiperArmExecutor", "PiperConfig", "PIPER_SCENE", "build_piper_mjcf",
]
