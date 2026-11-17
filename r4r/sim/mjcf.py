"""Generate a MuJoCo MJCF world from a SceneGraph.

- SURFACE nodes -> static boxes (the table).
- OBJECT nodes with geometry -> free rigid bodies (box geoms, density from
  material) that obey physics.
- CONTAINER nodes -> an open-top box (floor + 4 thin walls), static, so objects
  can be dropped into it and collide with the walls.
Robot / end-effector / knowledge nodes are not instantiated (execution uses the
kinematic attach-and-carry abstraction, not an articulated arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from r4r.materials import DEFAULT_DENSITY, DENSITY
from r4r.scene_graph.graph import SceneGraph
from r4r.schemas.nodes import NodeKind, SceneNode

_WALL_T = 0.012  # bin wall thickness (m)


@dataclass
class SimMeta:
    """Bookkeeping to interpret the sim after building it."""
    free_objects: List[str] = field(default_factory=list)
    table_top_z: float = 0.75
    # bin interior bounds: (xmin, xmax, ymin, ymax, floor_z, rim_z)
    containers: Dict[str, Tuple[float, float, float, float, float, float]] = field(default_factory=dict)


def _density(n: SceneNode) -> float:
    return DENSITY.get((n.material or "").lower(), DEFAULT_DENSITY)


def _box(name, size, pos, rgba, extra="") -> str:
    sx, sy, sz = size
    px, py, pz = pos
    return (f'<geom name="{name}" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
            f'pos="{px:.4f} {py:.4f} {pz:.4f}" rgba="{rgba}" {extra}/>')


def scene_elements(graph: SceneGraph) -> Tuple[List[str], List[str], SimMeta]:
    """Build the static geoms (surfaces + container walls) and dynamic free
    bodies (objects) for a graph. Shared by the plain scene and the Piper scene
    so object naming/geometry is identical for both executors.

    Returns (static_xml, body_xml, meta)."""
    meta = SimMeta()
    body_xml: List[str] = []
    static_xml: List[str] = []

    # table / surfaces (static)
    surfaces = graph.nodes_by_kind(NodeKind.SURFACE)
    if surfaces:
        meta.table_top_z = max((s.pose.z + s.dims.z / 2.0) for s in surfaces if s.pose and s.dims)
    for s in surfaces:
        if s.pose is None or s.dims is None:
            continue
        hx, hy, hz = s.dims.half()
        static_xml.append(_box(s.object_id, (hx, hy, hz),
                               (s.pose.x, s.pose.y, s.pose.z), "0.6 0.45 0.3 1"))

    # containers (static open-top box): floor + 4 walls
    for c in graph.nodes_by_kind(NodeKind.CONTAINER):
        if c.pose is None or c.dims is None:
            continue
        hx, hy, hz = c.dims.half()
        bx, by, bz = c.pose.x, c.pose.y, c.pose.z
        floor_z = bz - hz
        rim_z = bz + hz
        t = _WALL_T
        static_xml.append(_box(f"{c.object_id}_floor", (hx, hy, t / 2),
                               (bx, by, floor_z + t / 2), "0.2 0.4 0.8 1"))
        static_xml.append(_box(f"{c.object_id}_wx+", (t / 2, hy, hz), (bx + hx - t / 2, by, bz), "0.2 0.4 0.8 0.6"))
        static_xml.append(_box(f"{c.object_id}_wx-", (t / 2, hy, hz), (bx - hx + t / 2, by, bz), "0.2 0.4 0.8 0.6"))
        static_xml.append(_box(f"{c.object_id}_wy+", (hx, t / 2, hz), (bx, by + hy - t / 2, bz), "0.2 0.4 0.8 0.6"))
        static_xml.append(_box(f"{c.object_id}_wy-", (hx, t / 2, hz), (bx, by - hy + t / 2, bz), "0.2 0.4 0.8 0.6"))
        meta.containers[c.object_id] = (bx - hx + t, bx + hx - t, by - hy + t, by + hy - t,
                                        floor_z + t, rim_z)

    # free objects (dynamic)
    for o in graph.nodes_by_kind(NodeKind.OBJECT):
        if o.pose is None or o.dims is None:
            continue
        hx, hy, hz = o.dims.half()
        body_xml.append(
            f'<body name="{o.object_id}" pos="{o.pose.x:.4f} {o.pose.y:.4f} {o.pose.z:.4f}">'
            f'<freejoint name="{o.object_id}_free"/>'
            f'<geom name="{o.object_id}_g" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
            f'density="{_density(o):.1f}" rgba="0.8 0.3 0.3 1" friction="1 0.5 0.1"/>'
            f'</body>'
        )
        meta.free_objects.append(o.object_id)

    return static_xml, body_xml, meta


def build_mjcf(graph: SceneGraph) -> Tuple[str, SimMeta]:
    static_xml, body_xml, meta = scene_elements(graph)
    xml = f"""<mujoco model="r4r_scene">
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <light pos="0.5 -0.5 3" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-0.5 0.5 2.5" dir="0.2 -0.2 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.35 0.37 0.4 1"/>
    <camera name="fixed" pos="-0.2 -1.25 1.45" xyaxes="1 0 0 0 0.5 0.86"/>
    {''.join(static_xml)}
    {''.join(body_xml)}
  </worldbody>
</mujoco>"""
    return xml, meta
