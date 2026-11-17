"""Shared material properties — single source of truth.

Used by the validator (conservative payload mass estimate) and the simulator
(MuJoCo body density → mass). Densities are solid-material values in kg/m^3;
real objects are often hollow, so these OVER-estimate mass, which is the safe
direction for payload checks.
"""

from __future__ import annotations

from typing import Optional

DENSITY = {
    "ceramic": 2300.0,
    "glass": 2500.0,
    "aluminium": 2700.0,
    "metal": 7800.0,
    "metal_plastic": 3000.0,
    "wood": 650.0,
    "cardboard": 700.0,
    "plastic": 950.0,
}
DEFAULT_DENSITY = 500.0           # unknown material (sim only; validator warns instead)
FRAGILE_MATERIALS = {"ceramic", "glass"}


def density_of(material: Optional[str]) -> Optional[float]:
    """Known density for a material, or None if unrecognised."""
    return DENSITY.get((material or "").lower())


def is_fragile(material: Optional[str]) -> bool:
    return (material or "").lower() in FRAGILE_MATERIALS
