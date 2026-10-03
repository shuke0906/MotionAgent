"""Canonical HumanML3D/TMR/GEM ID mapping utilities."""

from __future__ import annotations

from motion_agent.retrieval.schemas import MotionIdMap, MotionRecord


def mapping_coverage(mappings: list[MotionIdMap]) -> dict[str, float | int]:
    total = len(mappings)
    mapped = sum(1 for item in mappings if item.gem_mid)
    unmapped = total - mapped
    coverage = (mapped / total * 100.0) if total else 0.0
    return {"total": total, "mapped": mapped, "unmapped": unmapped, "coverage": coverage}


def validate_motion_mapping(motions: list[MotionRecord], mappings: list[MotionIdMap]) -> list[str]:
    by_motion_id = {item.canonical_motion_id: item for item in mappings}
    errors: list[str] = []
    for motion in motions:
        mapping = by_motion_id.get(motion.motion_id)
        if motion.smpl_handle and mapping is None:
            errors.append(f"{motion.motion_id} has smpl_handle but no canonical mapping")
        if motion.smpl_handle and mapping is not None and not mapping.gem_mid:
            errors.append(f"{motion.motion_id} has smpl_handle but mapping has no gem_mid")
    return errors
