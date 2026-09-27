"""Canonical ID helpers."""

from __future__ import annotations

import uuid


ID_PREFIXES = {
    "run": "run",
    "plan": "plan",
    "segment": "seg",
    "reference": "ref",
    "constraint": "constraint",
    "keyframe": "keyframe",
    "generation": "gen",
    "candidate": "cand",
    "verification": "verify",
    "diagnosis": "diag",
    "artifact": "artifact",
    "motion_spec": "spec",
    "gem_text_condition": "textcond",
    "tournament": "tour",
    "repair": "repair",
    "event": "event",
    "trace": "trace",
    "span": "span",
}


def new_id(kind: str) -> str:
    prefix = ID_PREFIXES.get(kind, kind)
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def id_from_fingerprint(kind: str, fingerprint: str) -> str:
    prefix = ID_PREFIXES.get(kind, kind)
    return f"{prefix}_{fingerprint[:16]}"


def new_run_id() -> str:
    return new_id("run")


def new_artifact_id(artifact_type: str, fingerprint: str | None = None) -> str:
    if fingerprint:
        return id_from_fingerprint(artifact_type, fingerprint)
    return new_id(artifact_type)

