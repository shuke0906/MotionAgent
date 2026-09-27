"""Stable fingerprint helpers for cache keys and idempotency."""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any


def _to_canonical(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_canonical(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _to_canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "__array_like__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_sha256": hashlib.sha256(pickle.dumps(value)).hexdigest(),
        }
    return repr(value)


def stable_fingerprint(value: Any, *, namespace: str = "motionagent") -> str:
    canonical = _to_canonical({"namespace": namespace, "value": value})
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def retrieval_signature(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="retrieval_signature")


def condition_fingerprint(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="condition_fingerprint")


def candidate_fingerprint(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="candidate_fingerprint")


def evidence_fingerprint(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="evidence_fingerprint")


def failure_signature(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="failure_signature")


def repair_signature(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="repair_signature")


def criteria_fingerprint(payload: Any) -> str:
    return stable_fingerprint(payload, namespace="criteria_fingerprint")

