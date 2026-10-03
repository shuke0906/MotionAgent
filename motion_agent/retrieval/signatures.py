"""Retrieval idempotency helpers."""

from __future__ import annotations

import re
from typing import Any

from motion_agent.common.fingerprints import retrieval_signature


def normalize_query_text(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def make_retrieval_signature(
    *,
    target_segment: int,
    normalized_query: str,
    retrieval_type: str,
    purpose: str,
    corpus_version: str,
) -> str:
    payload: dict[str, Any] = {
        "target_segment": target_segment,
        "normalized_query": normalize_query_text(normalized_query),
        "retrieval_type": retrieval_type,
        "purpose": purpose,
        "corpus_version": corpus_version,
    }
    return retrieval_signature(payload)
