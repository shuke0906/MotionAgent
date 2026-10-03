"""Offline retrieval-index construction helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.retrieval.id_mapping import mapping_coverage
from motion_agent.retrieval.schemas import CaptionRecord, MotionIdMap, MotionRecord, RetrievalIndex
from motion_agent.retrieval.tmr_encoder import DeterministicTextEncoder


def build_index_from_records(
    *,
    motions: list[MotionRecord],
    captions: list[CaptionRecord],
    id_map: list[MotionIdMap],
    corpus_version: str,
    tmr_model_version: str,
) -> RetrievalIndex:
    encoder = DeterministicTextEncoder()
    index = RetrievalIndex(
        motions=motions,
        captions=captions,
        motion_embeddings=[encoder.encode_motion_caption(motion.captions) for motion in motions],
        caption_embeddings=[encoder.encode_text(caption.caption) for caption in captions],
        motion_index_to_id={idx: motion.motion_id for idx, motion in enumerate(motions)},
        motion_id_to_index={motion.motion_id: idx for idx, motion in enumerate(motions)},
        caption_index_to_id={idx: caption.caption_id for idx, caption in enumerate(captions)},
        caption_id_to_index={caption.caption_id: idx for idx, caption in enumerate(captions)},
        id_map=id_map,
        manifest={},
    )
    fingerprint = stable_fingerprint(
        {
            "motions": [motion.model_dump(mode="json") for motion in motions],
            "captions": [caption.model_dump(mode="json") for caption in captions],
            "id_map": [item.model_dump(mode="json") for item in id_map],
        },
        namespace="retrieval_index",
    )
    manifest: dict[str, Any] = {
        "corpus_version": corpus_version,
        "tmr_model_version": tmr_model_version,
        "split_policy": {"allowed_splits": ["train"]},
        "index_fingerprint": fingerprint,
        "embedding_dim": len(index.motion_embeddings[0]) if index.motion_embeddings else 0,
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "motion_count": len(motions),
        "caption_count": len(captions),
        "mapping": mapping_coverage(id_map),
    }
    return index.model_copy(update={"manifest": manifest})


def save_index(index: RetrievalIndex, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    (output / "index_manifest.json").write_text(json.dumps(index.manifest, indent=2), encoding="utf-8")


def load_index(path: str | Path) -> RetrievalIndex:
    value = Path(path)
    if value.is_dir():
        value = value / "index.json"
    return RetrievalIndex.model_validate(json.loads(value.read_text(encoding="utf-8")))
