"""Build the Phase 5 retrieval index from authorized local records.

This script intentionally does not download HumanML3D or TMR assets. Point it at
official/authorized preprocessed metadata exported as JSON records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motion_agent.retrieval.index_builder import build_index_from_records, save_index
from motion_agent.retrieval.id_mapping import validate_motion_mapping
from motion_agent.retrieval.schemas import CaptionRecord, MotionIdMap, MotionRecord


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions", required=True, help="JSON file containing MotionRecord list")
    parser.add_argument("--captions", required=True, help="JSON file containing CaptionRecord list")
    parser.add_argument("--id-map", required=True, help="JSON file containing MotionIdMap list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--tmr-model-version", required=True)
    args = parser.parse_args()

    motions = [MotionRecord.model_validate(item) for item in _load_json(args.motions)]
    captions = [CaptionRecord.model_validate(item) for item in _load_json(args.captions)]
    id_map = [MotionIdMap.model_validate(item) for item in _load_json(args.id_map)]
    errors = validate_motion_mapping(motions, id_map)
    if errors:
        raise SystemExit("ID mapping validation failed:\n" + "\n".join(errors))
    index = build_index_from_records(
        motions=motions,
        captions=captions,
        id_map=id_map,
        corpus_version=args.corpus_version,
        tmr_model_version=args.tmr_model_version,
    )
    save_index(index, args.output_dir)
    print(json.dumps(index.manifest, indent=2))
    return 0


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
