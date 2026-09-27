"""Filesystem-backed MotionCandidate persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from motion_agent.common.ids import new_artifact_id
from motion_agent.generation.schemas import CandidateMetadata, CandidateStoreRecord, MotionCandidate


class CandidateStore:
    def __init__(self, root: str | Path = "artifacts/phase4_validation") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _candidate_dir(self, candidate_id: str) -> Path:
        return self.root / candidate_id

    def get_by_fingerprint(self, fingerprint: str) -> MotionCandidate | None:
        for manifest in self.root.glob("cand_*/manifest.json"):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            candidate = MotionCandidate.model_validate(data["candidate"])
            if candidate.fingerprint == fingerprint:
                return candidate
        return None

    def save(
        self,
        *,
        tensors: dict[str, Any],
        metadata: CandidateMetadata,
    ) -> CandidateStoreRecord:
        candidate_id = new_artifact_id("candidate", metadata.candidate_fingerprint)
        candidate_dir = self._candidate_dir(candidate_id)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        motion_repr_uri = candidate_dir / "motion_repr.pt"
        smpl_global_uri = candidate_dir / "smpl_global.pt"
        smpl_incam_uri = candidate_dir / "smpl_incam.pt"
        metadata_uri = candidate_dir / "metadata.json"
        manifest_uri = candidate_dir / "manifest.json"

        torch.save(tensors["motion_repr"], motion_repr_uri)
        torch.save(tensors["body_params_global"], smpl_global_uri)
        smpl_incam_value = tensors.get("body_params_incam")
        saved_smpl_incam_uri: str | None = None
        if smpl_incam_value:
            torch.save(smpl_incam_value, smpl_incam_uri)
            saved_smpl_incam_uri = str(smpl_incam_uri)

        metadata_uri.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        candidate = MotionCandidate(
            candidate_id=candidate_id,
            generation_id=metadata.generation_id,
            fingerprint=metadata.candidate_fingerprint,
            motion_repr_uri=str(motion_repr_uri),
            smpl_global_uri=str(smpl_global_uri),
            smpl_incam_uri=saved_smpl_incam_uri,
            metadata_uri=str(metadata_uri),
            metadata=metadata,
        )
        record = CandidateStoreRecord(candidate=candidate, manifest_uri=str(manifest_uri))
        manifest_uri.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def load(self, candidate_id: str) -> dict[str, Any]:
        manifest = self._candidate_dir(candidate_id) / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        candidate = MotionCandidate.model_validate(data["candidate"])
        return {
            "candidate": candidate,
            "motion_repr": torch.load(candidate.motion_repr_uri, map_location="cpu", weights_only=False),
            "body_params_global": torch.load(candidate.smpl_global_uri, map_location="cpu", weights_only=False),
            "body_params_incam": torch.load(candidate.smpl_incam_uri, map_location="cpu", weights_only=False)
            if candidate.smpl_incam_uri
            else None,
        }
