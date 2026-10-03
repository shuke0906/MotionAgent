"""Safe loading and content identity for actual candidate artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.generation.schemas import MotionCandidate
from motion_agent.verification.schemas import VerificationRequest


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(request: VerificationRequest) -> dict:
    candidate = request.candidate
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, MotionCandidate):
        raise FileNotFoundError("candidate artifact is not supplied")
    if candidate.candidate_id != request.candidate_id:
        raise ValueError("candidate_id does not match artifact")
    return {
        "motion_repr": torch.load(candidate.motion_repr_uri, map_location="cpu", weights_only=True),
        "body_params_global": torch.load(candidate.smpl_global_uri, map_location="cpu", weights_only=True),
        "candidate": candidate,
    }


def candidate_identity(request: VerificationRequest) -> str:
    candidate = request.candidate
    if isinstance(candidate, MotionCandidate):
        return stable_fingerprint({"declared": candidate.fingerprint, "motion": file_digest(candidate.motion_repr_uri),
                                   "smpl": file_digest(candidate.smpl_global_uri)})
    return stable_fingerprint({"declared": request.candidate_fingerprint, "payload": evidence_identity(candidate)})


def evidence_identity(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: evidence_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_identity(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
