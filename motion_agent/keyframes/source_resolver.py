"""Resolve keyframe sources without calling retrieval or generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.keyframes.schemas import KeyframeRequest
from motion_agent.state.schemas import MotionAgentState


@dataclass(frozen=True)
class KeyframeSourceResult:
    status: str
    source_type: str | None = None
    pose: dict[str, Any] | None = None
    source_reference_ids: list[str] = field(default_factory=list)
    source_candidate_id: str | None = None
    source_frame: int | None = None
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


class KeyframeSourceResolver:
    def __init__(self, candidate_store: CandidateStore | None = None) -> None:
        self.candidate_store = candidate_store

    def resolve(self, state: MotionAgentState, request: KeyframeRequest, target_frame: int) -> KeyframeSourceResult:
        if request.explicit_pose is not None:
            return KeyframeSourceResult(
                status="resolved",
                source_type="explicit_pose",
                pose=self._pose_from_payload(request.explicit_pose, state.task.total_frames),
                confidence=1.0,
            )

        references = request.retrieved_references
        if request.source_preference in {"retrieval", "auto"} and references:
            pose_ref = next((ref for ref in references if ref.get("pose_handle")), None)
            if pose_ref:
                return KeyframeSourceResult(
                    status="resolved",
                    source_type="retrieval_pose",
                    pose=self._pose_from_reference(pose_ref, state.task.total_frames, target_frame),
                    source_reference_ids=[pose_ref.get("ref_id", "fixture_reference")],
                    source_frame=pose_ref.get("metadata", {}).get("frame", target_frame),
                    confidence=pose_ref.get("retrieval_score"),
                    warnings=["DEFERRED_PHASE5_DATA: fixture retrieved pose used"],
                )
            motion_ref = next((ref for ref in references if ref.get("motion_handle")), None)
            if motion_ref:
                return KeyframeSourceResult(
                    status="resolved",
                    source_type="retrieval_motion",
                    pose=self._pose_from_reference(motion_ref, state.task.total_frames, target_frame),
                    source_reference_ids=[motion_ref.get("ref_id", "fixture_reference")],
                    source_frame=motion_ref.get("metadata", {}).get("frame", target_frame),
                    confidence=motion_ref.get("retrieval_score"),
                    warnings=["DEFERRED_PHASE5_DATA: fixture retrieved motion frame used"],
                )

        if request.source_preference in {"generation", "auto"}:
            candidate_id = request.source_candidate_id or state.generation.champion_candidate_id
            if candidate_id and self.candidate_store is not None:
                try:
                    loaded = self.candidate_store.load(candidate_id)
                except FileNotFoundError:
                    loaded = None
                if loaded is not None:
                    return KeyframeSourceResult(
                        status="resolved",
                        source_type="candidate",
                        pose=self._pose_from_candidate(loaded["body_params_global"], target_frame),
                        source_candidate_id=candidate_id,
                        source_frame=target_frame,
                        confidence=0.8,
                    )
            if request.source_preference == "generation":
                return KeyframeSourceResult(status="needs_candidate", warnings=["source candidate is unavailable"])

        if request.source_preference == "ik" and not references and request.explicit_pose is None:
            return KeyframeSourceResult(status="needs_reference", warnings=["IK keyframes need a base pose"])
        return KeyframeSourceResult(status="needs_reference", warnings=["no executable keyframe source available"])

    def _pose_from_payload(self, payload: dict[str, Any], total_frames: int) -> dict[str, torch.Tensor]:
        body_pose = self._coerce_body_pose(payload.get("body_pose"), total_frames)
        return {
            "body_pose": body_pose,
            "global_orient": self._coerce_feature(payload.get("global_orient"), 6),
            "betas": self._coerce_feature(payload.get("betas"), 10),
            "local_transl_vel": self._coerce_feature(payload.get("local_transl_vel"), 3),
        }

    def _pose_from_reference(self, ref: dict[str, Any], total_frames: int, target_frame: int) -> dict[str, torch.Tensor]:
        metadata = ref.get("metadata", {})
        if "body_pose" in metadata:
            return self._pose_from_payload(metadata, total_frames)
        body_pose = torch.zeros(21, 6, dtype=torch.float32)
        body_pose[:, 0] = float((target_frame % 7) + 1) * 0.01
        return self._pose_from_payload({"body_pose": body_pose}, total_frames)

    def _pose_from_candidate(self, body_params: dict[str, Any], target_frame: int) -> dict[str, torch.Tensor]:
        pose: dict[str, torch.Tensor] = {}
        for key, value in body_params.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim > 1:
                index = min(target_frame, tensor.shape[0] - 1)
                tensor = tensor[index]
            pose[key] = tensor
        return self._pose_from_payload(pose, total_frames=1)

    def _coerce_body_pose(self, value: Any, total_frames: int) -> torch.Tensor:
        if value is None:
            return torch.zeros(21, 6, dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor[min(total_frames - 1, tensor.shape[0] - 1)]
        if tensor.ndim == 1 and tensor.numel() == 126:
            tensor = tensor.reshape(21, 6)
        if tensor.ndim == 1 and tensor.numel() == 63:
            padded = torch.zeros(21, 6, dtype=torch.float32)
            padded.reshape(-1)[:63] = tensor
            tensor = padded
        if tensor.shape != (21, 6):
            raise ValueError("keyframe body_pose must resolve to [21, 6]")
        return tensor

    def _coerce_feature(self, value: Any, dim: int) -> torch.Tensor:
        if value is None:
            return torch.zeros(dim, dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        output = torch.zeros(dim, dtype=torch.float32)
        output[: min(dim, tensor.numel())] = tensor[: min(dim, tensor.numel())]
        return output
