"""CPU-only motion and visual surrogates for local control-plane validation."""

from pathlib import Path
import math
import time

import numpy as np
import torch

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.schemas import CandidateMetadata, GenerationResult, SamplerMetadata
from motion_agent.generation.validators import validate_generation_request


class CPUFixtureGenerator:
    def __init__(self, root, observed_counts=(3,)):
        self.root = Path(root)
        self.store = CandidateStore(self.root / "candidates")
        self.observed_counts = tuple(observed_counts)

    def generate(self, request, *, round_index):
        if request.num_candidates != 1 or request.strategy != "normal":
            raise ValueError("local fixture supports normal K=1 only")
        started = time.perf_counter()
        preflight = validate_generation_request(request)
        if preflight.status != "ready":
            raise ValueError("fixture generation preflight blocked")
        count = self.observed_counts[min(round_index, len(self.observed_counts) - 1)]
        actions = {s.action for s in request.motion_spec.segments}
        if not actions.issubset({"walk", "wave"}):
            raise ValueError("local fixture only supports walking and waving")
        frames = request.total_frames
        motion = torch.zeros(frames, 151, device="cpu")
        pose = torch.zeros(frames, 63, device="cpu")
        transl = torch.zeros(frames, 3, device="cpu")
        phase = torch.arange(frames, dtype=torch.float32, device="cpu") / (frames - 1)
        if "walk" in actions:
            transl[:, 2] = phase * 1.5
            pose[:, 0] = torch.sin(phase * 8 * math.pi) * 0.35
            pose[:, 3] = -pose[:, 0]
        if "wave" in actions:
            pose[:, 50] = 0.65 * torch.sin(2 * math.pi * count * phase)
        motion[:, :63] = pose
        motion[:, 148:] = transl
        params = {"body_pose": pose, "global_orient": torch.zeros(frames, 3, device="cpu"),
                  "transl": transl, "betas": torch.zeros(frames, 10, device="cpu")}
        fingerprint = stable_fingerprint({"request": request.model_dump(mode="json"), "count": count,
                                          "backend": "cpu_procedural_fixture_v1"})
        seed = request.seeds[0]
        metadata = CandidateMetadata(
            seed=seed, generation_id=request.generation_id,
            condition_fingerprint=request.condition_bundle.condition_fingerprint,
            candidate_fingerprint=fingerprint, checkpoint_version="MOCK_GEM_CPU_FIXTURE",
            sampler=SamplerMetadata(seed=seed, checkpoint_version="MOCK_GEM_CPU_FIXTURE"),
            scope=request.scope, target_segments=request.target_segments,
            postprocess_policy=request.postprocess_policy,
            runtime_ms=(time.perf_counter() - started) * 1000,
            frame_count=frames, motion_dim=151, technical_valid=True, generation_request=request,
        )
        record = self.store.save(tensors={"motion_repr": motion, "body_params_global": params}, metadata=metadata)
        video = self.root / "renders" / (record.candidate.candidate_id + ".avi")
        render_fixture(video, params, request.fps, walking="walk" in actions, waving="wave" in actions)
        events = {"present": sorted(actions), "counts": {"wave": count} if "wave" in actions else {},
                  "order": [s.action for s in request.motion_spec.segments]}
        return GenerationResult(
            generation_id=request.generation_id, status="success", candidates=[record.candidate],
            condition_fingerprint=request.condition_bundle.condition_fingerprint,
            runtime_ms=(time.perf_counter() - started) * 1000, preflight=preflight, inference_calls=1,
        ), {"path": str(video.resolve()), "source": "cpu_procedural_fixture_v1",
            "observed_events": events, "renderer": "two_view_skeleton_from_saved_fixture_parameters",
            "not_real_gem": True}


def render_fixture(path, params, fps, *, walking, waving):
    """Unlabeled motion, two camera views; no expected count is drawn into evidence."""
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (768, 512))
    if not writer.isOpened():
        raise RuntimeError("fixture video encoder unavailable")
    try:
        for index, pose in enumerate(params["body_pose"]):
            image = np.full((512, 768, 3), 248, dtype=np.uint8)
            cv2.line(image, (384, 0), (384, 512), (185, 185, 185), 1)
            cv2.putText(image, "Front", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 60, 60), 1)
            cv2.putText(image, "Side", (396, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 60, 60), 1)
            root = float(params["transl"][index, 2])
            # A fixed side camera shows root travel relative to the floor grid.
            for x in range(405, 768, 40):
                cv2.line(image, (x, 460), (x, 475), (150, 150, 150), 1)
            cv2.line(image, (395, 460), (760, 460), (130, 130, 130), 2)
            color = (45, 60, 80)
            cv2.circle(image, (192, 100), 24, color, 3)
            # Nose and eyes establish the face orientation without anatomical labels.
            cv2.circle(image, (183, 95), 2, color, -1)
            cv2.circle(image, (201, 95), 2, color, -1)
            cv2.line(image, (192, 99), (192, 109), color, 2)
            cv2.line(image, (192, 124), (192, 310), color, 4)
            for side, sign in (("right", -1), ("left", 1)):
                shoulder = (192 + sign * 30, 150)
                elbow = (192 + sign * 85, 165 if side == "right" and waving else 235)
                angle = float(pose[50]) if side == "right" and waving else 0
                wrist = ((int(elbow[0] + sign * 65 * math.sin(angle)), int(elbow[1] - 90 * math.cos(angle)))
                         if side == "right" and waving else (elbow[0], 295))
                for a, b in [((192, 150), shoulder), (shoulder, elbow), (elbow, wrist)]:
                    cv2.line(image, a, b, color, 5)
                cv2.circle(image, wrist, 8, color, -1)
                offset = int(math.sin(index / fps * 4.2) * 10) if walking else 0
                cv2.line(image, (192, 310), (192 + sign * 45 + offset, 445), color, 5)
            x = int(470 + root * 135)
            cv2.circle(image, (x, 100), 24, color, 3)
            cv2.line(image, (x + 22, 97), (x + 33, 103), color, 3)
            cv2.line(image, (x, 124), (x, 310), color, 5)
            for sign in (-1, 1):
                angle = float(pose[0]) * sign if walking else sign * 0.10
                knee = (int(x + math.sin(angle) * 85), 385)
                ankle = (int(x + math.sin(angle) * 140), 450)
                cv2.line(image, (x, 310), knee, color, 5)
                cv2.line(image, knee, ankle, color, 5)
                cv2.line(image, ankle, (ankle[0] + 20, ankle[1]), color, 5)
            cv2.line(image, (x, 150), (x + 12, 245), color, 5)
            cv2.line(image, (x + 12, 245), (x + 25, 295), color, 5)
            writer.write(image)
    finally:
        writer.release()
