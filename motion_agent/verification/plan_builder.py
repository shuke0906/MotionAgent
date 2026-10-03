"""Dynamic verifier plan selection for Phase 9."""

from __future__ import annotations

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.compiler.schemas import MotionSegment


def event_action(action: str) -> str:
    # Compiler secondary actions can include an anatomical suffix.
    for suffix in (" right hand", " left hand", " right arm", " left arm"):
        if action.endswith(suffix):
            return action[:-len(suffix)]
    return action
from motion_agent.verification.schemas import VerificationPlan, VerificationRequest, VerifierCheck


def _check(
    check_id: str,
    direction: str,
    evaluator: str,
    *,
    required: bool,
    critical: bool = False,
    target_segments: list[int] | None = None,
    body_parts: list[str] | None = None,
    source_spec_ids: list[str] | None = None,
    threshold_config: dict | None = None,
    render_needed: bool = False,
    metadata: dict | None = None,
) -> VerifierCheck:
    return VerifierCheck(
        check_id=check_id,
        direction=direction,
        evaluator=evaluator,
        required=required,
        critical=critical,
        target_segments=target_segments or [],
        body_parts=body_parts or [],
        source_spec_ids=source_spec_ids or [],
        threshold_config=threshold_config or {},
        render_needed=render_needed,
        metadata=metadata or {},
    )


def _has_temporal_order(segments: list[MotionSegment]) -> bool:
    if len(segments) >= 2:
        return True
    return any(segment.temporal_relation is not None for segment in segments)


class VerificationPlanBuilder:
    def __init__(self, *, learned_config=None):
        self.learned_config = learned_config

    def build(self, request: VerificationRequest) -> VerificationPlan:
        checks: list[VerifierCheck] = [
            _check("technical", "technical", "technical_v1", required=True, critical=True),
            _check("semantic", "semantic", "semantic_backend", required=True, render_needed=False),
            _check("naturalness_kinematic", "naturalness", "kinematic_v1", required=True),
        ]

        segments = request.motion_spec.segments
        eventful = len(segments) >= 2 or any(s.repetition or s.temporal_constraint or s.secondary_actions or s.temporal_relation for s in segments)
        if eventful:
            checks.append(
                _check(
                    "event_integrity",
                    "event_integrity",
                    "event_fixture_backend",
                    required=True,
                    target_segments=[segment.segment_id for segment in segments],
                    source_spec_ids=[str(segment.segment_id) for segment in segments],
                    render_needed=True,
                )
            )
        if _has_temporal_order(segments):
            checks.append(
                _check(
                    "event_temporal",
                    "event_temporal",
                    "event_fixture_backend",
                    required=True,
                    target_segments=[segment.segment_id for segment in segments],
                    source_spec_ids=[str(segment.segment_id) for segment in segments],
                    render_needed=True,
                )
            )
        for segment in segments:
            expected_count = segment.repetition
            if segment.temporal_constraint and segment.temporal_constraint.count:
                expected_count = segment.temporal_constraint.count
            if expected_count:
                checks.append(
                    _check(
                        f"event_frequency_{segment.segment_id}",
                        "event_frequency",
                        "event_fixture_backend",
                        required=True,
                        target_segments=[segment.segment_id],
                        body_parts=[str(part) for part in (segment.temporal_relation.body_parts
                                                          if segment.temporal_relation and segment.secondary_actions
                                                          else segment.body_parts)],
                        source_spec_ids=[str(segment.segment_id)],
                        render_needed=True,
                        metadata={"action": event_action(segment.temporal_relation.related_action
                                             if segment.temporal_relation and segment.temporal_relation.related_action
                                             and segment.secondary_actions else segment.action),
                                  "expected_count": expected_count},
                    )
                )

        for spec in request.constraint_verification_specs:
            checks.append(
                _check(
                    f"constraint_{spec.constraint_id}",
                    "constraint",
                    "constraint_metrics_v1",
                    required=True,
                    critical=True,
                    target_segments=spec.target_segments,
                    body_parts=spec.body_parts,
                    source_spec_ids=[spec.verification_spec_id, spec.constraint_id],
                    threshold_config={"pass_threshold": spec.pass_threshold, "units": spec.units},
                    metadata={"spec": spec.model_dump(mode="python")},
                )
            )

        for spec in request.keyframe_verification_specs:
            checks.append(
                _check(
                    f"keyframe_{spec.keyframe_id}",
                    "keyframe",
                    "keyframe_metrics_v1",
                    required=True,
                    critical=True,
                    source_spec_ids=[spec.keyframe_id, spec.pose_handle],
                    threshold_config={
                        "rotation_threshold_deg": spec.rotation_threshold_deg,
                        "position_threshold_m": spec.position_threshold_m,
                    },
                    metadata={"spec": spec.model_dump(mode="python")},
                )
            )

        if request.generation_scope == "segment":
            checks.append(
                _check(
                    "preservation",
                    "preservation",
                    "preservation_metrics_v1",
                    required=True,
                    critical=True,
                    target_segments=request.target_segments or [],
                    source_spec_ids=[request.previous_candidate_id or ""],
                    threshold_config={"outside_segment_rmse": request.thresholds.get("preservation_outside_rmse", 0.05)},
                )
            )

        for direction in request.force_checks:
            if direction == "physical" and not any(check.direction == "physical" for check in checks):
                checks.append(
                    _check(
                        "physical",
                        "physical",
                        "physical_metrics_v1",
                        required=True,
                        threshold_config={
                            "foot_skating": request.thresholds.get("foot_skating", 0.20),
                            "ground_penetration": request.thresholds.get("ground_penetration", 0.02),
                            "smoothness": request.thresholds.get("smoothness", 10.0),
                        },
                    )
                )

        if request.verifier_policy.enable_motioncritic:
            checks.append(
                _check(
                    "motioncritic",
                    "naturalness",
                    "motioncritic_adapter",
                    required=False,
                    threshold_config={"score": request.thresholds.get("motioncritic", 0.5)},
                )
            )

        if self.learned_config is not None:
            config = self.learned_config
            if request.verifier_policy.enable_tmr:
                checks[1].evaluator = "semantic_backend"
                checks[1].threshold_config = {"score": request.thresholds.get("tmr", config.tmr_threshold),
                                             "status": config.threshold_status}
            else:
                checks = [check for check in checks if check.check_id != "semantic"]
            for check in checks:
                if check.direction.startswith("event_"):
                    check.evaluator = "mllm_backend"
                if check.check_id == "motioncritic":
                    check.required = config.motioncritic_required
                    check.threshold_config = {"score": request.thresholds.get("motioncritic", config.motioncritic_threshold),
                                              "status": config.threshold_status}
            semantic_complexity = eventful or any(s.direction in {"left", "right", "clockwise", "counterclockwise"}
                                                 or s.orientation for s in segments)
            if semantic_complexity and request.verifier_policy.enable_mllm_semantic:
                checks.append(_check("semantic_compositional", "semantic", "mllm_backend", required=True, render_needed=True))
            for segment in segments:
                metric = f"signed_heading_change_{segment.segment_id}"
                if segment.action == "turn" and segment.direction in {"left", "right"} and metric in request.observed_measurements:
                    checks.append(_check(f"direction_{segment.segment_id}", "semantic", "heading_geometry_v1", required=True,
                                         metadata={"metric": metric, "direction": segment.direction},
                                         threshold_config={"minimum_turn_deg": request.thresholds.get("minimum_turn_deg", 0.0)}))
            if request.verifier_policy.enable_physical_metrics and not any(c.direction == "physical" for c in checks):
                checks.append(_check("physical", "physical", "physical_metrics_v1", required=True,
                                     threshold_config={"foot_skating": request.thresholds.get("foot_skating", 0.20),
                                                       "ground_penetration": request.thresholds.get("ground_penetration", 0.02),
                                                       "smoothness": request.thresholds.get("smoothness", 10.0)},
                                     metadata={"require_actual_evidence": True}))
            for check in checks:
                if check.direction == "physical":
                    check.metadata["require_actual_evidence"] = True
                if check.check_id == "naturalness_kinematic":
                    check.metadata["use_smpl_parameters"] = True
            if not request.verifier_policy.enable_atom_event_verifier:
                checks = [check for check in checks if not check.direction.startswith("event_")]

        required = [check.check_id for check in checks if check.required]
        optional = [check.check_id for check in checks if not check.required]
        fingerprint = stable_fingerprint(
            {
                "verification_id": request.verification_id,
                "candidate_id": request.candidate_id,
                "checks": [check.model_dump(mode="json") for check in checks],
                "threshold_profile": request.threshold_profile,
            }
        )
        return VerificationPlan(
            verification_id=request.verification_id,
            candidate_id=request.candidate_id,
            checks=checks,
            required_check_ids=required,
            optional_check_ids=optional,
            render_requirements=[{"check_id": c.check_id} for c in checks if c.render_needed],
            evidence_fingerprint=fingerprint,
        )
