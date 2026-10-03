from __future__ import annotations

import unittest

import torch

from motion_agent.compiler.schemas import MotionSegment, MotionSpecification, TemporalConstraint
from motion_agent.constraints.schemas import VerificationSpec
from motion_agent.generation.schemas import (
    CandidateMetadata,
    GenerationPreflightReport,
    GenerationResult,
    MotionCandidate,
    SamplerMetadata,
)
from motion_agent.keyframes.schemas import KeyframeVerificationSpec
from motion_agent.tournament import SingleCandidateSelector
from motion_agent.verification import (
    FixtureSemanticBackend,
    VerificationCache,
    VerificationPlanBuilder,
    VerificationRequest,
    VerificationService,
)


def spec_for(prompt: str, segments: list[MotionSegment] | None = None) -> MotionSpecification:
    if segments is None:
        segments = [
            MotionSegment(
                segment_id=0,
                action=prompt,
                start_s=0.0,
                end_s=6.0,
                start_frame=0,
                end_frame=180,
            )
        ]
    return MotionSpecification(
        original_request=prompt,
        duration_s=6.0,
        fps=30,
        total_frames=180,
        segments=segments,
    )


def valid_candidate(delta: float = 0.0) -> dict:
    motion = torch.zeros(180, 151) + delta
    return {
        "motion_repr": motion,
        "body_params_global": {
            "body_pose": torch.zeros(180, 63),
            "global_orient": torch.zeros(180, 3),
            "transl": torch.zeros(180, 3),
            "betas": torch.zeros(180, 10),
        },
    }


def request_for(
    motion_spec: MotionSpecification,
    *,
    candidate: dict | None = None,
    constraints: list[VerificationSpec] | None = None,
    keyframes: list[KeyframeVerificationSpec] | None = None,
    observed_events: dict | None = None,
    observed_measurements: dict | None = None,
    physical_evidence: dict | None = None,
    force_checks: list[str] | None = None,
    generation_scope: str = "full",
    previous_candidate: dict | None = None,
    target_segments: list[int] | None = None,
    threshold_profile: str = "phase9_default_v1",
    evaluator_versions: dict | None = None,
    prompt_versions: dict | None = None,
) -> VerificationRequest:
    return VerificationRequest(
        verification_id="verify_fixture",
        candidate_id="cand_fixture",
        original_request=motion_spec.original_request,
        motion_spec=motion_spec,
        candidate=candidate or valid_candidate(),
        constraint_verification_specs=constraints or [],
        keyframe_verification_specs=keyframes or [],
        observed_events=observed_events or {},
        observed_measurements=observed_measurements or {},
        physical_evidence=physical_evidence or {},
        force_checks=force_checks or [],
        generation_scope=generation_scope,
        previous_candidate=previous_candidate,
        target_segments=target_segments,
        threshold_profile=threshold_profile,
        evaluator_versions=evaluator_versions or {},
        prompt_versions=prompt_versions or {},
    )


def service(status: str = "pass", cache: VerificationCache | None = None):
    backend = FixtureSemanticBackend(status=status)
    return VerificationService(semantic_backend=backend, cache=cache), backend


class Phase9VerificationTests(unittest.TestCase):
    def test_9_1_plan_selection_golden_cases(self):
        builder = VerificationPlanBuilder()
        walk = builder.build(request_for(spec_for("walk forward")))
        self.assertEqual({c.direction for c in walk.checks if c.required}, {"technical", "semantic", "naturalness"})

        wave = spec_for(
            "wave right hand three times",
            [
                MotionSegment(
                    segment_id=0,
                    action="wave",
                    body_parts=["right_hand"],
                    temporal_constraint=TemporalConstraint(type="repetition", count=3),
                    start_s=0,
                    end_s=6,
                    start_frame=0,
                    end_frame=180,
                )
            ],
        )
        wave_plan = builder.build(request_for(wave))
        self.assertIn("event_frequency", {c.direction for c in wave_plan.checks})
        frequency = next(c for c in wave_plan.checks if c.direction == "event_frequency")
        self.assertEqual(frequency.metadata["expected_count"], 3)

        ordered = spec_for(
            "wave then walk then sit",
            [
                MotionSegment(segment_id=0, action="wave", start_s=0, end_s=2, start_frame=0, end_frame=60),
                MotionSegment(segment_id=1, action="walk", start_s=2, end_s=4, start_frame=60, end_frame=120),
                MotionSegment(segment_id=2, action="sit", start_s=4, end_s=6, start_frame=120, end_frame=180),
            ],
        )
        ordered_plan = builder.build(request_for(ordered))
        self.assertIn("event_integrity", {c.direction for c in ordered_plan.checks})
        self.assertIn("event_temporal", {c.direction for c in ordered_plan.checks})

        constraint = VerificationSpec(
            verification_spec_id="verspec_contact",
            constraint_id="constraint_contact",
            metric_type="contact_distance",
            target_segments=[0],
            pass_threshold=0.05,
            units="m",
        )
        self.assertIn("constraint", {c.direction for c in builder.build(request_for(walk.motion_spec if False else spec_for("contact"), constraints=[constraint])).checks})

        keyframe = KeyframeVerificationSpec(
            keyframe_id="keyframe_final",
            target_frame=179,
            temporal_tolerance_frames=3,
            pose_handle="pose_final",
            metrics=["joint_rotation_error"],
        )
        self.assertIn("keyframe", {c.direction for c in builder.build(request_for(spec_for("final pose"), keyframes=[keyframe])).checks})
        self.assertIn(
            "preservation",
            {c.direction for c in builder.build(request_for(spec_for("segment retry"), generation_scope="segment", target_segments=[0], previous_candidate=valid_candidate())).checks},
        )

    def test_9_2_synthetic_constraint_failure(self):
        verifier, _ = service()
        spec = VerificationSpec(
            verification_spec_id="verspec_joint",
            constraint_id="constraint_joint",
            metric_type="joint_position_error",
            target_segments=[0],
            pass_threshold=0.05,
            units="m",
        )
        report = verifier.verify(
            request_for(
                spec_for("right wrist target"),
                constraints=[spec],
                observed_measurements={"verspec_joint": 0.08},
            )
        )
        self.assertFalse(report.overall_pass)
        self.assertIn("constraint_constraint_joint", report.failed_required_checks)

    def test_9_3_event_failure_fixtures(self):
        verifier, _ = service()
        ordered = spec_for(
            "wave then walk then sit",
            [
                MotionSegment(segment_id=0, action="wave", start_s=0, end_s=2, start_frame=0, end_frame=60),
                MotionSegment(segment_id=1, action="walk", start_s=2, end_s=4, start_frame=60, end_frame=120),
                MotionSegment(segment_id=2, action="sit", start_s=4, end_s=6, start_frame=120, end_frame=180),
            ],
        )
        missing = verifier.verify(request_for(ordered, observed_events={"present": ["wave", "walk"]}))
        self.assertTrue(any(f.diagnostic_code == "EVENT_MISSING" for f in missing.findings))
        wrong_order = verifier.verify(request_for(ordered, observed_events={"order": ["sit", "walk", "wave"]}))
        self.assertTrue(any(f.diagnostic_code == "EVENT_ORDER_MISMATCH" for f in wrong_order.findings))

        wave = spec_for(
            "wave right hand three times",
            [
                MotionSegment(
                    segment_id=0,
                    action="wave",
                    temporal_constraint=TemporalConstraint(type="repetition", count=3),
                    start_s=0,
                    end_s=6,
                    start_frame=0,
                    end_frame=180,
                )
            ],
        )
        count = verifier.verify(request_for(wave, observed_events={"counts": {"0": 2}}))
        self.assertTrue(any(f.diagnostic_code == "EVENT_FREQUENCY_MISMATCH" for f in count.findings))

    def test_9_4_physical_failure_injection(self):
        verifier, _ = service()
        report = verifier.verify(
            request_for(
                spec_for("physical fixture"),
                force_checks=["physical"],
                physical_evidence={"foot_skating": 0.4, "ground_penetration": 0.04, "smoothness": 15.0},
            )
        )
        codes = {f.diagnostic_code for f in report.findings}
        self.assertIn("FOOT_SKATING", codes)
        self.assertIn("GROUND_PENETRATION", codes)
        self.assertIn("MOTION_JITTER", codes)
        self.assertFalse(report.overall_pass)

    def test_9_5_required_uncertain_and_9_6_required_error(self):
        uncertain, _ = service("uncertain")
        uncertain_report = uncertain.verify(request_for(spec_for("walk forward")))
        self.assertEqual(uncertain_report.status, "incomplete")
        self.assertFalse(uncertain_report.overall_pass)

        error, _ = service("error")
        error_report = error.verify(request_for(spec_for("walk forward")))
        self.assertEqual(error_report.status, "incomplete")
        self.assertFalse(error_report.overall_pass)

    def test_9_7_required_failure_and_9_8_all_required_pass(self):
        fail, _ = service("fail")
        failed = fail.verify(request_for(spec_for("walk forward")))
        self.assertFalse(failed.overall_pass)
        self.assertEqual(failed.status, "complete")

        ok, _ = service("pass")
        passed = ok.verify(request_for(spec_for("walk forward")))
        self.assertEqual(passed.status, "complete")
        self.assertTrue(passed.overall_pass)

    def test_9_9_cache_invalidation(self):
        cache = VerificationCache()
        verifier, backend = service("pass", cache=cache)
        first = request_for(spec_for("walk forward"), evaluator_versions={"semantic_backend": "v1"}, prompt_versions={"semantic": "p1"})
        verifier.verify(first)
        verifier.verify(first)
        self.assertEqual(backend.calls, 1)
        verifier.verify(request_for(spec_for("walk forward"), evaluator_versions={"semantic_backend": "v1"}, prompt_versions={"semantic": "p1"}, threshold_profile="phase9_default_v2"))
        self.assertEqual(backend.calls, 2)
        verifier.verify(request_for(spec_for("walk forward"), evaluator_versions={"semantic_backend": "v1"}, prompt_versions={"semantic": "p2"}))
        self.assertEqual(backend.calls, 3)
        verifier.verify(request_for(spec_for("walk forward"), evaluator_versions={"semantic_backend": "v2"}, prompt_versions={"semantic": "p2"}))
        self.assertEqual(backend.calls, 4)

    def test_9_10_k1_direct_handoff(self):
        metadata = CandidateMetadata(
            seed=1,
            generation_id="gen_k1",
            condition_fingerprint="condition",
            candidate_fingerprint="candidate",
            checkpoint_version="fake",
            sampler=SamplerMetadata(seed=1, checkpoint_version="fake"),
            scope="full",
            postprocess_policy="none",
            runtime_ms=1.0,
            frame_count=180,
            motion_dim=151,
            technical_valid=True,
        )
        candidate = MotionCandidate(
            candidate_id="cand_k1",
            generation_id="gen_k1",
            fingerprint="candidate",
            motion_repr_uri="motion.pt",
            smpl_global_uri="smpl.pt",
            metadata_uri="metadata.json",
            metadata=metadata,
        )
        result = GenerationResult(
            generation_id="gen_k1",
            status="success",
            candidates=[candidate],
            condition_fingerprint="condition",
            runtime_ms=1.0,
            preflight=GenerationPreflightReport(status="ready", estimated_frames=180, condition_fingerprint="condition"),
        )
        selected = SingleCandidateSelector().select(result)
        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.phase8_status, "BYPASSED_FOR_K1_MODE")
        self.assertEqual(selected.champion_candidate_id, "cand_k1")
        self.assertEqual(selected.pairwise_evidence, [])


if __name__ == "__main__":
    unittest.main()
