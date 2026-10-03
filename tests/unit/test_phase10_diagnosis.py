from __future__ import annotations

import unittest

from motion_agent.compiler.schemas import MotionSegment, MotionSpecification, TemporalConstraint
from motion_agent.diagnosis import (
    DiagnosisRequest,
    DiagnosisService,
    RepairHistoryEntry,
    classify_metric_progression,
    detect_regressions,
    normalize_failures,
    targeted_repair_success,
    validate_repair_proposal,
)
from motion_agent.diagnosis.history import RepairHistory
from motion_agent.diagnosis.proposal_validator import validate_repair_proposal
from motion_agent.verification.schemas import VerificationPlan, VerificationReport, VerifierCheck, VerifierFinding


def motion_spec(*, count: int | None = 3, direction: str | None = None, body_part: str = "right_hand") -> MotionSpecification:
    temporal = TemporalConstraint(type="repetition", count=count) if count else None
    return MotionSpecification(
        original_request="Walk forward while waving the right hand three times.",
        duration_s=6.0,
        fps=30,
        total_frames=180,
        segments=[
            MotionSegment(
                segment_id=0,
                action="walk",
                start_s=0,
                end_s=6,
                start_frame=0,
                end_frame=180,
            ),
            MotionSegment(
                segment_id=1,
                action="wave",
                body_parts=[body_part],
                direction=direction,
                temporal_constraint=temporal,
                repetition=count,
                start_s=0,
                end_s=6,
                start_frame=0,
                end_frame=180,
            ),
        ],
    )


def report_for(
    code: str,
    *,
    direction: str = "event_frequency",
    expected=None,
    observed=None,
    threshold: float | None = None,
    measured: float | None = None,
    required: bool = True,
) -> VerificationReport:
    check = VerifierCheck(
        check_id="check_1",
        direction=direction,  # type: ignore[arg-type]
        evaluator="fixture",
        required=required,
        critical=required,
        target_segments=[1],
        body_parts=["right_hand"],
        source_spec_ids=["spec_event_wave"],
        metadata={"event": "wave", "action": "wave"},
    )
    finding = VerifierFinding(
        check_id="check_1",
        direction=direction,  # type: ignore[arg-type]
        status="fail",
        required=required,
        diagnostic_code=code,
        expected=expected or {},
        observed=observed or {},
        threshold=threshold,
        measured_value=measured,
        evaluator_version="fixture_v1",
    )
    return VerificationReport(
        verification_id="verify_1",
        candidate_id="cand_1",
        plan=VerificationPlan(
            verification_id="verify_1",
            candidate_id="cand_1",
            checks=[check],
            required_check_ids=["check_1"] if required else [],
            optional_check_ids=[] if required else ["check_1"],
            evidence_fingerprint="evidence",
        ),
        findings=[finding],
        failed_required_checks=["check_1"] if required else [],
        status="complete",
        overall_pass=False,
        threshold_profile="phase10_fixture",
        evaluator_versions={"fixture": "v1"},
    )


def diagnose(report: VerificationReport, spec=None, generation_request=None, history=None):
    return DiagnosisService().diagnose(
        DiagnosisRequest(
            diagnosis_id="diag_1",
            verification_report=report,
            motion_spec=spec,
            generation_request=generation_request,
            repair_history=history or [],
        )
    )


class Phase10DiagnosisTests(unittest.TestCase):
    def test_10_1_rule_fixture_table_major_codes(self):
        cases = [
            ("EVENT_MISSING", "EVENT", "GENERATION_SEMANTIC_EXECUTION_FAILURE", "GENERATE"),
            ("EVENT_ORDER_MISMATCH", "EVENT", "TEMPORAL_RELATION_EXECUTION_FAILURE", "GENERATE"),
            ("EVENT_FREQUENCY_MISMATCH", "EVENT", "GENERATION_SEMANTIC_EXECUTION_FAILURE", "GENERATE"),
            ("DIRECTION_MISMATCH", "SEMANTIC", "GENERATION_SEMANTIC_EXECUTION_FAILURE", "GENERATE"),
            ("CONSTRAINT_CONTACT_ERROR", "CONSTRAINT", "CONSTRAINT_TOO_WEAK", "GENERATE"),
            ("KEYFRAME_MISMATCH", "KEYFRAME", "SAMPLING_VARIANCE", "GENERATE"),
            ("NATURALNESS_FAILURE", "NATURALNESS", "SAMPLING_VARIANCE", "GENERATE"),
            ("FOOT_SKATING", "PHYSICAL", "PHYSICAL_SAMPLING_ARTIFACT", "GENERATE"),
            ("PRESERVATION_FAILURE", "PRESERVATION", "SEGMENT_BOUNDARY_ARTIFACT", "GENERATE"),
        ]
        for code, family, cause, action in cases:
            direction = {
                "CONSTRAINT_CONTACT_ERROR": "constraint",
                "KEYFRAME_MISMATCH": "keyframe",
                "NATURALNESS_FAILURE": "naturalness",
                "FOOT_SKATING": "physical",
                "PRESERVATION_FAILURE": "preservation",
            }.get(code, "semantic" if "DIRECTION" in code else "event_frequency")
            result = diagnose(report_for(code, direction=direction, expected={"expected_count": 3}), motion_spec())
            self.assertEqual(result.failure_cases[0].failure_family, family, code)
            self.assertEqual(result.root_causes[0].cause_code, cause, code)
            self.assertEqual(result.repair_proposals[0].planner_action.value, action, code)

    def test_10_2_proposal_contract(self):
        result = diagnose(report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}), motion_spec())
        for proposal in result.repair_proposals:
            validation = validate_repair_proposal(proposal)
            self.assertEqual(validation.status, "accepted")

    def test_10_3_no_improvement_guard_blocks_same_family(self):
        result = diagnose(report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}), motion_spec())
        proposal = result.repair_proposals[0]
        history = [
            RepairHistoryEntry(
                diagnosis_round=1,
                failure_signature=proposal.failure_signature,
                repair_family=proposal.repair_family,
                planner_action=proposal.planner_action,
                outcome="unchanged",
                proposal_fingerprint=proposal.proposal_fingerprint,
            ),
            RepairHistoryEntry(
                diagnosis_round=2,
                failure_signature=proposal.failure_signature,
                repair_family=proposal.repair_family,
                planner_action=proposal.planner_action,
                outcome="worse",
                proposal_fingerprint="different",
            ),
        ]
        self.assertTrue(RepairHistory(history).no_improvement_states()[0].blocked)
        guarded = diagnose(report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}), motion_spec(), history=history)
        self.assertEqual(guarded.status, "no_valid_repair")

    def test_10_4_improvement_tracking(self):
        self.assertEqual(classify_metric_progression(previous_metric=None, current_metric=0.28, threshold=0.05), "failed")
        self.assertEqual(classify_metric_progression(previous_metric=0.28, current_metric=0.12, threshold=0.05), "improved")
        self.assertEqual(classify_metric_progression(previous_metric=0.12, current_metric=0.04, threshold=0.05), "resolved")

    def test_10_5_regression_tracking(self):
        before = normalize_failures(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", measured=0.28, threshold=0.05))
        after = normalize_failures(report_for("SEMANTIC_MISMATCH", direction="semantic"))
        regressions = detect_regressions(before, after)
        self.assertTrue(regressions)
        self.assertFalse(targeted_repair_success(target_resolved=True, regressions=regressions))

    def test_10_6_ambiguous_diagnosis_unknown(self):
        result = diagnose(report_for("UNMAPPED_UNKNOWN_FAILURE", direction="semantic"))
        self.assertEqual(result.root_causes[0].cause_code, "UNKNOWN")
        self.assertEqual(result.status, "ambiguous")

    def test_10_7_duplicate_proposal_rejected(self):
        result = diagnose(report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}), motion_spec())
        proposal = result.repair_proposals[0]
        validation = validate_repair_proposal(proposal, seen_fingerprints={proposal.proposal_fingerprint})
        self.assertEqual(validation.status, "rejected")
        self.assertIn("DUPLICATE_PROPOSAL", validation.reason_codes)

    def test_10_8_correct_compiler_vs_failed_generation(self):
        report = report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1})
        result = diagnose(report, motion_spec(), generation_request={"motion_spec": motion_spec().model_dump(mode="json")})
        self.assertEqual(result.root_causes[0].cause_code, "GENERATION_SEMANTIC_EXECUTION_FAILURE")
        self.assertEqual(result.repair_proposals[0].planner_action.value, "GENERATE")

    def test_10_9_compiler_lost_requirement(self):
        report = report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1})
        result = diagnose(report, motion_spec(count=None))
        self.assertIn(result.root_causes[0].cause_code, {"SEMANTIC_COMPILATION_LOSS", "SPECIFICATION_INCOMPLETE"})
        self.assertEqual(result.repair_proposals[0].planner_action.value, "COMPILE_MOTION")

    def test_10_10_constraint_validity_split(self):
        invalid = diagnose(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", expected={"spec_valid": False}))
        self.assertEqual(invalid.repair_proposals[0].planner_action.value, "BUILD_CONSTRAINT")
        valid = diagnose(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", expected={"spec_valid": True}))
        self.assertEqual(valid.repair_proposals[0].planner_action.value, "GENERATE")

    def test_10_11_planner_compatibility(self):
        for code, direction in [("KEYFRAME_MISMATCH", "keyframe"), ("PRESERVATION_FAILURE", "preservation"), ("FOOT_SKATING", "physical")]:
            result = diagnose(report_for(code, direction=direction), motion_spec())
            for proposal in result.repair_proposals:
                self.assertEqual(validate_repair_proposal(proposal).status, "accepted")

    def test_10_12_k1_verifier_to_diagnosis_handoff(self):
        report = report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1})
        result = diagnose(report, motion_spec(), generation_request={"num_candidates": 1, "motion_spec": motion_spec().model_dump(mode="json")})
        summary = result.summary()
        self.assertEqual(summary["status"], "diagnosed")
        self.assertEqual(summary["proposal_summaries"][0]["action"], "GENERATE")
        self.assertEqual(summary["primary_failures"], ["EVENT_FREQUENCY_MISMATCH"])


if __name__ == "__main__":
    unittest.main()
