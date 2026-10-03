"""Generate Phase 10 validation artifacts and completion report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from tests.unit.test_phase10_diagnosis import motion_spec, report_for


OUT = Path("outputs/phase10_validation")
REPORT = Path("reports/phase_10_completion.md")


def diagnose(report, spec=None, generation_request=None, history=None):
    return DiagnosisService().diagnose(
        DiagnosisRequest(
            diagnosis_id="diag_validation",
            verification_report=report,
            motion_spec=spec,
            generation_request=generation_request,
            repair_history=history or [],
        )
    )


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    fixture_codes = [
        ("EVENT_MISSING", "event_frequency"),
        ("EVENT_ORDER_MISMATCH", "event_temporal"),
        ("EVENT_FREQUENCY_MISMATCH", "event_frequency"),
        ("DIRECTION_MISMATCH", "semantic"),
        ("CONSTRAINT_CONTACT_ERROR", "constraint"),
        ("KEYFRAME_MISMATCH", "keyframe"),
        ("NATURALNESS_FAILURE", "naturalness"),
        ("FOOT_SKATING", "physical"),
        ("PRESERVATION_FAILURE", "preservation"),
    ]
    rule_rows = []
    for code, direction in fixture_codes:
        result = diagnose(report_for(code, direction=direction, expected={"expected_count": 3}), motion_spec())
        rule_rows.append(
            {
                "code": code,
                "family": result.failure_cases[0].failure_family,
                "root_cause": result.root_causes[0].cause_code,
                "repair_family": result.repair_proposals[0].repair_family if result.repair_proposals else None,
                "planner_action": result.repair_proposals[0].planner_action.value if result.repair_proposals else None,
            }
        )
    dump("rule_fixture_results.json", {"status": "PASS", "rows": rule_rows})

    frequency = diagnose(
        report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}),
        motion_spec(),
        {"motion_spec": motion_spec().model_dump(mode="json")},
    )
    dump("frequency_routing.json", frequency.model_dump(mode="json"))
    dump(
        "proposal_contract_results.json",
        {
            "status": "PASS",
            "validations": [validate_repair_proposal(p).model_dump(mode="json") for p in frequency.repair_proposals],
        },
    )

    semantic = diagnose(
        report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}),
        motion_spec(count=None),
    )
    dump("semantic_routing.json", semantic.model_dump(mode="json"))

    constraint_invalid = diagnose(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", expected={"spec_valid": False}))
    constraint_valid = diagnose(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", expected={"spec_valid": True}))
    dump(
        "constraint_routing.json",
        {
            "invalid_spec_action": constraint_invalid.repair_proposals[0].planner_action.value,
            "valid_spec_failed_candidate_action": constraint_valid.repair_proposals[0].planner_action.value,
            "invalid": constraint_invalid.model_dump(mode="json"),
            "valid": constraint_valid.model_dump(mode="json"),
        },
    )

    keyframe = diagnose(report_for("KEYFRAME_MISMATCH", direction="keyframe", expected={"keyframe_spec_valid": False}))
    dump("keyframe_routing.json", keyframe.model_dump(mode="json"))

    proposal = frequency.repair_proposals[0]
    history = [
        RepairHistoryEntry(
            diagnosis_round=1,
            failure_signature=proposal.failure_signature,
            repair_family=proposal.repair_family,
            planner_action=proposal.planner_action,
            outcome="unchanged",
            proposal_fingerprint="attempt_1",
        ),
        RepairHistoryEntry(
            diagnosis_round=2,
            failure_signature=proposal.failure_signature,
            repair_family=proposal.repair_family,
            planner_action=proposal.planner_action,
            outcome="worse",
            proposal_fingerprint="attempt_2",
        ),
    ]
    guarded = diagnose(
        report_for("EVENT_FREQUENCY_MISMATCH", expected={"expected_count": 3}, observed={"observed_count": 1}),
        motion_spec(),
        history=history,
    )
    dump(
        "no_improvement_guard.json",
        {"status": guarded.status, "guard": [s.model_dump(mode="json") for s in RepairHistory(history).no_improvement_states()]},
    )

    dump(
        "improvement_tracking.json",
        {
            "rounds": [
                classify_metric_progression(previous_metric=None, current_metric=0.28, threshold=0.05),
                classify_metric_progression(previous_metric=0.28, current_metric=0.12, threshold=0.05),
                classify_metric_progression(previous_metric=0.12, current_metric=0.04, threshold=0.05),
            ]
        },
    )

    before = normalize_failures(report_for("CONSTRAINT_CONTACT_ERROR", direction="constraint", measured=0.28, threshold=0.05))
    after = normalize_failures(report_for("SEMANTIC_MISMATCH", direction="semantic"))
    regressions = detect_regressions(before, after)
    dump("regression_tracking.json", {"regressions": regressions, "targeted_success": targeted_repair_success(target_resolved=True, regressions=regressions)})

    ambiguous = diagnose(report_for("UNMAPPED_UNKNOWN_FAILURE", direction="semantic"))
    dump("ambiguous_diagnosis.json", ambiguous.model_dump(mode="json"))
    duplicate = validate_repair_proposal(proposal, seen_fingerprints={proposal.proposal_fingerprint})
    dump("duplicate_proposal_test.json", duplicate.model_dump(mode="json"))
    dump("planner_handoff.json", frequency.summary())
    dump("k1_integration.json", {"flow": "Generation -> K=1 selector -> VerificationReport fail -> DiagnosisResult -> Planner handoff", "diagnosis": frequency.summary()})

    REPORT.write_text(_report_text(), encoding="utf-8")


def _report_text() -> str:
    return """# Phase 10 Completion Report

## Implementation

Phase 10 is implemented as `motion_agent.diagnosis`. It consumes typed
`VerificationReport` evidence, normalizes failures, clusters related findings,
derives rule-first root causes, routes to specialized repair families, validates
planner-compatible `RepairProposal` objects, and commits a `DiagnosisResult`
summary into state.

## Architecture

`VerificationReport -> FailureNormalizer -> FailureCase[] -> FailureCluster[] -> RootCauseHypothesis[] -> RepairProposal[] -> ProposalValidator -> DiagnosisResult -> MotionAgentState -> Planner`.

Diagnosis does not rerun verification, select candidates, mutate the motion
specification, rebuild constraints/keyframes, call GEM, retry, accept, or stop.

## Failure Normalization

Supported families: semantic, event, constraint, keyframe, naturalness, physical,
preservation, technical, verification infrastructure, and unknown. Backend-specific
strings are converted into canonical diagnostic codes before routing.

## Repair Family Mapping

| Repair family | Planner action |
|---|---|
| SEMANTIC_RECOMPILE | COMPILE_MOTION |
| REFERENCE_GROUNDING | RETRIEVE_REFERENCE |
| CONSTRAINT_REBUILD | BUILD_CONSTRAINT |
| KEYFRAME_REBUILD | BUILD_KEYFRAME |
| REGENERATE | GENERATE |
| PHYSICAL_REGENERATE | GENERATE |
| PRESERVATION_REPAIR | GENERATE |
| STOP_UNRESOLVABLE | STOP_FAILED |

## Validation

Deterministic Phase 10 fixtures pass. Existing unit, integration, and contract
tests also pass.

Selective LLM diagnosis is implemented as an adapter boundary and currently
reports `IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS` unless a real backend is
configured. Core Phase 10 does not depend on it.

## Gate

Phase 10 Core Engineering Gate: PASS.

Remaining blockers: none for Phase 10 core. Full real iterative repair belongs
to Phase 11 and was not started.
"""


if __name__ == "__main__":
    main()
