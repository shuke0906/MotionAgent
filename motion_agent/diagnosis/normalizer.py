"""Normalize Phase 9 verifier findings into canonical Phase 10 failures."""

from __future__ import annotations

from typing import Any

from motion_agent.common.fingerprints import failure_signature
from motion_agent.common.ids import new_id
from motion_agent.diagnosis.schemas import FailureCase, FailureFamily
from motion_agent.verification.schemas import VerificationReport, VerifierCheck, VerifierFinding


DIAGNOSTIC_FAMILY: dict[str, FailureFamily] = {
    "EVENT_MISSING": "EVENT",
    "EVENT_ORDER_MISMATCH": "EVENT",
    "EVENT_FREQUENCY_MISMATCH": "EVENT",
    "BODY_PART_MISMATCH": "SEMANTIC",
    "SEMANTIC_BODY_PART_MISMATCH": "SEMANTIC",
    "DIRECTION_MISMATCH": "SEMANTIC",
    "SEMANTIC_DIRECTION_MISMATCH": "SEMANTIC",
    "SEMANTIC_MISMATCH": "SEMANTIC",
    "SEMANTIC_FAILURE": "SEMANTIC",
    "UNCOMMANDED_HEADING_DRIFT": "SEMANTIC",
    "CONSTRAINT_JOINT_ERROR": "CONSTRAINT",
    "CONSTRAINT_POSITION_ERROR": "CONSTRAINT",
    "CONSTRAINT_CONTACT_ERROR": "CONSTRAINT",
    "CONSTRAINT_TRAJECTORY_ERROR": "CONSTRAINT",
    "KEYFRAME_MISMATCH": "KEYFRAME",
    "KEYFRAME_POSE_MISMATCH": "KEYFRAME",
    "NATURALNESS_FAILURE": "NATURALNESS",
    "MOTION_UNNATURAL": "NATURALNESS",
    "MOTION_DISCONTINUITY": "NATURALNESS",
    "JITTER_FAILURE": "NATURALNESS",
    "MOTION_JITTER": "NATURALNESS",
    "FOOT_SKATING": "PHYSICAL",
    "GROUND_PENETRATION": "PHYSICAL",
    "PHYSICAL_METRIC_FAILURE": "PHYSICAL",
    "PRESERVATION_FAILURE": "PRESERVATION",
    "TECHNICAL_FAILURE": "TECHNICAL",
    "TECHNICAL_INVALID": "TECHNICAL",
    "VERIFIER_UNCERTAIN": "VERIFICATION_INFRA",
    "VERIFIER_ERROR": "VERIFICATION_INFRA",
    "VERIFIER_EXECUTION_ERROR": "VERIFICATION_INFRA",
    "PHYSICAL_EVIDENCE_INCOMPLETE": "VERIFICATION_INFRA",
}

CODE_ALIASES = {
    "FREQUENCY_MISMATCH": "EVENT_FREQUENCY_MISMATCH",
    "TEMPORAL_ORDER_FAILURE": "EVENT_ORDER_MISMATCH",
    "KEYFRAME_POSE_MISMATCH": "KEYFRAME_MISMATCH",
    "CONSTRAINT_POSITION_ERROR": "CONSTRAINT_JOINT_ERROR",
    "MOTION_JITTER": "JITTER_FAILURE",
}


def canonical_diagnostic_code(finding: VerifierFinding) -> str:
    code = finding.diagnostic_code
    if not code:
        if finding.status == "uncertain":
            code = "VERIFIER_UNCERTAIN"
        elif finding.status == "error":
            code = "VERIFIER_ERROR"
        else:
            code = f"{finding.direction.upper()}_FAILURE"
    return CODE_ALIASES.get(code, code)


def failure_family_for_code(code: str) -> FailureFamily:
    if code in DIAGNOSTIC_FAMILY:
        return DIAGNOSTIC_FAMILY[code]
    if code.startswith("EVENT_"):
        return "EVENT"
    if code.startswith("CONSTRAINT_"):
        return "CONSTRAINT"
    if code.startswith("KEYFRAME_"):
        return "KEYFRAME"
    if code.startswith("PHYS") or "FOOT" in code or "GROUND" in code:
        return "PHYSICAL"
    if "NATURAL" in code or "JITTER" in code:
        return "NATURALNESS"
    if "PRESERV" in code:
        return "PRESERVATION"
    if "TECHNICAL" in code:
        return "TECHNICAL"
    if "VERIFIER" in code or "EVIDENCE" in code:
        return "VERIFICATION_INFRA"
    if "SEMANTIC" in code or "BODY_PART" in code or "DIRECTION" in code:
        return "SEMANTIC"
    return "UNKNOWN"


def normalize_failures(report: VerificationReport) -> list[FailureCase]:
    checks_by_id = {check.check_id: check for check in report.plan.checks}
    failures: list[FailureCase] = []
    for finding in report.findings:
        if finding.status not in {"fail", "uncertain", "error"}:
            continue
        check = checks_by_id.get(finding.check_id)
        code = canonical_diagnostic_code(finding)
        family = failure_family_for_code(code)
        target_segments = _target_segments(finding, check)
        body_parts = _body_parts(finding, check)
        expected = dict(finding.expected)
        observed = dict(finding.observed)
        if finding.threshold is not None and "threshold" not in expected:
            expected["threshold"] = finding.threshold
        if finding.measured_value is not None and "value" not in observed:
            observed["value"] = finding.measured_value
        signature_payload = {
            "diagnostic_code": code,
            "family": family,
            "target_segments": target_segments,
            "body_parts": body_parts,
            "source_spec_ids": check.source_spec_ids if check else [],
            "target_event": _target_event(expected, observed, check),
            "expected": _stable_requirement(expected),
            "threshold": finding.threshold,
            "threshold_profile": report.threshold_profile,
        }
        failures.append(
            FailureCase(
                failure_id=new_id("failure"),
                source_check_id=finding.check_id,
                verifier_type=finding.direction,
                diagnostic_code=code,
                failure_family=family,
                severity=_severity(finding, check),
                required=finding.required,
                candidate_id=report.candidate_id,
                target_segments=target_segments,
                target_body_parts=body_parts,
                target_event=_target_event(expected, observed, check),
                expected=expected,
                observed=observed,
                threshold=finding.threshold,
                measured_value=finding.measured_value,
                metrics=_metrics_from_finding(finding),
                evidence_refs=[*finding.evidence_refs, *finding.artifact_refs],
                status=finding.status,  # type: ignore[arg-type]
                source_evaluator_version=finding.evaluator_version,
                source_spec_ids=check.source_spec_ids if check else [],
                failure_signature=failure_signature(signature_payload),
            )
        )
    return failures


def _target_segments(finding: VerifierFinding, check: VerifierCheck | None) -> list[int]:
    expected = finding.expected
    observed = finding.observed
    value = expected.get("target_segments") or observed.get("target_segments")
    if isinstance(value, list):
        return [int(v) for v in value]
    segment = expected.get("segment_id") or observed.get("segment_id")
    if segment is not None:
        return [int(segment)]
    return list(check.target_segments) if check else []


def _body_parts(finding: VerifierFinding, check: VerifierCheck | None) -> list[str]:
    value = finding.expected.get("body_parts") or finding.expected.get("body_part")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    observed = finding.observed.get("body_parts") or finding.observed.get("body_part")
    if isinstance(observed, str):
        return [observed]
    return list(check.body_parts) if check else []


def _target_event(expected: dict[str, Any], observed: dict[str, Any], check: VerifierCheck | None) -> str | None:
    for key in ("event", "action", "expected_event", "missing_event"):
        value = expected.get(key) or observed.get(key)
        if value is not None:
            return str(value)
    if check and "event" in check.metadata:
        return str(check.metadata["event"])
    if check and "action" in check.metadata:
        return str(check.metadata["action"])
    return None


def _stable_requirement(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        key: expected[key]
        for key in sorted(expected)
        if key in {"count", "expected_count", "direction", "body_part", "body_parts", "event", "action", "constraint_id", "keyframe_id"}
    }


def _metrics_from_finding(finding: VerifierFinding) -> dict[str, Any]:
    metrics = {}
    if finding.measured_value is not None:
        metrics["measured_value"] = finding.measured_value
    if finding.threshold is not None:
        metrics["threshold"] = finding.threshold
    return metrics


def _severity(finding: VerifierFinding, check: VerifierCheck | None) -> str:
    if check and check.critical:
        return "critical"
    if finding.required:
        return "major"
    if finding.status == "uncertain":
        return "minor"
    return "minor"
