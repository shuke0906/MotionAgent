"""Required-check aggregation for Phase 9."""

from __future__ import annotations

from motion_agent.verification.schemas import (
    VerificationPlan,
    VerificationReport,
    VerificationRequest,
    VerifierFinding,
)


def aggregate_verification(
    request: VerificationRequest,
    plan: VerificationPlan,
    findings: list[VerifierFinding],
) -> VerificationReport:
    findings_by_id = {finding.check_id: finding for finding in findings}
    failed_required: list[str] = []
    incomplete = False
    invalid = False

    for check_id in plan.required_check_ids:
        finding = findings_by_id.get(check_id)
        if finding is None:
            incomplete = True
            failed_required.append(check_id)
            continue
        if finding.direction == "technical" and finding.status != "pass":
            invalid = True
        if finding.status in {"uncertain", "error"}:
            incomplete = True
            failed_required.append(check_id)
        elif finding.status == "fail":
            failed_required.append(check_id)
        elif finding.status != "pass":
            incomplete = True
            failed_required.append(check_id)

    if invalid:
        status = "invalid"
        overall_pass = False
    elif incomplete:
        status = "incomplete"
        overall_pass = False
    else:
        status = "complete"
        overall_pass = len(failed_required) == 0

    warnings = [
        f"{finding.check_id}:{finding.status}:{finding.diagnostic_code or finding.message or ''}"
        for finding in findings
        if not finding.required and finding.status != "pass"
    ]
    return VerificationReport(
        verification_id=request.verification_id,
        candidate_id=request.candidate_id,
        plan=plan,
        findings=findings,
        failed_required_checks=failed_required,
        warnings=warnings,
        status=status,
        overall_pass=overall_pass,
        threshold_profile=request.threshold_profile,
        evaluator_versions=request.evaluator_versions,
        artifact_refs=[request.candidate_id],
    )
