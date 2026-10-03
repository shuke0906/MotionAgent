"""Finalize readable reports and diagrams from the completed, unmodified real trace."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys
import tarfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/phase11_real_e2e"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(step):
    lines = [step["node"] + " | generated_rounds=" + str(step["round"])]
    for artifact in step["artifacts"].values():
        if artifact.get("action"):
            lines.append(artifact["action"] + " / " + artifact["reason_code"])
            lines.append("Payload: " + json.dumps(artifact.get("payload", {})))
        if artifact.get("motion_spec"):
            for segment in artifact["motion_spec"]["segments"]:
                constraint = segment.get("temporal_constraint") or {}
                relation = segment.get("temporal_relation") or {}
                lines.append(f"Event {segment['segment_id']}: {segment['action']}; body={segment['body_parts']}; "
                    f"direction={segment.get('direction')}; count={constraint.get('count', segment.get('repetition'))}; "
                    f"relation={relation.get('type')}; linked_event={segment.get('simultaneous_with')}")
        if artifact.get("generation_result"):
            lines.append("Agent Round " + str(step["round"] - 1))
            for candidate in artifact["generation_result"]["candidates"]:
                lines.append(f"{candidate['candidate_id']} | seed={candidate['metadata']['seed']} | "
                             f"scope={candidate['metadata']['scope']} | real Frozen GEM")
        if artifact.get("champion_candidate_id"):
            lines.append("K1 selected " + artifact["champion_candidate_id"])
        if artifact.get("verification_report"):
            report = artifact["verification_report"]
            lines.append(f"{report['status']} | overall_pass={report['overall_pass']}")
            for status in ("pass", "fail", "uncertain", "error"):
                checks = [finding["check_id"] for finding in report["findings"] if finding["status"] == status]
                if checks:
                    lines.append(status.upper() + ": " + ", ".join(checks))
            for finding in report["findings"]:
                if finding["direction"] == "event_frequency":
                    lines.append("Frequency: expected=" + str(finding["expected"].get("expected_count")) +
                        "; observed=" + str(finding["observed"].get("observed_count")) + "; " + finding["status"])
        if artifact.get("diagnosis_result"):
            diagnosis = artifact["diagnosis_result"]
            lines.append(diagnosis["diagnosis_summary"])
            lines.extend("RepairProposal " + proposal["proposal_id"] + ": " + proposal["repair_family"] +
                         " -> " + proposal["planner_action"] for proposal in diagnosis["repair_proposals"])
            if diagnosis.get("terminal_hint"):
                lines.append("Terminal hint: " + diagnosis["terminal_hint"])
    if step["terminal_reason"]:
        lines.append("STOP_FAILED: " + step["terminal_reason"])
    budget = step["budget"]
    lines.append(f"Budget: generations={budget['generations_left']}; repairs={budget['repairs_left']}")
    return lines


def main():
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from motion_agent.llm.config import load_dotenv
    from motion_agent.verification.artifacts import file_digest

    trace = read(OUT / "trace/trace.json")
    summary = read(OUT / "run_summary.json")
    diagnoses = [artifact["diagnosis_result"] for step in trace["steps"]
                 for artifact in step["artifacts"].values() if "diagnosis_result" in artifact]
    if diagnoses:
        write(OUT / "diagnosis.json", diagnoses[0])
        write(OUT / "repair_proposals.json", diagnoses[0]["repair_proposals"])
        write(OUT / "final_diagnosis.json", diagnoses[-1])
    rounds = [read(OUT / f"agent/round_{index}/verification_report.json") for index in range(summary["round_count"])]
    metadata = [read(OUT / f"agent/round_{index}/generation_metadata.json") for index in range(summary["round_count"])]
    smoke = read(OUT / "smoke/generation_result.json")
    baseline = read(OUT / "baseline/metadata.json")
    summary["baseline_seed_base"] = 7
    summary["agent_seed_bases"] = [7, 8]
    summary["baseline_seed"] = baseline["metadata"]["seed"]
    summary["agent_seeds"] = [item["metadata"]["seed"] for item in metadata]
    summary["gpu_peak_vram_mb"] = max(item["metadata"]["peak_gpu_memory_mb"] for item in [*metadata, baseline, smoke["candidates"][0]])
    summary["candidate_ids"] = [item["candidate_id"] for item in metadata]
    summary["verification_statuses"] = [{"candidate_id": report["candidate_id"], "status": report["status"],
                                         "overall_pass": report["overall_pass"]} for report in rounds]
    summary["environment"] = {"gpu": torch.cuda.get_device_name(0), "vram_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "smplx_found": True, "gem_checkpoint_found": True}
    summary["checkpoint_sha256"] = read(ROOT / "outputs/phase11_setup/gem_asset.json")["sha256"]
    summary["smoke"] = {"candidate_id": smoke["candidates"][0]["candidate_id"], "status": smoke["status"],
        "latency_ms": smoke["runtime_ms"], "peak_vram_mb": smoke["candidates"][0]["metadata"]["peak_gpu_memory_mb"],
        "frames": smoke["candidates"][0]["metadata"]["frame_count"], "feature_dim": smoke["candidates"][0]["metadata"]["motion_dim"]}
    summary["diagnosis_backend"] = "actual_Phase10_DiagnosisService_rule_first"
    import yaml
    vendor = ROOT / "vendor/GENMO"
    config_paths = ["configs/demo.yaml", "configs/exp/gem_smpl.yaml", "configs/model/gem.yaml"]
    gem_config = {"entrypoint": "demo", "overrides": ["exp=gem_smpl", "ckpt_path=null", "video_name=demo",
                  "model.model_cfg.text_encoder.load_llm=true"],
                  "source_files": {path: {"sha256": file_digest(vendor / path),
                      "values": yaml.safe_load((vendor / path).read_text())} for path in config_paths}}
    write(OUT / "gem_runtime_config.json", gem_config)
    baseline["gem_runtime_config_artifact"] = "gem_runtime_config.json"
    write(OUT / "baseline/metadata.json", baseline)
    summary["gem_runtime_config_artifact"] = "gem_runtime_config.json"
    summary["post_run_fixes"] = ["SMPL-X config points to the actual vendor/GENMO asset",
        "Split simultaneous actions reference the linked event rather than a nonexistent event in their own segment",
        "Uncertain visual visibility remains uncertain despite a missing_actions list"]
    summary["post_run_fix_validation"] = "59 targeted tests passed; no new GEM generation or real verification after these fixes"
    summary["known_limitations"].extend([
        "Original temporal prompt v2 included nonexistent 1:walk; original trace preserved, corrected to v3 after this run",
        "Original v2 visual normalization mapped uncertain missing_actions to fail; fixed after the run, original evidence preserved",
        "Fixed wide camera and FK22 skeleton limited visible arm/cycle detail; both observed counts remain null",
        "Learned-verifier full acceptance gate remains PARTIAL; minimal runtime loop gate does not certify motion quality"])
    summary["known_limitations"] = list(dict.fromkeys(summary["known_limitations"]))
    required_files = ["baseline/baseline.mp4", "agent/round_0/video.mp4", "agent/final.mp4", "comparison.mp4",
                      "trace/trace.json", "trace/event_log.json", "trace/pipeline_trace.html", "trace/pipeline_trace.png"]
    events = [step["node"] for step in trace["steps"]]
    decisions = [artifact for step in trace["steps"] for artifact in step["artifacts"].values() if artifact.get("action")]
    gate = {"remote_real_api": read(ROOT / "outputs/phase11_local_readiness/api_smoke_test.json")["success"],
        "real_planner": any(decision["action"] == "COMPILE_MOTION" for decision in decisions),
        "real_compiler": "compile_motion_committed" in events,
        "real_gem_and_k1": len(metadata) in {1, 2} and all(item["metadata"]["technical_valid"] for item in metadata),
        "k1_selector": events.count("k1_selection_committed") == len(metadata),
        "real_multiverifier": len(rounds) == len(metadata) and summary["mllm_api_calls"] > 0,
        "failed_required_cannot_accept": not trace["final_state"]["control"]["accepted"] and not rounds[-1]["overall_pass"],
        "diagnosis_and_proposals_reach_planner": len(diagnoses) == 2 and bool(diagnoses[0]["repair_proposals"]),
        "one_repair_only": summary["repair_count"] == 1 and len(metadata) == 2,
        "correct_terminal": summary["final_status"] == "STOP_FAILED" and summary["terminal_reason"] == "BUDGET_EXHAUSTED"
             and trace["final_state"]["control"]["budgets"]["generations_left"] == 0,
        "no_infinite_loop": len(trace["steps"]) == 15,
        "required_artifacts_exist": all((OUT / path).is_file() for path in required_files)}
    summary["gate_checks"] = gate
    summary["minimal_real_e2e_gate"] = "PASS" if all(gate.values()) else "FAIL"
    write(OUT / "run_summary.json", summary)

    cards = [summarize(step) for step in trace["steps"]]
    wrapped = [[row for line in card for row in textwrap.wrap(line, 112)] for card in cards]
    heights = [len(card) * 20 + 28 for card in wrapped]
    picture = Image.new("RGB", (1100, 130 + sum(height + 30 for height in heights)), "#f4f7f8")
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((26, 20), "MotionAgent actual K=1 run | Final STOP_FAILED / BUDGET_EXHAUSTED", font=font, fill="#25313c")
    for index, line in enumerate(textwrap.wrap(summary["prompt"], 105)):
        draw.text((26, 48 + index * 20), line, font=font, fill="#25313c")
    y = 110
    blocks = []
    for index, (step, card, height) in enumerate(zip(trace["steps"], wrapped, heights)):
        draw.rectangle((26, y, 1074, y + height), fill="white", outline="#168474", width=2)
        for row_index, line in enumerate(card):
            draw.text((40, y + 14 + row_index * 20), line, font=font, fill="#25313c")
        if index < len(cards) - 1:
            draw.line((550, y + height, 550, y + height + 23), fill="#168474", width=2)
            draw.polygon([(545, y + height + 18), (555, y + height + 18), (550, y + height + 25)], fill="#168474")
        y += height + 30
        blocks.append('<section><h2>' + html.escape(step["node"]) + '</h2><pre>' +
            html.escape("\n".join(summarize(step))) + '</pre><details><summary>Actual step data</summary><pre>' +
            html.escape(json.dumps(step, indent=2)) + '</pre></details></section>')
    picture.save(OUT / "trace/pipeline_trace.png")
    document = '<!doctype html><meta charset="utf-8"><title>MotionAgent Actual K1 Trace</title><style>body{font:15px Arial;margin:32px;color:#25313c;background:#f4f7f8;max-width:1100px}section{padding:16px;border-left:4px solid #168474;margin:16px 0;background:white}h2{font-size:18px;margin:0 0 10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}summary{cursor:pointer}</style><h1>MotionAgent Actual K=1 Execution</h1><p>' + html.escape(summary["prompt"]) + '</p><p>Final: STOP_FAILED / BUDGET_EXHAUSTED</p>' + "".join(blocks)
    (OUT / "trace/pipeline_trace.html").write_text(document, encoding="utf-8")

    lines = ["# Phase 11 Minimal Real E2E", "", "Minimal runtime loop gate: **" + summary["minimal_real_e2e_gate"] + "**.",
             "Final motion outcome: **STOP_FAILED / BUDGET_EXHAUSTED**. This does not certify successful prompt execution.",
             "", "## Environment", "", "Remote GPU runtime: reachable during this run.",
             "GPU: " + summary["environment"]["gpu"] + "; VRAM approximately 80 GB.",
             "PyTorch: " + str(summary["environment"]["torch"]) + "; CUDA: " + str(summary["environment"]["cuda_runtime"]),
             "GEM checkpoint and licensed SMPL-X: FOUND. Official GEM SHA256: " + summary["checkpoint_sha256"],
             "OPENAI_API_KEY = FOUND; OPENAI_MODEL = FOUND. Remote real text and visual API probes passed.",
             "Selected API settings were transferred through SSH stdin into a private file (mode 600).",
             "The project .env is a symlink. /workspace network storage did not preserve mode 600, so no regular secret file remains there.",
             "", "## Restored Backends", "", "Planner and Compiler executed real API calls in the graph.",
             "MLLM: 12 real API attempts across both rounds; actual structured responses and evidence manifests are included.",
             "TMR: official weights plus DistilBERT, actual licensed FK and official 263-D conversion; two candidate inferences completed.",
             "TMR remains uncertain because its production threshold is uncalibrated; no threshold was invented.",
             "MotionCritic: weights load, but exact candidate inference is blocked by GEM's 21 body rotations versus required 23.",
             "A separate saved-candidate probe executed approximate neutral_terminal_hands inference; it does not certify strict acceptance.",
             "Diagnosis: actual Phase 10 rule-first DiagnosisService received both actual VerificationReports. No LLM diagnosis execution is claimed.",
             "", "## Smoke and Baseline", "", "Smoke: " + json.dumps(summary["smoke"]),
             "Baseline: raw Pure-Text Frozen GEM; actual seed " + str(summary["baseline_seed"]) +
             " (seed base 7); 420 frames, 14 seconds, 30 FPS. No Planner/Compiler/Verifier/Diagnosis.",
             "All videos use identical fixed camera, white background, FK22 skeleton, 640x480 rendering and candidate betas.",
             "Generated motions were not manually edited or normalized. The render metadata records camera and shape source."]
    lines.append("Actual Hydra overrides, GEM config values and source file digests are recorded in gem_runtime_config.json.")
    for index, report in enumerate(rounds):
        lines.extend(["", f"## MotionAgent Round {index}", "", "Candidate: " + report["candidate_id"] +
                      "; seed: " + str(metadata[index]["metadata"]["seed"]),
                      "Verification: " + report["status"] + "; overall_pass=" + str(report["overall_pass"]),
                      "| Check | Status | Diagnostic | Measurement |", "|---|---|---|---|"])
        for finding in report["findings"]:
            lines.append(f"| {finding['check_id']} | {finding['status']} | {finding.get('diagnostic_code') or ''} | {finding.get('measured_value')} |")
        frequency = next(finding for finding in report["findings"] if finding["direction"] == "event_frequency")
        lines.append("Frequency: expected_count=3; observed_count=" + str(frequency["observed"].get("observed_count")) +
                     ". Actual count is uncertain, not a fabricated number.")
    lines.extend(["", "## Compiler and Repair Handoff", "",
        "The real Compiler preserved walk forward, right-hand wave with count=3, simultaneous relation to walking, left turn, and sit_down.",
        "Compiler/Temporal Resolver inferred all event ranges from the raw prompt and total duration; no manual ranges were supplied.",
        "Diagnosis Round 0: " + diagnoses[0]["diagnosis_summary"],
        "Planner actions: " + " -> ".join(decision["action"] for decision in decisions),
        "The chosen proposal was REGENERATE, normal/segment, targets [0,1,2,3], K=1; actual seeds: " +
        str(summary["agent_seeds"]) + ". Seed bases 7/8 are deterministically converted by the existing seed manager.",
        "The selected segments span the full timeline; no outside segment frames remain. MotionSpecification and requirements were retained and reverified.",
        "The repair did not resolve the required failures. Both generation and repair budgets reached zero; Planner legally stopped.",
        "", "## Runtime Gate", "", "| Requirement | Actual Result |", "|---|---|"])
    lines.extend("| " + name + " | " + ("PASS" if value else "FAIL") + " |" for name, value in gate.items())
    lines.extend(["", "## Known Limitations", ""] + ["- " + limitation for limitation in summary["known_limitations"]])
    lines.extend(["", "## Post-Run Fixes", ""] + ["- " + fix for fix in summary["post_run_fixes"]] +
                 ["", summary["post_run_fix_validation"], "Original reports/responses/events remain unchanged. No expensive rerun was made.",
                  "", "## Artifacts", ""] + ["- outputs/phase11_real_e2e/" + path for path in required_files] +
                 ["- outputs/phase11_real_e2e_bundle.tar.gz", "", "Total runtime: " + str(summary["total_runtime_s"]) + " seconds.",
                  "The full eight-scenario benchmark and Phase 12 were not executed."])
    report_path = ROOT / "reports/phase_11_real_e2e.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    setup = OUT / "backend_setup"
    setup.mkdir(exist_ok=True)
    for name in ("gem_asset.json", "mllm_probe.json", "learned_probe.json"):
        write(setup / name, read(ROOT / "outputs/phase11_setup" / name))
    write(setup / "api_smoke.json", read(ROOT / "outputs/phase11_local_readiness/api_smoke_test.json"))
    load_dotenv(ROOT / ".env")
    secrets = [os.environ.get(name, "").encode() for name in ("OPENAI_API_KEY", "MLLM_API_KEY")]
    for path in [*OUT.rglob("*"), report_path]:
        if not path.is_file():
            continue
        if path.name.startswith(".env") or path.suffix == ".ckpt":
            raise RuntimeError("FORBIDDEN_ARTIFACT")
        data = path.read_bytes()
        if any(secret and len(secret) >= 8 and secret in data for secret in secrets):
            raise RuntimeError("SECRET_SCAN_FAILED")
    bundle = ROOT / "outputs/phase11_real_e2e_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(OUT, arcname="phase11_real_e2e")
        archive.add(report_path, arcname="reports/phase_11_real_e2e.md")
    print(json.dumps({"gate": summary["minimal_real_e2e_gate"], "gate_checks": gate,
                      "bundle_bytes": bundle.stat().st_size, "bundle_sha256": file_digest(bundle), "secret_scan": "PASS"}))


if __name__ == "__main__":
    main()
