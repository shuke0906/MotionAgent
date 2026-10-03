# Phase 11: Minimal Real End-to-End Demo

## Goal and Prompt

Validate the smallest useful real loop, not the full eight-scenario benchmark or superiority over GEM.

Run: `phase11_minimal_20261003T003949Z`.

> A person walks forward while waving their right hand three times, then turns left and sits down.

**Runtime gate PASS; final motion STOP_FAILED / BUDGET_EXHAUSTED.** The system generated, verified, diagnosed, repaired once, reverified, and terminated without accepting failed required checks.

## Runtime Setup

Recorded environment: Linux GPU pod, Python 3.12.3, A100-SXM4-80GB, PyTorch 2.8.0+cu128, CUDA 12.8. Real API text and visual probes succeeded. Planner: `RealLLMPlannerBackend`; compiler: `RealLLMMotionCompiler`; API model: `gpt-6-sol`. Configure an available model with the required capabilities rather than assuming access to this model.

For the recorded CUDA stack, install PyTorch first:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-phase11.txt
```

`requirements-phase11.txt` records important compatibility versions, not a complete lockfile. Additional upstream imports/support assets may be needed as GENMO changes. The run used Transformers 4.46.3 and PyTorch Lightning 2.3.0; do not assume historical `requirements-phase9b.txt` is interchangeable. Text encoder and DistilBERT assets are external downloads. `ffmpeg` must be on PATH.

Follow [Quick Setup](../README.md#quick-setup) for external GENMO and patch installation. The small patch set preserves the loader used in the run and checksum-verified GVHMR support downloader. Install licensed SMPL-X at `vendor/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`.

For real learned verification:

```bash
git clone https://github.com/Mathux/TMR.git vendor/TMR
git clone https://github.com/ou524u/MotionCritic.git vendor/MotionCritic
```

Use TMR's `prepare/download_pretrain_models.sh` from its repository and MotionCritic's `prepare/prepare_pretrained.sh` from `vendor/MotionCritic/`. Check the upstream README before download/install scripts. [Configured](../configs/learned_verifiers.yaml) asset paths:

```text
vendor/TMR/models/tmr_humanml3d_guoh3dfeats/last_weights/
vendor/MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth
```

No checkpoints, body models, tensors, or private configuration are included publicly.

## Baseline and Agent Setup

Baseline: exact raw prompt -> one full-clip condition -> Pure-Text Frozen GEM, without Planner, structured semantics, verifier, diagnosis, or repair. Actual seed **695425565**.

Agent: K=1; `max_generation_rounds=2`, `max_repair_rounds=1`. Actual seeds **695425565** and **486847130**. Seed bases 7/8 are converted by the existing seed manager, not used directly as generator seeds.

Both paths: 14 seconds, 420 frames, 30 FPS. Identical FK22 renderer: 640x480, white background, camera `[8,5.5,9]`, target `[0,1,2.5]`, FOV 45 degrees; each candidate's own body shape. No motion editing/normalization. Three-column comparison: 1920x480, synchronized.

GEM checkpoint SHA256: `1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`.

One preceding smoke produced `cand_69f20489384351f8`: 90 frames, 151 features, finite motion and successful render, 3960.81 ms latency, 6805.35 MB peak allocation. Only four GEM generations occurred: smoke, baseline, Round 0, Round 1.

## Planner and Compiler

Actual decisions: `COMPILE_MOTION / INITIAL_COMPILE`, `GENERATE / CONDITIONS_READY`, repair `GENERATE / SEMANTIC_FAILURE`, then `STOP_FAILED / BUDGET_EXHAUSTED`.

| Segment | Action | Body / direction | Repetition / relation |
|---|---|---|---|
| 0 | walk | full_body / forward | Walk and wave overlap |
| 1 | wave | right_hand | count=3; simultaneous_with=0 |
| 2 | turn | full_body / left | Follows walk/wave |
| 3 | sit_down | full_body | Follows turn |

All internal ranges were inferred by the Temporal Resolver; only total duration was supplied. [MotionSpecification](../assets/demo/motion_specification.json) is actual compiler output, not manually repaired semantics.

## Round 0 and Verification

Candidate `cand_62029139ee35ecf7`: `status=incomplete`, `overall_pass=false`.

| Selected check | Round 0 | Round 1 | Evidence / qualification |
|---|---|---|---|
| Technical | pass | pass | Finite 420x151 motion and SMPL outputs |
| Semantic / TMR | uncertain | uncertain | Scores 0.730443 and 0.690396; uncalibrated threshold |
| Kinematic naturalness | fail | pass | Root jerk 224.052 -> 20.116 m/s^3; threshold 100 |
| Event integrity | fail* | fail* | Original EVENT_MISSING normalization |
| Event temporal | uncertain | uncertain | MLLM_UNCERTAIN; original v2 linking issue |
| Event frequency | uncertain | uncertain | expected_count=3; observed_count=null |
| MotionCritic | uncertain | uncertain | Strict representation unavailable; zero candidate inferences |
| Compositional semantics | fail* | fail* | Original SEMANTIC_MISMATCH normalization |
| Turn direction | fail | fail | Heading -148.967 and -16.259 degrees; left-positive convention |
| Foot skating | pass | pass | 0.075541 and 0.088778 |
| Ground penetration | pass | fail | 0 and 1.085083 |
| Physical smoothness | fail | fail | 12.573344 and 15.065989 |
| Physical aggregate | fail | fail | PHYSICAL_METRIC_FAILURE |

*Original reports are not definitive visual proof of missing actions: raw visual outputs were uncertain, but normalization mapped nonempty `missing_actions` to failure. This was corrected afterward. Original evidence is retained, not rewritten under the corrected policy.*

MLLM made 12 API attempts across both rounds. TMR performed two candidate inferences using licensed FK and official 263-D features. Exact MotionCritic was blocked by GEM's 21 body rotations versus required 23; loaded weights and an independent approximate probe do not establish strict acceptance.

## Diagnosis and Repair Proposal

Actual Phase 10 rule-first `DiagnosisService` received the Round 0 report. Status `ambiguous`, 10 failed/uncertain findings; no LLM diagnosis executed. Hypotheses included verifier infrastructure failure, sampling variance, semantic execution failure, physical sampling artifacts, and unknown causes. No single confirmed root cause was established.

The Planner chose this actual proposal (compact excerpt):

```json
{
  "proposal_id": "repair_fb6286551d08",
  "repair_family": "REGENERATE",
  "planner_action": "GENERATE",
  "reason_code": "SEMANTIC_FAILURE",
  "target_segments": [0, 1, 2, 3],
  "parameters": {
    "strategy": "normal",
    "scope": "segment",
    "num_candidates": 1,
    "reward_targets": []
  },
  "preserve_requirements": [
    "check:technical",
    "check:physical_foot_skating",
    "check:physical_ground_penetration"
  ]
}
```

Selected segments span the whole timeline, so this was not a small localized edit. The validated specification was retained. Preservation requirements were rechecked, not assumed; ground penetration subsequently regressed.

## Round 1 and Final Outcome

One repair produced `cand_d442a42f1adee871`. Reverification remained incomplete, `overall_pass=false`. A second diagnosis received that report; the Planner stopped with generation and repair budgets both zero.

**STOP_FAILED / BUDGET_EXHAUSTED** is correct bounded runtime behavior, not accepted motion or evidence that repair always helps.

Recorded runtime: **343.91 seconds**. Maximum candidate peak allocation: **6937.06 MB**, not total system VRAM. Main-loop MLLM attempts: **12**; TMR inferences: **2**; strict MotionCritic inferences: **0**. Exact total Planner/compiler API count was not recorded.

## Artifacts and Provenance

- [Baseline](../assets/demo/baseline.mp4), [Round 0](../assets/demo/agent_round0.mp4), [Final / Round 1](../assets/demo/agent_final.mp4)
- [Full comparison](../assets/demo/comparison.mp4), [repair comparison](../assets/demo/repair_comparison.mp4), [GIF](../assets/demo/repair_preview.gif)
- [Trace PNG](../assets/demo/pipeline_trace.png), [HTML](../assets/demo/pipeline_trace.html), [trace JSON](../assets/demo/trace.json), [EventLog](../assets/demo/event_log.json)
- [Summary](../assets/demo/demo_summary.json), [specification](../assets/demo/motion_specification.json), [verification excerpts](../assets/demo/verification.json), [diagnosis/proposals](../assets/demo/diagnosis.json)

Public JSON is a selected, sanitized derivative of the real run. IDs, timestamps, actions, budgets, and findings are preserved; large payloads are summarized, connection details and machine paths removed. Originals remain local. Repair MP4/GIF are layout/encoding derivatives of existing videos, not regenerated motions.

## Known Limitations and Post-Run Fixes

- TMR production threshold is uncalibrated; full learned-verifier acceptance remains partial.
- MotionCritic needs exact terminal-hand retargeting; approximate scores are not certification.
- Wide-camera FK22 evidence lacks finger detail and reliable wave counts.
- Original temporal prompt v2 referenced nonexistent `1:walk`. Linked-segment resolution was fixed; temporal/integrity/compositional prompts became v3.
- Uncertain visual observations now remain uncertain despite nonempty `missing_actions`. Original reports are unchanged.
- Fixes were unit-tested, without new real verification or GEM generation. Corrected code can produce different reports from this recorded demo.
- Dependencies are compatibility guidance, not a locked GPU image. Retrieval requires external real data/index assets.
- This is one minimal experiment, not the full Phase 11 benchmark. Phase 12 was not started.
