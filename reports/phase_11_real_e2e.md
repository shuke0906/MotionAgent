# Phase 11 Minimal Real E2E

Minimal runtime loop gate: **PASS**.
Final motion outcome: **STOP_FAILED / BUDGET_EXHAUSTED**. This does not certify successful prompt execution.

## Environment

SSH: <REMOTE_ENDPOINT>, reachable.
GPU: NVIDIA A100-SXM4-80GB; VRAM approximately 80 GB.
PyTorch: 2.8.0+cu128; CUDA: 12.8
GEM checkpoint and licensed SMPL-X: FOUND. Official GEM SHA256: 1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a
OPENAI_API_KEY = FOUND; OPENAI_MODEL = FOUND. Remote real text and visual API probes passed.
API settings are selected from local .env, sent through SSH stdin, stored at <PRIVATE_API_CONFIG> (600).
The project .env is a symlink. /workspace network storage did not preserve mode 600, so no regular secret file remains there.

## Restored Backends

Planner and Compiler executed real API calls in the graph.
MLLM: 12 real API attempts across both rounds; actual structured responses and evidence manifests are included.
TMR: official weights plus DistilBERT, actual licensed FK and official 263-D conversion; two candidate inferences completed.
TMR remains uncertain because its production threshold is uncalibrated; no threshold was invented.
MotionCritic: weights load, but exact candidate inference is blocked by GEM's 21 body rotations versus required 23.
A separate saved-candidate probe executed approximate neutral_terminal_hands inference; it does not certify strict acceptance.
Diagnosis: actual Phase 10 rule-first DiagnosisService received both actual VerificationReports. No LLM diagnosis execution is claimed.

## Smoke and Baseline

Smoke: {"candidate_id": "cand_69f20489384351f8", "status": "success", "latency_ms": 3960.811760276556, "peak_vram_mb": 6805.35205078125, "frames": 90, "feature_dim": 151}
Baseline: raw Pure-Text Frozen GEM; actual seed 695425565 (seed base 7); 420 frames, 14 seconds, 30 FPS. No Planner/Compiler/Verifier/Diagnosis.
All videos use identical fixed camera, white background, FK22 skeleton, 640x480 rendering and candidate betas.
Generated motions were not manually edited or normalized. The render metadata records camera and shape source.
Actual Hydra overrides, GEM config values and source file digests are recorded in gem_runtime_config.json.

## MotionAgent Round 0

Candidate: cand_62029139ee35ecf7; seed: 695425565
Verification: incomplete; overall_pass=False
| Check | Status | Diagnostic | Measurement |
|---|---|---|---|
| technical | pass |  | None |
| semantic | uncertain | TMR_THRESHOLD_UNCALIBRATED | 0.7304432988166809 |
| naturalness_kinematic | fail | KINEMATIC_JITTER | 224.05165100097656 |
| event_integrity | fail | EVENT_MISSING | None |
| event_temporal | uncertain | MLLM_UNCERTAIN | None |
| event_frequency_1 | uncertain | MLLM_UNCERTAIN | None |
| motioncritic | uncertain | MOTIONCRITIC_UNAVAILABLE | None |
| semantic_compositional | fail | SEMANTIC_MISMATCH | None |
| direction_2 | fail | DIRECTION_MISMATCH | -148.96707346952812 |
| physical_foot_skating | pass |  | 0.07554067671298981 |
| physical_ground_penetration | pass |  | 0.0 |
| physical_smoothness | fail | MOTION_JITTER | 12.573344230651855 |
| physical | fail | PHYSICAL_METRIC_FAILURE | None |
Frequency: expected_count=3; observed_count=None. Actual count is uncertain, not a fabricated number.

## MotionAgent Round 1

Candidate: cand_d442a42f1adee871; seed: 486847130
Verification: incomplete; overall_pass=False
| Check | Status | Diagnostic | Measurement |
|---|---|---|---|
| technical | pass |  | None |
| semantic | uncertain | TMR_THRESHOLD_UNCALIBRATED | 0.6903961300849915 |
| naturalness_kinematic | pass |  | 20.11578369140625 |
| event_integrity | fail | EVENT_MISSING | None |
| event_temporal | uncertain | MLLM_UNCERTAIN | None |
| event_frequency_1 | uncertain | MLLM_UNCERTAIN | None |
| motioncritic | uncertain | MOTIONCRITIC_UNAVAILABLE | None |
| semantic_compositional | fail | SEMANTIC_MISMATCH | None |
| direction_2 | fail | DIRECTION_MISMATCH | -16.259280213302432 |
| physical_foot_skating | pass |  | 0.08877800405025482 |
| physical_ground_penetration | fail | GROUND_PENETRATION | 1.0850834846496582 |
| physical_smoothness | fail | MOTION_JITTER | 15.065988540649414 |
| physical | fail | PHYSICAL_METRIC_FAILURE | None |
Frequency: expected_count=3; observed_count=None. Actual count is uncertain, not a fabricated number.

## Compiler and Repair Handoff

The real Compiler preserved walk forward, right-hand wave with count=3, simultaneous relation to walking, left turn, and sit_down.
Compiler/Temporal Resolver inferred all event ranges from the raw prompt and total duration; no manual ranges were supplied.
Diagnosis Round 0: Diagnosed 10 failure(s); root causes: VERIFIER_INFRASTRUCTURE_FAILURE, SAMPLING_VARIANCE, GENERATION_SEMANTIC_EXECUTION_FAILURE, VERIFIER_INFRASTRUCTURE_FAILURE, VERIFIER_INFRASTRUCTURE_FAILURE, UNKNOWN, GENERATION_SEMANTIC_EXECUTION_FAILURE, PHYSICAL_SAMPLING_ARTIFACT; proposed planner actions: STOP_FAILED, GENERATE, GENERATE, STOP_FAILED, STOP_FAILED, STOP_FAILED, GENERATE, GENERATE.
Planner actions: COMPILE_MOTION -> GENERATE -> GENERATE -> STOP_FAILED
The chosen proposal was REGENERATE, normal/segment, targets [0,1,2,3], K=1; actual seeds: [695425565, 486847130]. Seed bases 7/8 are deterministically converted by the existing seed manager.
The selected segments span the full timeline; no outside segment frames remain. MotionSpecification and requirements were retained and reverified.
The repair did not resolve the required failures. Both generation and repair budgets reached zero; Planner legally stopped.

## Runtime Gate

| Requirement | Actual Result |
|---|---|
| remote_real_api | PASS |
| real_planner | PASS |
| real_compiler | PASS |
| real_gem_and_k1 | PASS |
| k1_selector | PASS |
| real_multiverifier | PASS |
| failed_required_cannot_accept | PASS |
| diagnosis_and_proposals_reach_planner | PASS |
| one_repair_only | PASS |
| correct_terminal | PASS |
| no_infinite_loop | PASS |
| required_artifacts_exist | PASS |

## Known Limitations

- TMR threshold uncalibrated
- MotionCritic lacks full SMPL terminal hand retargeting
- Skeleton renderer shows actual FK22, no finger animation
- Original temporal prompt v2 included nonexistent 1:walk; original trace preserved, corrected to v3 after this run
- Original v2 visual normalization mapped uncertain missing_actions to fail; fixed after the run, original evidence preserved
- Fixed wide camera and FK22 skeleton limited visible arm/cycle detail; both observed counts remain null
- Learned-verifier full acceptance gate remains PARTIAL; minimal runtime loop gate does not certify motion quality

## Post-Run Fixes

- SMPL-X config points to the actual vendor/GENMO asset
- Split simultaneous actions reference the linked event rather than a nonexistent event in their own segment
- Uncertain visual visibility remains uncertain despite a missing_actions list

59 targeted tests passed; no new GEM generation or real verification after these fixes
Original reports/responses/events remain unchanged. No expensive rerun was made.

## Artifacts

- outputs/phase11_real_e2e/baseline/baseline.mp4
- outputs/phase11_real_e2e/agent/round_0/video.mp4
- outputs/phase11_real_e2e/agent/final.mp4
- outputs/phase11_real_e2e/comparison.mp4
- outputs/phase11_real_e2e/trace/trace.json
- outputs/phase11_real_e2e/trace/event_log.json
- outputs/phase11_real_e2e/trace/pipeline_trace.html
- outputs/phase11_real_e2e/trace/pipeline_trace.png
- outputs/phase11_real_e2e_bundle.tar.gz

Total runtime: 343.91 seconds.
The full eight-scenario benchmark and Phase 12 were not executed.
