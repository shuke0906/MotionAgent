# Phase 4 Controlled Comparison

Date: 2026-09-27

## Scope

This rerun validates Phase 4 only. It compares baseline Frozen GEM against the
real MotionAgent Motion Compiler using the same prompt, checkpoint, sampler,
seed, frame count, and fixed global camera. Phase 5 was not started and no
Phase 5+ module was modified.

## Controlled Inputs

- Prompt: `walk forward, wave the right hand, then sit down`
- Frames: 300
- Frame rate: 30 fps
- Duration: 10 seconds
- Sampler seed: 7
- Sampler: GEM checkpoint default
- Checkpoint: `/workspace/MotionAgent/vendor/GENMO/inputs/pretrained/gem_smpl.ckpt`
- Checkpoint SHA-256:
  `1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`
- Model state: evaluation mode with all GEM weights frozen

The baseline condition is one caption over `[0.0, 1.0]`. The Phase 4 condition
comes from the real Motion Compiler and contains these normalized windows:

| Caption | Start | End |
| --- | ---: | ---: |
| `a person walks forward` | 0.000000 | 0.430000 |
| `a person waves the right hand` | 0.430000 | 0.713333 |
| `a person sits down` | 0.713333 | 1.000000 |

The exact comma-separated prompt exposed a real Phase 3 parser bug: punctuation
normalization removed the clause boundaries before sequential splitting, so the
walk clause was omitted. The narrow fix preserves commas and recognizes
`A, B, then C`; a 300-frame regression test now covers the exact prompt. The
compiler's normalized duration allocation was otherwise preserved.

## Fixed Camera

Both renders use one comparison-specific GENMO/Open3D camera without per-clip
auto-framing:

- Position: `[8.0, 5.5, 9.0]`
- Target center: `[0.0, 1.0, 2.5]`
- Up vector: `[0.0, 1.0, 0.0]`
- Vertical field of view: 45 degrees
- Camera height: 5.5
- Viewing direction:
  `[-0.7112867591590193, -0.40009880202694836, -0.5779204918167031]`
- Ground center X/Z: `[0.0, 2.5]`
- Ground size: 12.0
- Render size: 640x360 per clip; 1280x360 side by side

The renderer reuses GENMO's SMPL-X, global-vertex normalization, camera sensor,
mesh, ground, side-by-side, and video utilities. Camera position, target, FOV,
height, and direction remain identical for every frame of both clips.

## Results

### Baseline GEM

- Mode: single caption
- Candidate: `cand_c0b8b8305b86194f`
- Motion shape: `[300, 151]`
- Motion finite: yes
- SMPL finite and structurally valid: yes
- Runtime: 5.98 seconds

The baseline produces a technically valid compound-prompt motion and reaches a
seated pose. Its action transitions are less explicitly controlled because the
entire prompt is supplied as one caption.

### MotionAgent Phase 4

- Mode: real Motion Compiler multi-text windows
- Candidate: `cand_b975711ed0e2474e`
- Motion shape: `[300, 151]`
- Motion finite: yes
- SMPL finite and structurally valid: yes
- Runtime: 7.60 seconds

The Phase 4 result visibly stages forward walking, a raised right-hand wave,
and a final sit-down. The body reaches and holds a seated pose in the final
segment. Because the earlier 180-frame sample returned to standing while this
controlled 300-frame sample completes the action, the earlier failure was
likely related to the shorter generation horizon.

## Reproducibility And Traceability

Same-seed reproducibility was tested with two actual uncached Phase 4 inference
runs using separate CandidateStore roots. Both produced the same candidate
fingerprint, exact tensor equality, maximum absolute difference `0.0`, and mean
absolute difference `0.0`.

The baseline, Phase 4, and repeat stores each retain metadata plus motion,
global SMPL, and in-camera SMPL artifacts. All motion tensors are `[300, 151]`
and finite; all SMPL outputs are finite with body pose `[300, 63]`, global
orientation `[300, 3]`, translation `[300, 3]`, and betas `[300, 10]`.

## Regression

- Local: `44 passed, 24 subtests passed in 9.15s`
- RunPod: `44 passed, 24 subtests passed in 7.97s`
- Covered: Phase 1-3 regressions, Phase 4 technical validation, finite
  `pred_x`, `last_dim == 151`, SMPL validity, CandidateStore traceability, and
  true same-seed inference reproducibility

## Local Outputs

- Baseline video:
  `outputs/render_comparison_fixed/baseline_compound_fixed_camera.mp4`
- Phase 4 video:
  `outputs/render_comparison_fixed/phase4_compound_fixed_camera.mp4`
- Side-by-side video:
  `outputs/render_comparison_fixed/side_by_side_fixed_camera.mp4`
- Metrics: `outputs/render_comparison_fixed/comparison_metrics.json`
- Baseline contact sheet:
  `outputs/render_comparison_fixed/baseline_fixed_contact_sheet.png`
- Phase 4 contact sheet:
  `outputs/render_comparison_fixed/phase4_fixed_contact_sheet.png`
- Final side-by-side preview:
  `outputs/render_comparison_fixed/side_by_side_final_preview.png`
- Candidate traces:
  `artifacts/phase4_controlled_comparison/run_20260926T231557Z/`

Video SHA-256 values:

- Baseline:
  `eb71bdebf7148faeae12e7dd46d398e98fe091002376c593d62db858792760d1`
- Phase 4:
  `31b70236cb62aabc8a5cebed0286bc470a790577b2d7a366be4058762840c731`
- Side by side:
  `67fd9ccd3edb0d5bf2d0975e862b7b9c7262a80199fdb579fb00f0fc92e47bbd`

## Remaining Limitation

This is a controlled qualitative A/B check for one fixed seed and prompt, not a
semantic-adherence benchmark across prompts or seeds. No Verifier, Repair,
Keyframe, or Phase 5 logic was added.
