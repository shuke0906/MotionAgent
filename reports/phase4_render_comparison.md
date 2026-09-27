# Phase 0 and Phase 4 Render Comparison

Date: 2026-09-27

## Scope

This is a qualitative render-and-compare check before Phase 5. Phase 5 was not
started. Existing real motion artifacts were reused; no motion was regenerated.

## Selected Samples

### Phase 0 baseline

- Prompt: `a person walks forward`
- Seed: 7
- Sample: validated run 1 from Phase 0
- Source: `artifacts/phase0_baseline/walk_seed7_run1.pt`
- Rendered tensor: `body_params_global`
- Frames: 300 at 30 fps (10 seconds)

### Phase 4 pipeline

- Prompt: `walk forward, wave the right hand, then sit down`
- Candidate: `cand_cdac10774732cf8a`
- Candidate seed: 971477687
- Source:
  `artifacts/phase4_validation/cand_cdac10774732cf8a/smpl_global.pt`
- Pipeline: Motion Compiler -> GEMTextCondition -> Frozen GEM -> MotionCandidate
- Frames: 180 at 30 fps (6 seconds)

The Phase 4 compiler windows were preserved as walk `0.000000-0.427778`, wave
`0.427778-0.716667`, and sit `0.716667-1.000000`.

## Rendering

Both samples rendered successfully on the RunPod A100 using GENMO's standard
global-motion path:

- `make_smplx("supermotion")`
- `normalize_global_verts`
- `render_global_frames`
- `save_video`

The two videos use the same body model, camera framing, ground normalization,
mesh material, 640x360 resolution, and 30 fps encoding. The Phase 4
CandidateStore did not persist `K_fullimg`, so the global renderer was selected
for both samples instead of reconstructing an in-camera view.

| Output | Frames | Duration | Resolution |
| --- | ---: | ---: | ---: |
| `phase0_render.mp4` | 300 | 10.0 s | 640x360 |
| `phase4_render.mp4` | 180 | 6.0 s | 640x360 |
| `comparison_side_by_side.mp4` | 300 | 10.0 s | 1280x360 |

In the side-by-side video, Phase 0 is on the left and Phase 4 is on the right.
The final Phase 4 frame is held from 6 to 10 seconds so both complete source
motions remain visible in one 10-second clip.

## Qualitative Comparison

The Phase 0 sample shows continuous forward locomotion across the ground plane.
Its behavior is coherent but intentionally single-stage.

The Phase 4 sample shows clearer temporal staging: forward stepping transitions
to a visibly raised right hand and wave-like gesture, followed by a return to an
upright neutral pose. This makes the walk-to-wave structure visually clearer
than the Phase 0 baseline's single action.

The final requested sit-down is not visibly completed. The last third settles
back to standing instead of reaching a seated posture. The selected Phase 4
candidate therefore demonstrates partial multi-stage adherence rather than a
clean realization of all three compiler segments.

## Local Outputs

- Phase 0 video: `outputs/render_comparison/phase0_render.mp4`
- Phase 4 video: `outputs/render_comparison/phase4_render.mp4`
- Side-by-side video:
  `outputs/render_comparison/comparison_side_by_side.mp4`
- Manifest: `outputs/render_comparison/render_manifest.json`
- Contact sheets: `outputs/render_comparison/phase0_contact_sheet.png`,
  `outputs/render_comparison/phase4_contact_sheet.png`, and
  `outputs/render_comparison/phase4_contact_sheet_detailed.png`
- Side-by-side preview: `outputs/render_comparison/comparison_preview.png`

## Caveats

- This is a qualitative check, not a scientific benchmark.
- The samples have different durations: 10 seconds for Phase 0 and 6 seconds
  for Phase 4.
- RunPod required the missing Ubuntu `libegl1` runtime for Open3D headless
  rendering.
- No Phase 5 code or architecture was introduced.
