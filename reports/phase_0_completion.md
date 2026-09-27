# Phase 0: GEM Baseline Freeze and Real GPU Validation

Status: **PASS**

Validated on 2026-09-27 in `/workspace/MotionAgent` on RunPod. This report covers
Phase 0 only; no Phase 1 work or architectural redesign was performed.

## Environment

- GPU: NVIDIA A100-SXM4-80GB
- VRAM: 81,920 MiB
- NVIDIA driver: 580.126.16
- Python: 3.10.18 (`vendor/GENMO/.venv`)
- PyTorch: 2.6.0+cu124
- PyTorch CUDA runtime: 12.4
- `torch.cuda.is_available()`: `True`
- Licensed SMPL-X model: present at
  `vendor/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`

## Checkpoint

- Source: `nvidia/GEM-X`, downloaded with `gem.utils.hf_utils.download_checkpoint`
- Path: `/workspace/MotionAgent/vendor/GENMO/inputs/pretrained/gem_smpl.ckpt`
- Size: 5,518,923,941 bytes (approximately 5.14 GiB)
- SHA256: `1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`
- HMR2, ViTPose, ONNX, and YOLOX assets were not downloaded or used.

## Tests

- Adapter unit tests: PASS, 3 tests (`python -m unittest tests.unit.test_gem_motionagent_adapter`)
- Syntax validation: PASS for `gem/motionagent.py`, `gem/gem.py`, and
  `scripts/demo/demo_motionagent_text.py`
- `import gem`: PASS
- `import gem.gem`: PASS
- CUDA availability: PASS
- Exact requested CLI real CUDA inference: PASS
- Output tensor and SMPL validation: PASS
- Same-seed reproducibility: PASS
- Different-seed diversity: PASS
- Additional semantic smoke tests: PASS

The first install attempt exposed Windows CRLF line endings in
`scripts/install_env.sh`; the script was normalized to LF and the official install
flow then completed. The exact unittest module path was initially shadowed by a
third-party `tests` package, so package markers were added under `tests/` and
`tests/unit/`. Deep import validation also exposed five support tensors referenced
by GENMO's GVHMR-derived body-model compatibility layer but omitted from the GENMO
tree. The exact tensors were restored in the RunPod runtime from the official
`zju3dv/GVHMR` repository.

## Generated Outputs

- Requested CLI output:
  `/workspace/MotionAgent/vendor/GENMO/outputs/motionagent_text/smpl_params.pt`
- Walk, seed 7, run 1:
  `/workspace/MotionAgent/vendor/GENMO/outputs/phase0_validation/walk_seed7_run1.pt`
- Walk, seed 7, run 2:
  `/workspace/MotionAgent/vendor/GENMO/outputs/phase0_validation/walk_seed7_run2.pt`
- Walk, seed 8:
  `/workspace/MotionAgent/vendor/GENMO/outputs/phase0_validation/walk_seed8.pt`
- Sit, seed 11:
  `/workspace/MotionAgent/vendor/GENMO/outputs/phase0_validation/sit_seed11.pt`
- Right arm, seed 11:
  `/workspace/MotionAgent/vendor/GENMO/outputs/phase0_validation/right_arm_seed11.pt`
- Preview rendering: deferred because Phase 0 requested `--no_render`; tensor and
  SMPL artifacts were validated directly.

## Numeric Validation

Main prompt: `a person walks forward`, seed 7, 300 frames.

- `pred_x` shape: `[1, 300, 151]`
- `pred_x` minimum: -2.240196704864502
- `pred_x` maximum: 2.0294535160064697
- `pred_x` mean: -0.011164313182234764
- `pred_x` standard deviation: 0.3349970281124115
- Finite: yes; no NaN or Inf
- Nondegenerate: yes
- SMPL frame count: 300, matching `pred_x`
- `body_params_global.body_pose`: `[300, 63]`
- `body_params_global.global_orient`: `[300, 3]`
- `body_params_global.transl`: `[300, 3]`
- `body_params_global.betas`: `[300, 10]`
- All global SMPL tensors are finite.

## Seed Validation

Same prompt and seed (`a person walks forward`, seed 7), two runs:

- Exact tensor equality: yes
- Mean absolute difference: 0.0
- Maximum absolute difference: 0.0

Seed 7 compared with seed 8 for the same prompt:

- Exact tensor equality: no
- Mean absolute difference: 0.12399036437273026
- Maximum absolute difference: 2.313023805618286

## Semantic Smoke Tests

`a person sits down` and `a person raises the right arm` both completed for 300
frames with finite, nondegenerate `[1, 300, 151]` outputs. Using seed 11 for both
prompts, their mean absolute difference was 0.2788711488246918 and maximum
absolute difference was 3.2747511863708496.

## Performance

- Requested CLI model inference: 5.487 seconds for 300 frames
- Instrumented end-to-end main validation call: 5.574 seconds for 300 frames
- Peak CUDA memory allocated: 6.727 GiB
- Baseline CUDA memory allocated after model load: 6.602 GiB
- Incremental peak during the measured call: 0.124 GiB

## Files Changed During Remote Validation

- `vendor/GENMO/scripts/install_env.sh` (CRLF-to-LF normalization)
- `tests/__init__.py`
- `tests/unit/__init__.py`
- Runtime-only: `vendor/GENMO/gem/utils/body_model/coco_aug_dict.pth`
- Runtime-only: `vendor/GENMO/gem/utils/body_model/smplx2smpl_sparse.pt`
- Runtime-only: `vendor/GENMO/gem/utils/body_model/smpl_coco17_J_regressor.pt`
- Runtime-only: `vendor/GENMO/gem/utils/body_model/smplx_verts437.pt`
- Runtime-only: `vendor/GENMO/gem/utils/body_model/smpl_neutral_J_regressor.pt`
- `reports/phase_0_completion.md`

Generated checkpoint, environment, caches, and output `.pt` files are runtime
artifacts and are not source changes.

## Phase 0 Gate

- [x] A100 GPU visible
- [x] Python 3.10 GEM environment works
- [x] CUDA PyTorch works
- [x] GEM imports successfully
- [x] `SMPLX_NEUTRAL.npz` is present
- [x] `gem_smpl.ckpt` downloaded from `nvidia/GEM-X`
- [x] Checkpoint SHA256 recorded
- [x] Existing adapter unit tests pass
- [x] Real CUDA pure-text GEM inference succeeds
- [x] `pred_x` exists and its final dimension is 151
- [x] No NaN or Inf
- [x] Global SMPL output exists and frame counts match
- [x] Seed 7 repeated run is exactly reproducible
- [x] Seed 8 differs from seed 7
- [x] Phase 0 completion report updated

**Phase 0 Gate: PASS**

Remaining blockers: none for Phase 0.

## Local Reconciliation

The validated MotionAgent source, adapter tests, package markers, and this report
were confirmed identical between the local Windows source of truth and RunPod.
No config changes or remote-only source fixes required copying back. The five
generated validation bundles remain on RunPod; their lightweight metrics are
preserved locally in `reports/phase_0_validation_summary.json`.

The GVHMR support tensors, GEM checkpoint, licensed SMPL-X file, virtual
environment, caches, and generated outputs are runtime-only and are not retained
as local source files. Use `vendor/GENMO/scripts/download_gvhmr_support.py` to
install the checksum-pinned GVHMR tensors in a GPU runtime when needed. See
`doc/PHASE0_GPU_SETUP.md` for the complete setup and source-control policy.
