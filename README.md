# MotionAgent

MotionAgent is an agentic control layer for structured human-motion generation built around a frozen NVIDIA GEM/GENMO generator. It compiles a user motion request into explicit timeline-aware conditioning, sends that condition to Frozen GEM, and stores reproducible motion candidates without finetuning GEM.

## Demo

![MotionAgent Demo](docs/assets/demo/motionagent_demo.gif)

Left: direct Frozen GEM baseline. Right: MotionAgent structured multi-stage conditioning.

This is a qualitative side-by-side demo for one controlled prompt and seed, not a formal superiority benchmark.

## Core Architecture

```text
User
  |
  v
LangGraph Planner
  |
  v
Motion Compiler
  |
  v
MotionSpecification + GEMTextCondition
  |
  v
Frozen GEM
  |
  v
MotionCandidate
  |
  v
CandidateStore
```

Planned later modules are kept separate from the implemented Phase 0-4 path:

```text
Retrieval
Constraint / Keyframe
Candidate Tournament
Multi-Verifier
Diagnosis / Repair
```

Phase 5 has not started in this repository. Cross-segment continuity appears in planning documentation, but it should be treated as design intent unless a corresponding code patch is present in the implementation.

## Implementation Status

| Phase | Module | Status |
|---|---|---|
| 0 | Frozen GEM baseline | Implemented |
| 1 | State / persistence | Implemented |
| 2 | LangGraph planner harness | Implemented |
| 3 | Motion Compiler | Implemented |
| 4 | Compiler to Frozen GEM generation | Implemented |
| 5+ | Retrieval / constraints / evaluation / repair | Planned |

## Technology Stack

- Python: core implementation and validation scripts.
- PyTorch: tensor data structures, GEM payload assembly, and model runtime integration.
- LangGraph: stateful planner orchestration, graph routing, checkpoint, and resume.
- Pydantic: typed schemas for state, compiler outputs, generation requests, and candidates.
- SQLite / LangGraph SqliteSaver: local checkpoint persistence for graph runs.
- NVIDIA GEM / GENMO: frozen base motion generator used by MotionAgent through adapter code.
- SMPL / SMPL-X runtime: body representation and rendering/runtime dependency for real GEM validation.
- CUDA: required for real GEM generation.
- RunPod: used for the completed GPU validation runs; no RunPod connection details are stored here.

## GEM / GENMO Integration

This repository does not vendor the full upstream GENMO project. It keeps MotionAgent-specific integration files as a small patch set:

- `motion_agent/generation/gem_adapter.py`
- `integrations/GENMO/gem/motionagent.py`
- `integrations/GENMO/scripts/demo/demo_motionagent_text.py`

Use the official upstream [NVIDIA GENMO repository](https://github.com/NVlabs/GENMO) for the complete GEM source, preserve its license notices, and copy the files under `integrations/GENMO/` into the matching upstream paths when reproducing the MotionAgent pure-text adapter. GEM checkpoints and SMPL-X body-model assets are not distributed in this repository.

## Setup

Create an environment and install the MotionAgent test/runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For real GEM generation, install the full upstream GENMO project under `vendor/GENMO` or otherwise make it importable, then apply/keep the MotionAgent adapter files listed above. Obtain `gem_smpl.ckpt` from the official GEM/GENMO distribution path, and obtain SMPL-X assets such as `SMPLX_NEUTRAL.npz` from the official restricted source. Do not commit these assets.

Expected local asset locations may follow the upstream GENMO conventions:

```text
vendor/GENMO/inputs/pretrained/gem_smpl.ckpt
vendor/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

CPU-only development can run the state, planner, compiler, schema, and adapter contract tests. Real generation and rendering require a CUDA GPU plus the GEM checkpoint and SMPL-X assets.

## Project Structure

```text
motion_agent/
├── agent/        # planner policy, guards, budgets, action helpers
├── app/          # orchestrator and graph runner entry points
├── common/       # IDs, fingerprints, shared errors
├── compiler/     # natural-language motion parsing and GEM text conditioning
├── generation/   # GenerationRequest, GEM adapter, candidate persistence
├── graph/        # LangGraph topology, state, routing, nodes, checkpointing
└── state/        # canonical state, reducer, event log, artifact store

tests/            # unit, integration, and contract tests
scripts/          # phase validation and rendering scripts
doc/              # implementation guide and module design notes
docs/assets/demo/ # lightweight README demo GIF
integrations/     # MotionAgent patch files for upstream GENMO
```

## Verified Results

Use only the validated facts below as current project claims:

- Real Frozen GEM CUDA generation: PASS.
- Compiler to GEM pipeline: PASS.
- Multi-text conditioning: PASS.
- Same-seed reproducibility: PASS.
- CandidateStore persistence and traceability: PASS.
- Phase 4 controlled comparison prompt: `walk forward, wave the right hand, then sit down`.
- Current local regression result: `50 passed, 24 subtests passed`.

No scientific SOTA claim is made here.

## Research Acknowledgements

Direct dependency / base model:

- [NVIDIA GEM / GENMO](https://github.com/NVlabs/GENMO): frozen human-motion generator used by MotionAgent.

Design inspirations and planned module references:

- [LAMP](https://cyberiada.github.io/LAMP/): inspired structured motion program / DSL design.
- [RAPO](https://arxiv.org/abs/2504.11739): inspired prompt and caption refinement ideas.
- [TMR](https://arxiv.org/abs/2305.00976): planned text-motion retrieval and semantic evidence.
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D): planned retrieval / motion-language corpus reference.
- [Retrieval-Guided DNO](https://hanchaoliu.github.io/RetrievalGuidedDNO/): inspired future training-free guided-generation ideas.
- [ReAlign](https://wengwanjiang.github.io/ReAlign-page/): inspired reward-guided generation concepts.
- [VISTA](https://www.emergentmind.com/papers/2510.15831): planned pairwise candidate tournament structure.
- [AToM](https://atom-motion.github.io/): planned event / temporal evaluation structure.
- [MotionCritic](https://motioncritic.github.io/): planned motion naturalness evidence.
- [GENMAC](https://arxiv.org/abs/2412.04440): inspired diagnosis and repair structure.
- [NEWTON](https://newton026.github.io/newton/): inspired planner/tool orchestration and verifier feedback loops.

These references are design inspirations unless explicitly represented by implemented code in this repository.

## Tests

Run the CPU-capable regression suite:

```bash
python -m pytest tests -v
```

No expensive GPU rerun is required for ordinary publishing checks.

## Third-Party Notice

Third-party components retain their own licenses and usage restrictions. This repository does not distribute GEM weights, SMPL-X assets, body-model files, or generated tensor bundles. Users must obtain restricted assets from their official sources and comply with the corresponding licenses.
