# GraspAnything

Open-vocabulary robot grasping: tell it what to pick up in plain English. Grounding DINO → SAM → MuJoCo + robosuite.

## Environment

```bash
conda activate grasp          # Python 3.10
# Key deps: mujoco>=3.3.0,<3.10 (tested: 3.5.0), robosuite==1.5.2
MUJOCO_GL=egl                  # required for offscreen rendering
```

**NOTE:** system Python 3.12 breaks robosuite EGL rendering. Must use conda Python 3.10. MuJoCo 3.10.0+ breaks `mj_fullM` API — pin to <3.10.

## Architecture

**Pipeline + Stage pattern.** Each stage has `run(ctx) -> ctx` with typed inputs/outputs via `PipelineContext`.

```
app.py (Gradio)
  → GraspPipeline
    → GroundingStage    (text → detections)
    → SegmentationStage (detections → masks)
    → ProjectionStage   (masks + depth → 3D poses)
    → ExecutionStage    (poses → robot grasp)
```

**Key principles:**
- Each stage independently testable — mock context, no GPU needed for unit tests
- `types.py` defines `Detection`, `Mask`, `GraspPose`, `GraspResult` dataclasses
- Errors: `StageError` (recoverable), `StageFatalError` (fatal), `GraspFailedError` (expected)
- Model loading: lazy, cached to `~/.cache/grasp-anything/models/`
- `GraspPipeline.run_stream()` yields intermediate results for Gradio step-by-step UI

## Phase 1 Scope

- 5 YCB objects, single-object grasping, fixed top-down grasp
- Step-by-step Gradio: 3 panels (detection, segmentation, result)
- Top-1 detection only, no orientation estimation
- No LLM agent — this is Phase 4+

## User Priorities (ranked)

1. Code quality — clean architecture, testable, easy to contribute
2. Demo experience — polished Gradio
3. Grasp success rate — good enough, not benchmark-chasing
4. Docs & distribution — README, Colab, HuggingFace Spaces

## Development

- **TDD:** test first, run to see it fail, implement, run to pass, commit
- **Tests:** `tests/` mirrors `grasp_anything/` structure. Unit tests use mock ctx — no GPU, no MuJoCo needed.
- **Lint/Type:** ruff + pyright
- **Commit style:** conventional commits (`feat:`, `fix:`, `test:`, `refactor:`)

## Future (Phase 4+)

- LLM Agent for complex instructions ("pick up the leftmost cup")
- Fine-tuned VLA model (research paper potential)
- Multi-object sequential grasping

## Docs

- `docs/superpowers/specs/2026-07-04-*.md` — original vision/positioning
- `docs/superpowers/specs/2026-07-13-*.md` — technical design spec
