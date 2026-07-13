# GraspAnything — Technical Design Spec

> Date: 2026-07-13
> Status: Approved
> Supersedes: `2026-07-04-open-vocabulary-grasping-design.md`

---

## 1. Overview

GraspAnything is an open-vocabulary robot grasping system. The user describes an object in natural English ("the red cup"), and the system finds it in a simulated environment, plans a grasp pose, and executes the grasp.

### 1.1 Scope (Phase 1)

- **5 YCB objects**: block, soda can, mug, banana, screwdriver
- **Single object per command** — top-1 detection
- **Fixed top-down grasp** — no orientation estimation yet
- **Step-by-step Gradio UI** — detection → mask → grasp result
- **No LLM agent** — complex instruction understanding is Phase 4+

### 1.2 User Priorities

| Priority | Dimension |
|----------|-----------|
| 1 | Code quality — clean architecture, testable, easy to fork/contribute |
| 2 | Demo experience — polished Gradio UI, smooth flow |
| 3 | Grasp success rate — good enough for demo, not benchmark-chasing |
| 4 | Docs & distribution — README, Colab, HuggingFace Spaces |

---

## 2. Architecture

### 2.1 Pattern: Pipeline + Stage

Each stage is an independent component with defined inputs and outputs. A `GraspPipeline` orchestrator composes them sequentially.

```
app.py (Gradio UI, ~50 lines)
    ↓
GraspPipeline (orchestrator)
    ↓
GroundingStage → SegmentationStage → ProjectionStage → ExecutionStage
```

### 2.2 Stage Interface

```python
class GraspStage(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext: ...

    def visualize(self, ctx: PipelineContext) -> Image | None:
        """Optional: return visualization for Gradio display."""
        return None
```

### 2.3 PipelineContext

All data flows through a shared context object. Each stage reads its inputs from and writes its outputs to the context.

```python
@dataclass
class PipelineContext:
    # Input (from simulation)
    image: np.ndarray          # RGB, H×W×3
    depth: np.ndarray          # Depth, H×W
    camera_intrinsics: np.ndarray  # 3×3 matrix
    text_prompt: str           # User input

    # Stage outputs
    detections: list[Detection] | None = None
    masks: list[Mask] | None = None
    grasp_poses: list[GraspPose] | None = None
    grasp_result: GraspResult | None = None
```

### 2.4 Pipeline Orchestrator

```python
class GraspPipeline:
    def __init__(self, stages: list[GraspStage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx

    def run_stream(self, ctx: PipelineContext):
        """Yield intermediate results for Gradio step-by-step display."""
        for stage in self.stages:
            ctx = stage.run(ctx)
            vis = stage.visualize(ctx)
            yield stage.name, vis, ctx
```

---

## 3. Data Types

All types are Python dataclasses defined in `grasp_anything/types.py`.

### 3.1 Detection

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str                                 # model-inferred phrase
    confidence: float

    @property
    def center_2d(self) -> tuple[float, float]: ...
    @property
    def box_size(self) -> tuple[float, float]: ...
```

### 3.2 Mask

```python
@dataclass
class Mask:
    data: np.ndarray       # H×W boolean
    center_2d: tuple[float, float]
    area: int              # pixel count
```

### 3.3 GraspPose

```python
@dataclass
class GraspPose:
    position: tuple[float, float, float]     # world coordinates
    rotation: tuple[float, float, float]     # Euler angles (0,0,0 in Phase 1)
    approach_vector: tuple[float, float, float]
    gripper_open: float                     # 0-1
```

### 3.4 GraspResult

```python
@dataclass
class GraspResult:
    success: bool
    object_lifted: bool
    frames: list[np.ndarray]   # keyframes for GIF
```

### 3.5 Phase 1 Constraints

- `rotation` is always `(0, 0, 0)` — top-down grasp
- Only top-1 detection used — highest confidence
- Field types defined to accommodate future PCA rotation and multi-object selection

### 3.6 Grasp Success Definition

- **`GraspResult.success`**: The full action sequence completed without errors (arm moved, gripper closed, arm lifted).
- **`GraspResult.object_lifted`**: The object is no longer in contact with the table surface after lifting. Checked by comparing object Z position to initial Z.
- **Success ≠ object lifted**: A grasp can execute cleanly but drop the object (`success=True, object_lifted=False`). Phase 3 target: >80% `object_lifted` rate.

---

## 4. Simulation Stack

### 4.1 Choice: MuJoCo + robosuite

MuJoCo is the physics engine (gravity, collision, contact dynamics). robosuite wraps it with robot models, gripper control, inverse kinematics, and camera utilities.

### 4.2 Ubuntu 24.04 Compatibility

**Problem**: Python 3.12 (default on 24.04) has ctypes changes that break robosuite's EGL offscreen rendering.

**Solution**: Use conda with Python 3.10:

```bash
conda create -n grasp python=3.10
conda activate grasp
pip install mujoco==3.1.1   # Pin — 3.1.5 is broken
pip install robosuite
sudo apt install libosmesa6-dev libgl1-mesa-dev libegl1-mesa-dev
export MUJOCO_GL=egl
```

### 4.3 Robot Environment

**Scene**: Custom tabletop scene with Franka Panda arm + 5 YCB objects. Not the default robosuite "Lift" task (which has a single cube). Built using robosuite's scene API or by extending an existing task template. Camera position: front-facing, angled down at the table.

```python
class RobotEnv:
    """Wraps robosuite for offscreen rendering."""

    def __init__(self):
        self.env = suite.make(
            "Lift",                        # Base task; customize scene
            has_renderer=False,            # no window
            has_offscreen_renderer=True,   # EGL offscreen
            render_camera="frontview",
            use_camera_obs=True,
        )

    def get_observation(self) -> dict:
        """Returns {"rgb": np.ndarray, "depth": np.ndarray, "intrinsics": np.ndarray}."""
        ...

    def execute_grasp(self, pose: GraspPose) -> GraspResult:
        """Move arm to pose, close gripper, lift, return result + keyframes."""
        ...
```

---

## 5. Rendering Pipeline (Gradio)

MuJoCo EGL offscreen → `get_observation()` → numpy arrays → `gr.Image` components.

### 5.1 Step-by-Step UI

Three `gr.Image` panels, updated sequentially:
1. **Detection** — bounding boxes drawn on RGB
2. **Segmentation** — mask overlay on RGB
3. **Grasp Result** — GIF of grasp execution

### 5.2 Key Technical Details

- Depth visualization: normalize to `(depth - min) / (max - min)` before display
- GIF generation: `imageio.mimsave` from `GraspResult.frames` keyframes
- MuJoCo rendering: requires `MUJOCO_GL=egl` environment variable

---

## 6. Model Management

### 6.1 Models

| Model | Variant | Size | Inference (CPU) |
|-------|---------|------|-----------------|
| Grounding DINO | Tiny | ~700MB | ~2s |
| SAM | ViT-H | ~2.4GB | ~3s |

### 6.2 Strategy

- **Lazy loading** — models loaded on first request, not at Gradio startup
- **Local cache** — `~/.cache/grasp-anything/models/` (XDG-compliant)
- **Download script** — `scripts/download_models.sh` for pre-download
- **EfficientSAM swap** — interface preserved; switch via one import change for HuggingFace Spaces deployment

```python
class ModelManager:
    MODEL_DIR = Path.home() / ".cache/grasp-anything/models"

    def get_grounding_dino(self) -> GroundingDINO:
        if "grounding_dino" not in self._models:
            self._ensure_downloaded("grounding_dino")
            self._models["grounding_dino"] = self._load_grounding_dino()
        return self._models["grounding_dino"]
```

---

## 7. Project Structure

```
grasp-anything/
├── README.md
├── pyproject.toml
├── LICENSE
├── .gitignore
├── .env.example                   # MUJOCO_GL=egl, etc.
├── app.py                         # Gradio entry point (~50 lines)
├── demo.ipynb                     # Colab notebook
│
├── grasp_anything/                # Core package
│   ├── __init__.py                # Public API exports
│   ├── pipeline.py                # GraspPipeline + PipelineContext
│   ├── types.py                   # Detection, Mask, GraspPose, GraspResult
│   ├── config.py                  # Paths, scene params, camera settings
│   │
│   ├── stages/                    # One file per stage
│   │   ├── __init__.py
│   │   ├── base.py                # GraspStage ABC
│   │   ├── grounding.py           # GroundingStage
│   │   ├── segmentation.py        # SegmentationStage
│   │   ├── projection.py          # ProjectionStage
│   │   └── execution.py           # ExecutionStage
│   │
│   ├── models/                    # Model wrappers (inference only)
│   │   ├── __init__.py
│   │   ├── manager.py             # ModelManager
│   │   ├── grounding_dino.py      # Grounding DINO wrapper
│   │   └── sam.py                 # SAM wrapper
│   │
│   └── robot/                     # robosuite simulation
│       ├── __init__.py
│       ├── env.py                 # RobotEnv
│       └── controller.py          # Gripper + motion commands
│
├── scripts/
│   ├── download_models.sh
│   └── check_install.py           # Env checker (Python version, deps, EGL)
│
├── tests/                         # Mirrors source structure
│   ├── __init__.py
│   ├── conftest.py                # Fixtures: mock ctx, fake images
│   ├── test_pipeline.py
│   ├── test_types.py
│   ├── stages/
│   │   ├── test_grounding.py
│   │   ├── test_segmentation.py
│   │   ├── test_projection.py
│   │   └── test_execution.py
│   └── robot/
│       ├── test_env.py
│       └── test_controller.py
│
├── docs/                          # MkDocs Material (Phase 4)
└── .github/
    └── workflows/                 # CI: lint + typecheck + tests
```

### 7.1 Key Differences from Original Design

| Original | New | Reason |
|----------|-----|--------|
| Single `perception.py` | `stages/grounding.py` + `stages/segmentation.py` | Independent testing per stage |
| Single `robot.py` | `robot/env.py` + `robot/controller.py` | Decouple env setup from control |
| No `types.py` | Central `types.py` | Single source of truth for data shapes |
| No `tests/` | Full `tests/` mirroring source | Code quality is #1 priority |
| `grasping.py` standalone | Merged into `projection.py` | 3D projection and grasp pose are one continuous step |

---

## 8. Error Handling

### 8.1 Exception Hierarchy

```python
class StageError(Exception):
    """Recoverable — user can correct and retry."""

class StageFatalError(Exception):
    """Non-recoverable — env/model issue, needs fixing."""

class GraspFailedError(StageError):
    """Grasp was attempted but failed — normal, counted in stats."""
```

### 8.2 Failure Scenarios

| Scenario | Stage | User sees |
|----------|-------|-----------|
| Model not downloaded | Grounding | "Model not found. Run: scripts/download_models.sh" |
| No object detected | Grounding | "Did not detect 'xxx' in the scene. Try a different description." |
| Low confidence | Grounding | Show all candidates, let user pick |
| Mask too small | Segmentation | "Segmentation result is poor — object may be too small or occluded." |
| Invalid depth | Projection | "Cannot compute 3D position — depth data is invalid." |
| Arm unreachable | Execution | "Grasp position is outside the robot's workspace." |
| Grasp dropped | Execution | "Object slipped during grasp." (not an error, counted in stats) |

---

## 9. Testing Strategy

### 9.1 Three Layers

| Layer | What | Requirements | Phase 1 target |
|-------|------|-------------|----------------|
| Unit tests | Each stage's `run()` with mock context | No GPU, no MuJoCo | ~15 tests |
| Integration tests | Full pipeline with real models + MuJoCo | EGL + models downloaded | ~3 tests (5 objects × 1 grasp each) |
| Visual verification | Each stage's `visualize()` output | Real scene | Manual |

### 9.2 Unit Test Example

```python
def test_projection_with_known_depth():
    """Depth=1.0 everywhere, mask at image center → pose at (0,0,1)."""
    stage = ProjectionStage()
    ctx = PipelineContext(
        image=mock_rgb(480, 640),
        depth=mock_depth(value=1.0),
        camera_intrinsics=mock_intrinsics(cx=320, cy=240, fx=500),
        masks=[mock_mask(center=(320, 240))],
    )
    ctx = stage.run(ctx)
    assert ctx.grasp_poses[0].position == (0.0, 0.0, 1.0)
```

### 9.3 CI Pipeline

- **Every PR**: ruff lint + pyright typecheck + unit tests (no GPU needed)
- **Push to main**: Full test suite (requires EGL/MuJoCo CI runner)
- **Before release**: Manual — 5 objects × 5 grasps each

---

## 10. Development Phases

### Phase 1: Minimum Viable (2-3 weeks, student schedule)

- [ ] MuJoCo + robosuite env with Franka Panda + 5 YCB objects on table
- [ ] Grounding DINO-T: text → detection boxes
- [ ] SAM ViT-H: box → mask
- [ ] Simple grasp: mask center → world coordinate → fixed top-down grasp
- [ ] Verify: grasp "red cube" successfully in simulation
- [ ] Pipeline + Stage architecture with unit tests

### Phase 2: Gradio UI (1-2 weeks)

- [ ] Step-by-step Gradio interface: text input + 3 image panels
- [ ] `pip install -e .` one-command install
- [ ] `python app.py` one-command launch
- [ ] `scripts/download_models.sh` pre-download script
- [ ] Offscreen rendering → Gradio Image flow working

### Phase 3: Quality (2-3 weeks)

- [ ] 10+ random object placements (YCB object set)
- [ ] PCA principal direction estimation for grasp rotation
- [ ] Collision detection: avoid knocking over other objects
- [ ] Multi-view: front view ↔ top-down toggle
- [ ] Target >80% grasp success rate

### Phase 4: Release (1-2 weeks)

- [ ] MkDocs documentation site
- [ ] README with GIF (success + failure comparison)
- [ ] Colab notebook
- [ ] HuggingFace Spaces deployment
- [ ] Publication: Twitter/X + Reddit + Zhihu

### Phase 4+ (Future Vision)

- [ ] LLM Agent for complex instruction understanding ("pick up the leftmost cup and place it on the right")
- [ ] Fine-tuned small VLA (Vision-Language-Action) model — research paper potential
- [ ] Multi-object sequential grasping

---

## 11. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Pipeline + Stage | Independent testing, Gradio step-by-step fits naturally, easy to extend |
| Simulation | MuJoCo + robosuite | Best contact dynamics, pip-installable, academic standard |
| Python version | 3.10 via conda | Avoids Ubuntu 24.04 Python 3.12 EGL bug |
| MuJoCo version | 3.1.1 (pinned) | 3.1.5 is broken; 3.1.1 is community-validated |
| Detection model | Grounding DINO-T (Tiny) | CPU-runnable, <3% accuracy loss vs Large |
| Segmentation model | SAM ViT-H → EfficientSAM | Phase 1 quality first; swap for Spaces deployment |
| Grasp strategy (Phase 1) | Fixed top-down | Simplest working strategy; rotation field reserved for Phase 3 PCA |
| Detection scope | Top-1 | Scene has 5 objects, user describes one at a time |
| Model loading | Lazy | Gradio starts instantly; first request triggers ~5-10s load |
| Interaction | Step-by-step | Each stage visualized; helps debugging and makes compelling demo |
| Error model | 3 exception types | Recoverable vs fatal vs expected grasp failure |
| Testing | 3-layer (unit/integration/visual) | Unit tests need no GPU; integration tests gate releases |

---

## 12. Technology Stack

| Layer | Technology | Version Constraint |
|-------|-----------|-------------------|
| Physics engine | MuJoCo | ==3.1.1 |
| Robot framework | robosuite | latest (from pip or source) |
| Open-vocab detection | Grounding DINO | Tiny variant |
| Segmentation | SAM / EfficientSAM | ViT-H (swap to EfficientSAM for Spaces) |
| Web UI | Gradio | latest |
| Package management | pip + conda | Python 3.10 |
| Linting | ruff | latest |
| Type checking | pyright | latest |
| Testing | pytest | latest |
| Documentation | MkDocs Material | Phase 4 |
| CI/CD | GitHub Actions | — |
