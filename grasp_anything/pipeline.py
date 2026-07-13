"""Pipeline orchestrator and shared context for GraspAnything."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from grasp_anything.types import Detection, GraspPose, GraspResult, Mask


@dataclass
class PipelineContext:
    """Shared state passed between pipeline stages.

    Input fields (set before pipeline runs):
        image: RGB image from the simulation camera (H×W×3, uint8).
        depth: Depth map from the simulation camera (H×W, float32, meters).
        camera_intrinsics: 3×3 camera intrinsic matrix.
        text_prompt: User's natural language description of the target object.

    Output fields (populated by stages):
        detections: Detection boxes from Grounding DINO.
        masks: Segmentation masks from SAM.
        grasp_poses: 3D grasp poses from projection.
        grasp_result: Final outcome from execution.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    image: np.ndarray
    depth: np.ndarray
    camera_intrinsics: np.ndarray
    text_prompt: str

    # ── Stage outputs ────────────────────────────────────────────────────────
    detections: Optional[list[Detection]] = None
    masks: Optional[list[Mask]] = None
    grasp_poses: Optional[list[GraspPose]] = None
    grasp_result: Optional[GraspResult] = None


class GraspPipeline:
    """Orchestrates a sequence of GraspStage instances.

    Usage:
        pipeline = GraspPipeline([
            GroundingStage(model_manager),
            SegmentationStage(model_manager),
            ProjectionStage(),
            ExecutionStage(robot_env),
        ])
        ctx = PipelineContext(image=..., depth=..., intrinsics=..., text_prompt="red cup")
        result = pipeline.run(ctx)

        # Or for step-by-step Gradio display:
        for stage_name, vis, ctx in pipeline.run_stream(ctx):
            update_gradio_panel(stage_name, vis)
    """

    def __init__(self, stages: list["GraspStage"]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all stages sequentially. Returns final context."""
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx

    def run_stream(self, ctx: PipelineContext):
        """Execute stages one at a time, yielding after each for Gradio UI.

        Yields:
            tuple[str, np.ndarray | None, PipelineContext]:
                (stage_name, visualization_image, updated_context)
        """
        for stage in self.stages:
            ctx = stage.run(ctx)
            vis = stage.visualize(ctx)
            yield stage.name, vis, ctx
