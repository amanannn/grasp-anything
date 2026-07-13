"""Tests for grasp_anything.pipeline — PipelineContext + GraspPipeline."""

import numpy as np
import pytest
from grasp_anything.pipeline import PipelineContext, GraspPipeline
from grasp_anything.stages.base import GraspStage
from grasp_anything.types import Detection, Mask, GraspPose, GraspResult


class CountingStage(GraspStage):
    """Records that it ran by incrementing a counter in ctx."""
    def __init__(self, name="counter"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, ctx: PipelineContext) -> PipelineContext:
        run_count = getattr(ctx, "_run_count", 0) + 1
        object.__setattr__(ctx, "_run_count", run_count)
        return ctx


class AppendStage(GraspStage):
    """Appends its name to a list stored in ctx."""
    def __init__(self, name="appender"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, ctx: PipelineContext) -> PipelineContext:
        history = list(getattr(ctx, "_history", []))
        history.append(self._name)
        object.__setattr__(ctx, "_history", history)
        return ctx

    def visualize(self, ctx: PipelineContext):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class TestPipelineContext:
    def test_empty_context(self):
        ctx = PipelineContext(
            image="img",
            depth="dep",
            camera_intrinsics="K",
            text_prompt="pick",
        )
        assert ctx.image == "img"
        assert ctx.text_prompt == "pick"

    def test_fields_default_to_none(self):
        ctx = PipelineContext(
            image=np.zeros((10, 10, 3)),
            depth=np.zeros((10, 10)),
            camera_intrinsics=np.eye(3),
            text_prompt="",
        )
        assert ctx.detections is None
        assert ctx.masks is None
        assert ctx.grasp_poses is None
        assert ctx.grasp_result is None


class TestGraspPipeline:
    def test_run_empty_stages(self):
        pipeline = GraspPipeline(stages=[])
        ctx = PipelineContext("img", "dep", "K", "prompt")
        result = pipeline.run(ctx)
        assert result is ctx

    def test_run_all_stages(self):
        pipeline = GraspPipeline([
            CountingStage("a"),
            CountingStage("b"),
            CountingStage("c"),
        ])
        ctx = PipelineContext(
            image=np.zeros((10, 10, 3)),
            depth=np.zeros((10, 10)),
            camera_intrinsics=np.eye(3),
            text_prompt="test",
        )
        result = pipeline.run(ctx)
        assert getattr(result, "_run_count") == 3

    def test_run_stream_yields_intermediate_results(self):
        pipeline = GraspPipeline([
            AppendStage("detect"),
            AppendStage("segment"),
            AppendStage("execute"),
        ])
        ctx = PipelineContext(
            image=np.zeros((10, 10, 3)),
            depth=np.zeros((10, 10)),
            camera_intrinsics=np.eye(3),
            text_prompt="test",
        )
        results = list(pipeline.run_stream(ctx))
        assert len(results) == 3
        # Each yield: (stage_name, visualization, updated_ctx)
        assert results[0][0] == "detect"
        assert results[1][0] == "segment"
        assert results[2][0] == "execute"
        # Final ctx has all 3 stages recorded
        assert getattr(results[2][2], "_history") == ["detect", "segment", "execute"]

    def test_run_stream_visualization(self):
        pipeline = GraspPipeline([AppendStage("a")])
        ctx = PipelineContext(
            image=np.zeros((10, 10, 3)),
            depth=np.zeros((10, 10)),
            camera_intrinsics=np.eye(3),
            text_prompt="test",
        )
        results = list(pipeline.run_stream(ctx))
        name, vis, updated_ctx = results[0]
        assert name == "a"
        assert isinstance(vis, np.ndarray)
