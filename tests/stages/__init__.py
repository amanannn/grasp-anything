"""Tests for grasp_anything.stages.base — GraspStage ABC."""

import numpy as np
import pytest
from grasp_anything.stages.base import GraspStage
from grasp_anything.pipeline import PipelineContext
from grasp_anything.types import Detection, Mask


class FakeStage(GraspStage):
    """Minimal stage implementation for testing the ABC."""

    @property
    def name(self) -> str:
        return "fake_stage"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        return ctx


class TestGraspStage:
    def test_name_property(self):
        stage = FakeStage()
        assert stage.name == "fake_stage"

    def test_run_returns_context(self):
        stage = FakeStage()
        ctx = PipelineContext(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640)),
            camera_intrinsics=np.eye(3),
            text_prompt="test",
        )
        result = stage.run(ctx)
        assert result is ctx

    def test_visualize_default_returns_none(self):
        stage = FakeStage()
        ctx = PipelineContext(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640)),
            camera_intrinsics=np.eye(3),
            text_prompt="test",
        )
        assert stage.visualize(ctx) is None

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            GraspStage()  # abstract — missing name and run
