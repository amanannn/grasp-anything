"""Tests for grasp_anything.types — dataclasses and error classes."""

import numpy as np
import pytest
from grasp_anything.types import (
    Detection,
    Mask,
    GraspPose,
    GraspResult,
    StageError,
    StageFatalError,
    GraspFailedError,
)


class TestDetection:
    def test_center_2d(self):
        d = Detection(bbox=(10, 20, 110, 120), label="red cup", confidence=0.95)
        assert d.center_2d == (60.0, 70.0)

    def test_box_size(self):
        d = Detection(bbox=(0, 0, 100, 50), label="cube", confidence=0.8)
        assert d.box_size == (100.0, 50.0)

    def test_immutable(self):
        d = Detection(bbox=(0, 0, 10, 10), label="test", confidence=0.5)
        with pytest.raises(Exception):
            d.label = "changed"


class TestMask:
    def test_basic(self):
        data = np.zeros((100, 100), dtype=bool)
        data[40:60, 40:60] = True
        m = Mask(data=data, center_2d=(50.0, 50.0), area=400)
        assert m.area == 400
        assert m.center_2d == (50.0, 50.0)
        assert m.data.sum() == 400


class TestGraspPose:
    def test_defaults(self):
        p = GraspPose(
            position=(0.5, 0.0, 0.3),
            rotation=(0.0, 0.0, 0.0),
            approach_vector=(0.0, 0.0, -1.0),
            gripper_open=1.0,
        )
        assert p.position == (0.5, 0.0, 0.3)
        assert p.rotation == (0.0, 0.0, 0.0)

    def test_phase1_default_rotation(self):
        """Phase 1: rotation always zero for fixed top-down grasp."""
        p = GraspPose(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            approach_vector=(0.0, 0.0, -1.0),
            gripper_open=1.0,
        )
        assert p.rotation == (0.0, 0.0, 0.0)


class TestGraspResult:
    def test_successful(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        r = GraspResult(success=True, object_lifted=True, frames=frames)
        assert r.success is True
        assert r.object_lifted is True
        assert len(r.frames) == 5

    def test_grasp_attempted_but_dropped(self):
        """Executed cleanly but object slipped — success=True, object_lifted=False."""
        r = GraspResult(success=True, object_lifted=False, frames=[])
        assert r.success is True
        assert r.object_lifted is False


class TestErrorHierarchy:
    def test_stage_error_is_exception(self):
        with pytest.raises(StageError):
            raise StageError("no object detected")

    def test_fatal_error_is_exception(self):
        with pytest.raises(StageFatalError):
            raise StageFatalError("model not found")

    def test_grasp_failed_is_stage_error(self):
        """GraspFailedError is a StageError (recoverable)."""
        err = GraspFailedError("object slipped")
        assert isinstance(err, StageError)
        assert isinstance(err, Exception)

    def test_stage_error_message(self):
        with pytest.raises(StageError, match="low confidence"):
            raise StageError("detection low confidence")
