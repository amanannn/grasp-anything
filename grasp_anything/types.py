"""Core data types and error classes for GraspAnything."""

from dataclasses import dataclass

import numpy as np


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """A detection result from Grounding DINO.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        label: Model-inferred phrase matching the text prompt.
        confidence: Detection confidence score [0, 1].
    """

    bbox: tuple[float, float, float, float]
    label: str
    confidence: float

    @property
    def center_2d(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def box_size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)


@dataclass(frozen=True)
class Mask:
    """A segmentation mask from SAM.

    Args:
        data: H×W boolean array where True = object pixels.
        center_2d: (x, y) centroid in pixel coordinates.
        area: Number of True pixels in the mask.
    """

    data: np.ndarray
    center_2d: tuple[float, float]
    area: int


@dataclass(frozen=True)
class GraspPose:
    """A 6-DOF grasp pose in world coordinates.

    Args:
        position: (x, y, z) in world frame (meters).
        rotation: (roll, pitch, yaw) Euler angles. Phase 1: always (0, 0, 0).
        approach_vector: (x, y, z) unit vector of grasp approach direction.
        gripper_open: Gripper aperture [0=closed, 1=fully open].
    """

    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    approach_vector: tuple[float, float, float]
    gripper_open: float


@dataclass(frozen=True)
class GraspResult:
    """Outcome of a grasp execution.

    Args:
        success: The full action sequence completed without errors.
        object_lifted: The object was lifted off the table surface.
        frames: Keyframe images for GIF playback.
    """

    success: bool
    object_lifted: bool
    frames: list[np.ndarray]


# ── Error types ──────────────────────────────────────────────────────────────


class StageError(Exception):
    """Recoverable error — user can correct and retry."""


class StageFatalError(Exception):
    """Non-recoverable error — environment or model issue, needs fixing."""


class GraspFailedError(StageError):
    """Grasp was attempted but the object slipped or was not lifted."""
