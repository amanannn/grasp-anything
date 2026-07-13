"""Abstract base class for all pipeline stages."""

from abc import ABC, abstractmethod

import numpy as np

from grasp_anything.pipeline import PipelineContext


class GraspStage(ABC):
    """Abstract base for a processing stage in the GraspPipeline.

    Each stage reads from PipelineContext, processes, and writes back.
    Stages must be independently testable with mocked contexts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging and Gradio display."""
        ...

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage's processing — read from ctx, write back to ctx."""
        ...

    def visualize(self, ctx: PipelineContext) -> "np.ndarray | None":
        """Return a visualization image for Gradio display. Optional."""
        return None
