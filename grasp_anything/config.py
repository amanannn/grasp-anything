"""Configuration for GraspAnything — paths, scene params, thresholds."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration with sensible defaults for Phase 1.

    All paths use pathlib.Path for cross-platform safety.
    """

    # Model paths
    model_dir: Path = Path.home() / ".cache" / "grasp-anything" / "models"

    # Scene
    robot_name: str = "Panda"
    camera_name: str = "frontview"
    image_width: int = 640
    image_height: int = 480

    # Grounding
    confidence_threshold: float = 0.3
    top_k_detections: int = 1  # Phase 1: only top-1

    # Segmentation
    min_mask_area: int = 100  # pixels — reject tiny noise masks

    # Grasp
    grasp_approach_distance: float = 0.05  # meters above object
    grasp_lift_height: float = 0.15  # meters to lift after grasp
    gripper_close_force: float = 1.0

    # Rendering
    offscreen_render: bool = True
    render_camera: str = "frontview"


DEFAULT_CONFIG = Config()
