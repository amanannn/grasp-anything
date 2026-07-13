"""Tests for grasp_anything.config."""

from pathlib import Path
from grasp_anything.config import Config, DEFAULT_CONFIG


class TestConfig:
    def test_model_dir_is_path(self):
        assert isinstance(DEFAULT_CONFIG.model_dir, Path)

    def test_model_dir_in_home_cache(self):
        assert ".cache" in str(DEFAULT_CONFIG.model_dir)
        assert "grasp-anything" in str(DEFAULT_CONFIG.model_dir)

    def test_confidence_threshold_is_float(self):
        assert isinstance(DEFAULT_CONFIG.confidence_threshold, float)
        assert 0.0 < DEFAULT_CONFIG.confidence_threshold < 1.0

    def test_scene_has_camera(self):
        assert DEFAULT_CONFIG.camera_name == "frontview"

    def test_scene_has_image_dimensions(self):
        assert DEFAULT_CONFIG.image_width == 640
        assert DEFAULT_CONFIG.image_height == 480

    def test_custom_config_override(self):
        cfg = Config(model_dir=Path("/tmp/models"), confidence_threshold=0.8)
        assert cfg.model_dir == Path("/tmp/models")
        assert cfg.confidence_threshold == 0.8
        # Un-overridden fields use defaults
        assert cfg.image_width == 640

    def test_grasp_params(self):
        assert DEFAULT_CONFIG.grasp_approach_distance > 0
        assert DEFAULT_CONFIG.grasp_lift_height > 0
