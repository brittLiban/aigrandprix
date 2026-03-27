"""Unit tests for aigrandprix.config — YAML loading and deep merge."""
from pathlib import Path

import pytest

from aigrandprix.config import (
    Config, load_config, default_config, _deep_merge,
)

BASE_YAML = Path(__file__).parents[2] / "configs" / "base.yaml"
R1_YAML = Path(__file__).parents[2] / "configs" / "r1_qualifier.yaml"
R2_YAML = Path(__file__).parents[2] / "configs" / "r2_qualifier.yaml"


class TestDefaultConfig:
    def test_returns_config_instance(self):
        cfg = default_config()
        assert isinstance(cfg, Config)

    def test_adapter_type_default(self):
        cfg = default_config()
        assert cfg.adapter.type == "mock"

    def test_sim_defaults(self):
        cfg = default_config()
        assert cfg.sim.fps == 60
        assert cfg.sim.total_gates == 10

    def test_vision_hsv_defaults(self):
        cfg = default_config()
        assert cfg.vision.hsv_lower == [0, 0, 200]
        assert cfg.vision.hsv_upper == [180, 30, 255]

    def test_controller_profiles_present(self):
        cfg = default_config()
        profiles = cfg.controller.profiles
        for state in ("SEARCH", "TRACK", "APPROACH", "COMMIT", "RECOVER"):
            assert state in profiles, f"Missing profile: {state}"
            assert "throttle" in profiles[state]

    def test_config_hash_is_stable(self):
        cfg = default_config()
        h1 = cfg.config_hash()
        h2 = cfg.config_hash()
        assert h1 == h2
        assert len(h1) == 12

    def test_config_hash_differs_on_change(self):
        cfg1 = default_config()
        cfg2 = default_config()
        cfg2.sim.fps = 120
        assert cfg1.config_hash() != cfg2.config_hash()


class TestLoadConfig:
    def test_load_base_yaml(self):
        cfg = load_config(BASE_YAML)
        assert cfg.adapter.type == "mock"
        assert cfg.sim.fps == 60
        assert cfg.vision.resize_h == 240

    def test_load_base_with_r1_override(self):
        cfg = load_config(BASE_YAML, R1_YAML)
        # R1 overrides confidence_ema_alpha to 0.4
        assert cfg.vision.confidence_ema_alpha == pytest.approx(0.4)
        # Base values not overridden are preserved
        assert cfg.sim.fps == 60

    def test_load_base_with_r2_override(self):
        cfg = load_config(BASE_YAML, R2_YAML)
        assert cfg.vision.confidence_ema_alpha == pytest.approx(0.25)

    def test_deep_merge_does_not_lose_nested_keys(self):
        cfg = load_config(BASE_YAML, R1_YAML)
        # R1 only overrides a few vision keys; others should still be present
        assert cfg.vision.resize_h == 240  # unchanged from base
        assert cfg.vision.resize_w == 320  # unchanged from base


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override_preserves_sibling(self):
        base = {"vision": {"hsv_lower": [0, 0, 200], "resize_h": 240}}
        override = {"vision": {"hsv_lower": [0, 0, 180]}}
        result = _deep_merge(base, override)
        assert result["vision"]["hsv_lower"] == [0, 0, 180]
        assert result["vision"]["resize_h"] == 240

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}
