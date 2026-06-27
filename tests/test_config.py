"""Tests for WhaleConfig - configuration management."""
import pytest
from whale_video.config import WhaleConfig, MODEL_REGISTRY, DEFAULT_CONFIG


class TestWhaleConfigDefaults:
    """Test default configuration values."""

    def test_default_config_values(self):
        cfg = WhaleConfig()
        assert cfg.get("device") in ("cuda", "cpu", "mps")
        assert cfg.get("dtype") == "float16"
        assert cfg.get("output_dir") == "./outputs"
        assert cfg.get("models_dir") == "./models"
        assert cfg.get("default_engine") == "wan2.2-t2v"
        assert cfg.get("enable_model_caching") is True
        assert cfg.get("enable_memory_optimization") is True
        assert cfg.get("max_video_length_seconds") == 30
        assert cfg.get("default_fps") == 24
        assert cfg.get("default_resolution") == [720, 1280]
        assert cfg.get("submodules_dir") == "./submodules"

    def test_get_unknown_key_returns_default(self):
        cfg = WhaleConfig()
        assert cfg.get("nonexistent_key") is None
        assert cfg.get("nonexistent_key", "fallback") == "fallback"

    def test_get_empty_values(self):
        cfg = WhaleConfig()
        # Keys that exist but could be empty/falsy
        assert cfg.get("default_resolution") is not None
        assert len(cfg.get("default_resolution")) == 2


class TestModelRegistry:
    """Test the model registry contains all expected entries."""

    EXPECTED_MODELS = [
        "wan2.2-t2v",
        "wan2.2-i2v",
        "cogvideox-2",
        "hunyuanvideo",
        "mochi",
        "ltx-video",
        "skyreels-v1",
        "liveportrait",
        "wav2lip",
        "animatediff",
        "rife",
    ]

    def test_model_registry_has_all_entries(self):
        assert len(MODEL_REGISTRY) == 11
        for model_name in self.EXPECTED_MODELS:
            assert model_name in MODEL_REGISTRY, f"Missing model: {model_name}"

    @pytest.mark.parametrize("model_name,expected", [
        ("wan2.2-t2v", {"architecture": "MoE", "type": "text-to-video", "parameters": "14B", "min_vram_gb": 24}),
        ("wan2.2-i2v", {"architecture": "MoE", "type": "image-to-video", "parameters": "14B", "min_vram_gb": 24}),
        ("cogvideox-2", {"architecture": "DiT", "type": "text-to-video", "parameters": "13B", "min_vram_gb": 16}),
        ("hunyuanvideo", {"architecture": "DiT", "type": "text-to-video", "parameters": "8.3B", "min_vram_gb": 12}),
        ("mochi", {"architecture": "DiT", "type": "text-to-video", "parameters": "10B", "min_vram_gb": 20}),
        ("ltx-video", {"architecture": "DiT", "type": "text-to-video", "parameters": "22B", "min_vram_gb": 16}),
        ("skyreels-v1", {"architecture": "DiT", "type": "text-to-video", "parameters": "13B", "min_vram_gb": 16}),
        ("liveportrait", {"architecture": "CNN", "type": "portrait-animation", "parameters": "~1B", "min_vram_gb": 8}),
        ("wav2lip", {"architecture": "CNN+LSTM", "type": "lip-sync", "parameters": "~0.2B", "min_vram_gb": 4}),
        ("animatediff", {"architecture": "UNet+Motion", "type": "animation", "parameters": "~1.5B", "min_vram_gb": 8}),
        ("rife", {"architecture": "IFNet", "type": "frame-interpolation", "parameters": "~7M", "min_vram_gb": 4}),
    ])
    def test_get_model_config(self, model_name, expected):
        """Verify get_model_config returns correct details for each model."""
        cfg = WhaleConfig()
        config = cfg.get_model_config(model_name)
        assert config is not None, f"get_model_config returned None for {model_name}"
        for key, value in expected.items():
            assert config[key] == value, f"{model_name}.{key}: expected {value}, got {config[key]}"

    def test_get_model_config_unknown(self):
        cfg = WhaleConfig()
        assert cfg.get_model_config("unknown-model-123") is None

    def test_list_available_engines_contains_all(self):
        cfg = WhaleConfig()
        engines = cfg.list_available_engines()
        assert "wan2.2-t2v" in engines
        assert "wan2.2-i2v" in engines
        assert "cogvideox-2" in engines
        assert "hunyuanvideo" in engines
        assert "mochi" in engines
        assert "ltx-video" in engines
        assert "skyreels-v1" in engines
        assert "liveportrait" in engines
        assert "wav2lip" in engines
        assert "animatediff" in engines
        assert "rife" in engines
        # Verify the values are human-readable names
        assert engines["wan2.2-t2v"] == "Wan2.2-T2V-14B"
        assert engines["rife"] == "RIFE"

    def test_list_available_engines_excludes_disabled(self):
        cfg = WhaleConfig()
        # Temporarily disable one model
        cfg.model_registry["rife"]["enabled"] = False
        engines = cfg.list_available_engines()
        assert "rife" not in engines
        assert "wan2.2-t2v" in engines  # Still has all the others
        # Reset
        cfg.model_registry["rife"]["enabled"] = True

    def test_list_available_engines_returns_dict(self):
        cfg = WhaleConfig()
        engines = cfg.list_available_engines()
        assert isinstance(engines, dict)

    def test_resolve_submodule_path(self):
        cfg = WhaleConfig()
        path = cfg.resolve_submodule_path("wan2.2-t2v")
        assert path is not None
        assert path.endswith("Wan2.2")
        assert "submodules" in path

    def test_resolve_submodule_path_unknown(self):
        cfg = WhaleConfig()
        assert cfg.resolve_submodule_path("nonexistent-model") is None

    def test_wav2lip_repo_id_is_none(self):
        cfg = WhaleConfig()
        wav2lip = cfg.get_model_config("wav2lip")
        assert wav2lip is not None
        assert wav2lip["repo_id"] is None

    def test_rife_repo_id_is_none(self):
        cfg = WhaleConfig()
        rife = cfg.get_model_config("rife")
        assert rife is not None
        assert rife["repo_id"] is None
