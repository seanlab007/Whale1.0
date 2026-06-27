"""Tests for VideoEngine base class."""
import pytest
from pathlib import Path
from whale_video.engine.base import VideoEngine


class SimpleTestEngine(VideoEngine):
    """Concrete engine for testing the abstract base class."""

    @property
    def model_name(self) -> str:
        return "test-engine"

    @property
    def engine_type(self) -> str:
        return "text-to-video"

    def load(self, **kwargs) -> None:
        self._loaded = True

    def generate(self, **kwargs) -> Path:
        return Path("/tmp/test_output.mp4")


class TestEngineBaseBasics:
    """Test basic engine properties."""

    def test_engine_cannot_be_instantiated_directly(self):
        """VideoEngine is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            VideoEngine()

    def test_engine_init_with_config(self):
        """Engine stores config dict."""
        engine = SimpleTestEngine({"device": "cpu", "dtype": "float32"})
        assert engine.config["device"] == "cpu"
        assert engine.config["dtype"] == "float32"

    def test_engine_init_without_config(self):
        """Engine uses empty dict when no config provided."""
        engine = SimpleTestEngine()
        assert engine.config == {}

    def test_engine_properties(self):
        """Engine returns correct model_name and engine_type."""
        engine = SimpleTestEngine()
        assert engine.model_name == "test-engine"
        assert engine.engine_type == "text-to-video"

    def test_engine_default_state(self):
        """Engine starts unloaded."""
        engine = SimpleTestEngine()
        assert engine._model is None
        assert engine.is_loaded is False

    def test_engine_load_sets_loaded_flag(self):
        """Engine.load() sets _loaded to True."""
        engine = SimpleTestEngine()
        assert engine.is_loaded is False
        engine.load()
        assert engine.is_loaded is True

    def test_engine_generate_returns_path(self):
        """Engine.generate() returns a Path."""
        engine = SimpleTestEngine()
        result = engine.generate()
        assert isinstance(result, Path)


class TestEngineUnload:
    """Test unload method without torch."""

    def test_unload_does_not_crash_without_model(self):
        """unload() on an engine with no model loaded."""
        engine = SimpleTestEngine()
        # Should not raise any exception
        engine.unload()
        assert engine._model is None
        assert engine.is_loaded is False

    def test_unload_after_load(self):
        """unload() after load() clears model and resets loaded flag."""
        engine = SimpleTestEngine()
        engine.load()
        assert engine.is_loaded is True
        engine.unload()
        assert engine._model is None
        assert engine.is_loaded is False

    def test_unload_called_twice(self):
        """Calling unload() twice is safe."""
        engine = SimpleTestEngine()
        engine.unload()
        engine.unload()  # Second call should not crash


class TestEngineToDict:
    """Test to_dict serialization."""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a proper dict."""
        engine = SimpleTestEngine()
        d = engine.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_keys(self):
        """to_dict() contains expected keys."""
        engine = SimpleTestEngine()
        d = engine.to_dict()
        assert d["model_name"] == "test-engine"
        assert d["engine_type"] == "text-to-video"
        assert d["loaded"] is False

    def test_to_dict_after_load(self):
        """to_dict() reflects loaded state after load()."""
        engine = SimpleTestEngine()
        engine.load()
        d = engine.to_dict()
        assert d["loaded"] is True

    def test_to_dict_after_unload(self):
        """to_dict() reflects unloaded state after unload()."""
        engine = SimpleTestEngine()
        engine.load()
        engine.unload()
        d = engine.to_dict()
        assert d["loaded"] is False

    def test_to_dict_with_custom_config(self):
        """to_dict() works with custom config."""
        engine = SimpleTestEngine({"device": "cuda", "variant": "t2v"})
        d = engine.to_dict()
        assert d["model_name"] == "test-engine"
        # config is stored but not serialized in to_dict
        assert engine.config["device"] == "cuda"
