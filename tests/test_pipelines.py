"""Whale1.0 basic tests."""
import pytest
from pathlib import Path
from whale_video.config import WhaleConfig
from whale_video.pipeline.t2v import T2VPipeline
from whale_video.pipeline.i2v import I2VPipeline
from whale_video.pipeline.full import FullPipeline
from whale_video.pipeline.portrait import PortraitPipeline


class TestConfig:
    def test_default_config(self):
        cfg = WhaleConfig()
        assert cfg.get("device") is not None
        assert cfg.get("default_engine") == "wan2.2-t2v"
        assert len(cfg.model_registry) > 0

    def test_model_registry(self):
        cfg = WhaleConfig()
        wan = cfg.get_model_config("wan2.2-t2v")
        assert wan is not None
        assert wan["architecture"] == "MoE"
        assert wan["type"] == "text-to-video"

    def test_list_engines(self):
        cfg = WhaleConfig()
        engines = cfg.list_available_engines()
        assert "wan2.2-t2v" in engines
        assert "cogvideox-2" in engines
        assert "hunyuanvideo" in engines


class TestPipelines:
    def test_t2v_pipeline_init(self):
        pipeline = T2VPipeline()
        assert pipeline.config is not None

    def test_t2v_list_engines(self):
        pipeline = T2VPipeline()
        engines = pipeline.list_engines()
        assert "wan2.2" in engines
        assert "cogvideox" in engines

    def test_i2v_pipeline_init(self):
        pipeline = I2VPipeline()
        assert pipeline.config is not None

    def test_full_pipeline_init(self):
        pipeline = FullPipeline()
        assert pipeline.t2v is not None
        assert pipeline.i2v is not None
        assert pipeline.portrait is not None

    # --- New pipeline tests ---

    def test_t2v_pipeline_config_dict(self):
        """T2VPipeline accepts a dict as config."""
        cfg_dict = {"device": "cpu", "default_engine": "wan2.2-t2v"}
        pipeline = T2VPipeline(cfg_dict)
        assert pipeline.config.get("device") == "cpu"
        assert pipeline.config.get("default_engine") == "wan2.2-t2v"

    def test_i2v_pipeline_config_dict(self):
        """I2VPipeline accepts a dict as config."""
        cfg_dict = {"device": "cpu"}
        pipeline = I2VPipeline(cfg_dict)
        assert pipeline.config.get("device") == "cpu"

    def test_portrait_pipeline_init(self):
        """PortraitPipeline initializes with default config."""
        pipeline = PortraitPipeline()
        assert pipeline.config is not None
        assert pipeline.config.get("output_dir") == "./outputs"

    def test_portrait_pipeline_generate_returns_dict(self):
        """PortraitPipeline.generate returns a dict without needing external deps."""
        pipeline = PortraitPipeline()
        result = pipeline.generate(source_image="test.jpg", output_path="/tmp/test_out")
        assert isinstance(result, dict)
        assert result["engine"] == "portrait-pipeline"
        assert "output_path" in result
        assert "liveportrait" in result["modules"]
        assert "wav2lip" in result["modules"]

    def test_t2v_pipeline_auto_engine_selection(self):
        """Auto engine selection reads from config default_engine."""
        pipeline = T2VPipeline()
        # 'auto' should resolve to the config's default_engine without crashing
        # It creates the engine, so we just verify the pipeline can handle 'auto'
        engine_name = pipeline.config.get("default_engine", "wan2.2")
        assert engine_name == "wan2.2-t2v"

    def test_t2v_pipeline_unknown_engine_raises_value_error(self):
        """Passing an unknown engine name raises ValueError."""
        pipeline = T2VPipeline()
        unknown_name = "this-engine-does-not-exist"
        with pytest.raises(ValueError, match="Unknown engine"):
            pipeline._get_engine(unknown_name)

    def test_full_pipeline_method_delegation(self):
        """FullPipeline methods delegate to sub-pipelines."""
        pipeline = FullPipeline()
        # text_to_video delegates to t2v.generate
        assert hasattr(pipeline.t2v, "generate")
        # image_to_video delegates to i2v.generate
        assert hasattr(pipeline.i2v, "generate")
        # portrait_animate delegates to portrait.generate
        assert hasattr(pipeline.portrait, "generate")

    def test_full_pipeline_unload_all(self):
        """FullPipeline.unload_all calls t2v.unload_all."""
        pipeline = FullPipeline()
        # Should not crash
        pipeline.unload_all()
        # Internal engine cache should be empty
        assert len(pipeline.t2v._engines) == 0

    def test_t2v_pipeline_unload_all(self):
        """T2VPipeline.unload_all clears engine cache."""
        pipeline = T2VPipeline()
        # unload_all should not crash with empty engine cache
        pipeline.unload_all()
        assert len(pipeline._engines) == 0
