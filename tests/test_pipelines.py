"""Whale1.0 basic tests."""
import pytest
from pathlib import Path
from whale_video.config import WhaleConfig
from whale_video.pipeline.t2v import T2VPipeline
from whale_video.pipeline.i2v import I2VPipeline
from whale_video.pipeline.full import FullPipeline


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
