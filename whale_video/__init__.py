"""
Whale1.0 — Open Source Video Generation Foundation Model

A unified platform integrating the best open-source video generation models,
providing a consistent API for text-to-video, image-to-video, portrait animation,
lip sync, and frame interpolation.
"""

__version__ = "1.0.0"

from whale_video.config import WhaleConfig
from whale_video.engine import VideoEngine, Wan2Engine, CogVideoEngine, HunyuanEngine, MochiEngine, LTXEngine
from whale_video.pipeline import T2VPipeline, I2VPipeline, PortraitPipeline, FullPipeline

__all__ = [
    "WhaleConfig",
    "VideoEngine",
    "Wan2Engine",
    "CogVideoEngine",
    "HunyuanEngine",
    "MochiEngine",
    "LTXEngine",
    "T2VPipeline",
    "I2VPipeline",
    "PortraitPipeline",
    "FullPipeline",
]
