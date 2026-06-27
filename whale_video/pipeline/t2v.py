"""Text-to-Video pipeline - unified interface for all T2V engines."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
from whale_video.config import WhaleConfig
from whale_video.engine.wan2_2 import Wan2Engine
from whale_video.engine.cogvideo import CogVideoEngine
from whale_video.engine.hunyuan import HunyuanEngine
from whale_video.engine.mochi import MochiEngine
from whale_video.engine.ltx import LTXEngine


class T2VPipeline:
    """Unified text-to-video pipeline supporting multiple backends."""

    ENGINE_MAP = {
        "wan2.2": Wan2Engine,
        "cogvideox": CogVideoEngine,
        "hunyuanvideo": HunyuanEngine,
        "mochi": MochiEngine,
        "ltx-video": LTXEngine,
    }

    def __init__(self, config: Optional[Union[WhaleConfig, Dict]] = None):
        if isinstance(config, dict):
            self.config = WhaleConfig()
            self.config.config.update(config)
        else:
            self.config = config or WhaleConfig()
        self._engines = {}

    def _get_engine(self, engine_name: str):
        if engine_name not in self._engines:
            cls = self.ENGINE_MAP.get(engine_name)
            if not cls:
                raise ValueError(f"Unknown engine: {engine_name}. Available: {list(self.ENGINE_MAP.keys())}")
            engine_cfg = self.config.get_model_config(engine_name) or {}
            self._engines[engine_name] = cls(engine_cfg)
        return self._engines[engine_name]

    def generate(
        self,
        prompt: str,
        engine: str = "auto",
        negative_prompt: Optional[str] = None,
        num_frames: int = 81,
        fps: int = 24,
        width: int = 1280,
        height: int = 720,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if engine == "auto":
            engine = self.config.get("default_engine", "wan2.2")

        eng = self._get_engine(engine)
        result_path = eng.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
        return {
            "engine": engine,
            "prompt": prompt,
            "output_path": str(result_path),
            "fps": fps,
            "num_frames": num_frames,
            "resolution": f"{width}x{height}",
        }

    def list_engines(self):
        return {k: v.__name__ for k, v in self.ENGINE_MAP.items()}

    def unload_all(self):
        for eng in self._engines.values():
            eng.unload()
        self._engines.clear()
