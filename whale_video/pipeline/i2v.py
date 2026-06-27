"""Image-to-Video pipeline."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
from whale_video.config import WhaleConfig
from whale_video.engine.wan2_2 import Wan2Engine


class I2VPipeline:
    """Image-to-video pipeline (primarily uses Wan2.2-I2V)."""

    def __init__(self, config: Optional[Union[WhaleConfig, Dict]] = None):
        if isinstance(config, dict):
            self.config = WhaleConfig()
            self.config.config.update(config)
        else:
            self.config = config or WhaleConfig()
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            cfg = self.config.get_model_config("wan2.2-i2v") or {}
            cfg["variant"] = "i2v"
            self._engine = Wan2Engine(cfg)
        return self._engine

    def generate(
        self,
        image: Union[str, Path],
        prompt: Optional[str] = None,
        num_frames: int = 81,
        fps: int = 16,
        width: int = 1280,
        height: int = 720,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        eng = self._get_engine()
        actual_prompt = prompt or "Generate a video from this image"
        result_path = eng.generate(
            prompt=actual_prompt,
            image=image,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
        return {
            "engine": "wan2.2-i2v",
            "prompt": actual_prompt,
            "input_image": str(image),
            "output_path": str(result_path),
            "fps": fps,
            "num_frames": num_frames,
        }
