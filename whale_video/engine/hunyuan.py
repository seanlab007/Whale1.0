"""HunyuanVideo engine - Lightweight DiT model (Tencent)."""
from pathlib import Path
from typing import Optional, Union
from whale_video.engine.base import VideoEngine
from whale_video.utils import get_torch


class HunyuanEngine(VideoEngine):
    """HunyuanVideo-1.5 engine: lightweight DiT for consumer GPUs."""

    @property
    def model_name(self) -> str: return "hunyuanvideo"

    @property
    def engine_type(self) -> str: return "text-to-video"

    def load(self, **kwargs) -> None:
        _torch = get_torch()
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", "float16")
        try:
            from diffusers import HunyuanVideoPipeline
        except ImportError:
            raise ImportError(
                "HunyuanVideo requires `diffusers`. Install with: pip install diffusers"
            )
        repo_id = self.config.get("repo_id", "Tencent/HunyuanVideo-1.5")
        self._model = HunyuanVideoPipeline.from_pretrained(
            repo_id, torch_dtype=getattr(_torch, dtype)
        ).to(device)
        self._model.enable_model_cpu_offload()
        self._loaded = True

    def generate(self, prompt: str, negative_prompt=None, num_frames=129, fps=24,
                 width=848, height=480, steps=50, guidance_scale=6.0, seed=None,
                 output_path=None, **kwargs) -> Path:
        if not self._loaded: self.load()
        _torch = get_torch()
        output_path = Path(output_path or self.config.get("output_dir", "./outputs"))
        output_path.mkdir(parents=True, exist_ok=True)
        generator = _torch.Generator().manual_seed(seed) if seed is not None else None
        result = self._model(prompt=prompt, negative_prompt=negative_prompt or "",
                             num_frames=num_frames, num_inference_steps=steps,
                             guidance_scale=guidance_scale, generator=generator)
        out_file = output_path / f"hunyuan_{prompt[:30].replace(' ', '_')}.mp4"
        import imageio
        writer = imageio.get_writer(str(out_file), fps=fps, codec="libx264")
        for frame in result.frames[0]: writer.append_data(frame)
        writer.close()
        return out_file
