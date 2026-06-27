"""LTX-Video engine - DiT with synchronized audio (Lightricks)."""
from pathlib import Path
from typing import Optional, Union
from whale_video.engine.base import VideoEngine
from whale_video.utils import get_torch


class LTXEngine(VideoEngine):
    """LTX-Video engine: DiT-based video generation with synchronized audio (Lightricks).

    Supports text-to-video generation with audio synchronization capabilities,
    built on the LTX architecture for efficient consumer-GPU inference.
    """

    @property
    def model_name(self) -> str: return "ltx-video"

    @property
    def engine_type(self) -> str: return "text-to-video"

    def load(self, **kwargs) -> None:
        _torch = get_torch()
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", "float16")
        try:
            from diffusers import LTXPipeline
        except ImportError:
            raise ImportError(
                "LTX-Video requires `diffusers`. Install with: pip install diffusers"
            )
        repo_id = self.config.get("repo_id", "Lightricks/LTX-Video")
        self._model = LTXPipeline.from_pretrained(
            repo_id, torch_dtype=getattr(_torch, dtype)
        ).to(device)
        self._loaded = True

    def generate(self, prompt: str, negative_prompt: Optional[str] = None,
                 num_frames=97, fps=30, width=1216, height=704,
                 steps=50, guidance_scale=4.5, seed=None,
                 output_path=None, **kwargs) -> Path:
        """Generate video from text prompt."""
        if not self._loaded: self.load()
        _torch = get_torch()
        output_path = Path(output_path or self.config.get("output_dir", "./outputs"))
        output_path.mkdir(parents=True, exist_ok=True)
        generator = _torch.Generator().manual_seed(seed) if seed is not None else None
        result = self._model(prompt=prompt, negative_prompt=negative_prompt or "",
                             num_frames=num_frames, width=width,
                             height=height, num_inference_steps=steps,
                             guidance_scale=guidance_scale, generator=generator, **kwargs)
        out_file = output_path / f"ltx_{prompt[:30].replace(' ', '_')}.mp4"
        import imageio
        writer = imageio.get_writer(str(out_file), fps=fps, codec="libx264")
        for frame in result.frames[0]: writer.append_data(frame)
        writer.close()
        return out_file
