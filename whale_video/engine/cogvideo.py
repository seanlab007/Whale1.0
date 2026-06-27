"""CogVideoX engine - DiT architecture with strong Chinese support (ZhipuAI/THU)."""
from pathlib import Path
from typing import Optional, Union
from whale_video.engine.base import VideoEngine
from whale_video.utils import get_torch


class CogVideoEngine(VideoEngine):
    """CogVideoX engine: DiT-based text-to-video generation with strong semantic understanding."""

    @property
    def model_name(self) -> str:
        return "cogvideox"

    @property
    def engine_type(self) -> str:
        return "text-to-video"

    def load(self, **kwargs) -> None:
        _torch = get_torch()
        device = kwargs.get("device", self.config.get("device", "cuda"))
        dtype = kwargs.get("dtype", self.config.get("dtype", "float16"))
        print(f"[CogVideoX] Loading model...")

        from diffusers import CogVideoXPipeline
        repo_id = self.config.get("repo_id", "THUDM/CogVideoX-2b")
        self._model = CogVideoXPipeline.from_pretrained(
            repo_id, torch_dtype=getattr(_torch, dtype),
        ).to(device)
        self._model.enable_sequential_cpu_offload()
        self._loaded = True
        print(f"[CogVideoX] Loaded successfully on {device}")

    def generate(
        self, prompt: str, negative_prompt: Optional[str] = None,
        num_frames: int = 49, fps: int = 8, width: int = 720, height: int = 480,
        num_inference_steps: int = 50, guidance_scale: float = 6.0,
        seed: Optional[int] = None, output_path: Optional[Union[str, Path]] = None, **kwargs,
    ) -> Path:
        if not self._loaded: self.load()
        _torch = get_torch()
        output_path = Path(output_path or self.config.get("output_dir", "./outputs"))
        output_path.mkdir(parents=True, exist_ok=True)
        generator = _torch.Generator().manual_seed(seed) if seed is not None else None

        result = self._model(
            prompt=prompt, negative_prompt=negative_prompt or "",
            num_frames=num_frames, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )
        out_file = output_path / f"cogvideo_{prompt[:30].replace(' ', '_')}.mp4"
        import imageio
        writer = imageio.get_writer(str(out_file), fps=fps, codec="libx264")
        for frame in result.frames[0]: writer.append_data(frame)
        writer.close()
        return out_file
