"""Wan2.2 engine - MoE architecture video generation (Alibaba)."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
from whale_video.engine.base import VideoEngine
from whale_video.utils import get_torch


class Wan2Engine(VideoEngine):
    """Wan2.2 engine: MoE-based text-to-video and image-to-video generation."""

    @property
    def model_name(self) -> str:
        return "wan2.2"

    @property
    def engine_type(self) -> str:
        variant = self.config.get("variant", "t2v")
        return "text-to-video" if variant == "t2v" else "image-to-video"

    def load(self, **kwargs) -> None:
        _torch = get_torch()
        model_path = kwargs.get("model_path", self.config.get("model_path"))
        device = kwargs.get("device", self.config.get("device", "cuda"))
        dtype = kwargs.get("dtype", self.config.get("dtype", "float16"))

        print(f"[Wan2.2] Loading model from {model_path or 'default'}...")

        try:
            from wan.pipeline import WanPipeline
            from wan.config import WanConfig

            cfg = WanConfig.from_pretrained(model_path) if model_path else WanConfig()
            self._model = WanPipeline(cfg, device=device, dtype=getattr(_torch, dtype))
            self._model.to(device)
            self._loaded = True
            print(f"[Wan2.2] Loaded successfully on {device}")
        except ImportError:
            print("[Wan2.2] Full model import failed, falling back to diffusers pipeline")
            self._load_diffusers(device, dtype, _torch)

    def _load_diffusers(self, device: str, dtype: str, _torch) -> None:
        from diffusers import DiffusionPipeline
        repo_id = self.config.get("repo_id", "Wan-AI/Wan2.2-T2V-14B")
        self._model = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=getattr(_torch, dtype),
            trust_remote_code=True,
        ).to(device)
        self._loaded = True

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[str, Path]] = None,
        num_frames: int = 81,
        fps: int = 16,
        width: int = 1280,
        height: int = 720,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Path:
        if not self._loaded:
            self.load()

        _torch = get_torch()
        output_path = Path(output_path or self.config.get("output_dir", "./outputs"))
        output_path.mkdir(parents=True, exist_ok=True)

        generator = None
        if seed is not None:
            generator = _torch.Generator().manual_seed(seed)

        print(f"[Wan2.2] Generating video from {'image' if image else 'text'} prompt...")

        if hasattr(self._model, "__call__"):
            result = self._model(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                image=Path(image) if image else None,
                num_frames=num_frames,
                fps=fps,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            result = self._model(
                prompt=prompt,
                image=image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        video_frames = result.frames[0]
        out_file = output_path / f"wan2.2_{prompt[:30].replace(' ', '_')}.mp4"

        import imageio
        writer = imageio.get_writer(str(out_file), fps=fps, codec="libx264")
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()

        print(f"[Wan2.2] Video saved to {out_file}")
        return out_file
