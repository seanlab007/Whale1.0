"""Whale1.0 configuration management."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Default model registry
MODEL_REGISTRY = {
    "wan2.2-t2v": {
        "name": "Wan2.2-T2V-14B",
        "type": "text-to-video",
        "architecture": "MoE",
        "parameters": "14B",
        "min_vram_gb": 24,
        "repo_id": "Wan-AI/Wan2.2-T2V-14B",
        "submodule": "Wan2.2",
        "enabled": True,
    },
    "wan2.2-i2v": {
        "name": "Wan2.2-I2V-14B",
        "type": "image-to-video",
        "architecture": "MoE",
        "parameters": "14B",
        "min_vram_gb": 24,
        "repo_id": "Wan-AI/Wan2.2-I2V-14B",
        "submodule": "Wan2.2",
        "enabled": True,
    },
    "cogvideox-2": {
        "name": "CogVideoX-2",
        "type": "text-to-video",
        "architecture": "DiT",
        "parameters": "13B",
        "min_vram_gb": 16,
        "repo_id": "THUDM/CogVideoX-2b",
        "submodule": "CogVideo",
        "enabled": True,
    },
    "hunyuanvideo": {
        "name": "HunyuanVideo-1.5",
        "type": "text-to-video",
        "architecture": "DiT",
        "parameters": "8.3B",
        "min_vram_gb": 12,
        "repo_id": "Tencent/HunyuanVideo-1.5",
        "submodule": "HunyuanVideo",
        "enabled": True,
    },
    "mochi": {
        "name": "Mochi-1",
        "type": "text-to-video",
        "architecture": "DiT",
        "parameters": "10B",
        "min_vram_gb": 20,
        "repo_id": "genmo/mochi-1-preview",
        "submodule": "mochi",
        "enabled": True,
    },
    "ltx-video": {
        "name": "LTX-Video-2.3",
        "type": "text-to-video",
        "architecture": "DiT",
        "parameters": "22B",
        "min_vram_gb": 16,
        "repo_id": "Lightricks/LTX-Video",
        "submodule": "LTX-Video",
        "enabled": True,
    },
    "skyreels-v1": {
        "name": "SkyReels-V1",
        "type": "text-to-video",
        "architecture": "DiT",
        "parameters": "13B",
        "min_vram_gb": 16,
        "repo_id": "Skywork/SkyReels-V1-Hunyuan-T2V",
        "submodule": "SkyReels-V1",
        "enabled": True,
    },
    "liveportrait": {
        "name": "LivePortrait",
        "type": "portrait-animation",
        "architecture": "CNN",
        "parameters": "~1B",
        "min_vram_gb": 8,
        "repo_id": "KwaiVGI/LivePortrait",
        "submodule": "LivePortrait",
        "enabled": True,
    },
    "wav2lip": {
        "name": "Wav2Lip",
        "type": "lip-sync",
        "architecture": "CNN+LSTM",
        "parameters": "~0.2B",
        "min_vram_gb": 4,
        "repo_id": None,
        "submodule": "Wav2Lip",
        "enabled": True,
    },
    "animatediff": {
        "name": "AnimateDiff",
        "type": "animation",
        "architecture": "UNet+Motion",
        "parameters": "~1.5B",
        "min_vram_gb": 8,
        "repo_id": "guoyww/animatediff",
        "submodule": "AnimateDiff",
        "enabled": True,
    },
    "rife": {
        "name": "RIFE",
        "type": "frame-interpolation",
        "architecture": "IFNet",
        "parameters": "~7M",
        "min_vram_gb": 4,
        "repo_id": None,
        "submodule": "ECCV2022-RIFE",
        "enabled": True,
    },
}

DEFAULT_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "dtype": "float16",
    "output_dir": "./outputs",
    "models_dir": "./models",
    "submodules_dir": "./submodules",
    "default_engine": "wan2.2-t2v",
    "enable_model_caching": True,
    "enable_memory_optimization": True,
    "max_video_length_seconds": 30,
    "default_fps": 24,
    "default_resolution": [720, 1280],  # [H, W]
}


@dataclass
class WhaleConfig:
    """Unified configuration for Whale1.0."""
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    model_registry: Dict[str, Dict] = field(default_factory=lambda: MODEL_REGISTRY.copy())

    @classmethod
    def from_yaml(cls, path: str) -> "WhaleConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if data:
            if "config" in data:
                cfg.config.update(data["config"])
            if "model_registry" in data:
                cfg.model_registry.update(data["model_registry"])
        return cfg

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def get_model_config(self, model_name: str) -> Optional[Dict]:
        return self.model_registry.get(model_name)

    def list_available_engines(self) -> Dict[str, str]:
        return {
            name: info["name"]
            for name, info in self.model_registry.items()
            if info.get("enabled", True)
        }

    def resolve_submodule_path(self, model_name: str) -> Optional[str]:
        info = self.model_registry.get(model_name)
        if not info:
            return None
        sub = info.get("submodule", "")
        if not sub:
            return None
        return os.path.join(self.config.get("submodules_dir", "./submodules"), sub)
