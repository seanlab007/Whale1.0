"""Model weight download utilities."""
import os
from pathlib import Path


def ensure_models_dir(path: str = "./models") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model_size(model_name: str) -> str:
    sizes = {
        "wan2.2-t2v": "~30GB",
        "wan2.2-i2v": "~30GB",
        "cogvideox-2": "~20GB",
        "hunyuanvideo": "~16GB",
        "mochi": "~25GB",
        "ltx-video": "~40GB",
    }
    return sizes.get(model_name, "unknown")
