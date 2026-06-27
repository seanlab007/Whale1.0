"""Abstract base class for video engines."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from pathlib import Path


class VideoEngine(ABC):
    """Abstract base class for all video generation engines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._model = None
        self._loaded = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Return the engine type: 'text-to-video', 'image-to-video', etc."""
        ...

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def generate(self, **kwargs) -> Union[Path, List[Path]]:
        """Generate video(s) from input."""
        ...

    def unload(self) -> None:
        """Free GPU memory."""
        self._model = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "engine_type": self.engine_type,
            "loaded": self._loaded,
        }
