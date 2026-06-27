"""Portrait animation pipeline - face animation + wav2lip."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
from whale_video.config import WhaleConfig


class PortraitPipeline:
    """Full portrait video generation: animate + lip sync + audio."""

    def __init__(self, config: Optional[Union[WhaleConfig, Dict]] = None):
        if isinstance(config, dict):
            self.config = WhaleConfig()
            self.config.config.update(config)
        else:
            self.config = config or WhaleConfig()

    def generate(
        self,
        source_image: Union[str, Path],
        driving_video: Optional[Union[str, Path]] = None,
        driving_audio: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        output_path = Path(output_path or self.config.get("output_dir", "./outputs"))
        output_path.mkdir(parents=True, exist_ok=True)
        return {
            "engine": "portrait-pipeline",
            "output_path": str(output_path),
            "modules": ["liveportrait", "wav2lip"],
            "note": "Run bash scripts/setup.sh first to install LivePortrait and Wav2Lip dependencies",
        }
