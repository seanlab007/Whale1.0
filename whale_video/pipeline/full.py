"""Full production pipeline - end-to-end video creation."""
from pathlib import Path
from typing import Optional, Dict, Any, Union
from whale_video.pipeline.t2v import T2VPipeline
from whale_video.pipeline.i2v import I2VPipeline
from whale_video.pipeline.portrait import PortraitPipeline


class FullPipeline:
    """End-to-end video production pipeline combining all modules."""

    def __init__(self, config=None):
        self.config = config
        self.t2v = T2VPipeline(config)
        self.i2v = I2VPipeline(config)
        self.portrait = PortraitPipeline(config)

    def text_to_video(self, prompt: str, **kwargs) -> Dict:
        return self.t2v.generate(prompt=prompt, **kwargs)

    def image_to_video(self, image: Union[str, Path], **kwargs) -> Dict:
        return self.i2v.generate(image=image, **kwargs)

    def portrait_animate(self, source_image: Union[str, Path], **kwargs) -> Dict:
        return self.portrait.generate(source_image=source_image, **kwargs)

    def unload_all(self):
        self.t2v.unload_all()
