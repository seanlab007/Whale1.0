"""Video processing utilities."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union


def extract_frames(video_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Extract all frames from a video file."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    out_dir = Path(output_dir) if output_dir else Path(video_path).parent / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = out_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        i += 1
    cap.release()
    return frames


def frames_to_video(frames: List[Union[np.ndarray, Path]], output_path: Union[str, Path], fps: int = 24):
    """Convert frame list to video."""
    import cv2
    import numpy as np
    import imageio
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    for frame in frames:
        if isinstance(frame, Path):
            frame = cv2.imread(str(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)
    writer.close()
    return output_path


def get_video_info(video_path: Union[str, Path]) -> dict:
    """Get video metadata."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }
    cap.release()
    return info
