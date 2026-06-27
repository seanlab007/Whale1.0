"""Whale1.0 FastAPI server — exposes video generation as an HTTP API.

Run:
    python -m whale_video.server --port 7999

Endpoints:
    POST /api/generate      — Submit video generation task
    GET  /api/status/{id}   — Query task status & video URL
    GET  /api/video/{id}    — Download generated video file
    GET  /api/models        — List available engines
    GET  /api/health        — Health check
"""

from __future__ import annotations

import os
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Whale1.0 Video Generation API",
    version="1.0.0",
    description="Unified API for all Whale1.0 video generation engines (Wan2.2, CogVideoX, HunyuanVideo, Mochi, LTX-Video).",
)

# In-memory task store: { task_id: { ... } }
_tasks: dict[str, dict] = {}
_lock = threading.Lock()


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the video")
    engine: str = Field("auto", description="Engine to use: auto | wan2.2 | cogvideox | hunyuanvideo | mochi | ltx-video")
    negative_prompt: Optional[str] = Field(None, description="What to avoid in the video")
    num_frames: int = Field(81, ge=1, le=300, description="Number of frames to generate")
    fps: int = Field(24, ge=1, le=60, description="Frames per second")
    width: int = Field(1280, ge=256, le=4096, description="Video width (px)")
    height: int = Field(720, ge=256, le=4096, description="Video height (px)")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Inference steps (higher = better quality but slower)")
    guidance_scale: float = Field(5.0, ge=0.0, le=30.0, description="Prompt guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    output_dir: Optional[str] = Field(None, description="Override output directory (default: ./outputs)")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A golden retriever running on a beach at sunset, cinematic quality",
                "engine": "auto",
                "num_frames": 81,
                "fps": 24,
                "width": 1280,
                "height": 720,
            }
        }


class GenerateResponse(BaseModel):
    task_id: str
    status: str  # PENDING


class StatusResponse(BaseModel):
    task_id: str
    status: str  # PENDING | PROCESSING | SUCCEEDED | FAILED
    progress: Optional[float] = None
    video_url: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float


# ── Background generation ─────────────────────────────────────────────────────

def _run_generation(task_id: str, req: GenerateRequest) -> None:
    """Execute video generation in a background thread."""
    try:
        with _lock:
            _tasks[task_id]["status"] = "PROCESSING"
            _tasks[task_id]["progress"] = 0.0

        # Lazy imports — torch only loads here
        from whale_video import T2VPipeline, WhaleConfig

        config = WhaleConfig()
        if req.output_dir:
            config.config["output_dir"] = req.output_dir

        pipeline = T2VPipeline(config)

        output_dir = Path(req.output_dir or config.get("output_dir", "./outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.mp4"

        result = pipeline.generate(
            prompt=req.prompt,
            engine=req.engine,
            negative_prompt=req.negative_prompt,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_path=str(output_path),
        )

        with _lock:
            _tasks[task_id]["status"] = "SUCCEEDED"
            _tasks[task_id]["progress"] = 100.0
            _tasks[task_id]["result"] = result
            _tasks[task_id]["video_url"] = f"/api/video/{task_id}"

    except Exception as exc:
        with _lock:
            _tasks[task_id]["status"] = "FAILED"
            _tasks[task_id]["error"] = f"{type(exc).__name__}: {exc}"
            _tasks[task_id]["progress"] = 0.0


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Submit a video generation task. Returns immediately with a task_id."""
    task_id = str(uuid.uuid4())
    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "status": "PENDING",
            "progress": 0.0,
            "video_url": None,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(target=_run_generation, args=(task_id, req), daemon=True)
    thread.start()

    return GenerateResponse(task_id=task_id, status="PENDING")


@app.get("/api/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Get the status and result of a video generation task."""
    with _lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return StatusResponse(**task)


@app.get("/api/video/{task_id}")
async def serve_video(task_id: str):
    """Download the generated video file."""
    with _lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task["status"] != "SUCCEEDED":
        raise HTTPException(status_code=400, detail=f"Task status is {task['status']}, not SUCCEEDED")

    result = task["result"]
    if not result:
        raise HTTPException(status_code=500, detail="Task succeeded but no result data")

    video_path = Path(result.get("output_path", ""))
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found at {video_path}")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    """List all available video generation engines."""
    try:
        from whale_video.config import MODEL_REGISTRY

        models = []
        for key, info in MODEL_REGISTRY.items():
            models.append(ModelInfo(
                id=key,
                name=info.get("name", key),
                type=info.get("type", "unknown"),
            ))
        return models
    except ImportError:
        # Fallback: return hardcoded engine list
        return [
            ModelInfo(id="wan2.2", name="Wan2.2-T2V-14B", type="text-to-video"),
            ModelInfo(id="cogvideox", name="CogVideoX-2", type="text-to-video"),
            ModelInfo(id="hunyuanvideo", name="HunyuanVideo-1.5", type="text-to-video"),
            ModelInfo(id="mochi", name="Mochi-1", type="text-to-video"),
            ModelInfo(id="ltx-video", name="LTX-Video-2.3", type="text-to-video"),
        ]


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — returns server status and version."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime=time.time() - _start_time,
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

_start_time = time.time()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Whale1.0 Video Generation API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=7999, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")

    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "whale_video.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
