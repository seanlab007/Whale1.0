#!/usr/bin/env python3
"""Face API Server — MaoAI 面部处理服务

提供换脸融合、面部增强、人脸分析三大能力，对标手机端妙鸭/Remini/美图的技术路线。

Endpoints:
    POST /api/face/swap       - FaceSwap: 将源人脸替换到目标图像
    POST /api/face/enhance    - 面部增强 (GFPGAN / CodeFormer)
    POST /api/face/analyze    - 提取人脸特征向量 (ArcFace embedding)
    GET  /api/health          - 健康检查
    GET  /api/models          - 可用模型列表

Run:
    python -m face_tools.server --port 8001
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="MaoAI Face Tools API",
    version="1.0.0",
    description="换脸融合 (FaceSwap) + 面部增强 (GFPGAN) + 人脸分析 (ArcFace)",
)

OUTPUT_DIR = Path(os.environ.get("FACE_OUTPUT_DIR", "./outputs/face"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_CACHE_DIR = Path(os.environ.get("FACE_MODELS_DIR", "./models/face"))
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_start_time = time.time()


# ── Response Models ──────────────────────────────────────────────────────────

class SwapResponse(BaseModel):
    success: bool
    output_url: Optional[str] = None
    output_b64: Optional[str] = None  # base64 encoded result image
    error: Optional[str] = None
    method: str = "insightface-inswapper"


class EnhanceResponse(BaseModel):
    success: bool
    output_url: Optional[str] = None
    output_b64: Optional[str] = None
    error: Optional[str] = None
    method: str = "gfpgan"


class FaceInfo(BaseModel):
    bbox: List[int]          # [x, y, w, h]
    landmarks: List[List[float]]  # 5 keypoints
    embedding: Optional[List[float]] = None  # 512-dim ArcFace embedding
    confidence: float


class AnalyzeResponse(BaseModel):
    success: bool
    faces: List[FaceInfo] = []
    face_count: int = 0
    error: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str


# ── Lazy model loading ──────────────────────────────────────────────────────

_models: Dict[str, Any] = {}


def _get_insightface():
    """Lazy load InsightFace models."""
    if "insightface" in _models:
        return _models["insightface"]

    try:
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        # Face analysis (detection + landmarks + ArcFace embedding)
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=str(MODELS_CACHE_DIR),
        )
        analyzer.prepare(ctx_id=0, det_size=(640, 640))

        # InSwapper model
        swapper_path = MODELS_CACHE_DIR / "inswapper_128.onnx"
        if not swapper_path.exists():
            print(f"[FaceAPI] ⚠️ InSwapper model not found at {swapper_path}")
            print(f"[FaceAPI] Download from: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx")
            swapper = None
        else:
            swapper = get_model(str(swapper_path), providers=["CPUExecutionProvider"])

        _models["insightface"] = (analyzer, swapper)
        return analyzer, swapper
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"InsightFace not installed. Run: pip install insightface onnxruntime"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"InsightFace load error: {e}")


def _get_gfpgan():
    """Lazy load GFPGAN for face enhancement."""
    if "gfpgan" in _models:
        return _models["gfpgan"]

    try:
        from gfpgan import GFPGANer
        model_path = MODELS_CACHE_DIR / "GFPGANv1.4.pth"
        if not model_path.exists():
            print(f"[FaceAPI] ⚠️ GFPGAN model not found at {model_path}")
            _models["gfpgan"] = None
            return None

        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _models["gfpgan"] = restorer
        return restorer
    except ImportError:
        print("[FaceAPI] ⚠️ GFPGAN not installed. Run: pip install gfpgan")
        _models["gfpgan"] = None
        return None
    except Exception as e:
        print(f"[FaceAPI] GFPGAN load error: {e}")
        _models["gfpgan"] = None
        return None


# ── FaceSwap Engine ──────────────────────────────────────────────────────────

def _load_image_from_upload(file: UploadFile) -> "np.ndarray":
    """Read uploaded file into numpy array (BGR)."""
    import numpy as np
    contents = file.file.read()
    # Try PIL first, fallback to cv2
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        # RGB to BGR
        return arr[:, :, ::-1].copy()
    except Exception:
        import cv2
        arr = np.frombuffer(contents, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _image_to_b64(arr: "np.ndarray") -> str:
    """Convert numpy BGR array to base64 JPEG string."""
    import cv2
    import base64
    _, buf = cv2.imencode(".jpg", arr)
    return base64.b64encode(buf).decode("utf-8")


def _do_faceswap(source_img: "np.ndarray", target_img: "np.ndarray") -> "np.ndarray":
    """Core face swap using InsightFace InSwapper."""
    import cv2
    import numpy as np

    analyzer, swapper = _get_insightface()

    if swapper is None:
        raise HTTPException(
            status_code=503,
            detail="InSwapper model not found. Download inswapper_128.onnx to models/face/"
        )

    # Detect faces in both images
    src_faces = analyzer.get(source_img)
    tgt_faces = analyzer.get(target_img)

    if not src_faces:
        raise HTTPException(status_code=400, detail="No face found in source image")
    if not tgt_faces:
        raise HTTPException(status_code=400, detail="No face found in target image")

    # Use the first face from each
    src_face = src_faces[0]
    tgt_face = tgt_faces[0]

    # Perform swap
    result = swapper.get(target_img, tgt_face, src_face, paste_back=True)
    return result


# ── Face Enhancement Engine ──────────────────────────────────────────────────

def _do_enhance(img: "np.ndarray") -> "np.ndarray":
    """Enhance face quality using GFPGAN."""
    restorer = _get_gfpgan()
    if restorer is None:
        raise HTTPException(
            status_code=503,
            detail="GFPGAN model not available. Download GFPGANv1.4.pth to models/face/"
        )

    _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False)
    if restored is None or len(restored) == 0:
        raise HTTPException(status_code=500, detail="GFPGAN enhancement failed")

    return restored[0]


# ── Face Analysis ────────────────────────────────────────────────────────────

def _do_analyze(img: "np.ndarray") -> List[FaceInfo]:
    """Detect faces and extract embeddings."""
    import numpy as np
    analyzer, _ = _get_insightface()
    faces = analyzer.get(img)

    results = []
    for face in faces:
        info = FaceInfo(
            bbox=[int(v) for v in face.bbox],
            landmarks=face.kps.tolist() if face.kps is not None else [],
            embedding=face.embedding.tolist() if face.embedding is not None else None,
            confidence=float(face.det_score) if hasattr(face, "det_score") else 1.0,
        )
        results.append(info)

    return results


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/face/swap", response_model=SwapResponse)
async def face_swap(
    source: UploadFile = File(..., description="Source face image (the person's face)"),
    target: UploadFile = File(..., description="Target image to put the face onto"),
):
    """
    换脸融合 - 将 source 图片的人脸替换到 target 图片上。
    
    对标手机端 FaceSwap 路线：先生成底图，再用 InsightFace InSwapper 精准融合。
    五官100%保留。
    """
    try:
        src_img = _load_image_from_upload(source)
        tgt_img = _load_image_from_upload(target)
        result = _do_faceswap(src_img, tgt_img)

        # Save to disk
        output_path = OUTPUT_DIR / f"swap_{int(time.time())}.jpg"
        import cv2
        cv2.imwrite(str(output_path), result)

        b64 = _image_to_b64(result)
        return SwapResponse(
            success=True,
            output_url=f"/outputs/face/{output_path.name}",
            output_b64=b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        return SwapResponse(success=False, error=str(e))


@app.post("/api/face/enhance", response_model=EnhanceResponse)
async def face_enhance(
    image: UploadFile = File(..., description="Image to enhance"),
):
    """
    面部增强 - 使用 GFPGAN 修复模糊/低质量人脸。
    
    对标手机端 Remini 的面部修复功能。
    """
    try:
        img = _load_image_from_upload(image)
        result = _do_enhance(img)

        output_path = OUTPUT_DIR / f"enhance_{int(time.time())}.jpg"
        import cv2
        cv2.imwrite(str(output_path), result)

        b64 = _image_to_b64(result)
        return EnhanceResponse(
            success=True,
            output_url=f"/outputs/face/{output_path.name}",
            output_b64=b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        return EnhanceResponse(success=False, error=str(e))


@app.post("/api/face/analyze", response_model=AnalyzeResponse)
async def face_analyze(
    image: UploadFile = File(..., description="Image to analyze"),
    extract_embedding: bool = Form(False, description="Extract ArcFace 512-dim embedding"),
):
    """
    人脸分析 - 检测人脸、提取关键点和 ArcFace 特征向量。
    
    特征向量可用于 InstantID / IP-Adapter 身份注入。
    """
    try:
        img = _load_image_from_upload(image)
        faces = _do_analyze(img)

        if not extract_embedding:
            for f in faces:
                f.embedding = None

        return AnalyzeResponse(
            success=True,
            faces=faces,
            face_count=len(faces),
        )
    except HTTPException:
        raise
    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))


@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """返回可用的面部处理模型列表。"""
    return [
        ModelInfo(id="insightface-inswapper", name="InsightFace InSwapper 128", type="face-swap"),
        ModelInfo(id="gfpgan-v1.4", name="GFPGAN v1.4", type="face-enhance"),
        ModelInfo(id="arcface-buffalo_l", name="ArcFace (buffalo_l)", type="face-embedding"),
    ]


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "uptime": time.time() - _start_time}


# ── CLI Entry ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MaoAI Face Tools API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Dev mode auto-reload")

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("face_tools.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
