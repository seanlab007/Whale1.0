#!/usr/bin/env python3
"""
RunPod Serverless Worker — Whale1.0 Face Tools

RunPod handler that receives face_swap / face_enhance / face_analyze requests
via the RunPod serverless protocol and returns results.

Deploy:
    1. Create RunPod template with image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-runtime-22.04
    2. Set worker command: python scripts/runpod_worker.py
    3. Set environment:
       FACE_MODELS_DIR=/models/face
       HF_TOKEN=hf_xxx  (for downloading inswapper model)
    4. Volume mount: /models → RunPod network volume (persistent across cold starts)

RunPod handler interface:
    rp_handler(job) → dict with output
"""

import base64
import io
import os
import json
import time
import sys
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(os.environ.get("FACE_MODELS_DIR", "/models/face"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_output_dir = Path("/outputs")
_output_dir.mkdir(parents=True, exist_ok=True)


def _setup_models():
    """Lazy download of required ONNX models on first request."""
    import urllib.request

    swapper_path = MODELS_DIR / "inswapper_128.onnx"
    if not swapper_path.exists():
        print("[RunPod Worker] Downloading InSwapper model...")
        # Try HF mirror first (no auth needed), fallback to HF with token
        token = os.environ.get("HF_TOKEN", "")
        urls = [
            ("https://hf-mirror.com/deepinsight/inswapper/resolve/main/inswapper_128.onnx", {}),
        ]
        if token:
            urls.append((
                "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
                {"Authorization": f"Bearer {token}"},
            ))

        downloaded = False
        for url, headers in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    swapper_path.write_bytes(resp.read())
                downloaded = True
                print(f"[RunPod Worker] InSwapper downloaded: {swapper_path.stat().st_size} bytes")
                break
            except Exception as e:
                print(f"[RunPod Worker] Download failed from {url[:50]}...: {e}")

        if not downloaded:
            raise RuntimeError(
                "Cannot download inswapper_128.onnx. "
                "Set HF_TOKEN env var or pre-download to /models/face/"
            )

    # InsightFace will auto-download detection models on first use
    os.environ["INSIGHTFACE_HOME"] = str(MODELS_DIR)


def _b64_decode(data: str) -> "np.ndarray":
    """Decode base64 image string to numpy BGR array."""
    import numpy as np
    import cv2

    if data.startswith("data:"):
        data = data.split(",", 1)[1]

    img_bytes = base64.b64decode(data)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _b64_encode(img: "np.ndarray") -> str:
    """Encode numpy BGR array to base64 JPEG."""
    import cv2
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


def _load_insightface():
    """Lazy load InsightFace (GPU accelerated)."""
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    analyzer = FaceAnalysis(name="buffalo_l", providers=providers, root=str(MODELS_DIR))
    analyzer.prepare(ctx_id=0, det_size=(640, 640))

    swapper = get_model(str(MODELS_DIR / "inswapper_128.onnx"), providers=providers)

    return analyzer, swapper


# ── Handlers ────────────────────────────────────────────────────────────────


def handle_face_swap(input_data: dict) -> dict:
    """Face swap: source_face → target_image → swapped result."""
    import cv2
    import numpy as np

    source_b64 = input_data.get("source_image", "")
    target_b64 = input_data.get("target_image", "")

    if not source_b64 or not target_b64:
        return {"error": "Missing source_image or target_image"}

    src_img = _b64_decode(source_b64)
    tgt_img = _b64_decode(target_b64)

    analyzer, swapper = _load_insightface()

    src_faces = analyzer.get(src_img)
    tgt_faces = analyzer.get(tgt_img)

    if not src_faces:
        return {"error": "No face found in source image"}
    if not tgt_faces:
        return {"error": "No face found in target image"}

    result = swapper.get(tgt_img, tgt_faces[0], src_faces[0], paste_back=True)

    return {"output_image": _b64_encode(result)}


def handle_face_enhance(input_data: dict) -> dict:
    """Face enhancement using GFPGAN."""
    import cv2

    img_b64 = input_data.get("image", "")
    if not img_b64:
        return {"error": "Missing image"}

    img = _b64_decode(img_b64)

    # Try GFPGAN, fallback to OpenCV enhancement
    try:
        from gfpgan import GFPGANer
        gfpgan_path = MODELS_DIR / "GFPGANv1.4.pth"
        if gfpgan_path.exists():
            restorer = GFPGANer(
                model_path=str(gfpgan_path),
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=True)
            if restored is not None and len(restored) > 0:
                return {"output_image": _b64_encode(restored[0])}
    except ImportError:
        pass

    # Fallback: basic OpenCV enhancement
    import numpy as np
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return {"output_image": _b64_encode(sharpened)}


def handle_face_analyze(input_data: dict) -> dict:
    """Face detection + embedding extraction."""
    img_b64 = input_data.get("image", "")
    if not img_b64:
        return {"error": "Missing image"}

    img = _b64_decode(img_b64)
    analyzer, _ = _load_insightface()
    faces = analyzer.get(img)

    results = []
    for i, face in enumerate(faces):
        info = {
            "index": i,
            "bbox": [int(v) for v in face.bbox],
            "confidence": float(getattr(face, "det_score", 1.0)),
        }
        if input_data.get("extract_embedding") and face.embedding is not None:
            info["embedding"] = face.embedding.tolist()
        results.append(info)

    return {"faces": results, "face_count": len(results)}


# ── RunPod Serverless Handler ────────────────────────────────────────────────

def rp_handler(job: dict) -> dict:
    """
    RunPod serverless entry point.

    Input format (job["input"]):
    {
        "action": "face_swap" | "face_enhance" | "face_analyze",
        "source_image": "<base64>",
        "target_image": "<base64>",
        "image": "<base64>",
        "extract_embedding": false
    }

    Output format:
    {
        "output_image": "<base64>" | None,
        "error": "<message>" | None
    }
    """
    start_time = time.time()

    try:
        _setup_models()
    except Exception as e:
        return {"error": f"Model setup failed: {e}"}

    job_input = job.get("input", {})

    # Support both dict and string JSON input
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}

    action = job_input.get("action", "face_swap")

    try:
        if action == "face_swap":
            result = handle_face_swap(job_input)
        elif action == "face_enhance":
            result = handle_face_enhance(job_input)
        elif action == "face_analyze":
            result = handle_face_analyze(job_input)
        else:
            result = {"error": f"Unknown action: {action}. Supported: face_swap, face_enhance, face_analyze"}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"[RunPod Worker] action={action} done in {elapsed:.1f}s")

    return result


# ── Local test mode ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runpod_worker.py [--test]")
        print("  --test: Start local handler for testing")
        print("")
        print("RunPod deployment: set this as worker command in template")
        sys.exit(0)

    if sys.argv[1] == "--test":
        # Local test: swap two local images
        import cv2

        _setup_models()

        # Test with any two images
        test_src = sys.argv[2] if len(sys.argv) > 2 else "/inputs/source.jpg"
        test_tgt = sys.argv[3] if len(sys.argv) > 3 else "/inputs/target.jpg"

        if not Path(test_src).exists():
            print(f"Source image not found: {test_src}")
            sys.exit(1)
        if not Path(test_tgt).exists():
            print(f"Target image not found: {test_tgt}")
            sys.exit(1)

        src_img = cv2.imread(test_src)
        tgt_img = cv2.imread(test_tgt)

        src_b64 = _b64_encode(src_img)
        tgt_b64 = _b64_encode(tgt_img)

        result = rp_handler({
            "input": {
                "action": "face_swap",
                "source_image": src_b64,
                "target_image": tgt_b64,
            }
        })

        if result.get("output_image"):
            output_path = _output_dir / f"swap_{int(time.time())}.jpg"
            output_path.write_bytes(base64.b64decode(result["output_image"]))
            print(f"✅ Face swap done! Saved to: {output_path}")
        else:
            print(f"❌ Failed: {result.get('error', 'unknown')}")
    else:
        # Start RunPod serverless handler
        import runpod
        runpod.serverless.start({"handler": rp_handler})
