#!/usr/bin/env python3
"""Run Whale1.0 video generation pipeline from CLI."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whale_video import T2VPipeline, I2VPipeline, FullPipeline
from whale_video.utils import get_device, optimize_memory


def main():
    parser = argparse.ArgumentParser(description="Whale1.0 Video Generation Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--engine", type=str, default="auto",
                        choices=["auto", "wan2.2", "cogvideox", "hunyuanvideo", "mochi", "ltx-video"],
                        help="Video generation engine")
    parser.add_argument("--image", type=str, help="Input image path (for I2V)")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--list-engines", action="store_true", help="List available engines")

    args = parser.parse_args()

    if args.list_engines:
        pipeline = T2VPipeline()
        print("Available engines:")
        for name, cls in pipeline.list_engines().items():
            print(f"  - {name} ({cls})")
        return

    # Optimize memory
    if get_device() == "cuda":
        optimize_memory()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        print(f"[Whale1.0] Image-to-Video mode: {args.image}")
        pipeline = I2VPipeline()
        result = pipeline.generate(
            image=args.image,
            prompt=args.prompt,
            num_frames=args.num_frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            seed=args.seed,
            output_path=output_dir,
        )
    else:
        print(f"[Whale1.0] Text-to-Video mode: {args.prompt}")
        pipeline = T2VPipeline()
        result = pipeline.generate(
            prompt=args.prompt,
            engine=args.engine,
            num_frames=args.num_frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            output_path=output_dir,
        )

    print(f"\n✅ Video generated!")
    print(f"   Engine: {result['engine']}")
    print(f"   Output: {result['output_path']}")
    print(f"   Resolution: {result.get('resolution', 'N/A')}")
    print(f"   FPS: {result.get('fps', 'N/A')} / Frames: {result.get('num_frames', 'N/A')}")


if __name__ == "__main__":
    main()
