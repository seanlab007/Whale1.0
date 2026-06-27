"""Whale1.0 CLI entry point: python -m whale_video [command]

Commands:
  serve    Start the FastAPI server (default)
  cli      Run the CLI pipeline
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI pipeline
        from whale_video.server import main as server_main
        print("[Whale] Use `python -m whale_video.server` for the API server")
        print("[Whale] Use `python scripts/run_pipeline.py` for the CLI")
        sys.exit(0)
    else:
        # Default: start server
        from whale_video.server import main as server_main
        server_main()


if __name__ == "__main__":
    main()
