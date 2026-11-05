"""Run from the repository root: python run_web.py."""

import argparse
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from web.app import build_app, launch_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuYu Web demonstration")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(max_size=16).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        run_history=False,
        **launch_options(),
    )
