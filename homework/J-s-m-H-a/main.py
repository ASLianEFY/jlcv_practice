"""CLI entry point for the week1 green-ball tracker.

The implementation lives in the :mod:`greenball` package; this module only
parses arguments and runs the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from greenball.config import DEFAULT_OUT_DIR, DEFAULT_VIDEO, Config
from greenball.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect the green ball and annotate its image-space motion.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="input video (default: repository root homework.mp4)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default: homework/J-s-m-H-a/output)",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="skip sample frames and the contact sheet",
    )
    parser.add_argument(
        "--trail",
        action="store_true",
        help="draw the last N ball centres as a thin trail",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="dump masks, candidates and ROI to <out-dir>/debug",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments and run the pipeline."""
    args = parse_args(argv)
    if not args.video.exists():
        print(f"error: video not found: {args.video}")
        return 1
    run_pipeline(
        args.video,
        args.out_dir,
        Config(),
        no_frames=args.no_frames,
        trail=args.trail,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
