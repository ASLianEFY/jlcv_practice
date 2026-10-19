"""week1 — green ball detection and image-motion annotation.

The file is intentionally a single module.  Each section below is delimited
with a banner so the pipeline stage is easy to locate:

    1. Configuration
    2. Video I/O
    3. Green segmentation
    4. Ball candidate detection
    5. Kalman tracking
    6. Speed gating
    7. Annotation
    8. Export
    9. CLI

Only section 1 and 9 exist in this commit; later commits fill in the rest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2  # noqa: F401  (used from section 2 onward)
import numpy as np  # noqa: F401  (used from section 3 onward)


# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    """All tunable parameters.

    Keep every magic number here so that tuning is a one-line, separately
    reviewable change.
    """

    # --- 3. segmentation -----------------------------------------------------
    hsv_low: tuple[int, int, int] = (40, 80, 40)
    hsv_high: tuple[int, int, int] = (90, 255, 255)
    morph_kernel: int = 3

    # --- 4. candidate detection ---------------------------------------------
    area_min_global: float = 800.0
    area_max_global: float = 40000.0
    radius_scale_min: float = 0.7
    radius_scale_max: float = 1.4
    circularity_perimeter_min: float = 0.6
    circularity_enclosing_min: float = 0.6
    aspect_min: float = 0.7
    aspect_max: float = 1.4
    below_green_max: float = 0.6
    hough_downscale: float = 0.5
    hough_dp: float = 1.2
    hough_param1: float = 100.0
    hough_param2: float = 22.0
    hough_radius_min_factor: float = 0.6
    hough_radius_max_factor: float = 1.3
    hough_green_fill_min: float = 0.6
    roi_radius_factor: float = 2.5

    # --- 5. Kalman tracking --------------------------------------------------
    meas_noise_px: float = 3.0
    accel_noise: float = 500.0
    mahalanobis_gate: float = 9.21
    euclid_radius_factor: float = 1.5
    euclid_motion_factor: float = 1.5
    euclid_extra_px: float = 10.0
    radius_consistency: float = 0.30
    hits_required: int = 2
    max_misses: int = 5
    adaptive_q_factor: float = 4.0
    radius_ema_alpha: float = 0.3

    # --- 6./7. speed gating and annotation -----------------------------------
    min_arrow_speed: float = 60.0
    speed_clamp: float = 2000.0
    sigma_v_arrow_max: float = 300.0
    min_updates_arrow: int = 3
    arrow_scale: float = 0.25
    arrow_min_len: float = 20.0
    arrow_max_len: float = 220.0
    arrow_thickness: int = 3
    arrow_tip: float = 0.25
    trail_length: int = 15

    # --- 8. export -----------------------------------------------------------
    sample_frames: tuple[int, ...] = (0, 90, 180, 270, 360, 435)
    contact_sheet_frames: int = 16
    contact_sheet_cols: int = 4
    fps_fallback: float = 30.0


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REPO_ROOT / "homework.mp4"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"


# =============================================================================
# 2. Video I/O
# =============================================================================


# =============================================================================
# 3. Green segmentation
# =============================================================================
def clamp_roi(
    roi: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a half-open ``(x0, y0, x1, y1)`` ROI to the image bounds."""
    x0, y0, x1, y1 = (int(round(v)) for v in roi)
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def green_mask(
    bgr: np.ndarray,
    cfg: Config,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Segment green pixels as a binary ``uint8`` mask (0 or 255).

    A small 3x3 opening removes isolated specks without merging the ball and
    the green cylinder when they touch.  When ``roi`` is given, only that
    half-open region is retained; everything outside is zero.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_low), np.array(cfg.hsv_high))

    if cfg.morph_kernel > 1:
        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if roi is not None:
        x0, y0, x1, y1 = clamp_roi(roi, bgr.shape[1], bgr.shape[0])
        limited = np.zeros_like(mask)
        if x1 > x0 and y1 > y0:
            limited[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = limited

    return mask


# =============================================================================
# 4. Ball candidate detection
# =============================================================================


# =============================================================================
# 5. Kalman tracking
# =============================================================================


# =============================================================================
# 6. Speed gating
# =============================================================================


# =============================================================================
# 7. Annotation
# =============================================================================


# =============================================================================
# 8. Export
# =============================================================================


# =============================================================================
# 9. CLI
# =============================================================================
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
    """Entry point.  The pipeline is filled in by later commits."""
    args = parse_args(argv)
    print("week1 green ball tracker: pipeline not implemented yet")
    print(f"video={args.video}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
