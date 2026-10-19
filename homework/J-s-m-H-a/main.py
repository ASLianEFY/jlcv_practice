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
@dataclass
class Candidate:
    """One green circular proposal plus the features used to score it."""

    cx: float
    cy: float
    r: float
    area: float
    circ_perimeter: float
    circ_enclosing: float
    aspect: float
    below_green: float
    green_fill: float
    score: float
    source: str  # "contour" or "hough"


def below_green_ratio(mask: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Fraction of green pixels in the strip just below a candidate circle.

    A green cylinder top has its body below it, so the ratio is high; the ball
    sits on the wooden table, so the ratio is low.  This is the main geometric
    cue that separates the two same-coloured objects.
    """
    height, width = mask.shape[:2]
    x0 = max(0, int(round(cx - 0.8 * r)))
    x1 = min(width, int(round(cx + 0.8 * r)))
    y0 = max(0, int(round(cy + 1.0 * r)))
    y1 = min(height, int(round(cy + 2.2 * r)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.mean(mask[y0:y1, x0:x1] > 0))


def _green_fill(mask: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Fraction of green pixels inside a circle (used for Hough candidates)."""
    height, width = mask.shape[:2]
    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(width, int(np.ceil(cx + r)) + 1)
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(height, int(np.ceil(cy + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not disk.any():
        return 0.0
    return float(np.mean(mask[y0:y1, x0:x1][disk] > 0))


def _radius_filter(r: float, cfg: Config, r_hint: float | None) -> bool:
    if r_hint is not None:
        return (
            cfg.radius_scale_min * r_hint <= r <= cfg.radius_scale_max * r_hint
        )
    area = np.pi * r * r
    return cfg.area_min_global <= area <= cfg.area_max_global


def _contour_candidates(
    mask: np.ndarray,
    full_mask: np.ndarray,
    cfg: Config,
    r_hint: float | None,
) -> list[Candidate]:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.area_min_global:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(contour)
        if r <= 0 or not _radius_filter(r, cfg, r_hint):
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circ_perimeter = 4.0 * np.pi * area / (perimeter * perimeter)
        circ_enclosing = area / (np.pi * r * r)

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue
        aspect = w / float(h)
        below = below_green_ratio(full_mask, cx, cy, r)

        if circ_perimeter < cfg.circularity_perimeter_min:
            continue
        if circ_enclosing < cfg.circularity_enclosing_min:
            continue
        if not (cfg.aspect_min <= aspect <= cfg.aspect_max):
            continue
        if below >= cfg.below_green_max:
            continue

        score = (
            0.5 * circ_perimeter
            + 0.5 * circ_enclosing
            + (1.0 - below)
            - 0.5 * abs(aspect - 1.0)
        )
        candidates.append(
            Candidate(
                cx=cx,
                cy=cy,
                r=r,
                area=area,
                circ_perimeter=circ_perimeter,
                circ_enclosing=circ_enclosing,
                aspect=aspect,
                below_green=below,
                green_fill=circ_enclosing,
                score=score,
                source="contour",
            )
        )
    return candidates


def _hough_candidates(
    mask: np.ndarray,
    full_mask: np.ndarray,
    cfg: Config,
    r_hint: float | None,
) -> list[Candidate]:
    """Fallback for merged masks (ball touching the green cylinder).

    Hough runs on a half-resolution blurred mask; radius bounds are tied to
    the tracked radius when available.
    """
    scale = cfg.hough_downscale
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (9, 9), 2)

    if r_hint is not None:
        min_radius = max(4, int(cfg.hough_radius_min_factor * r_hint * scale))
        max_radius = max(
            min_radius + 1, int(cfg.hough_radius_max_factor * r_hint * scale)
        )
        min_dist = max(10, int(1.5 * r_hint * scale))
    else:
        min_radius = max(4, int(np.sqrt(cfg.area_min_global / np.pi) * scale))
        max_radius = max(
            min_radius + 1, int(np.sqrt(cfg.area_max_global / np.pi) * scale)
        )
        min_dist = max(10, 2 * min_radius)

    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    out: list[Candidate] = []
    for cx_s, cy_s, r_s in circles[0]:
        cx, cy, r = float(cx_s / scale), float(cy_s / scale), float(r_s / scale)
        if r <= 0 or not _radius_filter(r, cfg, r_hint):
            continue
        green_fill = _green_fill(full_mask, cx, cy, r)
        if green_fill < cfg.hough_green_fill_min:
            continue
        below = below_green_ratio(full_mask, cx, cy, r)
        if below >= cfg.below_green_max:
            continue
        score = green_fill + (1.0 - below)
        out.append(
            Candidate(
                cx=cx,
                cy=cy,
                r=r,
                area=float(np.pi * r * r),
                circ_perimeter=0.0,
                circ_enclosing=green_fill,
                aspect=1.0,
                below_green=below,
                green_fill=green_fill,
                score=score,
                source="hough",
            )
        )
    return out


def detect_candidates(
    bgr: np.ndarray,
    cfg: Config,
    roi: tuple[int, int, int, int] | None = None,
    r_hint: float | None = None,
) -> list[Candidate]:
    """Return ball candidates, best first.

    ``roi`` limits where contours/Hough are searched; the below-green and fill
    features are always evaluated on the full-frame mask.  ``r_hint`` is the
    tracked radius, which tightens the size filter once a track exists.
    """
    full_mask = green_mask(bgr, cfg)
    if roi is None:
        search_mask = full_mask
    else:
        x0, y0, x1, y1 = clamp_roi(roi, bgr.shape[1], bgr.shape[0])
        search_mask = np.zeros_like(full_mask)
        if x1 > x0 and y1 > y0:
            search_mask[y0:y1, x0:x1] = full_mask[y0:y1, x0:x1]

    candidates = _contour_candidates(search_mask, full_mask, cfg, r_hint)
    if not candidates:
        candidates = _hough_candidates(search_mask, full_mask, cfg, r_hint)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


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
