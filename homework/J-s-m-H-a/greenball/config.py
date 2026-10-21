"""Configuration for the green-ball tracker.

All tunable parameters live in :class:`Config`.  Paths are resolved relative
to the repository layout so the CLI works from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    euclid_radius_factor: float = 2.5
    euclid_motion_factor: float = 1.5
    euclid_extra_px: float = 10.0
    radius_consistency: float = 0.30
    hits_required: int = 2
    max_misses: int = 5
    adaptive_q_factor: float = 4.0
    radius_ema_alpha: float = 0.3

    # --- 6. speed gating -----------------------------------------------------
    min_arrow_speed: float = 60.0
    speed_clamp: float = 2000.0
    sigma_v_arrow_max: float = 300.0
    min_updates_arrow: int = 3

    # --- 7. annotation -------------------------------------------------------
    arrow_scale: float = 0.25
    arrow_min_len: float = 20.0
    arrow_max_len: float = 220.0
    arrow_thickness: int = 3
    arrow_tip: float = 0.25
    trail_length: int = 15
    color_ball: tuple[int, int, int] = (0, 255, 255)
    color_arrow: tuple[int, int, int] = (0, 255, 255)
    color_pred: tuple[int, int, int] = (128, 128, 128)
    color_trail: tuple[int, int, int] = (255, 200, 0)
    color_text: tuple[int, int, int] = (255, 255, 255)
    color_panel: tuple[int, int, int] = (30, 30, 30)
    panel_alpha: float = 0.55
    font_scale: float = 0.7
    font_thickness: int = 2
    label_scale: float = 0.5
    label_thickness: int = 1
    dash_length: float = 8.0
    dash_gap: float = 6.0

    # --- 8. export -----------------------------------------------------------
    sample_frames: tuple[int, ...] = (0, 90, 180, 270, 360, 435)
    contact_sheet_frames: int = 16
    contact_sheet_cols: int = 4
    contact_tile_size: tuple[int, int] = (320, 180)
    fps_fallback: float = 30.0


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO = REPO_ROOT / "homework.mp4"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "output"
