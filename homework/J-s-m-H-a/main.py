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
import csv
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
class KalmanTracker:
    """Constant-velocity Kalman tracker for the ball centre.

    The state is ``[cx, cy, vx, vy]`` in image coordinates.  Apparent
    (camera-induced) motion is the signal of interest, matching the
    assignment's image-motion semantics.

    Lifecycle per frame:
        tracker.predict()
        candidate, recovery = tracker.associate(candidates)
        if candidate: tracker.update(candidate, recovery)
        else:         tracker.mark_miss()
    """

    def __init__(
        self,
        cfg: Config,
        cx: float,
        cy: float,
        r: float,
        dt: float,
    ) -> None:
        self.cfg = cfg
        self.dt = float(dt) if dt > 0 else 1.0 / cfg.fps_fallback
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        self.R = np.eye(2) * (cfg.meas_noise_px**2)
        self.Q_base = self._process_noise(cfg.accel_noise)

        self.x = self._initial_state(cx, cy)
        self.P = self._initial_covariance()
        self._x_pred = self.x.copy()
        self._P_pred = self.P.copy()

        self.r = float(r)
        self.hits = 1
        self.misses = 0
        self.updates = 1
        self.confirmed = False
        self.status = "detected"
        self.q_scale = 1.0

    # -- construction helpers ------------------------------------------------
    def _initial_state(self, cx: float, cy: float) -> np.ndarray:
        return np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)

    def _initial_covariance(self) -> np.ndarray:
        return np.diag(
            [
                self.cfg.meas_noise_px**2,
                self.cfg.meas_noise_px**2,
                1e6,
                1e6,
            ]
        )

    def _process_noise(self, sigma_a: float) -> np.ndarray:
        dt = self.dt
        q = sigma_a**2
        return q * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ]
        )

    # -- prediction / association -------------------------------------------
    def predict(self) -> None:
        """Advance the state one frame and store the prior for association."""
        self.x = self.F @ self.x
        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q_base * self.q_scale
        )
        # The adaptive scaling applies to one prediction only.
        self.q_scale = 1.0
        self._x_pred = self.x.copy()
        self._P_pred = self.P.copy()

    def _innovation_stats(self, candidate: Candidate) -> tuple[float, float]:
        z = np.array([[candidate.cx], [candidate.cy]], dtype=np.float64)
        innovation = z - self.H @ self._x_pred
        S = self.H @ self._P_pred @ self.H.T + self.R
        d2 = float((innovation.T @ np.linalg.inv(S) @ innovation)[0, 0])
        euclid = float(np.linalg.norm(innovation))
        return d2, euclid

    def euclidean_gate(self) -> float:
        speed = self.speed()
        return max(
            self.cfg.euclid_radius_factor * self.r,
            self.cfg.euclid_motion_factor * speed * self.dt
            + self.cfg.euclid_extra_px,
        )

    def associate(
        self, candidates: list[Candidate]
    ) -> tuple[Candidate | None, bool]:
        """Pick the best gated candidate.

        The **Euclidean gate** is the hard association gate: it is sized to
        the search ROI (``2.5 * r``).  The Mahalanobis distance ranks the
        remaining candidates and marks a ``recovery`` frame (``d^2`` above the
        threshold), which scales Q on the next prediction.

        A hard Mahalanobis gate would reject the true ball when a hand-held
        camera reverses the apparent direction: the constant-velocity prior
        points the wrong way and the innovation is legitimately large.  Shape,
        radius consistency and the Euclidean gate already reject the green
        cylinder, so the Mahalanobis gate is deliberately soft here.
        """
        gate_e = self.euclidean_gate()
        accepted: list[tuple[float, Candidate]] = []

        for candidate in candidates:
            if abs(candidate.r - self.r) > self.cfg.radius_consistency * self.r:
                continue
            d2, euclid = self._innovation_stats(candidate)
            if euclid <= gate_e:
                accepted.append((d2, candidate))

        if not accepted:
            return None, False

        d2, candidate = min(accepted, key=lambda item: item[0])
        return candidate, d2 > self.cfg.mahalanobis_gate

    # -- correction / lifecycle ---------------------------------------------
    def update(self, candidate: Candidate, recovery: bool = False) -> None:
        """Correct the state with an associated candidate."""
        z = np.array([[candidate.cx], [candidate.cy]], dtype=np.float64)
        innovation = z - self.H @ self._x_pred
        S = self.H @ self._P_pred @ self.H.T + self.R
        K = self._P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = self._x_pred + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self._P_pred

        self.r = (
            (1.0 - self.cfg.radius_ema_alpha) * self.r
            + self.cfg.radius_ema_alpha * candidate.r
        )
        self.hits += 1
        self.misses = 0
        self.updates += 1
        self.confirmed = self.hits >= self.cfg.hits_required
        self.status = "detected"
        self.q_scale = (
            self.cfg.adaptive_q_factor if recovery else 1.0
        )

    def mark_miss(self) -> None:
        """Called when no candidate is associated in this frame."""
        self.misses += 1
        if self.misses > self.cfg.max_misses:
            self.status = "lost"
            self.confirmed = False
        else:
            self.status = "predicted"

    def reacquire(self, cx: float, cy: float, r: float) -> None:
        """Reset the filter on a confident candidate after a loss."""
        self.x = self._initial_state(cx, cy)
        self.P = self._initial_covariance()
        self._x_pred = self.x.copy()
        self._P_pred = self.P.copy()
        self.r = float(r)
        self.hits = 1
        self.misses = 0
        self.updates = 1
        self.confirmed = False
        self.status = "detected"
        self.q_scale = 1.0

    # -- read-only views -----------------------------------------------------
    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    def speed(self) -> float:
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))

    def sigma_v(self) -> float:
        """One-sigma bound on the speed from the velocity covariance."""
        return float(np.sqrt(self.P[2, 2] + self.P[3, 3]))

    def arrow_ready(self) -> bool:
        """Whether the velocity estimate is trustworthy enough to draw."""
        return (
            self.confirmed
            and self.status in ("detected", "predicted")
            and self.updates >= self.cfg.min_updates_arrow
            and self.sigma_v() <= self.cfg.sigma_v_arrow_max
        )

    def roi(self) -> tuple[int, int, int, int] | None:
        """Search region for the next frame; None means full frame."""
        if not self.confirmed or self.status == "lost":
            return None
        cx, cy = self.position()
        pad = self.cfg.roi_radius_factor * self.r
        return (
            int(round(cx - pad)),
            int(round(cy - pad)),
            int(round(cx + pad)),
            int(round(cy + pad)),
        )


# =============================================================================
# 6. Speed gating
# =============================================================================
def clamp_speed(speed: float, cfg: Config) -> float:
    """Clamp a speed to the display/prediction range."""
    return float(min(max(speed, 0.0), cfg.speed_clamp))


def display_speed(tracker: KalmanTracker, cfg: Config) -> float:
    """Speed used for the arrow and the on-screen number."""
    return clamp_speed(tracker.speed(), cfg)


def arrow_visible(tracker: KalmanTracker, cfg: Config) -> bool:
    """Whether the velocity estimate is reliable enough to draw an arrow.

    Combines the Kalman readiness (confirmed, enough updates, small velocity
    covariance) with the minimum-speed threshold.
    """
    return tracker.arrow_ready() and tracker.speed() >= cfg.min_arrow_speed


def speed_text(tracker: KalmanTracker, cfg: Config) -> str:
    """Top-left HUD text for the current frame."""
    if tracker.status == "lost":
        return "v = --"
    value = display_speed(tracker, cfg)
    if tracker.status == "predicted":
        return f"v ≈ {value:.0f} px/s (pred)"
    return f"v = {value:.0f} px/s"


# =============================================================================
# 7. Annotation
# =============================================================================
def arrow_length(speed: float, cfg: Config) -> float:
    """Map speed (px/s) to arrow length (px) with clamps.

    Below the minimum speed no arrow is drawn.  The 0.25 scale keeps a typical
    apparent speed (~380 px/s) at roughly 95 px, which extends beyond the
    ~60 px ball radius.
    """
    if speed < cfg.min_arrow_speed:
        return 0.0
    return float(
        min(max(cfg.arrow_scale * speed, cfg.arrow_min_len), cfg.arrow_max_len)
    )


def dash_segments(
    start: tuple[float, float],
    end: tuple[float, float],
    dash: float,
    gap: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Split a line into dash segments (floats; drawing rounds them)."""
    x0, y0 = start
    x1, y1 = end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0.0 or dash <= 0.0:
        return []
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    offset = 0.0
    while offset < length:
        segment_end = min(offset + dash, length)
        segments.append(
            (
                (x0 + ux * offset, y0 + uy * offset),
                (x0 + ux * segment_end, y0 + uy * segment_end),
            )
        )
        offset += dash + gap
    return segments


def _draw_arrow(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    cfg: Config,
    dashed: bool = False,
) -> None:
    if not dashed:
        cv2.arrowedLine(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (0, 0, 0),
            cfg.arrow_thickness + 2,
            cv2.LINE_AA,
            tipLength=cfg.arrow_tip,
        )
        cv2.arrowedLine(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            cfg.arrow_thickness,
            cv2.LINE_AA,
            tipLength=cfg.arrow_tip,
        )
        return

    segments = dash_segments(start, end, cfg.dash_length, cfg.dash_gap)
    for p0, p1 in segments:  # outline pass
        cv2.line(
            frame,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (0, 0, 0),
            cfg.arrow_thickness + 2,
            cv2.LINE_AA,
        )
    for p0, p1 in segments:  # colour pass
        cv2.line(
            frame,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            color,
            cfg.arrow_thickness,
            cv2.LINE_AA,
        )

    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if length > 0:
        ux = (end[0] - start[0]) / length
        uy = (end[1] - start[1]) / length
        head_len = min(16.0, 0.25 * length)
        head_start = (end[0] - ux * head_len, end[1] - uy * head_len)
        cv2.arrowedLine(
            frame,
            (int(round(head_start[0])), int(round(head_start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (0, 0, 0),
            cfg.arrow_thickness + 2,
            cv2.LINE_AA,
            tipLength=0.5,
        )
        cv2.arrowedLine(
            frame,
            (int(round(head_start[0])), int(round(head_start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            cfg.arrow_thickness,
            cv2.LINE_AA,
            tipLength=0.5,
        )


def _draw_text_panel(frame: np.ndarray, text: str, cfg: Config) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, cfg.font_scale, cfg.font_thickness
    )
    x0, y0 = 8, 8
    x1 = x0 + text_w + 16
    y1 = y0 + text_h + baseline + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), cfg.color_panel, -1)
    cv2.addWeighted(
        overlay, cfg.panel_alpha, frame, 1.0 - cfg.panel_alpha, 0.0, frame
    )
    cv2.putText(
        frame,
        text,
        (x0 + 8, y0 + text_h + 6),
        font,
        cfg.font_scale,
        cfg.color_text,
        cfg.font_thickness,
        cv2.LINE_AA,
    )


def draw_annotation(
    frame: np.ndarray,
    tracker: KalmanTracker,
    cfg: Config,
    trail_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw the ball, arrow and HUD on ``frame`` in place.

    Draw order is trail -> circle -> label -> arrow -> text, so the arrow is
    always on top of the ball (option a: centre-anchored, longer than the
    radius).  Lost frames show only the HUD.
    """
    if tracker.status == "lost":
        _draw_text_panel(frame, speed_text(tracker, cfg), cfg)
        return frame

    cx, cy = tracker.position()
    radius = max(1, int(round(tracker.r)))
    predicted = tracker.status == "predicted"
    body_color = cfg.color_pred if predicted else cfg.color_ball

    if trail_points:
        recent = [
            (int(round(x)), int(round(y)))
            for x, y in trail_points[-cfg.trail_length :]
        ]
        if len(recent) >= 2:
            cv2.polylines(
                frame,
                [np.array(recent, dtype=np.int32)],
                False,
                cfg.color_trail,
                1,
                cv2.LINE_AA,
            )

    centre = (int(round(cx)), int(round(cy)))
    cv2.circle(frame, centre, radius, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, centre, radius, body_color, 2, cv2.LINE_AA)

    label_x = int(round(cx + radius + 4))
    if label_x + 40 > frame.shape[1]:
        label_x = int(round(cx - radius - 40))
    label_pos = (label_x, int(round(cy)))
    cv2.putText(
        frame,
        "ball",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.label_scale,
        (0, 0, 0),
        cfg.label_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "ball",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.label_scale,
        body_color,
        cfg.label_thickness,
        cv2.LINE_AA,
    )

    if arrow_visible(tracker, cfg):
        speed = display_speed(tracker, cfg)
        length = arrow_length(speed, cfg)
        vx, vy = tracker.velocity()
        norm = float(np.hypot(vx, vy))
        if length > 0.0 and norm > 1e-6:
            end = (cx + length * vx / norm, cy + length * vy / norm)
            arrow_color = cfg.color_pred if predicted else cfg.color_arrow
            _draw_arrow(
                frame,
                (cx, cy),
                end,
                arrow_color,
                cfg,
                dashed=predicted,
            )

    _draw_text_panel(frame, speed_text(tracker, cfg), cfg)
    return frame


# =============================================================================
# 8. Export
# =============================================================================
CSV_COLUMNS = (
    "frame",
    "time_s",
    "cx",
    "cy",
    "r",
    "vx",
    "vy",
    "speed_px_s",
    "status",
)


@dataclass
class FrameRow:
    """One CSV row (one video frame)."""

    frame: int
    time_s: float
    cx: float
    cy: float
    r: float
    vx: float
    vy: float
    speed_px_s: float
    status: str


@dataclass
class PipelineStats:
    """Aggregate numbers printed at the end of a run."""

    frames: int
    detected: int
    predicted: int
    lost: int
    detection_rate: float
    max_lost_streak: int
    speed_median: float
    speed_p90: float
    output_video: str


def write_csv(path: Path, rows: list[FrameRow]) -> None:
    """Write the per-frame CSV (``nan`` for lost frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.frame,
                    f"{row.time_s:.3f}",
                    f"{row.cx:.2f}",
                    f"{row.cy:.2f}",
                    f"{row.r:.2f}",
                    f"{row.vx:.2f}",
                    f"{row.vy:.2f}",
                    f"{row.speed_px_s:.2f}",
                    row.status,
                ]
            )


def open_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path]:
    """Open an MP4 writer, falling back to XVID/AVI when necessary."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create output directory: {path.parent}") from exc

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if writer.isOpened():
        return writer, path
    writer.release()

    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, size
    )
    if writer.isOpened():
        return writer, fallback
    writer.release()
    raise RuntimeError(
        f"cannot open a video writer for {path} (mp4v and XVID both failed)"
    )


def make_contact_sheet(
    frames: list[np.ndarray],
    cols: int,
    tile_size: tuple[int, int],
) -> np.ndarray:
    """Tile annotated frames into a single review image."""
    if not frames:
        return np.zeros((0, 0, 3), np.uint8)
    width, height = tile_size
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * height, cols * width, 3), 255, np.uint8)
    for index, frame in enumerate(frames):
        tile = cv2.resize(frame, (width, height))
        row, col = divmod(index, cols)
        sheet[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = tile
    return sheet


def sample_frame_path(out_dir: Path, index: int) -> Path:
    return out_dir / "frames" / f"frame_{index:04d}.png"


def _save_debug(
    out_dir: Path,
    frame: np.ndarray,
    mask: np.ndarray,
    candidates: list[Candidate],
    roi: tuple[int, int, int, int] | None,
    index: int,
) -> None:
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"mask_{index:04d}.png"), mask)

    overlay = frame.copy()
    if roi is not None:
        cv2.rectangle(
            overlay,
            (int(roi[0]), int(roi[1])),
            (int(roi[2]), int(roi[3])),
            (255, 0, 0),
            1,
        )
    for candidate in candidates:
        cv2.circle(
            overlay,
            (int(round(candidate.cx)), int(round(candidate.cy))),
            int(round(candidate.r)),
            (0, 0, 255),
            1,
        )
    cv2.imwrite(str(debug_dir / f"candidates_{index:04d}.png"), overlay)

    lines = [f"roi={roi}"]
    for candidate in candidates:
        lines.append(
            f"{candidate.source} c=({candidate.cx:.1f},{candidate.cy:.1f}) "
            f"r={candidate.r:.1f} score={candidate.score:.3f} "
            f"below={candidate.below_green:.2f}"
        )
    (debug_dir / f"info_{index:04d}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _track_frame(
    frame: np.ndarray,
    tracker: KalmanTracker | None,
    cfg: Config,
    dt: float,
) -> tuple[KalmanTracker | None, list[Candidate], tuple[int, int, int, int] | None]:
    """Advance the tracker by one frame; returns tracker, candidates, ROI."""
    if tracker is None:
        candidates = detect_candidates(frame, cfg)
        if candidates:
            best = candidates[0]
            tracker = KalmanTracker(
                cfg, best.cx, best.cy, best.r, dt
            )
        return tracker, candidates, None

    tracker.predict()
    roi = tracker.roi()
    candidates = detect_candidates(
        frame, cfg, roi=roi, r_hint=tracker.r
    )
    chosen, recovery = tracker.associate(candidates)
    if chosen is not None:
        tracker.update(chosen, recovery)
    else:
        tracker.mark_miss()
        if tracker.status == "lost":
            candidates = detect_candidates(frame, cfg, r_hint=tracker.r)
            if candidates:
                best = candidates[0]
                tracker.reacquire(best.cx, best.cy, best.r)
    return tracker, candidates, roi


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    cfg: Config,
    no_frames: bool = False,
    trail: bool = False,
    debug: bool = False,
) -> tuple[PipelineStats, list[FrameRow]]:
    """Run detection, tracking, annotation and export for one video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or cfg.fps_fallback
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt = 1.0 / fps

    out_dir.mkdir(parents=True, exist_ok=True)
    writer, video_out = open_writer(
        out_dir / "annotated.mp4", fps, (width, height)
    )

    if frame_count > 0 and not no_frames:
        contact_indices = {
            int(round(value))
            for value in np.linspace(
                0, max(0, frame_count - 1), cfg.contact_sheet_frames
            )
        }
    else:
        contact_indices = set()

    tracker: KalmanTracker | None = None
    trail_buffer: list[tuple[float, float]] = []
    rows: list[FrameRow] = []
    contact_frames: list[np.ndarray] = []
    speeds: list[float] = []
    counts = {"detected": 0, "predicted": 0, "lost": 0}
    lost_streak = 0
    max_lost_streak = 0
    index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        tracker, candidates, roi = _track_frame(frame, tracker, cfg, dt)

        if tracker is None:
            status = "lost"
            cx = cy = r = vx = vy = speed = float("nan")
            _draw_text_panel(frame, "v = --", cfg)
            trail_buffer.clear()
        else:
            status = tracker.status
            if status == "lost":
                cx = cy = r = vx = vy = speed = float("nan")
                trail_buffer.clear()
            else:
                cx, cy = tracker.position()
                r = tracker.r
                vx, vy = tracker.velocity()
                speed = float(np.hypot(vx, vy))
                speeds.append(speed)
                trail_buffer.append((cx, cy))
                if len(trail_buffer) > cfg.trail_length:
                    trail_buffer = trail_buffer[-cfg.trail_length :]
            draw_annotation(
                frame,
                tracker,
                cfg,
                trail_points=trail_buffer if trail else None,
            )

        counts[status] += 1
        if status == "lost":
            lost_streak += 1
            max_lost_streak = max(max_lost_streak, lost_streak)
        else:
            lost_streak = 0

        rows.append(
            FrameRow(
                frame=index,
                time_s=index / fps,
                cx=cx,
                cy=cy,
                r=r,
                vx=vx,
                vy=vy,
                speed_px_s=speed,
                status=status,
            )
        )

        if not no_frames and index in cfg.sample_frames:
            path = sample_frame_path(out_dir, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)

        if not no_frames and index in contact_indices:
            contact_frames.append(frame.copy())

        if debug and index in cfg.sample_frames:
            _save_debug(
                out_dir,
                frame,
                green_mask(frame, cfg),
                candidates,
                roi,
                index,
            )

        writer.write(frame)
        index += 1

    capture.release()
    writer.release()

    write_csv(out_dir / "speed.csv", rows)

    if not no_frames and contact_frames:
        sheet = make_contact_sheet(
            contact_frames, cfg.contact_sheet_cols, cfg.contact_tile_size
        )
        cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)

    speed_array = np.array(speeds, dtype=float) if speeds else np.zeros(1)
    stats = PipelineStats(
        frames=len(rows),
        detected=counts["detected"],
        predicted=counts["predicted"],
        lost=counts["lost"],
        detection_rate=counts["detected"] / max(1, len(rows)),
        max_lost_streak=max_lost_streak,
        speed_median=float(np.median(speed_array)),
        speed_p90=float(np.percentile(speed_array, 90)),
        output_video=str(video_out),
    )

    print(
        "frames={frames} detected={detected} predicted={predicted} "
        "lost={lost} detection_rate={rate:.1%} max_lost_streak={streak}".format(
            frames=stats.frames,
            detected=stats.detected,
            predicted=stats.predicted,
            lost=stats.lost,
            rate=stats.detection_rate,
            streak=stats.max_lost_streak,
        )
    )
    print(
        "speed px/s: median={median:.1f} p90={p90:.1f}".format(
            median=stats.speed_median, p90=stats.speed_p90
        )
    )
    print(f"output: {stats.output_video}")

    return stats, rows


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
