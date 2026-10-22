"""Ball candidate detection.

Contour proposals are filtered by two circularity metrics, aspect ratio, the
below-green ratio (which rejects the same-coloured cylinder) and radius
consistency.  A half-resolution Hough fallback handles merged masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .segmentation import clamp_roi, green_mask


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

