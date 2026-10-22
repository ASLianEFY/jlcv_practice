"""Green-pixel segmentation.

HSV thresholding, a small 3x3 opening pass (which avoids merging the ball and
the green cylinder), and optional ROI limiting.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config


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


