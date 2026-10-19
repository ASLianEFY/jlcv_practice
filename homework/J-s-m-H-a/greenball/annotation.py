"""Annotation drawing.

Draws the ball outline, the centre-anchored speed arrow (option a), dashed
predicted arrows and the top-left HUD.  The arrow is drawn after the ball so
it stays on top.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .speed import arrow_visible, display_speed, speed_text
from .tracking import KalmanTracker


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


def draw_text_panel(frame: np.ndarray, text: str, cfg: Config) -> None:
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
        draw_text_panel(frame, speed_text(tracker, cfg), cfg)
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

    draw_text_panel(frame, speed_text(tracker, cfg), cfg)
    return frame


