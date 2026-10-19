"""U5: arrow length mapping, draw order, dashed predictions and HUD."""

from __future__ import annotations

import numpy as np

from greenball.annotation import arrow_length, dash_segments, draw_annotation
from greenball.config import Config
from greenball.tracking import KalmanTracker

DT = 1.0 / 30.0
GREEN = (60, 180, 80)


def ready_tracker(
    vx: float = 400.0,
    status: str = "detected",
    cx: float = 100.0,
    cy: float = 100.0,
    r: float = 20.0,
) -> KalmanTracker:
    tracker = KalmanTracker(Config(), cx, cy, r, DT)
    tracker.confirmed = True
    tracker.hits = 5
    tracker.updates = 5
    tracker.status = status
    tracker.x[0, 0] = cx
    tracker.x[1, 0] = cy
    tracker.x[2, 0] = vx
    tracker.x[3, 0] = 0.0
    tracker.P[2, 2] = 100.0
    tracker.P[3, 3] = 100.0
    return tracker


def green_frame() -> np.ndarray:
    return np.full((240, 320, 3), GREEN, np.uint8)


def has_colour(patch: np.ndarray, colour: tuple[int, int, int], tol: int = 20) -> bool:
    target = np.array(colour)
    return bool(
        np.any(np.all(np.abs(patch.astype(int) - target) <= tol, axis=2))
    )


def test_arrow_length_mapping() -> None:
    cfg = Config()
    assert arrow_length(0.0, cfg) == 0.0
    assert arrow_length(50.0, cfg) == 0.0
    assert arrow_length(60.0, cfg) == cfg.arrow_min_len
    assert arrow_length(100.0, cfg) == 25.0
    assert arrow_length(400.0, cfg) == 100.0
    assert arrow_length(2000.0, cfg) == cfg.arrow_max_len


def test_dash_segments_cover_the_line() -> None:
    segments = dash_segments((0.0, 0.0), (100.0, 0.0), dash=8.0, gap=6.0)

    assert segments
    assert segments[0][0] == (0.0, 0.0)
    assert abs(segments[-1][1][0] - 100.0) < 1e-9
    for (start, end), (next_start, _) in zip(segments, segments[1:]):
        assert 0.0 <= end[0] - start[0] <= 8.0 + 1e-9
        assert abs(next_start[0] - end[0] - 6.0) < 1e-9


def test_arrow_is_drawn_on_top_of_the_ball() -> None:
    cfg = Config()
    image = green_frame()
    tracker = ready_tracker(vx=400.0)

    draw_annotation(image, tracker, cfg)

    # (110, 100) is inside the r=20 ball, where only the arrow can paint yellow.
    patch = image[96:105, 105:116]
    assert has_colour(patch, cfg.color_arrow)
    # HUD panel darkens the top-left corner.
    assert int(image[10, 10].sum()) < 300


def test_predicted_arrow_is_gray_and_dashed() -> None:
    cfg = Config()
    image = green_frame()
    tracker = ready_tracker(vx=400.0, status="predicted")

    draw_annotation(image, tracker, cfg)

    # Outside the ball (x >= 130), only the arrow can paint gray.
    patch = image[95:106, 130:170]
    assert has_colour(patch, cfg.color_pred, tol=30)


def test_lost_frame_draws_no_ball_or_arrow() -> None:
    cfg = Config()
    image = green_frame()
    tracker = ready_tracker(vx=400.0, status="lost")

    draw_annotation(image, tracker, cfg)

    assert tuple(image[100, 150]) == GREEN


def test_trail_is_drawn() -> None:
    cfg = Config()
    image = green_frame()
    tracker = ready_tracker(vx=400.0)
    trail = [(40.0, 40.0), (50.0, 50.0), (60.0, 60.0), (70.0, 70.0)]

    draw_annotation(image, tracker, cfg, trail_points=trail)

    assert has_colour(image[55:66, 55:66], cfg.color_trail, tol=30)
