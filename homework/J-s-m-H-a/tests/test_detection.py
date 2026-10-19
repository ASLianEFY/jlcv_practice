"""U2: ball-vs-cylinder discrimination, Hough fallback and radius adaptivity."""

from __future__ import annotations

import cv2
import numpy as np

from greenball.config import Config
from greenball.detection import below_green_ratio, detect_candidates

GREEN = (60, 180, 80)
RED = (40, 40, 200)
WHITE = (240, 240, 240)
BACKGROUND = (100, 120, 140)


def canvas(height: int = 260, width: int = 360) -> np.ndarray:
    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    cv2.circle(image, (40, 220), 18, RED, -1)
    cv2.rectangle(image, (300, 180), (340, 230), WHITE, -1)
    return image


def near(candidates, x: float, y: float, tol: float = 8.0):
    return [c for c in candidates if abs(c.cx - x) <= tol and abs(c.cy - y) <= tol]


def test_ball_accepted_and_tall_cylinder_rejected() -> None:
    image = canvas()
    cv2.circle(image, (80, 80), 20, GREEN, -1)  # ball
    cv2.rectangle(image, (220, 40), (250, 180), GREEN, -1)  # cylinder

    candidates = detect_candidates(image, Config())

    assert near(candidates, 80, 80), "ball should be detected"
    assert not near(candidates, 235, 110, tol=20), "cylinder must be rejected"


def test_ball_in_front_of_cylinder_is_kept() -> None:
    """The ball may sit below a green cylinder: above-green is high, below is not."""
    image = canvas()
    cv2.rectangle(image, (140, 20), (180, 120), GREEN, -1)  # cylinder body
    cv2.circle(image, (160, 150), 20, GREEN, -1)  # ball below it

    candidates = detect_candidates(image, Config())

    assert near(candidates, 160, 150), "ball in front of the cylinder must be kept"


def test_touching_ball_uses_hough_fallback() -> None:
    """A merged ball+cylinder contour has no valid shape; Hough recovers the ball."""
    image = canvas()
    cv2.rectangle(image, (140, 20), (180, 120), GREEN, -1)
    cv2.circle(image, (160, 135), 20, GREEN, -1)  # overlaps the cylinder bottom

    candidates = detect_candidates(image, Config())

    balls = near(candidates, 160, 135, tol=10)
    assert balls, "Hough fallback should recover the touching ball"
    assert any(abs(c.r - 20) <= 6 for c in balls)
    assert any(c.source == "hough" for c in balls)


def test_radius_hint_rejects_oversized_candidate() -> None:
    image = canvas()
    cv2.circle(image, (80, 80), 20, GREEN, -1)  # ball
    cv2.circle(image, (270, 80), 45, GREEN, -1)  # large green blob

    without_hint = detect_candidates(image, Config())
    assert near(without_hint, 80, 80)
    assert near(without_hint, 270, 80, tol=10)

    with_hint = detect_candidates(image, Config(), r_hint=20.0)
    assert near(with_hint, 80, 80)
    assert not near(with_hint, 270, 80, tol=10)


def test_below_green_ratio_separates_ball_from_cylinder() -> None:
    mask = np.zeros((260, 360), np.uint8)
    cv2.circle(mask, (100, 100), 20, 255, -1)
    assert below_green_ratio(mask, 100, 100, 20) < 0.1

    cv2.rectangle(mask, (80, 122), (120, 220), 255, -1)  # cylinder body
    assert below_green_ratio(mask, 100, 100, 20) > 0.6


def test_roi_restricts_detection() -> None:
    image = canvas()
    cv2.circle(image, (80, 80), 20, GREEN, -1)
    cv2.circle(image, (270, 80), 20, GREEN, -1)

    candidates = detect_candidates(image, Config(), roi=(40, 40, 130, 130))

    assert near(candidates, 80, 80)
    assert not near(candidates, 270, 80, tol=10)
