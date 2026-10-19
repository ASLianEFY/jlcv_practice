"""U1: HSV segmentation, opening and ROI behaviour."""

from __future__ import annotations

import cv2
import numpy as np

from greenball.config import Config
from greenball.segmentation import clamp_roi, green_mask

GREEN = (60, 180, 80)  # BGR -> H~70, S~170, V~180
RED = (40, 40, 200)
WHITE = (240, 240, 240)
BACKGROUND = (100, 120, 140)  # low saturation, not green


def make_scene() -> np.ndarray:
    """Ball at (80, 80) r=20; cylinder at x=200..240; distractor colours."""
    image = np.full((240, 320, 3), BACKGROUND, np.uint8)
    cv2.circle(image, (80, 80), 20, GREEN, -1)
    cv2.rectangle(image, (200, 40), (240, 180), GREEN, -1)
    cv2.circle(image, (80, 190), 18, RED, -1)
    cv2.rectangle(image, (150, 150), (185, 200), WHITE, -1)
    image[10:12, 10:12] = GREEN  # isolated 2x2 speck
    return image


def test_green_objects_are_kept() -> None:
    mask = green_mask(make_scene(), Config())
    assert mask[80, 80] == 255  # ball centre
    assert mask[100, 220] == 255  # cylinder body


def test_non_green_objects_are_rejected() -> None:
    mask = green_mask(make_scene(), Config())
    assert mask[190, 80] == 0  # red circle
    assert mask[175, 165] == 0  # white rectangle
    assert mask[230, 300] == 0  # background


def test_opening_removes_isolated_speck() -> None:
    mask = green_mask(make_scene(), Config())
    assert int(mask[10:12, 10:12].sum()) == 0


def test_roi_limits_the_mask() -> None:
    mask = green_mask(make_scene(), Config(), roi=(40, 40, 130, 130))
    assert mask[80, 80] == 255
    assert mask[100, 220] == 0


def test_mask_is_binary() -> None:
    mask = green_mask(make_scene(), Config())
    assert set(np.unique(mask)).issubset({0, 255})


def test_clamp_roi() -> None:
    assert clamp_roi((-10, -5, 400, 300), 320, 240) == (0, 0, 320, 240)
    assert clamp_roi((50, 60, 70, 80), 320, 240) == (50, 60, 70, 80)
