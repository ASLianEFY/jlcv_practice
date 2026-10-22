"""Speed gating and HUD text."""

from __future__ import annotations

from .config import Config
from .tracking import KalmanTracker


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


