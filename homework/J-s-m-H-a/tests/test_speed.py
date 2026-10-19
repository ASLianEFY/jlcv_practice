"""U4: speed clamp, early suppression and HUD text."""

from __future__ import annotations

from main import (
    Config,
    KalmanTracker,
    arrow_visible,
    clamp_speed,
    display_speed,
    speed_text,
)

DT = 1.0 / 30.0


def ready_tracker(
    vx: float,
    vy: float = 0.0,
    status: str = "detected",
    updates: int = 5,
    velocity_var: float = 100.0,
) -> KalmanTracker:
    tracker = KalmanTracker(Config(), 100.0, 100.0, 20.0, DT)
    tracker.confirmed = True
    tracker.hits = 5
    tracker.updates = updates
    tracker.status = status
    tracker.x[2, 0] = vx
    tracker.x[3, 0] = vy
    tracker.P[2, 2] = velocity_var
    tracker.P[3, 3] = velocity_var
    return tracker


def test_clamp_speed() -> None:
    cfg = Config()
    assert clamp_speed(-10.0, cfg) == 0.0
    assert clamp_speed(500.0, cfg) == 500.0
    assert clamp_speed(2500.0, cfg) == cfg.speed_clamp


def test_arrow_hidden_below_min_speed() -> None:
    cfg = Config()
    assert not arrow_visible(ready_tracker(50.0), cfg)
    assert arrow_visible(ready_tracker(100.0), cfg)


def test_arrow_hidden_while_velocity_is_uncertain() -> None:
    cfg = Config()
    assert not arrow_visible(ready_tracker(500.0, updates=2), cfg)
    assert not arrow_visible(ready_tracker(500.0, velocity_var=100000.0), cfg)
    assert not arrow_visible(ready_tracker(500.0, status="lost"), cfg)


def test_arrow_visible_on_predicted_frame() -> None:
    cfg = Config()
    assert arrow_visible(ready_tracker(500.0, status="predicted"), cfg)


def test_display_speed_clamped() -> None:
    cfg = Config()
    assert display_speed(ready_tracker(2500.0), cfg) == cfg.speed_clamp


def test_speed_text_variants() -> None:
    cfg = Config()
    assert speed_text(ready_tracker(123.4), cfg) == "v = 123 px/s"
    assert speed_text(ready_tracker(123.4, status="predicted"), cfg) == (
        "v ≈ 123 px/s (pred)"
    )
    assert speed_text(ready_tracker(123.4, status="lost"), cfg) == "v = --"
    assert speed_text(ready_tracker(2500.0), cfg) == "v = 2000 px/s"


def test_fresh_tracker_has_zero_speed_and_no_arrow() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    assert tracker.speed() == 0.0
    assert not arrow_visible(tracker, cfg)
    assert speed_text(tracker, cfg) == "v = 0 px/s"
