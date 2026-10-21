"""U3 — Kalman 跟踪测试。

覆盖匀速/反向/加速轨迹、连续丢失与重捕获、hits/misses、欧氏硬门控、
Mahalanobis 排序/自适应 Q、镜头反向回归；对应 PLAN §8 U3。
"""

from __future__ import annotations

import numpy as np

from greenball.config import Config
from greenball.detection import Candidate
from greenball.tracking import KalmanTracker

DT = 1.0 / 30.0


def cand(cx: float, cy: float, r: float = 20.0) -> Candidate:
    return Candidate(
        cx=cx,
        cy=cy,
        r=r,
        area=float(np.pi * r * r),
        circ_perimeter=0.8,
        circ_enclosing=0.95,
        aspect=1.0,
        below_green=0.0,
        green_fill=0.95,
        score=1.0,
        source="contour",
    )


def run_track(tracker: KalmanTracker, points: list[tuple[float, float]]) -> None:
    for x, y in points:
        tracker.predict()
        chosen, recovery = tracker.associate([cand(x, y)])
        assert chosen is not None
        tracker.update(chosen, recovery)


def test_constant_velocity_converges() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    points = [(100.0 + 200.0 * k * DT, 100.0 + 100.0 * k * DT) for k in range(1, 61)]
    run_track(tracker, points)

    px, py = tracker.position()
    vx, vy = tracker.velocity()
    assert abs(px - points[-1][0]) < 2.0
    assert abs(py - points[-1][1]) < 2.0
    assert abs(vx - 200.0) < 20.0
    assert abs(vy - 100.0) < 20.0
    assert tracker.confirmed
    assert tracker.status == "detected"


def test_velocity_reversal() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    points: list[tuple[float, float]] = []
    x = 100.0
    for k in range(1, 31):
        x += 150.0 * DT
        points.append((x, 100.0))
    for k in range(31, 71):
        x -= 150.0 * DT
        points.append((x, 100.0))
    run_track(tracker, points)

    vx, _ = tracker.velocity()
    assert vx < -100.0
    assert abs(tracker.position()[0] - points[-1][0]) < 10.0


def test_constant_acceleration_is_tracked_with_bounded_lag() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 50.0, 100.0, 20.0, DT)
    points: list[tuple[float, float]] = []
    for k in range(1, 61):
        t = k * DT
        points.append((50.0 + 50.0 * t + 0.5 * 100.0 * t * t, 100.0))
    run_track(tracker, points)

    vx, _ = tracker.velocity()
    assert abs(tracker.position()[0] - points[-1][0]) < 20.0
    assert abs(vx - 250.0) < 50.0


def test_predict_only_then_lost() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0 + 10.0 * k, 100.0) for k in range(1, 6)])

    for _ in range(3):
        tracker.predict()
        tracker.mark_miss()
    assert tracker.status == "predicted"
    assert tracker.misses == 3
    assert tracker.roi() is not None

    for _ in range(3):
        tracker.predict()
        tracker.mark_miss()
    assert tracker.status == "lost"
    assert tracker.confirmed is False
    assert tracker.roi() is None


def test_reacquire_needs_two_hits() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0 + 10.0 * k, 100.0) for k in range(1, 6)])
    for _ in range(6):
        tracker.predict()
        tracker.mark_miss()
    assert tracker.status == "lost"

    tracker.reacquire(250.0, 200.0, 20.0)
    assert tracker.status == "detected"
    assert tracker.hits == 1
    assert tracker.confirmed is False

    tracker.predict()
    chosen, recovery = tracker.associate([cand(250.0, 200.0)])
    assert chosen is not None
    tracker.update(chosen, recovery)
    assert tracker.hits == 2
    assert tracker.confirmed is True


def test_euclidean_gate_rejects_far_candidate() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0 + 10.0 * k, 100.0) for k in range(1, 20)])

    tracker.predict()
    px, py = tracker.position()
    chosen, recovery = tracker.associate([cand(px + 200.0, py)])
    assert chosen is None
    assert recovery is False


def test_radius_consistency_rejects_wrong_size() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0 + 5.0 * k, 100.0) for k in range(1, 10)])

    tracker.predict()
    chosen, _ = tracker.associate([cand(140.0, 100.0, r=40.0)])
    assert chosen is None


def test_recovery_gate_sets_adaptive_q() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0, 100.0) for _ in range(20)])

    tracker.predict()
    # 15 px is inside the Euclidean gate but outside the 9.21 Mahalanobis gate
    # once the filter has converged (d^2 ~= 13.6).
    chosen, recovery = tracker.associate([cand(115.0, 100.0)])
    assert chosen is not None
    assert recovery is True

    tracker.update(chosen, recovery)
    assert tracker.q_scale == cfg.adaptive_q_factor
    tracker.predict()
    assert tracker.q_scale == 1.0


def test_direction_reversal_is_not_rejected() -> None:
    """Camera reversals create huge innovations; the ball must still be kept.

    Regression for frames ~401-406 of homework.mp4, where the image motion
    reversed and a hard Mahalanobis gate rejected the true ball.
    """
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    y = 100.0
    for _ in range(20):
        y += 10.0
        tracker.predict()
        chosen, recovery = tracker.associate([cand(100.0, y)])
        assert chosen is not None
        tracker.update(chosen, recovery)

    # The apparent motion reverses sharply (20 px/frame upwards).
    for _ in range(10):
        y -= 20.0
        tracker.predict()
        chosen, recovery = tracker.associate([cand(100.0, y)])
        assert chosen is not None, "reversed motion must not be rejected"
        tracker.update(chosen, recovery)

    assert tracker.status == "detected"
    assert abs(tracker.position()[1] - y) < 10.0


def test_arrow_ready_needs_enough_updates() -> None:
    cfg = Config()
    tracker = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    run_track(tracker, [(100.0 + 5.0 * k, 100.0) for k in range(1, 3)])
    assert tracker.confirmed
    assert tracker.updates == 3
    assert tracker.arrow_ready()

    fresh = KalmanTracker(cfg, 100.0, 100.0, 20.0, DT)
    assert not fresh.arrow_ready()
