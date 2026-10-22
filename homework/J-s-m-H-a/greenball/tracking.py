"""Constant-velocity Kalman tracking.

The Euclidean gate sized to the search ROI is the hard association gate;
Mahalanobis distance ranks candidates and flags recovery frames for adaptive
Q.  This keeps the true ball when a hand-held camera reverses the apparent
motion direction.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .detection import Candidate


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


