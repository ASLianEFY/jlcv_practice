"""End-to-end detection, tracking, annotation and export pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .annotation import draw_annotation, draw_text_panel
from .config import Config
from .detection import detect_candidates
from .export import (
    FrameRow,
    PipelineStats,
    make_contact_sheet,
    open_writer,
    sample_frame_path,
    save_debug,
    write_csv,
)
from .segmentation import green_mask
from .tracking import KalmanTracker


def _track_frame(
    frame: np.ndarray,
    tracker: KalmanTracker | None,
    cfg: Config,
    dt: float,
) -> tuple[KalmanTracker | None, list[Candidate], tuple[int, int, int, int] | None]:
    """Advance the tracker by one frame; returns tracker, candidates, ROI."""
    if tracker is None:
        candidates = detect_candidates(frame, cfg)
        if candidates:
            best = candidates[0]
            tracker = KalmanTracker(
                cfg, best.cx, best.cy, best.r, dt
            )
        return tracker, candidates, None

    tracker.predict()
    roi = tracker.roi()
    candidates = detect_candidates(
        frame, cfg, roi=roi, r_hint=tracker.r
    )
    chosen, recovery = tracker.associate(candidates)
    if chosen is not None:
        tracker.update(chosen, recovery)
    else:
        tracker.mark_miss()
        if tracker.status == "lost":
            candidates = detect_candidates(frame, cfg, r_hint=tracker.r)
            if candidates:
                best = candidates[0]
                tracker.reacquire(best.cx, best.cy, best.r)
    return tracker, candidates, roi


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    cfg: Config,
    no_frames: bool = False,
    trail: bool = False,
    debug: bool = False,
) -> tuple[PipelineStats, list[FrameRow]]:
    """Run detection, tracking, annotation and export for one video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or cfg.fps_fallback
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt = 1.0 / fps

    out_dir.mkdir(parents=True, exist_ok=True)
    writer, video_out = open_writer(
        out_dir / "annotated.mp4", fps, (width, height)
    )

    if frame_count > 0 and not no_frames:
        contact_indices = {
            int(round(value))
            for value in np.linspace(
                0, max(0, frame_count - 1), cfg.contact_sheet_frames
            )
        }
    else:
        contact_indices = set()

    tracker: KalmanTracker | None = None
    trail_buffer: list[tuple[float, float]] = []
    rows: list[FrameRow] = []
    contact_frames: list[np.ndarray] = []
    speeds: list[float] = []
    counts = {"detected": 0, "predicted": 0, "lost": 0}
    lost_streak = 0
    max_lost_streak = 0
    index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        tracker, candidates, roi = _track_frame(frame, tracker, cfg, dt)

        if tracker is None:
            status = "lost"
            cx = cy = r = vx = vy = speed = float("nan")
            draw_text_panel(frame, "v = --", cfg)
            trail_buffer.clear()
        else:
            status = tracker.status
            if status == "lost":
                cx = cy = r = vx = vy = speed = float("nan")
                trail_buffer.clear()
            else:
                cx, cy = tracker.position()
                r = tracker.r
                vx, vy = tracker.velocity()
                speed = float(np.hypot(vx, vy))
                speeds.append(speed)
                trail_buffer.append((cx, cy))
                if len(trail_buffer) > cfg.trail_length:
                    trail_buffer = trail_buffer[-cfg.trail_length :]
            draw_annotation(
                frame,
                tracker,
                cfg,
                trail_points=trail_buffer if trail else None,
            )

        counts[status] += 1
        if status == "lost":
            lost_streak += 1
            max_lost_streak = max(max_lost_streak, lost_streak)
        else:
            lost_streak = 0

        rows.append(
            FrameRow(
                frame=index,
                time_s=index / fps,
                cx=cx,
                cy=cy,
                r=r,
                vx=vx,
                vy=vy,
                speed_px_s=speed,
                status=status,
            )
        )

        if not no_frames and index in cfg.sample_frames:
            path = sample_frame_path(out_dir, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)

        if not no_frames and index in contact_indices:
            contact_frames.append(frame.copy())

        if debug and index in cfg.sample_frames:
            save_debug(
                out_dir,
                frame,
                green_mask(frame, cfg),
                candidates,
                roi,
                index,
            )

        writer.write(frame)
        index += 1

    capture.release()
    writer.release()

    write_csv(out_dir / "speed.csv", rows)

    if not no_frames and contact_frames:
        sheet = make_contact_sheet(
            contact_frames, cfg.contact_sheet_cols, cfg.contact_tile_size
        )
        cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)

    speed_array = np.array(speeds, dtype=float) if speeds else np.zeros(1)
    stats = PipelineStats(
        frames=len(rows),
        detected=counts["detected"],
        predicted=counts["predicted"],
        lost=counts["lost"],
        detection_rate=counts["detected"] / max(1, len(rows)),
        max_lost_streak=max_lost_streak,
        speed_median=float(np.median(speed_array)),
        speed_p90=float(np.percentile(speed_array, 90)),
        output_video=str(video_out),
    )

    print(
        "frames={frames} detected={detected} predicted={predicted} "
        "lost={lost} detection_rate={rate:.1%} max_lost_streak={streak}".format(
            frames=stats.frames,
            detected=stats.detected,
            predicted=stats.predicted,
            lost=stats.lost,
            rate=stats.detection_rate,
            streak=stats.max_lost_streak,
        )
    )
    print(
        "speed px/s: median={median:.1f} p90={p90:.1f}".format(
            median=stats.speed_median, p90=stats.speed_p90
        )
    )
    print(f"output: {stats.output_video}")

    return stats, rows


