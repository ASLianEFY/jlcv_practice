"""I7: full CLI pipeline on a synthetic video with a known trajectory."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from main import Config, run_pipeline

GREEN = (60, 180, 80)
BACKGROUND = (100, 120, 140)
FPS = 30.0
FRAMES = 60
BALL_VX = 100.0
BALL_VY = 40.0
BALL_R = 20


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240)
    )
    assert writer.isOpened()
    for index in range(FRAMES):
        image = np.full((240, 320, 3), BACKGROUND, np.uint8)
        # Same-colour cylinder distractor, away from the trajectory.
        cv2.rectangle(image, (10, 150), (35, 230), GREEN, -1)
        t = index / FPS
        cx = int(round(50 + BALL_VX * t))
        cy = int(round(100 + BALL_VY * t))
        cv2.circle(image, (cx, cy), BALL_R, GREEN, -1)
        writer.write(image)
    writer.release()


def test_synthetic_video_end_to_end(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    make_video(video)
    out_dir = tmp_path / "out"

    stats, rows = run_pipeline(
        video, out_dir, Config(), no_frames=True
    )

    assert stats.frames == FRAMES
    assert len(rows) == FRAMES
    assert stats.detection_rate >= 0.95
    assert stats.lost == 0
    assert stats.max_lost_streak == 0

    true_speed = float(np.hypot(BALL_VX, BALL_VY))
    speeds = [
        row.speed_px_s
        for row in rows[15:]
        if row.status != "lost"
    ]
    assert speeds
    median_speed = float(np.median(speeds))
    assert abs(median_speed - true_speed) / true_speed < 0.10

    csv_path = out_dir / "speed.csv"
    assert csv_path.exists()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == FRAMES + 1

    capture = cv2.VideoCapture(stats.output_video)
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAMES
    capture.release()
