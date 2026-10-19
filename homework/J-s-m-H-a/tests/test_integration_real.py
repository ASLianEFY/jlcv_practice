"""I8/I9: real-video acceptance metrics and hand-checked ball centres.

The ground-truth fixture was produced by the pipeline, then every centre was
visually checked against a crosshair overlay; it is a regression fixture, not
an independent annotation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import pytest

from greenball.config import Config
from greenball.export import FrameRow, PipelineStats
from greenball.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO = REPO_ROOT / "homework.mp4"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ball_centers.csv"

pytestmark = pytest.mark.skipif(
    not VIDEO.exists(), reason="homework.mp4 is not present"
)


@pytest.fixture(scope="module")
def pipeline_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PipelineStats, list[FrameRow], Path]:
    out_dir = tmp_path_factory.mktemp("real_out")
    stats, rows = run_pipeline(VIDEO, out_dir, Config(), no_frames=True)
    return stats, rows, out_dir


def test_real_video_acceptance_metrics(
    pipeline_result: tuple[PipelineStats, list[FrameRow], Path],
) -> None:
    stats, rows, out_dir = pipeline_result

    assert stats.frames == 473
    assert len(rows) == 473
    assert stats.detection_rate >= 0.95
    assert stats.max_lost_streak <= 5

    csv_path = out_dir / "speed.csv"
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 474

    capture = cv2.VideoCapture(stats.output_video)
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 473
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1280
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 720
    finally:
        capture.release()


def test_hand_checked_ball_centres(
    pipeline_result: tuple[PipelineStats, list[FrameRow], Path],
) -> None:
    _, rows, _ = pipeline_result
    by_frame = {row.frame: row for row in rows}

    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        ground_truth = list(csv.DictReader(handle))

    assert len(ground_truth) == 12
    for item in ground_truth:
        frame = int(item["frame"])
        row = by_frame[frame]
        assert row.status != "lost"
        error = math.hypot(row.cx - float(item["cx"]), row.cy - float(item["cy"]))
        assert error <= 10.0, f"frame {frame}: centre error {error:.1f} px"
