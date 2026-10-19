"""U6: CSV schema, contact sheet, sample naming and writer checks."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from greenball.export import (
    CSV_COLUMNS,
    FrameRow,
    make_contact_sheet,
    open_writer,
    sample_frame_path,
    write_csv,
)


def test_csv_schema_and_nan() -> None:
    rows = [
        FrameRow(0, 0.0, 100.0, 200.0, 20.0, 10.0, 20.0, 22.36, "detected"),
        FrameRow(
            1,
            1.0 / 30.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            "lost",
        ),
    ]
    path = Path("unused")
    tmp = pytest.importorskip("tempfile").TemporaryDirectory()
    path = Path(tmp.name) / "speed.csv"
    write_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        parsed = list(reader)

    assert parsed[0]["status"] == "detected"
    assert float(parsed[0]["speed_px_s"]) == pytest.approx(22.36)
    assert parsed[1]["status"] == "lost"
    assert math.isnan(float(parsed[1]["cx"]))


def test_contact_sheet_shape() -> None:
    frames = [np.zeros((240, 320, 3), np.uint8) for _ in range(16)]
    sheet = make_contact_sheet(frames, cols=4, tile_size=(320, 180))
    assert sheet.shape == (720, 1280, 3)


def test_sample_frame_path() -> None:
    path = sample_frame_path(Path("out"), 90)
    assert path.name == "frame_0090.png"
    assert path.parent.name == "frames"


def test_open_writer_creates_a_readable_video(tmp_path: Path) -> None:
    writer, actual = open_writer(tmp_path / "out.mp4", 30.0, (320, 240))
    try:
        assert writer.isOpened()
        writer.write(np.zeros((240, 320, 3), np.uint8))
    finally:
        writer.release()

    assert actual.exists()
    assert actual.stat().st_size > 0


def test_open_writer_rejects_a_file_as_parent(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        open_writer(blocker / "out.mp4", 30.0, (320, 240))
