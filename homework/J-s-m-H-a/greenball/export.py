"""Video, CSV and review-image export."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .detection import Candidate


CSV_COLUMNS = (
    "frame",
    "time_s",
    "cx",
    "cy",
    "r",
    "vx",
    "vy",
    "speed_px_s",
    "status",
)


@dataclass
class FrameRow:
    """One CSV row (one video frame)."""

    frame: int
    time_s: float
    cx: float
    cy: float
    r: float
    vx: float
    vy: float
    speed_px_s: float
    status: str


@dataclass
class PipelineStats:
    """Aggregate numbers printed at the end of a run."""

    frames: int
    detected: int
    predicted: int
    lost: int
    detection_rate: float
    max_lost_streak: int
    speed_median: float
    speed_p90: float
    output_video: str


def write_csv(path: Path, rows: list[FrameRow]) -> None:
    """Write the per-frame CSV (``nan`` for lost frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.frame,
                    f"{row.time_s:.3f}",
                    f"{row.cx:.2f}",
                    f"{row.cy:.2f}",
                    f"{row.r:.2f}",
                    f"{row.vx:.2f}",
                    f"{row.vy:.2f}",
                    f"{row.speed_px_s:.2f}",
                    row.status,
                ]
            )


def open_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path]:
    """Open an MP4 writer, falling back to XVID/AVI when necessary."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create output directory: {path.parent}") from exc

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if writer.isOpened():
        return writer, path
    writer.release()

    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, size
    )
    if writer.isOpened():
        return writer, fallback
    writer.release()
    raise RuntimeError(
        f"cannot open a video writer for {path} (mp4v and XVID both failed)"
    )


def make_contact_sheet(
    frames: list[np.ndarray],
    cols: int,
    tile_size: tuple[int, int],
) -> np.ndarray:
    """Tile annotated frames into a single review image."""
    if not frames:
        return np.zeros((0, 0, 3), np.uint8)
    width, height = tile_size
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * height, cols * width, 3), 255, np.uint8)
    for index, frame in enumerate(frames):
        tile = cv2.resize(frame, (width, height))
        row, col = divmod(index, cols)
        sheet[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = tile
    return sheet


def sample_frame_path(out_dir: Path, index: int) -> Path:
    return out_dir / "frames" / f"frame_{index:04d}.png"


def save_debug(
    out_dir: Path,
    frame: np.ndarray,
    mask: np.ndarray,
    candidates: list[Candidate],
    roi: tuple[int, int, int, int] | None,
    index: int,
) -> None:
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"mask_{index:04d}.png"), mask)

    overlay = frame.copy()
    if roi is not None:
        cv2.rectangle(
            overlay,
            (int(roi[0]), int(roi[1])),
            (int(roi[2]), int(roi[3])),
            (255, 0, 0),
            1,
        )
    for candidate in candidates:
        cv2.circle(
            overlay,
            (int(round(candidate.cx)), int(round(candidate.cy))),
            int(round(candidate.r)),
            (0, 0, 255),
            1,
        )
    cv2.imwrite(str(debug_dir / f"candidates_{index:04d}.png"), overlay)

    lines = [f"roi={roi}"]
    for candidate in candidates:
        lines.append(
            f"{candidate.source} c=({candidate.cx:.1f},{candidate.cy:.1f}) "
            f"r={candidate.r:.1f} score={candidate.score:.3f} "
            f"below={candidate.below_green:.2f}"
        )
    (debug_dir / f"info_{index:04d}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


