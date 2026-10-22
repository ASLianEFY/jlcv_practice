#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track the green ball in a video and annotate its motion with arrows.

The arrow points in the ball's motion direction.  Its length is proportional
to the ball's speed: longer arrows mean faster motion and shorter arrows mean
slower motion.  This is exactly the notation required by the assignment.

Usage
-----
    python3 homework/jiangaojie/jiangao.py

The script only needs OpenCV and NumPy.  It writes an annotated MP4, a CSV
with the per-frame track, and a preview JPEG into ``output/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = WORKSPACE / "homework.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# The green ball and the green cylinder have almost the same hue.  They are
# separated by shape: the sphere is round (width / height ~= 1), while the
# cylinder is tall and narrow.
HSV_LOWER = (35, 80, 40)
HSV_UPPER = (90, 255, 255)

MIN_AREA = 900
MAX_AREA = 60_000
MIN_ASPECT = 0.55
MAX_ASPECT = 1.45
MIN_CIRCULARITY = 0.42

# Arrow length = speed * this value (seconds).  At 800 px/s the arrow is
# 144 px long on a 1280 px wide frame.
ARROW_SECONDS = 0.18
MAX_ARROW_PX = 160.0
ARROW_COLOUR = (0, 200, 255)  # BGR: bright amber, visible on wood and props


@dataclass
class Candidate:
    x: float
    y: float
    radius: float
    area: int
    width: int
    height: int
    aspect: float
    circularity: float


@dataclass
class Detection:
    frame: int
    time_s: float
    x: float
    y: float
    radius: float
    area: int
    aspect: float
    circularity: float
    found: bool


def _odd_window(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Fill NaNs by linear interpolation, then apply a centred moving average."""
    values = np.asarray(values, dtype=np.float64)
    idx = np.arange(values.size)
    good = np.isfinite(values)
    if good.sum() == 0:
        return np.zeros_like(values)
    if good.sum() == 1:
        return np.full_like(values, values[good][0])
    filled = np.interp(idx, idx[good], values[good])
    window = _odd_window(window)
    if window <= 1 or values.size < 3:
        return filled
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(filled, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


class GreenBallDetector:
    """HSV segmentation + shape filtering + temporal nearest-neighbour tracking."""

    def __init__(
        self,
        min_area: int = MIN_AREA,
        max_area: int = MAX_AREA,
        min_circularity: float = MIN_CIRCULARITY,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.prev: Optional[Detection] = None
        self.vx = 0.0
        self.vy = 0.0
        self.lost = 0
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    def _candidates(self, frame: np.ndarray) -> list[Candidate]:
        mask = self._mask(frame)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[Candidate] = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < self.min_area or area > self.max_area:
                continue
            if w < 16 or h < 16:
                continue
            aspect = w / float(h)
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            if circularity < self.min_circularity:
                continue
            candidates.append(
                Candidate(
                    x=float(centroids[label][0]),
                    y=float(centroids[label][1]),
                    radius=math.sqrt(area / math.pi),
                    area=area,
                    width=w,
                    height=h,
                    aspect=aspect,
                    circularity=circularity,
                )
            )
        return candidates

    def _initial_pick(self, candidates: list[Candidate]) -> Optional[Candidate]:
        if not candidates:
            return None
        # The sphere is the only large, round green object.  Prefer roundness,
        # then size, and ignore anything that looks like the cylinder.
        return max(
            candidates,
            key=lambda c: (c.circularity * 2.0 - abs(c.aspect - 1.0), c.area),
        )

    def _track_pick(
        self, candidates: list[Candidate], pred_x: float, pred_y: float
    ) -> Optional[Candidate]:
        assert self.prev is not None
        gate = max(110.0, 3.0 * self.prev.radius)
        best: Optional[Candidate] = None
        best_cost = float("inf")
        for cand in candidates:
            distance = math.hypot(cand.x - pred_x, cand.y - pred_y)
            if distance > gate:
                continue
            size_penalty = abs(cand.area / max(1.0, self.prev.area) - 1.0)
            cost = (
                distance / gate
                + 1.3 * abs(cand.aspect - 1.0)
                + 1.0 * size_penalty
                - 0.35 * cand.circularity
            )
            if cost < best_cost:
                best_cost = cost
                best = cand
        return best

    def detect(self, frame: np.ndarray, frame_index: int, fps: float) -> Detection:
        candidates = self._candidates(frame)
        time_s = frame_index / fps
        pick: Optional[Candidate] = None

        if self.prev is None:
            pick = self._initial_pick(candidates)
        else:
            pred_x = self.prev.x + self.vx
            pred_y = self.prev.y + self.vy
            pick = self._track_pick(candidates, pred_x, pred_y)

        if pick is None:
            self.lost += 1
            if self.prev is None:
                # Nothing to predict from: report the image centre until the
                # ball appears.  This is only relevant for a broken first frame.
                return Detection(
                    frame_index, time_s, frame.shape[1] / 2.0, frame.shape[0] / 2.0,
                    0.0, 0, 1.0, 0.0, False,
                )
            detection = Detection(
                frame=frame_index,
                time_s=time_s,
                x=self.prev.x + self.vx,
                y=self.prev.y + self.vy,
                radius=self.prev.radius,
                area=self.prev.area,
                aspect=1.0,
                circularity=0.0,
                found=False,
            )
            self.prev = detection
            self.vx *= 0.6
            self.vy *= 0.6
            if self.lost > 30:
                # Re-acquire if the ball was lost for a full second.
                self.prev = None
                self.vx = self.vy = 0.0
                self.lost = 0
            return detection

        if self.prev is not None:
            measured_vx = pick.x - self.prev.x
            measured_vy = pick.y - self.prev.y
            # A little smoothing avoids arrow jitter from a single noisy mask.
            self.vx = 0.55 * self.vx + 0.45 * measured_vx
            self.vy = 0.55 * self.vy + 0.45 * measured_vy
        else:
            self.vx = self.vy = 0.0

        detection = Detection(
            frame=frame_index,
            time_s=time_s,
            x=pick.x,
            y=pick.y,
            radius=pick.radius,
            area=pick.area,
            aspect=pick.aspect,
            circularity=pick.circularity,
            found=True,
        )
        self.prev = detection
        self.lost = 0
        return detection


def _arrow_end(
    center: tuple[float, float],
    unit: tuple[float, float],
    start_distance: float,
    length: float,
    width: int,
    height: int,
    margin: int = 12,
) -> Optional[tuple[int, int]]:
    """Return an end point that stays inside the frame, preserving direction."""
    if length <= 1.0:
        return None
    max_travel = float("inf")
    for coordinate, direction, limit in (
        (center[0], unit[0], width - margin),
        (center[1], unit[1], height - margin),
    ):
        if direction > 1e-6:
            max_travel = min(max_travel, (limit - coordinate) / direction)
        elif direction < -1e-6:
            max_travel = min(max_travel, (margin - coordinate) / direction)
    end_distance = min(start_distance + length, max_travel)
    if end_distance <= start_distance + 2.0:
        return None
    return (
        int(round(center[0] + unit[0] * end_distance)),
        int(round(center[1] + unit[1] * end_distance)),
    )


def _draw_hud(
    frame: np.ndarray,
    speed: float,
    speed_max: float,
    time_s: float,
) -> None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = 24, 20, 430, 130
    roi = frame[y0:y1, x0:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.30, dark, 0.70, 0.0, roi)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (105, 105, 105), 1, cv2.LINE_AA)

    cv2.putText(
        frame, "GREEN BALL", (x0 + 16, y0 + 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"SPEED  {speed:4.0f} px/s", (x0 + 16, y0 + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, ARROW_COLOUR, 2, cv2.LINE_AA,
    )

    bar_x0, bar_x1 = x0 + 16, x1 - 16
    bar_y0, bar_y1 = y0 + 84, y0 + 96
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (80, 80, 80), -1)
    fill = int(round((bar_x1 - bar_x0) * float(np.clip(speed / max(1.0, speed_max), 0, 1))))
    if fill > 0:
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill, bar_y1), ARROW_COLOUR, -1)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (140, 140, 140), 1)

    time_text = f"{time_s:05.2f}s"
    text_size, _ = cv2.getTextSize(
        time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    )
    cv2.putText(
        frame, time_text, (width - text_size[0] - 28, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )


def annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    xs: np.ndarray,
    ys: np.ndarray,
    radii: np.ndarray,
    vxs: np.ndarray,
    vys: np.ndarray,
    speeds: np.ndarray,
    speed_max: float,
    fps: float,
    arrow_seconds: float,
    max_arrow_px: float,
) -> np.ndarray:
    output = frame.copy()
    centre = (float(xs[frame_index]), float(ys[frame_index]))
    radius = max(8.0, float(radii[frame_index]))
    speed = float(speeds[frame_index])

    # Ring around the ball: dark outline, white middle, dark inner edge.
    ring_radius = int(round(radius + 5))
    cv2.circle(output, (int(centre[0]), int(centre[1])), ring_radius,
               (25, 25, 25), 5, cv2.LINE_AA)
    cv2.circle(output, (int(centre[0]), int(centre[1])), ring_radius,
               (255, 255, 255), 3, cv2.LINE_AA)

    # Arrow direction = ball motion direction; arrow length ∝ speed.
    if speed >= 25.0 and math.hypot(vxs[frame_index], vys[frame_index]) > 1e-6:
        unit = (
            float(vxs[frame_index]) / speed,
            float(vys[frame_index]) / speed,
        )
        start_distance = radius + 10.0
        length = float(np.clip(speed * arrow_seconds, 15.0, max_arrow_px))
        end = _arrow_end(
            centre, unit, start_distance, length,
            output.shape[1], output.shape[0],
        )
        if end is not None:
            start = (
                int(round(centre[0] + unit[0] * start_distance)),
                int(round(centre[1] + unit[1] * start_distance)),
            )
            cv2.arrowedLine(
                output, start, end, (18, 18, 18), 10, cv2.LINE_AA, tipLength=0.24
            )
            cv2.arrowedLine(
                output, start, end, ARROW_COLOUR, 5, cv2.LINE_AA, tipLength=0.24
            )
            cv2.circle(output, start, 4, ARROW_COLOUR, -1, cv2.LINE_AA)

    _draw_hud(output, speed, speed_max, frame_index / fps)
    return output


def _resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and annotate the green ball's motion in a video."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"input video (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "green_ball_annotated.mp4",
        help="annotated video path",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_OUTPUT_DIR / "green_ball_track.csv",
        help="per-frame track CSV path",
    )
    parser.add_argument(
        "--preview", type=Path, default=DEFAULT_OUTPUT_DIR / "preview.jpg",
        help="preview JPEG path",
    )
    parser.add_argument(
        "--preview-time", type=float, default=6.0,
        help="preview timestamp in seconds; negative = frame with maximum speed",
    )
    parser.add_argument(
        "--smooth", type=int, default=9,
        help="centred moving-average window in frames (default: 9)",
    )
    parser.add_argument(
        "--arrow-seconds", type=float, default=ARROW_SECONDS,
        help="arrow length = speed * this value (default: 0.18)",
    )
    parser.add_argument(
        "--max-arrow", type=float, default=MAX_ARROW_PX,
        help="maximum arrow length in pixels (default: 150)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print detection diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_path = _resolve_path(args.input, Path.cwd())
    output_path = _resolve_path(args.output, Path.cwd())
    csv_path = _resolve_path(args.csv, Path.cwd())
    preview_path = _resolve_path(args.preview, Path.cwd())

    if not input_path.exists():
        print(f"error: input video not found: {input_path}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"error: cannot open video: {input_path}", file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 1e-6:
        fps = 30.0
    if width <= 0 or height <= 0:
        print("error: invalid video dimensions", file=sys.stderr)
        return 2

    detector = GreenBallDetector()
    detections: list[Detection] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detection = detector.detect(frame, frame_index, fps)
        detections.append(detection)
        frame_index += 1
    cap.release()

    total = len(detections)
    if total == 0:
        print("error: no frames decoded", file=sys.stderr)
        return 2

    found_count = sum(1 for d in detections if d.found)
    print(
        f"read {total} frames, {width}x{height} @ {fps:.2f} fps; "
        f"ball detected in {found_count}/{total} frames "
        f"({100.0 * found_count / total:.1f}%)"
    )

    raw_x = np.array([d.x if d.found else np.nan for d in detections])
    raw_y = np.array([d.y if d.found else np.nan for d in detections])
    raw_r = np.array([d.radius if d.found else np.nan for d in detections])
    xs = _smooth(raw_x, args.smooth)
    ys = _smooth(raw_y, args.smooth)
    radii = _smooth(raw_r, args.smooth)

    vxs = np.gradient(xs) * fps
    vys = np.gradient(ys) * fps
    # Smooth velocity as well: position smoothing alone can still leave a
    # one-frame kink in the arrow direction.
    vxs = _smooth(vxs, 5)
    vys = _smooth(vys, 5)
    speeds = np.hypot(vxs, vys)
    speed_max = max(1.0, float(np.max(speeds)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- CSV -----------------------------------------------------------
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame", "time_s", "x_px", "y_px", "vx_px_s", "vy_px_s",
             "speed_px_s", "radius_px", "detected"]
        )
        for i, detection in enumerate(detections):
            writer.writerow([
                i,
                f"{i / fps:.6f}",
                f"{xs[i]:.3f}",
                f"{ys[i]:.3f}",
                f"{vxs[i]:.3f}",
                f"{vys[i]:.3f}",
                f"{speeds[i]:.3f}",
                f"{radii[i]:.3f}",
                int(detection.found),
            ])

    # ---- annotated video ----------------------------------------------
    # H.264 is the most portable choice for browsers; if the local OpenCV /
    # FFmpeg build cannot encode it, fall back to MPEG-4 Part 2 (mp4v).
    writer = None
    video_codec = ""
    for codec_name in ("avc1", "mp4v"):
        candidate = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec_name),
            fps, (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            video_codec = codec_name
            break
        candidate.release()
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
    if writer is None:
        print(f"error: cannot create output video: {output_path}", file=sys.stderr)
        return 2
    print(f"video codec: {video_codec}")

    preview_index = (
        int(np.argmax(speeds)) if args.preview_time < 0
        else int(round(args.preview_time * fps))
    )
    preview_index = max(0, min(total - 1, preview_index))
    preview_saved = False

    cap = cv2.VideoCapture(str(input_path))
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        annotated = annotate_frame(
            frame=frame,
            frame_index=i,
            xs=xs,
            ys=ys,
            radii=radii,
            vxs=vxs,
            vys=vys,
            speeds=speeds,
            speed_max=speed_max,
            fps=fps,
            arrow_seconds=args.arrow_seconds,
            max_arrow_px=args.max_arrow,
        )
        writer.write(annotated)
        if i == preview_index:
            cv2.imwrite(
                str(preview_path), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 93],
            )
            preview_saved = True
        if args.debug and (i % 60 == 0):
            print(
                f"  frame {i:4d}  xy=({xs[i]:6.1f},{ys[i]:6.1f})  "
                f"r={radii[i]:5.1f}  speed={speeds[i]:6.1f}"
            )
    cap.release()
    writer.release()

    if not preview_saved:
        print("warning: preview frame was not saved", file=sys.stderr)

    # ---- run summary ---------------------------------------------------
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "frames": total,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": total / fps,
        "video_codec": video_codec,
        "detected_frames": found_count,
        "detection_rate": found_count / total,
        "speed_px_s": {
            "min": float(np.min(speeds)),
            "mean": float(np.mean(speeds)),
            "p95": float(np.percentile(speeds, 95)),
            "max": float(np.max(speeds)),
        },
        "arrow_seconds": args.arrow_seconds,
        "smooth_window": _odd_window(args.smooth),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"annotated video : {output_path}")
    print(f"track CSV       : {csv_path}")
    print(f"preview         : {preview_path}")
    print(f"summary         : {summary_path}")
    print(
        f"speed (px/s)    : min {np.min(speeds):.1f}  "
        f"mean {np.mean(speeds):.1f}  max {np.max(speeds):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
