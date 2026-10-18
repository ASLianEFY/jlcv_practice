#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 green_ball_tracker.py 的检测结果，绘制速度-时间曲线 speed_curve.png。

单序列折线图：一条蓝色主线、发丝网格、单一 y 轴；只直接标注峰值，
不逐点标数字。运行：
  python3 plot_speed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from green_ball_tracker import compute_motion, detect_trajectory

# 调色板（BGR）：参考数据可视化规范的浅色主题
SURFACE = (251, 252, 252)      # #fcfcfb 图表背景
INK_PRIMARY = (11, 11, 11)     # #0b0b0b 标题
INK_SECONDARY = (78, 81, 82)   # #52514e 副标题/标注
INK_MUTED = (129, 135, 137)    # #898781 轴标签
GRID = (217, 224, 225)         # #e1e0d9 发丝网格
AXIS = (183, 194, 195)         # #c3c2b7 基线
SERIES = (214, 120, 42)        # #2a78d6 数据线

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_speed_curve(speed: np.ndarray, fps: float, output_path: Path) -> None:
    width, height = 1200, 480
    left, right, top, bottom = 92, 1150, 92, 404
    y_max = 900.0
    # x 轴取整到整秒，避免最后一个刻度溢出绘图区
    t_max = max(float(np.ceil(len(speed) / fps)), 1.0)

    canvas = np.full((height, width, 3), SURFACE, dtype=np.uint8)

    def to_px(t: float, v: float) -> tuple[int, int]:
        x = left + (t / t_max) * (right - left)
        y = bottom - (v / y_max) * (bottom - top)
        return int(round(x)), int(round(y))

    # 标题与副标题（单序列不需要图例，标题即序列名）
    cv2.putText(canvas, "Apparent speed of the green ball (image space)",
                (left, 42), FONT, 0.85, INK_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"median {np.median(speed):.0f} px/s   |   max {speed.max():.0f} px/s"
                f"   |   {len(speed)} frames @ {fps:.0f} FPS",
                (left, 68), FONT, 0.52, INK_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Speed (px/s)", (left, 86), FONT, 0.5, INK_MUTED, 1, cv2.LINE_AA)

    # 横向发丝网格 + y 轴刻度（solid hairline，不画虚线）
    for v in range(0, int(y_max) + 1, 200):
        _, y = to_px(0, v)
        cv2.line(canvas, (left, y), (right, y), GRID, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(v), (left - 46, y + 5), FONT, 0.46, INK_MUTED, 1, cv2.LINE_AA)

    # x 轴刻度
    for t in range(0, int(np.ceil(t_max)) + 1, 2):
        x, _ = to_px(t, 0)
        cv2.line(canvas, (x, bottom), (x, bottom + 5), AXIS, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(t), (x - 8, bottom + 26), FONT, 0.46, INK_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Time (s)", ((left + right) // 2 - 30, 452),
                FONT, 0.5, INK_MUTED, 1, cv2.LINE_AA)

    # 基线
    cv2.line(canvas, (left, bottom), (right, bottom), AXIS, 1, cv2.LINE_AA)

    # 数据线：2px，圆角连接由 LINE_AA 提供
    times = np.arange(len(speed)) / fps
    points = np.array([to_px(t, v) for t, v in zip(times, speed)], dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=False, color=SERIES,
                  thickness=2, lineType=cv2.LINE_AA)

    # 只直接标注峰值：先画背景色圆环，再画数据色圆点
    peak = int(np.argmax(speed))
    px, py = to_px(times[peak], speed[peak])
    cv2.circle(canvas, (px, py), 7, SURFACE, -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 4, SERIES, -1, cv2.LINE_AA)

    text = f"peak {speed[peak]:.0f} px/s"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    tx = px - tw - 14 if px > (left + right) // 2 else px + 14
    ty = max(py - 12, top + 14)
    cv2.putText(canvas, text, (tx, ty), FONT, 0.52, INK_SECONDARY, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"无法写入图片：{output_path}")
    print(f"已生成 {output_path}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="绘制绿球速度曲线")
    parser.add_argument("--input", default=str(repo_root / "homework.mp4"))
    parser.add_argument("--output", default=str(script_dir / "speed_curve.png"))
    args = parser.parse_args()

    detections, fps, _ = detect_trajectory(Path(args.input))
    _, _, _, _, _, speed = compute_motion(detections, fps)
    draw_speed_curve(speed, fps, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
