#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JLCV 自瞄培训作业：绿球检测 + 运动方向和速度箭头标注

思路概述（详见同目录 README.md）：
  1. HSV 颜色分割提取绿色区域；
  2. 形态学开/闭运算去掉噪点、填补空洞；
  3. 对每个绿色轮廓计算形状特征（圆度、宽高比），区分绿球与绿圆柱；
  4. 用"预测位置 + 形状得分"做逐帧关联，得到稳定轨迹；
  5. 中值滤波 + 滑动平均平滑轨迹，中心差分求速度；
  6. 箭头方向 = 速度方向，箭头长度与速度成正比（带上下限）。

运行：
  python3 green_ball_tracker.py
  python3 green_ball_tracker.py --input ../../homework.mp4 --output result.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 参数区（所有可调参数集中在这里，方便按视频微调）
# ---------------------------------------------------------------------------

# OpenCV 的 HSV 范围：H ∈ [0,179], S,V ∈ [0,255]
# 绿色在 H≈60 附近，放宽到 [35,95] 以覆盖不同光照。
HSV_LOWER = np.array([35, 70, 40], dtype=np.uint8)
HSV_UPPER = np.array([95, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 1200      # 小于该面积的绿色块视为噪声（小球面积约 1 万像素）
MORPH_KERNEL = 5             # 形态学核尺寸

# 形状判据：球接近正圆（宽高比≈1、圆度高），圆柱偏竖长（宽高比≈0.6~0.85）
SHAPE_MAX_ASPECT_PENALTY = 1.5
SHAPE_GATE = 0.32            # 低于该得分的候选直接忽略，防止误锁绿圆柱
MAX_ASSOCIATION_DIST = 220.0 # 相邻帧关联允许的最大跳变（本视频最大约 28 px/帧）

# 轨迹平滑与箭头显示
MEDIAN_WINDOW = 5            # 中值滤波窗口，去掉单帧检测跳变
SMOOTH_WINDOW = 5            # 滑动平均窗口，抑制质心抖动
ARROW_SCALE = 0.25           # 箭头长度(px) = 速度(px/s) * ARROW_SCALE
ARROW_MIN = 25.0             # 最短箭头，保证慢速时方向仍可见
ARROW_MAX = 200.0            # 最长箭头，避免箭头冲出画面

ARROW_COLOR = (255, 0, 255)  # BGR 品红，与绿色球、木纹背景对比明显
OUTLINE_COLOR = (40, 40, 40)
BALL_CIRCLE_COLOR = (0, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)


class Candidate(NamedTuple):
    """一帧中的一个绿色候选块。"""
    x: float
    y: float
    radius: float
    area: float
    aspect: float
    circularity: float
    shape_score: float


# ---------------------------------------------------------------------------
# 1. 颜色分割 + 轮廓检测
# ---------------------------------------------------------------------------

def build_green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """返回绿色区域的二值掩码。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    # 先开后闭：去掉孤立噪点，再填补球面高光造成的小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_candidates(frame_bgr: np.ndarray) -> list[Candidate]:
    """找出画面中所有绿色候选块，并计算形状特征。"""
    mask = build_green_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        # 圆度：4πA/P²，正圆为 1，越不规则越小
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

        _, _, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh) if bh > 0 else 0.0

        # 综合形状分：圆度越高越好，宽高比越接近 1 越好
        shape_score = circularity - SHAPE_MAX_ASPECT_PENALTY * abs(aspect - 1.0)

        candidates.append(
            Candidate(x, y, radius, area, aspect, circularity, shape_score)
        )
    return candidates


# ---------------------------------------------------------------------------
# 2. 逐帧关联：在多个绿色块中锁定"球"
# ---------------------------------------------------------------------------

def track_ball(candidates: list[Candidate],
               last: tuple[float, float] | None,
               velocity: tuple[float, float]) -> Candidate | None:
    """
    从当前帧候选里选出球。

    - 第一帧：取形状得分最高的候选；
    - 后续帧：用上一帧速度预测球的位置，综合"离预测点近"和"形状像球"打分。
    """
    # 先用形状门槛筛掉明显是圆柱的候选
    plausible = [c for c in candidates if c.shape_score >= SHAPE_GATE]
    if not plausible:
        return None  # 本帧没有像球的绿色块，交给上层做匀速推断

    if last is None:
        return max(plausible, key=lambda c: c.shape_score)

    pred_x = last[0] + velocity[0]
    pred_y = last[1] + velocity[1]

    def cost(c: Candidate) -> float:
        dist = math.hypot(c.x - pred_x, c.y - pred_y)
        # 距离为主，形状分为奖励项（250 是让形状分与像素距离同量级）
        return dist - 250.0 * c.shape_score

    best = min(plausible, key=cost)

    # 防止场景突变时乱跳：跳得太远就退回"形状最像球"的候选
    if math.hypot(best.x - last[0], best.y - last[1]) > MAX_ASSOCIATION_DIST:
        best = max(plausible, key=lambda c: c.shape_score)
    return best


def detect_trajectory(video_path: Path) -> tuple[list[tuple[float, float, float] | None], float, int]:
    """第一遍读视频：返回逐帧 (x, y, r)、fps、帧数。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detections: list[tuple[float, float, float] | None] = []

    last: tuple[float, float] | None = None
    velocity = (0.0, 0.0)
    detected_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        best = track_ball(find_candidates(frame), last, velocity)

        if best is None:
            # 本帧没检测到：沿用上一帧位置（短暂遮挡时轨迹不会断）
            if last is not None:
                detections.append((last[0], last[1], detections[-1][2] if detections else 0.0))
            else:
                detections.append(None)
        else:
            detected_frames += 1
            if last is not None:
                # 用短时 EMA 更新运动趋势，供下一帧位置预测
                velocity = (
                    0.6 * velocity[0] + 0.4 * (best.x - last[0]),
                    0.6 * velocity[1] + 0.4 * (best.y - last[1]),
                )
            last = (best.x, best.y)
            detections.append((best.x, best.y, best.radius))

        if len(detections) % 100 == 0:
            print(f"  检测进度：{len(detections)} 帧")

    cap.release()
    print(f"  有效检测 {detected_frames}/{len(detections)} 帧"
          f"（其余帧沿用上一帧位置）")
    return detections, float(fps), len(detections)


# ---------------------------------------------------------------------------
# 3. 轨迹平滑与速度计算
# ---------------------------------------------------------------------------

def _fill_missing(values: np.ndarray) -> np.ndarray:
    """对缺失值（NaN）做最近邻填充。"""
    idx = np.arange(len(values))
    valid = ~np.isnan(values)
    if not valid.any():
        return np.zeros_like(values)
    return np.interp(idx, idx[valid], values[valid])


def _median_filter(values: np.ndarray, k: int) -> np.ndarray:
    """一维中值滤波（边缘复制填充），用于剔除单帧跳变。"""
    if k <= 1:
        return values.copy()
    pad = k // 2
    padded = np.pad(values, pad, mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(values))])


def _moving_average(values: np.ndarray, k: int) -> np.ndarray:
    """一维滑动平均（边缘复制填充）。"""
    if k <= 1:
        return values.copy()
    pad = k // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")


def compute_motion(detections, fps: float):
    """由逐帧检测结果得到平滑后的位置、半径、速度和速率。"""
    n = len(detections)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    for i, det in enumerate(detections):
        if det is not None:
            xs[i], ys[i], rs[i] = det

    xs, ys, rs = _fill_missing(xs), _fill_missing(ys), _fill_missing(rs)

    # 先中值滤波去跳变，再滑动平均抑制抖动
    xs = _moving_average(_median_filter(xs, MEDIAN_WINDOW), SMOOTH_WINDOW)
    ys = _moving_average(_median_filter(ys, MEDIAN_WINDOW), SMOOTH_WINDOW)
    rs = _moving_average(_median_filter(rs, MEDIAN_WINDOW), SMOOTH_WINDOW)

    # 中心差分求速度（单位：像素/秒），比单边差分更准确且无相位延迟
    vx = np.zeros(n)
    vy = np.zeros(n)
    if n >= 3:
        vx[1:-1] = (xs[2:] - xs[:-2]) * fps / 2.0
        vy[1:-1] = (ys[2:] - ys[:-2]) * fps / 2.0
        vx[0], vx[-1] = vx[1], vx[-2]
        vy[0], vy[-1] = vy[1], vy[-2]

    speed = np.hypot(vx, vy)
    return xs, ys, rs, vx, vy, speed


# ---------------------------------------------------------------------------
# 4. 绘制
# ---------------------------------------------------------------------------

def put_text(img, text: str, org: tuple[int, int], scale: float = 0.7) -> None:
    """带深色描边的文字，保证在任何背景上都看得清。"""
    cv2.putText(img, text, org, FONT, scale, OUTLINE_COLOR, 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, FONT_COLOR, 2, cv2.LINE_AA)


def draw_annotation(frame, x, y, r, vx, vy, speed):
    """在球的位置画检测圆、速度箭头和速度数值。"""
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, int(round(r)) + 3, BALL_CIRCLE_COLOR, 2, cv2.LINE_AA)

    if speed > 1e-6:
        # 箭头长度与速度成正比，并限制在 [ARROW_MIN, ARROW_MAX]
        length = float(np.clip(speed * ARROW_SCALE, ARROW_MIN, ARROW_MAX))
        ux, uy = vx / speed, vy / speed
        tip = (int(round(x + ux * length)), int(round(y + uy * length)))

        # 先画粗深色描边，再画亮色箭头，增强对比度
        cv2.arrowedLine(frame, center, tip, OUTLINE_COLOR, 9, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(frame, center, tip, ARROW_COLOR, 5, cv2.LINE_AA, tipLength=0.22)

    # 速度数值放在球上方（自动避开画面边界）
    h, w = frame.shape[:2]
    tx = int(np.clip(x - 80, 8, w - 300))
    ty = int(np.clip(y - r - 18, 30, h - 12))
    put_text(frame, f"v = {speed:4.0f} px/s", (tx, ty), 0.7)


def render(video_path: Path, output_path: Path, xs, ys, rs, vx, vy, speed, fps: float):
    """第二遍读视频：绘制标注并写出结果视频。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_annotation(frame, xs[idx], ys[idx], rs[idx], vx[idx], vy[idx], speed[idx])

        # 左上角信息栏
        put_text(frame, f"frame {idx:03d}  |  speed = {speed[idx]:.0f} px/s", (18, 34), 0.65)
        cv2.putText(frame, "arrow: direction of motion, length proportional to speed",
                    (18, 64), FONT, 0.55, OUTLINE_COLOR, 4, cv2.LINE_AA)
        cv2.putText(frame, "arrow: direction of motion, length proportional to speed",
                    (18, 64), FONT, 0.55, FONT_COLOR, 2, cv2.LINE_AA)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"完成：共 {idx} 帧 -> {output_path}")


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    global ARROW_SCALE, ARROW_MIN, ARROW_MAX

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="绿球检测与运动箭头标注")
    parser.add_argument("--input", default=str(repo_root / "homework.mp4"),
                        help="输入视频路径（默认：仓库根目录的 homework.mp4）")
    parser.add_argument("--output", default=str(script_dir / "green_ball_annotated.mp4"),
                        help="输出视频路径")
    parser.add_argument("--arrow-scale", type=float, default=ARROW_SCALE,
                        help="箭头长度与速度的比例（px 每 px/s）")
    parser.add_argument("--min-arrow", type=float, default=ARROW_MIN)
    parser.add_argument("--max-arrow", type=float, default=ARROW_MAX)
    args = parser.parse_args()

    # 允许从命令行覆盖模块级显示参数
    ARROW_SCALE, ARROW_MIN, ARROW_MAX = args.arrow_scale, args.min_arrow, args.max_arrow

    video_path = Path(args.input)
    output_path = Path(args.output)
    if not video_path.exists():
        print(f"错误：找不到输入视频 {video_path}", file=sys.stderr)
        return 1

    print(f"输入：{video_path}")
    print("第一遍：检测并跟踪绿球 ...")
    detections, fps, n = detect_trajectory(video_path)
    if n == 0:
        print("错误：视频没有可读帧", file=sys.stderr)
        return 1
    print(f"  共 {n} 帧，{fps:.1f} FPS")

    xs, ys, rs, vx, vy, speed = compute_motion(detections, fps)
    print(f"  速度范围：{speed.min():.0f} ~ {speed.max():.0f} px/s，"
          f"中位数 {np.median(speed):.0f} px/s")

    print("第二遍：绘制箭头并写出视频 ...")
    render(video_path, output_path, xs, ys, rs, vx, vy, speed, fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
