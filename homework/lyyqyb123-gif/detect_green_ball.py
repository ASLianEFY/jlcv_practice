#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
week0-jlcv 作业：识别视频中的绿球，并用箭头标注运动方向与速度变化。

思路
----
1. 把图像转到 HSV，用绿色范围做阈值分割；
2. 用面积、圆度、宽高比过滤候选区域，排除绿色圆柱等干扰；
3. 首帧选出最像球的候选，后续帧用“位置预测 + 最近邻 + 形状得分”跟踪；
4. 对轨迹做滑动平均，计算速度和加速度；
5. 逐帧绘制：球心、轮廓、速度箭头（箭头长度正比于速度）、
   速度/加速度文字，以及右下角的速度曲线。

箭头颜色：
    红色 = 加速    蓝色 = 减速    黄色 = 速度基本不变

用法
----
    python3 detect_green_ball.py
    python3 detect_green_ball.py --input /path/to/homework.mp4 --output out.mp4
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

# ----------------------- 可调参数 -----------------------
HSV_LOW = (35, 60, 50)          # 绿色下限 (OpenCV: H 0~179)
HSV_HIGH = (90, 255, 255)       # 绿色上限
MIN_AREA = 400                  # 候选轮廓最小面积
MAX_AREA = 40000                # 候选轮廓最大面积
MIN_CIRCULARITY = 0.50          # 最小圆度 4*pi*A/P^2
ASPECT_RANGE = (0.70, 1.40)     # 宽高比范围，排除圆柱侧面
MAX_JUMP = 120.0                # 相邻帧允许的最大位移（像素）
SMOOTH_WINDOW = 15              # 轨迹平滑窗口（奇数）
MAX_LOST = 15                   # 连续丢失多少帧后重新捕获
ARROW_MAX = 140                 # 箭头最大长度（像素）
ARROW_MIN = 18                  # 箭头最小长度（像素）
ACCEL_THRESHOLD = 300.0         # 加速度阈值（px/s^2），用于颜色判定
# --------------------------------------------------------


def candidates_in_frame(frame):
    """返回当前帧里所有“像绿球”的候选。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(max(h, 1))
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        result.append(
            {
                "center": np.array([cx, cy], dtype=np.float64),
                "area": area,
                "circularity": circularity,
                "aspect": aspect,
                "radius": math.sqrt(area / math.pi),
            }
        )
    return result


def initial_score(candidate):
    """首帧候选打分：圆、宽高比接近 1、面积不要太大。"""
    return (
        candidate["circularity"]
        - 0.5 * abs(candidate["aspect"] - 1.0)
        - candidate["area"] / 200000.0
    )


def track_video(path):
    """第一遍：逐帧检测并跟踪，返回轨迹（平滑前）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    positions = []
    found_flags = []
    radii = []

    prev = None
    velocity = np.zeros(2, dtype=np.float64)
    lost = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cands = candidates_in_frame(frame)

        if prev is None:
            # 首帧：选最像球的候选
            best = max(cands, key=initial_score) if cands else None
            if best is not None:
                pos = best["center"]
                radius = best["radius"]
                found = True
            else:
                pos = np.array([frame.shape[1] / 2.0, frame.shape[0] / 2.0])
                radius = 30.0
                found = False
        else:
            # 用“上一帧位置 + 速度”做预测，再找最近的候选
            predicted = prev + velocity
            best = None
            best_cost = None
            for cand in cands:
                distance = np.linalg.norm(cand["center"] - predicted)
                if distance > MAX_JUMP:
                    continue
                shape_bonus = (
                    0.4 * cand["circularity"]
                    + 0.2 * max(0.0, 1.0 - abs(cand["aspect"] - 1.0))
                )
                cost = distance - 60.0 * shape_bonus
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = cand

            if best is not None:
                pos = best["center"]
                radius = best["radius"]
                found = True
                lost = 0
            else:
                # 丢失时按预测位置滑行
                pos = predicted
                radius = radii[-1] if radii else 30.0
                found = False
                lost += 1
                if lost > MAX_LOST and cands:
                    # 长时间丢失：重新按首帧规则捕获
                    best = max(cands, key=initial_score)
                    pos = best["center"]
                    radius = best["radius"]
                    found = True
                    lost = 0

        if found:
            velocity = 0.7 * velocity + 0.3 * (pos - prev if prev is not None else 0)
        else:
            velocity = 0.9 * velocity

        positions.append(pos.copy())
        found_flags.append(found)
        radii.append(radius)
        prev = pos
        frame_index += 1

    cap.release()
    positions = np.array(positions, dtype=np.float64)
    return fps, total, positions, np.array(found_flags), np.array(radii)


def smooth_positions(positions, window):
    """对 x/y 分别做中心滑动平均。"""
    if window <= 1 or len(positions) < 3:
        return positions.copy()

    window = min(window, len(positions) - (1 - len(positions) % 2))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return positions.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    smoothed = np.empty_like(positions)
    for axis in range(2):
        signal = positions[:, axis]
        padded = np.pad(signal, (pad, pad), mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def compute_kinematics(positions, fps):
    """由平滑后的轨迹计算速度、速率和加速度。"""
    vx = np.gradient(positions[:, 0]) * fps
    vy = np.gradient(positions[:, 1]) * fps
    speed = np.hypot(vx, vy)
    acceleration = np.gradient(speed) * fps
    return vx, vy, speed, acceleration


def arrow_color(acceleration):
    """按加速度给箭头上色（BGR）。"""
    if acceleration > ACCEL_THRESHOLD:
        return (0, 0, 255), "accelerating"
    if acceleration < -ACCEL_THRESHOLD:
        return (255, 0, 0), "decelerating"
    return (0, 255, 255), "steady"


def draw_speed_graph(frame, speeds, current_index, max_speed):
    """在右下角画速度曲线。"""
    height, width = frame.shape[:2]
    box_w, box_h = 280, 130
    x0 = width - box_w - 20
    y0 = height - box_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame, "speed (px/s)", (x0 + 8, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    left, right = x0 + 10, x0 + box_w - 10
    top, bottom = y0 + 30, y0 + box_h - 12
    cv2.line(frame, (left, bottom), (right, bottom), (160, 160, 160), 1)
    cv2.line(frame, (left, top), (left, bottom), (160, 160, 160), 1)

    n = len(speeds)
    if n > 1 and max_speed > 1e-6:
        # 只绘制到当前帧，曲线随视频播放逐渐增长
        upto = min(current_index, n - 1)
        points = []
        for i in range(upto + 1):
            px = left + int((right - left) * i / (n - 1))
            py = bottom - int((bottom - top) * min(speeds[i], max_speed) / max_speed)
            points.append((px, py))
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False,
                          (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, points[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)


def render_video(input_path, output_path, fps, positions, found_flags, radii,
                 speeds, vx, vy, acceleration):
    """第二遍：读取视频并绘制标注。"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    # 箭头长度随速度自适应：让常见高速对应的箭头约 100 px
    nonzero = speeds[speeds > 1.0]
    reference_speed = float(np.percentile(nonzero, 90)) if nonzero.size else 100.0
    reference_speed = max(reference_speed, 1.0)
    arrow_scale = 100.0 / reference_speed
    max_speed = float(np.percentile(speeds, 99)) if speeds.size else 1.0

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index >= len(positions):
            break

        center = tuple(np.round(positions[index]).astype(int))
        speed = float(speeds[index])
        accel = float(acceleration[index])
        color, _state = arrow_color(accel)
        radius = max(12, int(round(radii[index])))

        # 球体标记
        cv2.circle(frame, center, radius + 4, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, center, 4, (255, 0, 255), -1, cv2.LINE_AA)

        # 速度箭头
        if speed > 5.0:
            direction = np.array([vx[index], vy[index]], dtype=np.float64)
            direction /= max(np.linalg.norm(direction), 1e-9)
            length = int(np.clip(speed * arrow_scale, ARROW_MIN, ARROW_MAX))
            tip = (
                int(round(center[0] + direction[0] * length)),
                int(round(center[1] + direction[1] * length)),
            )
            cv2.arrowedLine(frame, center, tip, color, 4, cv2.LINE_AA, tipLength=0.25)

        # 速度/加速度文字
        text_x = min(max(center[0] + radius + 12, 10), width - 260)
        text_y = max(center[1] - 10, 25)
        cv2.putText(
            frame, f"v = {speed:5.0f} px/s", (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"a = {accel:+6.0f} px/s^2", (text_x, text_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

        if not found_flags[index]:
            cv2.putText(
                frame, "detecting...", (text_x, text_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,
            )

        # 左上角图例
        legend = frame.copy()
        cv2.rectangle(legend, (10, 10), (330, 92), (0, 0, 0), -1)
        cv2.addWeighted(legend, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "Green ball: arrow = motion direction",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "arrow length ~ speed | red=accel blue=decel",
                    (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {index:03d}/{len(positions)-1:03d}  t={index/fps:5.2f}s",
                    (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        draw_speed_graph(frame, speeds, index, max_speed)
        writer.write(frame)
        index += 1

    cap.release()
    writer.release()
    return index


def main():
    here = Path(__file__).resolve().parent
    # 仓库根目录: homework/lyyqyb123-gif -> 上两级
    default_input = here.parents[1] / "homework.mp4"

    parser = argparse.ArgumentParser(description="识别绿球并标注运动方向与速度")
    parser.add_argument("--input", default=str(default_input), help="输入视频路径")
    parser.add_argument("--output", default=str(here / "output_annotated.mp4"),
                        help="输出标注视频路径")
    parser.add_argument("--csv", default=str(here / "trajectory.csv"),
                        help="输出轨迹数据 CSV")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    csv_path = Path(args.csv).expanduser()

    print(f"[1/3] 读取并跟踪: {input_path}")
    fps, total, positions, found_flags, radii = track_video(input_path)
    print(f"      共 {len(positions)} 帧, {fps:.1f} fps, 检测率 "
          f"{found_flags.mean() * 100:.1f}%")

    smoothed = smooth_positions(positions, SMOOTH_WINDOW)
    vx, vy, speed, acceleration = compute_kinematics(smoothed, fps)

    print(f"[2/3] 速度范围: {speed.min():.1f} ~ {speed.max():.1f} px/s, "
          f"平均 {speed.mean():.1f} px/s")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "x", "y", "vx", "vy", "speed_px_s",
                         "accel_px_s2", "detected"])
        for i in range(len(smoothed)):
            writer.writerow([
                i, f"{i / fps:.4f}",
                f"{smoothed[i, 0]:.2f}", f"{smoothed[i, 1]:.2f}",
                f"{vx[i]:.2f}", f"{vy[i]:.2f}",
                f"{speed[i]:.2f}", f"{acceleration[i]:.2f}",
                int(found_flags[i]),
            ])

    print(f"[3/3] 生成标注视频: {output_path}")
    rendered = render_video(input_path, output_path, fps, smoothed, found_flags,
                            radii, speed, vx, vy, acceleration)
    print(f"      完成，共写入 {rendered} 帧")
    print(f"      标注视频: {output_path}")
    print(f"      轨迹数据: {csv_path}")


if __name__ == "__main__":
    main()
