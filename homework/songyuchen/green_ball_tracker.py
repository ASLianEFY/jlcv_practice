#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿球识别与运动方向 / 速度标注
============================

作业要求：识别视频中的绿球，并用箭头标注其运动方向和速度变化。

处理流程：
    1. BGR -> HSV，用绿色阈值分割出所有绿色区域；
    2. 形态学开/闭运算去掉噪点、补上球面高光造成的小洞；
    3. 对每个轮廓计算 "圆度" 和 "长宽比"：
         - 绿球是正圆，圆度接近 1，长宽比接近 1；
         - 绿色圆柱的投影是圆角矩形，圆度明显更低；
       用这两个特征把圆柱筛掉，只留下球；
    4. 用上一帧的速度预测这一帧球的位置，在候选里选离预测最近的，
       避免画面里出现别的绿色物体时箭头乱跳；
    5. 速度 = 相邻两帧球心位移 × fps，再做指数平滑，避免箭头抖动；
    6. 画箭头（方向 = 速度方向，长度 ∝ 速度大小）、轨迹和速度曲线。

运行示例：
    python3 green_ball_tracker.py                       # 弹窗播放
    python3 green_ball_tracker.py --headless --output out.mp4 --csv track.csv
    python3 green_ball_tracker.py --show-mask           # 右上角显示绿色掩膜
"""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
# OpenCV 的 H 范围是 0~179（不是常见的 0~360），纯绿色 H≈60。
# 下限放宽到 35、上限放到 90，是为了兼容球面明暗过渡（边缘偏黄绿/青绿）。
GREEN_LOWER = np.array([35, 60, 50], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

MIN_AREA = 800          # 小于这个面积当作噪点（球最小时约 8800 像素）
MAX_AREA = 80000        # 大于这个面积当作误检（球最大时约 29000 像素）
MIN_CIRCULARITY = 0.80  # 圆度 = 轮廓面积 / 最小外接圆面积；球实测 0.86~0.94
MAX_ASPECT_RATIO = 1.30 # 最小外接矩形长宽比；球≈1.0，圆柱实测 1.4~1.8

VELOCITY_SMOOTHING = 0.55  # 速度指数平滑系数：新值占 55%，旧值占 45%
MAX_LOST_FRAMES = 10       # 连续丢失多少帧后放弃目标
ARROW_SCALE = 0.22         # 箭头长度 = 速度(px/s) × 该系数
ARROW_MIN_LEN = 30         # 箭头最短长度，保证低速时也看得见方向
ARROW_MAX_LEN = 260        # 箭头最长长度，避免高速时箭头冲出画面


# ---------------------------------------------------------------------------
# 检测：从一帧中找出所有“像绿球”的候选
# ---------------------------------------------------------------------------
def build_green_mask(frame):
    """把画面里的绿色区域变成二值掩膜。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # 椭圆核比矩形核更贴合球体边缘，不会把圆“削方”。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去零星绿点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 补高光小洞
    return mask


def find_ball_candidates(mask):
    """返回所有通过面积、圆度、长宽比筛选的候选。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # 最小外接圆：用来算圆度
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if radius < 1:
            continue
        circularity = area / (np.pi * radius * radius)

        # 最小外接矩形：用来算长宽比，进一步排除圆柱/长条
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        short_side = max(min(rect_w, rect_h), 1e-6)
        aspect_ratio = max(rect_w, rect_h) / short_side

        if circularity < MIN_CIRCULARITY or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # 用图像矩求质心，比外接圆圆心更稳定
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        candidates.append({
            "center": np.array([cx, cy], dtype=np.float64),
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
        })
    return candidates


# ---------------------------------------------------------------------------
# 跟踪：连续性选择 + 速度估计
# ---------------------------------------------------------------------------
class BallTracker:
    """维护球的位置和速度，负责“选哪个候选”和“丢了怎么办”。"""

    def __init__(self):
        self.center = None                 # 当前球心 (x, y)
        self.velocity = np.zeros(2)        # 速度，单位 px/帧
        self.lost_frames = 0
        self.detected = False              # 本帧是否真的检测到（而非预测）
        self.last_candidate = None

    def predict(self):
        """按匀速模型预测下一帧位置。"""
        if self.center is None:
            return None
        return self.center + self.velocity

    def update(self, candidates):
        """输入本帧所有候选，更新位置/速度，返回被选中的候选或 None。"""
        predicted = self.predict()

        if candidates:
            if predicted is None:
                # 第一帧：没有历史信息，取圆度最高的候选
                best = max(candidates, key=lambda c: c["circularity"])
            else:
                # 后面每帧：圆度高 + 离预测位置近来打分
                def score(c):
                    distance = np.linalg.norm(c["center"] - predicted)
                    continuity = max(0.0, 1.0 - distance / 300.0)
                    return c["circularity"] + 0.8 * continuity

                best = max(candidates, key=score)

            new_center = best["center"]
            if self.center is None:
                self.velocity[:] = 0.0
            else:
                raw_velocity = new_center - self.center          # px/帧
                # 指数平滑：箭头方向才不会因为球面高光抖动而乱摆
                self.velocity = (VELOCITY_SMOOTHING * raw_velocity
                                 + (1.0 - VELOCITY_SMOOTHING) * self.velocity)
            self.center = new_center
            self.lost_frames = 0
            self.detected = True
            self.last_candidate = best
            return best

        # 没检测到：短时间用预测值续接，避免箭头突然消失
        self.lost_frames += 1
        self.detected = False
        if self.center is not None and self.lost_frames <= MAX_LOST_FRAMES:
            self.center = predicted
            return None

        # 丢失太久，清空状态，等待重新捕获
        self.center = None
        self.velocity[:] = 0.0
        self.last_candidate = None
        return None


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def speed_to_color(speed, max_speed=900.0):
    """速度映射到颜色：慢 -> 黄，快 -> 红。"""
    t = float(np.clip(speed / max_speed, 0.0, 1.0))
    # BGR：黄色 (0,255,255) -> 红色 (0,0,255)
    return (0, int(255 * (1.0 - t)), 255)


def draw_arrow(frame, center, velocity, speed):
    """画运动箭头：方向 = 速度方向，长度 ∝ 速度大小。"""
    length = float(np.clip(speed * ARROW_SCALE, ARROW_MIN_LEN, ARROW_MAX_LEN))
    norm = np.linalg.norm(velocity)
    if norm < 1e-6:
        return
    direction = velocity / norm
    start = (int(round(center[0])), int(round(center[1])))
    end = (int(round(center[0] + direction[0] * length)),
           int(round(center[1] + direction[1] * length)))
    color = speed_to_color(speed)
    cv2.arrowedLine(frame, start, end, color, 5, cv2.LINE_AA, tipLength=0.28)


def draw_hud(frame, speed, accel, detected, candidate, lost_frames=0):
    """左上角文字信息。"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (470, 175), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    status = "TRACKING" if detected else f"PREDICTING (lost {lost_frames}f)"
    color = (0, 255, 0) if detected else (0, 200, 255)
    cv2.putText(frame, f"Speed : {speed:7.1f} px/s", (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Accel : {accel:+7.1f} px/s^2", (22, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status, (22, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if candidate is not None:
        info = (f"circ={candidate['circularity']:.2f}  "
                f"aspect={candidate['aspect_ratio']:.2f}  "
                f"r={candidate['radius']:.0f}px")
        cv2.putText(frame, info, (22, 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)


def draw_speed_graph(frame, history):
    """左下角画速度-时间曲线，直观展示“速度变化”。"""
    if len(history) < 2:
        return
    h, w = frame.shape[:2]
    x0, y0 = 20, h - 150
    gw, gh = 320, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + gw, y0 + gh), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    max_speed = max(300.0, max(history))
    pts = []
    for i, value in enumerate(history):
        px = int(x0 + 8 + (gw - 16) * i / max(1, len(history) - 1))
        py = int(y0 + gh - 10 - (gh - 28) * min(value, max_speed) / max_speed)
        pts.append((px, py))
    cv2.polylines(frame, [np.array(pts, np.int32)], False, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"speed history (max {max_speed:.0f})", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def draw_trail(frame, trail):
    """画球心轨迹，线段颜色跟随当时的速度。"""
    for i in range(1, len(trail)):
        (x1, y1, _), (x2, y2, speed) = trail[i - 1], trail[i]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                 speed_to_color(speed), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def default_video_path():
    """脚本放在 homework/<name>/ 下，仓库根目录在往上两级。"""
    return Path(__file__).resolve().parents[2] / "homework.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="识别绿球并标注运动方向/速度")
    parser.add_argument("--video", type=Path, default=None,
                        help="输入视频，默认使用仓库根目录的 homework.mp4")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出标注后的视频，例如 out.mp4")
    parser.add_argument("--csv", type=Path, default=None,
                        help="把逐帧轨迹和速度导出为 CSV")
    parser.add_argument("--headless", action="store_true",
                        help="不打开窗口（SSH/无显示器时使用）")
    parser.add_argument("--show-mask", action="store_true",
                        help="右上角显示绿色掩膜小窗")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="显示窗口缩放比例，例如 0.75")
    parser.add_argument("--trail", type=int, default=45,
                        help="轨迹保留的帧数")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video if args.video is not None else default_video_path()
    if not video_path.exists():
        print(f"[错误] 找不到视频: {video_path}")
        print("请用 --video 指定视频路径。")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[错误] 无法创建输出视频: {args.output}")
            return 1

    csv_file = csv_writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_s", "x", "y", "vx_px_s", "vy_px_s",
                             "speed_px_s", "accel_px_s2", "radius", "area",
                             "circularity", "detected"])

    tracker = BallTracker()
    trail = deque(maxlen=args.trail)
    speed_history = deque(maxlen=240)
    prev_speed = 0.0
    smoothed_accel = 0.0
    total_frames = 0
    detected_frames = 0
    speeds = []

    print(f"视频: {video_path}   {width}x{height} @ {fps:.1f} fps")
    print("开始处理，按 q 或 ESC 退出播放窗口 ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        mask = build_green_mask(frame)
        candidates = find_ball_candidates(mask)
        chosen = tracker.update(candidates)

        speed = float(np.linalg.norm(tracker.velocity) * fps)
        raw_accel = (speed - prev_speed) * fps
        smoothed_accel = 0.75 * smoothed_accel + 0.25 * raw_accel
        prev_speed = speed

        if tracker.detected:
            detected_frames += 1
            speeds.append(speed)
            trail.append((tracker.center[0], tracker.center[1], speed))
            speed_history.append(speed)

        # ---- 绘制 ----
        if tracker.center is not None:
            cx, cy = int(tracker.center[0]), int(tracker.center[1])
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
            if tracker.detected and chosen is not None:
                cv2.circle(frame, (cx, cy), int(chosen["radius"]) + 4,
                           (255, 255, 0), 2, cv2.LINE_AA)
            # 预测状态下速度是“旧值”，画箭头会误导，所以只在真正检出时画
            if tracker.detected:
                draw_arrow(frame, tracker.center, tracker.velocity, speed)

        draw_trail(frame, trail)
        draw_speed_graph(frame, speed_history)
        draw_hud(frame, speed, smoothed_accel, tracker.detected, chosen,
                 tracker.lost_frames)

        if args.show_mask:
            small = cv2.resize(mask, (width // 4, height // 4))
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            frame[10:10 + small.shape[0], width - 10 - small.shape[1]:width - 10] = small

        if csv_writer is not None:
            frame_index = total_frames - 1
            if tracker.center is None:
                row = [frame_index, frame_index / fps, "", "", "", "",
                       "", "", "", "", "", 0]
            else:
                row = [frame_index, frame_index / fps,
                       f"{tracker.center[0]:.2f}", f"{tracker.center[1]:.2f}",
                       f"{tracker.velocity[0] * fps:.2f}",
                       f"{tracker.velocity[1] * fps:.2f}", f"{speed:.2f}",
                       f"{smoothed_accel:.2f}",
                       f"{chosen['radius']:.2f}" if chosen else "",
                       f"{chosen['area']:.1f}" if chosen else "",
                       f"{chosen['circularity']:.4f}" if chosen else "",
                       1 if tracker.detected else 0]
            csv_writer.writerow(row)

        if writer is not None:
            writer.write(frame)

        if not args.headless:
            show = frame
            if args.scale != 1.0:
                show = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            try:
                cv2.imshow("Green Ball Tracker", show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            except cv2.error:
                print("[提示] 当前 OpenCV 无法打开窗口（headless 版本），"
                      "自动切换为无窗口模式；结果请用 --output 保存。")
                args.headless = True

        if total_frames % 50 == 0:
            print(f"  已处理 {total_frames} 帧，当前速度 {speed:.0f} px/s")

    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()
    if not args.headless:
        cv2.destroyAllWindows()

    # ---- 统计 ----
    print("\n===== 处理结果 =====")
    print(f"总帧数      : {total_frames}")
    print(f"检出帧数    : {detected_frames} "
          f"({detected_frames / max(1, total_frames) * 100:.1f}%)")
    if speeds:
        print(f"速度均值    : {np.mean(speeds):.1f} px/s")
        print(f"速度中位数  : {np.median(speeds):.1f} px/s")
        print(f"速度最大值  : {np.max(speeds):.1f} px/s")
    if args.output is not None:
        print(f"输出视频    : {args.output}")
    if args.csv is not None:
        print(f"轨迹 CSV    : {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
