#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
week0-jlcv 作业：绿球检测 + 卡尔曼滤波实时估计

相对原版（中心滑动平均方案）的变化
--------------------------------
1. 用 6 状态卡尔曼滤波同时估计位置、速度、加速度：

       x = [px, py, vx, vy, ax, ay]^T

   预测模型为匀加速模型，过程噪声由 jerk 标准差控制。
2. 所有估计都是因果的：第 t 帧只使用第 0~t 帧，不再做需要未来帧的
   中心滑动平均，因此可以实时逐帧输出。
3. 单遍完成检测、滤波、绘制、写视频和 CSV，不需要第二遍读视频。
4. 同时输出切向加速度 a_t（速率变化率）和加速度向量模 |a|。

可视化说明
----------
- 洋红圆圈：卡尔曼滤波估计位置
- 绿色小十字：本帧检测到的原始质心（测量值）
- 青色十字：卡尔曼滤波对下一帧的预测位置
- 箭头：速度方向，长度正比于速率
- 箭头颜色：红=加速，蓝=减速，黄=速度基本不变

用法
----
    python3 detect_green_ball.py
    python3 detect_green_ball.py --input /path/to/homework.mp4 \\
        --output kf.mp4 --csv kf.csv

可调参数
--------
    --jerk-std   过程噪声（jerk 标准差）。越大越信任测量、响应越快，
                 但估计越抖；越小越平滑，但转向/加速时滞后。
    --meas-std   测量噪声（质心标准差，像素）。检测越准可以设得越小。
    --show       实时显示处理窗口（需要图形界面，按 q 退出）。
"""

import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ----------------------- 检测参数（与原版一致） -----------------------
HSV_LOW = (35, 60, 50)          # 绿色下限 (OpenCV: H 0~179)
HSV_HIGH = (90, 255, 255)       # 绿色上限
MIN_AREA = 400                  # 候选轮廓最小面积
MAX_AREA = 40000                # 候选轮廓最大面积
MIN_CIRCULARITY = 0.50          # 最小圆度 4*pi*A/P^2
ASPECT_RANGE = (0.70, 1.40)     # 宽高比范围，排除圆柱侧面

# ----------------------- 跟踪参数 -----------------------
MAX_JUMP = 120.0                # 候选与预测位置的最大距离（像素）
MAX_LOST = 15                   # 连续丢失多少帧后重新捕获
REACQUIRE_GATE = 300.0          # 重捕获时优先考虑预测位置附近的范围

# ----------------------- 卡尔曼参数 -----------------------
DEFAULT_JERK_STD = 1500.0       # 过程噪声：jerk 标准差 (px/s^3)
DEFAULT_MEAS_STD = 3.0          # 测量噪声：质心标准差 (px)
INITIAL_VEL_STD = 400.0         # 初始速度不确定度 (px/s)
INITIAL_ACC_STD = 1500.0        # 初始加速度不确定度 (px/s^2)

# ----------------------- 可视化参数 -----------------------
ARROW_MAX = 140                 # 箭头最大长度（像素）
ARROW_MIN = 18                  # 箭头最小长度（像素）
ACCEL_THRESHOLD = 600.0         # 加速度阈值（px/s^2），用于颜色判定
                                # KF 的瞬时加速度比原版平滑轨迹更敏感，
                                # 阈值设高一些可以减少箭头颜色来回闪
GRAPH_WINDOW_FRAMES = 300       # 速度曲线窗口（30fps 下约 10 秒）
DEFAULT_REFERENCE_SPEED = 300.0 # 速度窗口样本不足时的箭头参考速度
DEFAULT_GRAPH_MAX = 600.0       # 速度窗口样本不足时的曲线量程


# ======================================================================
# 检测与候选打分
# ======================================================================

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
    """没有位置先验时的候选打分：越圆、越方、面积越适中越好。"""
    return (
        candidate["circularity"]
        - 0.5 * abs(candidate["aspect"] - 1.0)
        - candidate["area"] / 200000.0
    )


def select_candidate(candidates, predicted):
    """在预测位置附近选代价最小的候选。"""
    best = None
    best_cost = None
    for cand in candidates:
        distance = float(np.linalg.norm(cand["center"] - predicted))
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
    return best


def select_reacquire(candidates, predicted):
    """长时间丢失后的重捕获：优先选预测位置附近的候选。"""
    near = [
        cand for cand in candidates
        if float(np.linalg.norm(cand["center"] - predicted)) <= REACQUIRE_GATE
    ]
    pool = near if near else candidates
    return max(pool, key=initial_score)


# ======================================================================
# 卡尔曼滤波
# ======================================================================

class BallKalmanFilter:
    """6 状态卡尔曼滤波：位置、速度、加速度。

    状态
    ----
        x = [px, py, vx, vy, ax, ay]^T

    预测（dt 为帧间隔）
    -------------------
        px' = px + vx*dt + 0.5*ax*dt^2
        py' = py + vy*dt + 0.5*ay*dt^2
        vx' = vx + ax*dt
        vy' = vy + ay*dt
        ax' = ax
        ay' = ay

    观测
    ----
        z = [px, py]^T + 噪声
    """

    def __init__(self, dt, jerk_std=DEFAULT_JERK_STD,
                 measurement_std=DEFAULT_MEAS_STD):
        self.dt = float(dt)
        self.measurement_std = float(measurement_std)
        self.initialized = False

        dt = self.dt
        half_dt2 = 0.5 * dt * dt

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0, half_dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, half_dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # 连续白噪声 jerk 模型下的离散过程噪声。
        # 对每个轴，状态 [p, v, a] 的 3x3 块为：
        #   q * [[dt^5/20, dt^4/8, dt^3/6],
        #        [dt^4/8,  dt^3/3, dt^2/2],
        #        [dt^3/6,  dt^2/2, dt    ]]
        q = float(jerk_std) ** 2
        block = q * np.array(
            [
                [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
            ],
            dtype=np.float64,
        )

        self.Q = np.zeros((6, 6), dtype=np.float64)
        idx_x = [0, 2, 4]       # px, vx, ax
        idx_y = [1, 3, 5]       # py, vy, ay
        self.Q[np.ix_(idx_x, idx_x)] = block
        self.Q[np.ix_(idx_y, idx_y)] = block

        self.R = np.eye(2, dtype=np.float64) * (self.measurement_std ** 2)

        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1e6

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------
    @property
    def position(self):
        return self.x[0:2, 0].copy()

    @property
    def velocity(self):
        return self.x[2:4, 0].copy()

    @property
    def acceleration(self):
        return self.x[4:6, 0].copy()

    # ------------------------------------------------------------------
    # 初始化 / 重置
    # ------------------------------------------------------------------
    def initialize(self, position):
        """用第一个测量初始化滤波器。速度、加速度先置零，但方差很大。"""
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[0, 0] = float(position[0])
        self.x[1, 0] = float(position[1])
        self.P = np.diag(
            [
                self.measurement_std ** 2,
                self.measurement_std ** 2,
                INITIAL_VEL_STD ** 2,
                INITIAL_VEL_STD ** 2,
                INITIAL_ACC_STD ** 2,
                INITIAL_ACC_STD ** 2,
            ]
        ).astype(np.float64)
        self.initialized = True

    def reset(self, position):
        """重捕获时调用：位置重置到新测量，速度/加速度清零。"""
        self.initialize(position)

    # ------------------------------------------------------------------
    # 预测 / 更新
    # ------------------------------------------------------------------
    def predict(self):
        """时间更新：把状态推进一帧，返回预测位置。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.position

    def predict_next(self):
        """返回下一帧的预测位置（不改变滤波器状态）。"""
        return (self.F @ self.x)[0:2, 0].copy()

    def update(self, position):
        """测量更新：用本帧检测到的质心修正状态。"""
        z = np.asarray(position, dtype=np.float64).reshape(2, 1)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation

        # Joseph 形式，数值上更稳定
        i_kh = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = i_kh @ self.P @ i_kh.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ======================================================================
# 绘制
# ======================================================================

def arrow_color(acceleration):
    """按切向加速度给箭头上色（BGR）。"""
    if acceleration > ACCEL_THRESHOLD:
        return (0, 0, 255), "accelerating"       # 红
    if acceleration < -ACCEL_THRESHOLD:
        return (255, 0, 0), "decelerating"       # 蓝
    return (0, 255, 255), "steady"               # 黄


def draw_speed_graph(frame, speeds, max_speed):
    """在右下角画最近一段时间的速度曲线（实时滚动）。"""
    height, width = frame.shape[:2]
    box_w, box_h = 280, 130
    x0 = width - box_w - 20
    y0 = height - box_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame, "speed (px/s, last 10 s)", (x0 + 8, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )

    left, right = x0 + 10, x0 + box_w - 10
    top, bottom = y0 + 30, y0 + box_h - 12
    cv2.line(frame, (left, bottom), (right, bottom), (160, 160, 160), 1)
    cv2.line(frame, (left, top), (left, bottom), (160, 160, 160), 1)

    n = len(speeds)
    if n > 1 and max_speed > 1e-6:
        points = []
        denominator = max(GRAPH_WINDOW_FRAMES - 1, 1)
        for i, speed in enumerate(speeds):
            px = left + int((right - left) * i / denominator)
            py = bottom - int(
                (bottom - top) * min(float(speed), max_speed) / max_speed
            )
            points.append((px, py))

        if len(points) > 1:
            cv2.polylines(
                frame, [np.array(points, dtype=np.int32)], False,
                (0, 200, 255), 2, cv2.LINE_AA,
            )
        cv2.circle(frame, points[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_annotations(frame, pos, vel, speed, accel_t, accel_mag,
                     measurement, predicted_next, radius, status,
                     arrow_scale, speeds_window, graph_max,
                     frame_index, fps):
    """把估计、测量、预测、箭头、文字、图例和速度曲线画到帧上。"""
    center = (int(round(pos[0])), int(round(pos[1])))
    color, _state = arrow_color(accel_t)

    # KF 估计位置
    cv2.circle(frame, center, radius + 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 4, (255, 0, 255), -1, cv2.LINE_AA)

    # 原始测量：绿色十字 + 黑色描边，避免叠在绿球上看不清
    if measurement is not None:
        m = (int(round(measurement[0])), int(round(measurement[1])))
        cv2.drawMarker(frame, m, (0, 0, 0), cv2.MARKER_CROSS, 16, 4,
                       cv2.LINE_AA)
        cv2.drawMarker(frame, m, (0, 255, 0), cv2.MARKER_CROSS, 16, 2,
                       cv2.LINE_AA)

    # 下一帧预测：青色斜十字，和测量十字区分开
    if predicted_next is not None:
        p = (int(round(predicted_next[0])), int(round(predicted_next[1])))
        cv2.drawMarker(frame, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 4,
                       cv2.LINE_AA)
        cv2.drawMarker(frame, p, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2,
                       cv2.LINE_AA)

    # 速度箭头
    if speed > 5.0:
        direction = vel / max(float(np.linalg.norm(vel)), 1e-9)
        length = int(np.clip(speed * arrow_scale, ARROW_MIN, ARROW_MAX))
        tip = (
            int(round(center[0] + direction[0] * length)),
            int(round(center[1] + direction[1] * length)),
        )
        cv2.arrowedLine(frame, center, tip, color, 4, cv2.LINE_AA,
                        tipLength=0.25)

    # 速度 / 加速度文字
    height, width = frame.shape[:2]
    text_x = min(max(center[0] + radius + 12, 10), width - 310)
    text_y = max(center[1] - 10, 30)
    cv2.putText(
        frame, f"v = {speed:5.0f} px/s", (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"a_t = {accel_t:+6.0f} px/s^2", (text_x, text_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"|a| = {accel_mag:5.0f} px/s^2", (text_x, text_y + 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, status, (text_x, text_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,
    )

    # 左上角图例
    legend = frame.copy()
    cv2.rectangle(legend, (10, 10), (600, 112), (0, 0, 0), -1)
    cv2.addWeighted(legend, 0.55, frame, 0.45, 0, frame)
    lines = [
        "magenta circle = KF estimate   green + = measurement   cyan x = next prediction",
        "arrow length ~ speed, color: red=accel blue=decel yellow=steady",
        f"frame {frame_index:04d}   t={frame_index / fps:6.2f}s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (18, 34 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA,
        )

    # 右下角速度曲线
    draw_speed_graph(frame, speeds_window, graph_max)


# ======================================================================
# 主流程
# ======================================================================

def main():
    here = Path(__file__).resolve().parent
    # 仓库根目录: homework/lyyqyb123-gif -> 上两级
    default_input = here.parents[1] / "homework.mp4"

    parser = argparse.ArgumentParser(
        description="绿球检测 + 卡尔曼滤波实时估计位置/速度/加速度"
    )
    parser.add_argument("--input", default=str(default_input),
                        help="输入视频路径")
    parser.add_argument("--output",
                        default=str(here / "output_annotated_kalman.mp4"),
                        help="输出标注视频路径")
    parser.add_argument("--csv",
                        default=str(here / "trajectory_kalman.csv"),
                        help="输出轨迹数据 CSV")
    parser.add_argument("--jerk-std", type=float, default=DEFAULT_JERK_STD,
                        help="过程噪声：jerk 标准差 (px/s^3)")
    parser.add_argument("--meas-std", type=float, default=DEFAULT_MEAS_STD,
                        help="测量噪声：质心标准差 (px)")
    parser.add_argument("--show", action="store_true",
                        help="实时显示处理窗口（按 q 退出）")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    csv_path = Path(args.csv).expanduser()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频尺寸: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    kf = BallKalmanFilter(
        dt=1.0 / fps,
        jerk_std=args.jerk_std,
        measurement_std=args.meas_std,
    )

    speeds_window = deque(maxlen=GRAPH_WINDOW_FRAMES)
    lost = 0
    last_radius = 30.0
    frame_index = 0
    detected_count = 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    print(f"[1/1] 实时检测 + 卡尔曼滤波: {input_path}")
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame", "time_s", "x", "y", "vx", "vy", "ax", "ay",
                "speed_px_s", "accel_t_px_s2", "accel_mag_px_s2", "detected",
            ])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                candidates = candidates_in_frame(frame)
                measurement = None
                found = False
                reacquired = False

                # ---------------- 第一帧：初始化 ----------------
                if not kf.initialized:
                    if candidates:
                        best = max(candidates, key=initial_score)
                        kf.initialize(best["center"])
                        measurement = best["center"]
                        last_radius = best["radius"]
                        found = True
                        detected_count += 1
                        lost = 0

                    if kf.initialized:
                        pos = kf.position
                        vel = kf.velocity
                        acc = kf.acceleration
                    else:
                        # 还没有任何测量，先用画面中心占位
                        pos = np.array([width / 2.0, height / 2.0],
                                       dtype=np.float64)
                        vel = np.zeros(2, dtype=np.float64)
                        acc = np.zeros(2, dtype=np.float64)

                # ---------------- 后续帧：预测 → 匹配 → 更新 ----------------
                else:
                    predicted = kf.predict()
                    best = select_candidate(candidates, predicted)

                    if best is not None:
                        kf.update(best["center"])
                        measurement = best["center"]
                        last_radius = best["radius"]
                        found = True
                        detected_count += 1
                        lost = 0
                    else:
                        lost += 1
                        # 长时间丢失：优先在预测位置附近重新捕获
                        if lost > MAX_LOST and candidates:
                            best = select_reacquire(candidates, predicted)
                            kf.reset(best["center"])
                            measurement = best["center"]
                            last_radius = best["radius"]
                            found = True
                            reacquired = True
                            detected_count += 1
                            lost = 0

                    pos = kf.position
                    vel = kf.velocity
                    acc = kf.acceleration

                predicted_next = kf.predict_next() if kf.initialized else None

                # ---------------- 运动学量 ----------------
                speed = float(np.linalg.norm(vel))
                # 切向加速度：速率的变化率，带正负号
                accel_t = float(np.dot(vel, acc) / max(speed, 1e-6))
                # 加速度向量模：包含转弯带来的法向分量
                accel_mag = float(np.linalg.norm(acc))

                speeds_window.append(speed)
                speed_values = np.fromiter(speeds_window, dtype=np.float64)

                nonzero = speed_values[speed_values > 1.0]
                if nonzero.size:
                    reference_speed = max(
                        100.0, float(np.percentile(nonzero, 90))
                    )
                else:
                    reference_speed = DEFAULT_REFERENCE_SPEED
                arrow_scale = 100.0 / reference_speed

                if speed_values.size >= 5:
                    graph_max = max(
                        100.0, float(np.percentile(speed_values, 99))
                    )
                else:
                    graph_max = DEFAULT_GRAPH_MAX

                # ---------------- 状态文字 ----------------
                if not kf.initialized:
                    status = "searching..."
                elif reacquired:
                    status = "reacquired"
                elif found:
                    status = "tracking"
                else:
                    status = "KF predicting..."

                # ---------------- 绘制 ----------------
                draw_annotations(
                    frame, pos, vel, speed, accel_t, accel_mag,
                    measurement, predicted_next,
                    max(12, int(round(last_radius))), status,
                    arrow_scale, speeds_window, graph_max,
                    frame_index, fps,
                )

                # ---------------- 输出 ----------------
                writer.write(frame)
                csv_writer.writerow([
                    frame_index, f"{frame_index / fps:.4f}",
                    f"{pos[0]:.2f}", f"{pos[1]:.2f}",
                    f"{vel[0]:.2f}", f"{vel[1]:.2f}",
                    f"{acc[0]:.2f}", f"{acc[1]:.2f}",
                    f"{speed:.2f}",
                    f"{accel_t:.2f}", f"{accel_mag:.2f}",
                    int(found),
                ])
                frame_index += 1

                if args.show:
                    cv2.imshow("green ball - kalman", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    detection_rate = 100.0 * detected_count / max(frame_index, 1)
    process_fps = frame_index / max(elapsed, 1e-9)

    print(f"      共处理 {frame_index} 帧，检测率 {detection_rate:.1f}%")
    print(f"      耗时 {elapsed:.2f}s，平均 {process_fps:.1f} fps，"
          f"实时倍率 {process_fps / fps:.2f}x")
    print(f"      标注视频: {output_path}")
    print(f"      轨迹数据: {csv_path}")


if __name__ == "__main__":
    main()
