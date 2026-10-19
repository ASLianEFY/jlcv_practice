#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业：识别视频中的绿球，并用箭头标注其运动方向与速度变化
================================================================

运行
----
    python3 green_ball_tracker.py                                  # 弹窗播放
    python3 green_ball_tracker.py --output out.mp4                  # 播放 + 保存
    python3 green_ball_tracker.py --headless --output out.mp4       # 无窗口（SSH/服务器）
    python3 green_ball_tracker.py --video other.mp4 --show-mask     # 指定视频 + 显示掩膜

算法流程
--------
    1. 颜色分割   BGR -> HSV，用绿色阈值取出候选像素（HSV 只看色相，抗明暗变化）
    2. 形态学去噪 开运算去孤立噪点，闭运算补球面高光造成的空洞
    3. 形状筛选   圆度 = 面积 / 最小外接圆面积，把绿球和场上的绿色圆柱分开 ★核心
    4. 候选打分   圆度 + 位置连续性，避免箭头在多目标间来回跳
    5. 卡尔曼滤波  常速度模型平滑位置、估计速度；短时遮挡可预测续接
    6. 可视化     箭头（长度∝速度）+ 变速轨迹 + 速度曲线 + HUD

参数标定依据（本仓库 homework.mp4 实测，1280x720 / 30fps / 473 帧）
--------------------------------------------------------------------
    球    圆度 min=0.7537  median=0.9012  max=0.9353   半径 56~91px
    圆柱  圆度 max=0.7204  p99=0.7112                   <- 与球完全分离
    间隙 0.033，阈值取 0.737 落在正中间，473/473 帧全部正确区分。

屏幕文字为什么是英文：cv2.putText 只支持 Hershey 点阵字体，没有中文字形，
写中文会变成一串 "?"。所以画面内统一用 ASCII，终端日志仍用中文。

环境：Python 3.8+ / OpenCV 4.x / NumPy。已在本机 Ubuntu 22.04 + OpenCV 4.5.4 验证。

作者：southwardaai
"""

import argparse
import csv
import math
import os
import sys
from collections import deque

import cv2
import numpy as np

# ===========================================================================
# 一、可调参数（换视频 / 换光线后，需要动的就是这一段）
# ===========================================================================

# 绿色 HSV 范围。OpenCV 的 H 是 0~179，纯绿 H=60。
# 上下放宽到 35~95 是为了兼容球面明暗过渡：亮部偏黄绿、暗部偏青绿。
GREEN_LOWER = np.array([35, 50, 50], dtype=np.uint8)
GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# 圆度阈值：面积 / 最小外接圆面积，完美圆 = 1.0。
# 实测球最低 0.7537、圆柱最高 0.7204，取中间值分开。
MIN_CIRCULARITY = 0.737

# 面积上下限（px^2）。实测球 8961~23632；下限留余量滤噪点，上限滤桌面反光粘连。
MIN_AREA = 3000
MAX_AREA = 60000

MAX_LOST_FRAMES = 12     # 连续丢失超过这么多帧，才认为目标真的消失

# ---- 箭头与速度显示 ----
ARROW_GAIN = 0.30        # 箭头长度(px) = 速度(px/s) × 该系数
ARROW_MIN = 18           # 箭头最短长度，避免低速时箭头缩成一个点
ARROW_MAX = 220          # 箭头最长长度，避免高速时冲出画面
SPEED_MIN_SHOW = 20.0    # 低于此速度(px/s)视为静止，不画箭头
SPEED_VMAX = 900.0       # 颜色映射上限：0 → 绿，900 → 红

TRAIL_LEN = 50           # 轨迹保留帧数
SPEED_HISTORY_LEN = 180  # 速度曲线保留帧数


# ===========================================================================
# 二、小工具
# ===========================================================================

def clamp(value, low, high):
    """把 value 限制在 [low, high] 区间。"""
    return max(low, min(high, value))


def lerp_color(c1, c2, t):
    """两个 BGR 颜色按 t∈[0,1] 线性插值。"""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def speed_color(speed_px_s):
    """速度 -> BGR 颜色：慢=绿，中=黄，快=红。"""
    t = clamp(speed_px_s / SPEED_VMAX, 0.0, 1.0)
    if t < 0.5:
        return lerp_color((60, 200, 60), (0, 220, 255), t / 0.5)
    return lerp_color((0, 220, 255), (40, 50, 255), (t - 0.5) / 0.5)


def find_default_video():
    """脚本在 homework/<name>/ 下，视频在仓库根目录，按相对位置向上找。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "homework.mp4"),
        os.path.join(os.getcwd(), "homework.mp4"),
        os.path.join(os.getcwd(), "..", "..", "homework.mp4"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)
    return os.path.normpath(candidates[0])


# ===========================================================================
# 三、检测器：颜色分割 + 形态学 + 形状筛选
# ===========================================================================

class GreenBallDetector:
    """
    从一帧里找出"最像绿球"的候选。

    为什么不能只做颜色阈值？
        场上有一根绿色圆柱，颜色和球几乎一样，光靠 HSV 阈值一定把它一起框进来。
        区别在形状：圆柱的正面投影是圆角矩形，球是正圆。
        用圆度（面积/最小外接圆面积）就能把两者分开，这是本作业的关键一步。
    """

    def __init__(self):
        # 椭圆结构元比矩形更贴合球体边缘，不会把圆的边"削方"
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def build_mask(self, frame):
        """返回绿色二值掩膜（白=绿色像素）。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        # 开运算（先腐蚀后膨胀）：去掉零散噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        # 闭运算（先膨胀后腐蚀）：补上球面高光/阴影造成的小洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    @staticmethod
    def contour_features(contour):
        """计算单个轮廓的 (面积, 外接圆半径, 圆心, 圆度)。"""
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0, 0.0, (0.0, 0.0), 0.0
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius <= 1e-6:
            return area, 0.0, (cx, cy), 0.0
        circularity = area / (math.pi * radius * radius)   # 完美圆 = 1.0
        return area, radius, (cx, cy), circularity

    def detect(self, frame, hint=None):
        """
        返回最可信的候选 dict，没有则返回 None。
        hint：上一帧预测的球心 (x, y)。画面里有多个绿色候选时用它做连续性打分，
              避免这一帧锁球、下一帧跳到圆柱上。
        """
        mask = self.build_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0
        for contour in contours:
            area, radius, center, circularity = self.contour_features(contour)

            # --- 三重筛选 ---
            if not (MIN_AREA <= area <= MAX_AREA):
                continue                     # 太小是噪点，太大是反光/粘连
            if circularity < MIN_CIRCULARITY:
                continue                     # 不够圆 -> 多半是绿色圆柱

            # --- 打分：圆度为主，离上一帧越近加分越多 ---
            score = circularity
            if hint is not None:
                dist = math.hypot(center[0] - hint[0], center[1] - hint[1])
                score += 0.8 * max(0.0, 1.0 - dist / 260.0)

            if score > best_score:
                best_score = score
                best = {
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "contour": contour,
                }
        return best


# ===========================================================================
# 四、跟踪器：卡尔曼滤波 + 速度估计
# ===========================================================================

class BallTracker:
    """
    状态向量 X = [x, y, vx, vy]（像素、像素/帧），常速度模型：
        x' = x + vx,  y' = y + vy,  vx' = vx,  vy' = vy

    每帧两步：predict() 外推，correct() 用检测结果修正。
    - 有检测：正常更新，位置和速度都被平滑；
    - 短时丢失（手遮挡等）：用预测值顶上，最多撑 MAX_LOST_FRAMES 帧。
    """

    def __init__(self, fps, vel_alpha=0.55):
        self.fps = max(float(fps), 1.0)
        self.vel_alpha = vel_alpha    # 速度指数平滑系数：越大越跟手，越小越平滑

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        self.initialized = False
        self.lost_frames = 0
        self.center = None         # 当前帧球心（滤波后）
        self.prev_center = None    # 上一个有效检测帧的球心
        self.vx = 0.0              # 平滑后速度，px/帧
        self.vy = 0.0
        self.accel = 0.0           # 速率变化率 px/s^2

    # ---- 内部 ----
    def _init(self, center):
        self.kf.statePost = np.array(
            [[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True
        self.lost_frames = 0

    def _predict(self):
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def _reset(self):
        self.initialized = False
        self.center = None
        self.prev_center = None
        self.vx = self.vy = 0.0
        self.accel = 0.0
        self.lost_frames = 0

    # ---- 对外 ----
    def predicted_center(self):
        """
        只按当前状态外推一步，返回预测球心，**不修改滤波器状态**。

        注意：不能直接调 cv2.KalmanFilter.predict() 来取预测值 —— 它会同时推进
        内部 statePost，一帧里调两次就等于外推了两步，速度会翻倍。
        """
        if not self.initialized:
            return None
        s = self.kf.statePost
        return (float(s[0, 0] + s[2, 0]), float(s[1, 0] + s[3, 0]))

    def update(self, measured):
        """
        measured：本帧检测到的球心，None 表示本帧没检测到。
        返回 (center, detected)。
        """
        if not self.initialized:
            if measured is None:
                return None, False
            self._init(measured)
            self.center = measured
            self.prev_center = measured
            return measured, True

        if measured is not None:
            self._predict()          # 先外推
            self.kf.correct(np.array([[measured[0]], [measured[1]]], dtype=np.float32))
            state = self.kf.statePost
            center = (float(state[0, 0]), float(state[1, 0]))
            self.lost_frames = 0
            detected = True
        else:
            # 丢失：用预测值顶上，速度慢慢衰减，箭头不会突然乱指
            center = self._predict()
            self.lost_frames += 1
            if self.lost_frames > MAX_LOST_FRAMES:
                self._reset()
                return None, False
            self.vx *= 0.85
            self.vy *= 0.85
            detected = False

        # ---- 速度：用滤波后的帧间位移，再做指数平滑 ----
        if detected and self.prev_center is not None:
            dvx = center[0] - self.prev_center[0]
            dvy = center[1] - self.prev_center[1]

            speed_before = self.speed_px_s
            self.vx = self.vel_alpha * dvx + (1.0 - self.vel_alpha) * self.vx
            self.vy = self.vel_alpha * dvy + (1.0 - self.vel_alpha) * self.vy
            speed_after = self.speed_px_s

            # 加速度只用于显示，再平滑一次，否则数字跳得看不清
            dt = 1.0 / self.fps
            self.accel = 0.25 * ((speed_after - speed_before) / dt) + 0.75 * self.accel

        if detected:
            self.prev_center = center
        self.center = center
        return center, detected

    # ---- 供绘制使用的派生量 ----
    @property
    def speed_px_s(self):
        """速率，像素/秒。"""
        return math.hypot(self.vx, self.vy) * self.fps

    @property
    def direction(self):
        """单位方向向量 (dx, dy)；速度≈0 时返回 (0, 0)。"""
        norm = math.hypot(self.vx, self.vy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (self.vx / norm, self.vy / norm)


# ===========================================================================
# 五、可视化
# ===========================================================================

class Visualizer:
    """把结果画到画面上：箭头、变速轨迹、速度曲线、HUD。"""

    def __init__(self, trail_len=TRAIL_LEN, history_len=SPEED_HISTORY_LEN):
        self.trail = deque(maxlen=max(trail_len, 1))       # [(x, y, speed)]
        self.speeds = deque(maxlen=max(history_len, 2))    # 速度时间序列

    def push(self, center, speed):
        self.trail.append((int(center[0]), int(center[1]), speed))
        self.speeds.append(speed)

    def draw_trail(self, frame):
        """轨迹线按"当时的速率"着色，一眼看出哪段快、哪段慢。"""
        pts = list(self.trail)
        for i in range(1, len(pts)):
            x0, y0, s0 = pts[i - 1]
            x1, y1, s1 = pts[i]
            cv2.line(frame, (x0, y0), (x1, y1), speed_color(0.5 * (s0 + s1)), 2, cv2.LINE_AA)

    @staticmethod
    def arrow_end(center, direction, speed):
        """按当前速率算箭头终点；画箭头和写标注共用，保证两者对齐。"""
        length = clamp(speed * ARROW_GAIN, ARROW_MIN, ARROW_MAX)
        return (center[0] + direction[0] * length,
                center[1] + direction[1] * length)

    @classmethod
    def draw_arrow(cls, frame, center, direction, speed):
        """
        箭头起点=球心，方向=速度方向，长度∝速率，颜色随速率由绿到红。
        先画深色描边再画彩色，任何背景上都看得清。
        """
        if speed < SPEED_MIN_SHOW:
            return
        start = (int(round(center[0])), int(round(center[1])))
        ex, ey = cls.arrow_end(center, direction, speed)
        end = (int(round(ex)), int(round(ey)))

        color = speed_color(speed)
        cv2.arrowedLine(frame, start, end, (20, 20, 20), 6, cv2.LINE_AA, tipLength=0.24)
        cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.24)
        # 起点小圆点：强调"箭头原点就是球心"
        cv2.circle(frame, start, 4, (0, 0, 255), -1, cv2.LINE_AA)

    @classmethod
    def draw_speed_label(cls, frame, center, direction, speed):
        """在箭头末端标出速率数值，带深色底框，压在轨迹上也看得清。"""
        if speed < SPEED_MIN_SHOW:
            return
        text = "%.0f px/s" % speed
        scale, thickness = 0.6, 2
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        ex, ey = cls.arrow_end(center, direction, speed)
        nx, ny = -direction[1], direction[0]     # 箭头方向的法向量，把标签挪到侧面
        h, w = frame.shape[:2]
        tx = int(clamp(ex + nx * 22 + 6, 6, w - tw - 12))
        ty = int(clamp(ey + ny * 22 - 6, th + 8, h - 8))

        cv2.rectangle(frame, (tx - 5, ty - th - 6), (tx + tw + 5, ty + base), (25, 25, 25), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    speed_color(speed), thickness, cv2.LINE_AA)

    @staticmethod
    def put_text(frame, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
        """带黑色描边的字，深浅背景都能读。"""
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)

    def draw_speed_curve(self, frame):
        """
        左下角"速度-时间"曲线，直接对应作业要求的"速度变化"。
        纵轴自动缩放到历史最大值，横轴是最近 SPEED_HISTORY_LEN 帧。
        """
        speeds = list(self.speeds)
        if len(speeds) < 2:
            return
        h, w = frame.shape[:2]
        if h < 240 or w < 380:
            return      # 分辨率太小放不下曲线，跳过（不影响主功能）

        panel_w, panel_h = 320, 130
        x0, y0 = 10, h - panel_h - 12
        roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.0, 0.0, roi)   # 半透明黑底
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1),
                      (90, 90, 90), 1, cv2.LINE_AA)

        vmax = max(max(speeds), SPEED_VMAX * 0.25)
        plot_x, plot_y = x0 + 34, y0 + 10
        plot_w, plot_h = panel_w - 46, panel_h - 34

        # 网格与纵轴刻度
        for frac in (0.0, 0.5, 1.0):
            gy = int(plot_y + plot_h * (1.0 - frac))
            cv2.line(frame, (plot_x, gy), (plot_x + plot_w, gy), (70, 70, 70), 1, cv2.LINE_AA)
            self.put_text(frame, "%d" % int(vmax * frac), (x0 + 4, gy + 5),
                          scale=0.38, color=(200, 200, 200))
        self.put_text(frame, "speed px/s", (plot_x, y0 + panel_h - 4),
                      scale=0.4, color=(180, 180, 180))

        # 曲线：每个点用速度对应的颜色
        n = len(speeds)
        pts = []
        for i, s in enumerate(speeds):
            px = plot_x + int(plot_w * i / (n - 1))
            py = plot_y + int(plot_h * (1.0 - clamp(s / vmax, 0.0, 1.0)))
            pts.append((px, py, s))
        for i in range(1, n):
            cv2.line(frame, pts[i - 1][:2], pts[i][:2], speed_color(pts[i][2]), 2, cv2.LINE_AA)

        # 当前值：末尾画点并标数字
        last_x, last_y, last_s = pts[-1]
        cv2.circle(frame, (last_x, last_y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        self.put_text(frame, "%.0f" % last_s, (min(last_x + 6, x0 + panel_w - 40), last_y + 5),
                      scale=0.45, color=speed_color(last_s))

    def draw_hud(self, frame, tracker, detected, frame_idx, total):
        if detected:
            state = "TRACKING"
        elif tracker.center is not None:
            state = "PREDICTING"
        else:
            state = "LOST"
        lines = [
            "frame %d/%d" % (frame_idx + 1, total),
            "state: %s" % state,
            "speed: %.0f px/s" % tracker.speed_px_s,
            "accel: %+.0f px/s^2" % tracker.accel,
        ]
        for i, text in enumerate(lines):
            self.put_text(frame, text, (12, 26 + i * 26), scale=0.62)

    @staticmethod
    def draw_target(frame, center, radius, contour):
        """目标标记：外接圆 + 轮廓 + 十字准星（自瞄里常用的瞄准标记）。"""
        c = (int(round(center[0])), int(round(center[1])))
        cv2.circle(frame, c, int(max(radius, 6)), (0, 255, 0), 2, cv2.LINE_AA)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (c[0] - 10, c[1]), (c[0] + 10, c[1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (c[0], c[1] - 10), (c[0], c[1] + 10), (0, 255, 0), 1, cv2.LINE_AA)


# ===========================================================================
# 六、主流程
# ===========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="绿球识别与运动方向/速度标注（NUDT 自瞄方向 CV 作业）")
    parser.add_argument("--video", default=None, help="输入视频，默认用仓库根目录的 homework.mp4")
    parser.add_argument("--output", default=None, help="结果视频保存路径，如 out.mp4")
    parser.add_argument("--headless", action="store_true", help="不开窗口，只处理/保存（SSH、服务器用）")
    parser.add_argument("--show-mask", action="store_true", help="右上角显示绿色掩膜小窗，调阈值用")
    parser.add_argument("--csv", default=None, help="逐帧数据导出 csv，便于写报告")
    parser.add_argument("--scale", type=float, default=1.0, help="显示窗口缩放，如 0.75")
    parser.add_argument("--trail", type=int, default=TRAIL_LEN, help="轨迹保留帧数")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    video_path = args.video or find_default_video()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 打不开视频：%s" % video_path)
        print("        用 --video 指定路径，或把 homework.mp4 放到仓库根目录。")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] 视频: %s" % video_path)
    print("[INFO] 分辨率 %dx%d, %.2f fps, 共 %d 帧" % (width, height, fps, total))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("[ERROR] 无法创建输出视频：%s" % args.output)
            cap.release()
            return 1

    detector = GreenBallDetector()
    tracker = BallTracker(fps)
    viz = Visualizer(trail_len=args.trail)

    csv_rows = []
    frame_idx = 0
    detected_frames = 0
    speeds = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) 检测：用卡尔曼预测位置当 hint，多个绿色候选时保持连续性
        det = detector.detect(frame, tracker.predicted_center() or tracker.center)

        # 2) 卡尔曼更新，顺便算速度
        center, detected = tracker.update(det["center"] if det else None)
        if detected:
            detected_frames += 1
            speeds.append(tracker.speed_px_s)

        radius = 0.0
        if det is not None:
            # 画圆用"等效面积半径" sqrt(area/pi)，比最小外接圆更贴合球的实际边界
            radius = math.sqrt(det["area"] / math.pi)

        # 3) 可视化
        if center is not None:
            if det is not None and detected:
                viz.draw_target(frame, center, radius, det["contour"])

            viz.push(center, tracker.speed_px_s)
            viz.draw_trail(frame)
            viz.draw_arrow(frame, center, tracker.direction, tracker.speed_px_s)
            viz.draw_speed_label(frame, center, tracker.direction, tracker.speed_px_s)

        viz.draw_hud(frame, tracker, detected, frame_idx, total)
        viz.draw_speed_curve(frame)

        if args.show_mask and width > 300:
            mask = detector.build_mask(frame)
            small = cv2.resize(mask, (256, 144), interpolation=cv2.INTER_NEAREST)
            frame[10:154, width - 266:width - 10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

        if args.csv:
            csv_rows.append([
                frame_idx, round(frame_idx / fps, 3),
                "" if center is None else round(center[0], 2),
                "" if center is None else round(center[1], 2),
                round(tracker.speed_px_s, 2), round(tracker.accel, 2),
                round(radius, 2), int(detected),
            ])

        if writer is not None:
            writer.write(frame)

        if not args.headless:
            shown = frame
            if args.scale != 1.0:
                shown = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            cv2.imshow("Green Ball Tracking (press q/ESC to quit)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if args.csv and csv_rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["frame", "time_s", "x", "y", "speed_px_s",
                        "accel_px_s2", "radius_px", "detected"])
            w.writerows(csv_rows)
        print("[INFO] 逐帧数据已保存：%s" % args.csv)

    print("[INFO] 处理完成：共 %d 帧，检出目标 %d 帧 (%.1f%%)" % (
        frame_idx, detected_frames, 100.0 * detected_frames / max(frame_idx, 1)))
    if speeds:
        arr = np.array(speeds)
        print("[INFO] 速度统计 px/s：均值 %.0f | 中位数 %.0f | 峰值 %.0f" % (
            arr.mean(), np.median(arr), arr.max()))
    if args.output:
        print("[INFO] 结果视频：%s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
