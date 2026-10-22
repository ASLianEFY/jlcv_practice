# -*- coding: utf-8 -*-
"""
作业：识别视频中的绿球，并用箭头标注其运动方向与速度变化
================================================================

运行
----
    python3 green_ball_tracker.py                        # 弹出窗口播放
    python3 green_ball_tracker.py --output out.mp4       # 播放 + 保存结果视频
    python3 green_ball_tracker.py --headless --output out.mp4   # 无显示器（服务器/SSH）
    python3 green_ball_tracker.py --compensate           # 消除相机运动，只看球相对桌面的运动

算法流程（六个模块，与下面 6 个类/函数一一对应）
--------------------------------------------------
    1. GreenBallDetector  颜色分割：BGR -> HSV 阈值，绿色像素得到候选掩膜
    2. GreenBallDetector  形态学去噪：开运算去噪点，闭运算补球面高光
    3. GreenBallDetector  形状筛选：用"圆度"把绿球和绿色圆柱分开（本作业的核心）
    4. BallTracker        卡尔曼滤波：常速度模型平滑位置/速度，短时遮挡可预测续接
    5. CamMotionCompensator（可选）LK 光流 + RANSAC 估计相机运动并补偿
    6. Visualizer         箭头 / 变速轨迹 / 速度曲线 / HUD 可视化

屏幕上的字为什么是英文：cv2.putText 使用的 Hershey 字体不含中文字形，写中文会变问号，
所以画面内文本统一用 ASCII（终端里打印的中文日志不受影响）。

作者：zhounuolin
AI 工具使用说明：本文件的代码骨架、注释与参数调试在 AI 编程助手 **DeepSeek Harness
（DeepSeek 模型）** 辅助下完成；算法原理、参数含义与逐行讲解已由本人理解并在 README.md
的「代码讲解」一节整理，面试时可据此复述。

环境：Python 3.8+ / OpenCV 4.x 或 5.x / NumPy，Ubuntu 与 Windows 均可运行。
"""

import argparse
import csv
import math
import os
import sys
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 一、可调参数（换环境/换光线后需要微调的就是这一段）
# ---------------------------------------------------------------------------

# 绿色 HSV 范围。H 用 OpenCV 的 0~179 标度，绿色的中心在 60 附近（纯绿 = 60）。
# 下限放得比较宽（40~90）是为了兼容球面的明暗过渡和高光边缘。
GREEN_LOWER = np.array([40, 60, 60], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# 圆度 = 轮廓面积 / 最小外接圆面积，完美圆为 1.0。
# 实测本视频：绿球 0.87~0.95，绿色圆柱 0.59~0.72 —— 阈值取中间值 0.74 可完全分开。
MIN_CIRCULARITY = 0.74

MIN_AREA = 500        # 小于此面积的轮廓当作噪点（球半径 >13px 才有这个面积）
MAX_AREA = 60000      # 大于此面积的轮廓当作桌面反光/多个物体粘连

MAX_LOST_FRAMES = 12  # 连续丢失超过这么多帧才认为目标真的消失

# ---- 箭头与速度显示 ----
ARROW_GAIN = 0.30     # 箭头长度(px) = 速度(px/s) * 该系数
ARROW_MIN = 16        # 箭头最短长度，避免速度很小时箭头缩成一个点
ARROW_MAX = 230       # 箭头最长长度，避免高速时箭头冲出画面
SPEED_MIN_SHOW = 22.0 # 低于该速度(px/s)认为基本静止，不画箭头
SPEED_VMAX = 900.0    # 颜色映射的速度上限：0 -> 绿，900 -> 红

TRAIL_LEN = 45        # 运动轨迹保留的帧数
SPEED_HISTORY_LEN = 150  # 速度曲线保留的帧数


# ---------------------------------------------------------------------------
# 二、小工具
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    """把 value 限制在 [low, high] 区间内。"""
    return max(low, min(high, value))


def lerp_color(c1, c2, t):
    """两个 BGR 颜色按 t∈[0,1] 线性插值。"""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def speed_color(speed_px_s):
    """速度 -> BGR 颜色：慢=绿，中=黄，快=红。用于箭头和轨迹。"""
    t = clamp(speed_px_s / SPEED_VMAX, 0.0, 1.0)
    if t < 0.5:
        return lerp_color((60, 190, 60), (0, 220, 255), t / 0.5)   # 绿 -> 黄
    return lerp_color((0, 220, 255), (30, 50, 255), (t - 0.5) / 0.5)  # 黄 -> 红


def find_default_video():
    """
    脚本放在 homework/<name>/ 下、视频在仓库根目录，按相对位置向上找。
    找不到就返回一个用于报错的默认路径。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "homework.mp4"),      # homework/<name>/ -> 根目录
        os.path.join(here, "..", "..", "..", "homework.mp4"),
        os.path.join(os.getcwd(), "homework.mp4"),
        os.path.join(os.getcwd(), "..", "homework.mp4"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)
    return os.path.normpath(candidates[0])


# ---------------------------------------------------------------------------
# 三、检测器：颜色分割 + 形态学 + 形状筛选
# ---------------------------------------------------------------------------

class GreenBallDetector:
    """
    从一帧里找出"最像绿球"的候选。

    为什么不能只做颜色阈值：
        场上有绿色圆柱体，它的颜色和球一样，只靠 HSV 阈值一定会把它一起框进来。
        圆柱的正面投影是"圆角矩形"，而球是正圆 —— 用圆度（面积/最小外接圆面积）
        就能把两者分开，这一步是本作业的关键。
    """

    def __init__(self):
        # 椭圆结构元比矩形更贴合球体边缘，不容易把圆的边削方
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def build_mask(self, frame):
        """返回绿色的二值掩膜（白色=绿色像素）。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        # 开运算：先腐蚀再膨胀，去掉零散噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        # 闭运算：先膨胀再腐蚀，补上球面高光/阴影造成的小洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    @staticmethod
    def contour_features(contour):
        """计算单个轮廓的 (面积, 最小外接圆半径, 圆心, 圆度)。"""
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
        返回最可信的候选 dict，或 None。
        hint：上一帧的球心 (x, y)。场上如果有多个绿色候选，用它做"连续性"打分，
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
                continue                      # 太小是噪点，太大是反光/粘连
            if circularity < MIN_CIRCULARITY:
                continue                      # 不够圆 -> 多半是绿色圆柱体

            # --- 打分：圆度为主，离上一帧位置越近加分越多 ---
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


# ---------------------------------------------------------------------------
# 四、相机运动补偿（可选，--compensate 时启用）
# ---------------------------------------------------------------------------

class CamMotionCompensator:
    """
    手持/云台相机本身在动，直接拿相邻两帧的球心相减，会把"相机运动"误当成"球运动"。

    做法：在背景上取稀疏角点 -> LK 光流跟踪到当前帧 -> RANSAC 拟合相似变换
         （平移+旋转+缩放，4 个自由度）-> 把上一帧的球心投影到当前帧坐标系，
         再和当前帧球心相减，得到球相对桌面的真实位移。

    注意：本作业视频里球其实静止在桌面，画面里看到的"移动"几乎全是相机在动，
         所以开启这个模式后箭头会变得很短 —— 这恰好证明补偿是对的（见 README 讨论）。
    """

    def __init__(self, max_corners=300, min_keep=12):
        self.max_corners = max_corners
        self.min_keep = min_keep

    def estimate(self, prev_gray, cur_gray, ball_prev, ball_radius):
        """返回 2x3 相似变换矩阵；估计失败返回 None（调用方当作相机静止处理）。"""
        if prev_gray is None or cur_gray is None:
            return None

        # 把球所在区域挖掉，别让球自身的像素参与"背景"运动估计
        mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
        if ball_prev is not None:
            r = int(max(ball_radius * 1.8, 24))
            cv2.circle(mask, (int(ball_prev[0]), int(ball_prev[1])), r, 0, -1)

        pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=self.max_corners,
            qualityLevel=0.01, minDistance=10, mask=mask,
        )
        if pts is None or len(pts) < self.min_keep:
            return None

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None:
            return None

        good = status.ravel() == 1
        if good.sum() < self.min_keep:
            return None

        src = pts[good].reshape(-1, 2)
        dst = nxt[good].reshape(-1, 2)
        transform, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if transform is None:
            return None
        # 相邻两帧的相机位移不可能太大，过大的估计多半是匹配置信度崩了
        if abs(transform[0, 2]) > 120 or abs(transform[1, 2]) > 120:
            return None
        return transform


def apply_affine(transform, point):
    """把点按 2x3 仿射矩阵变换。"""
    x = transform[0, 0] * point[0] + transform[0, 1] * point[1] + transform[0, 2]
    y = transform[1, 0] * point[0] + transform[1, 1] * point[1] + transform[1, 2]
    return (x, y)


# ---------------------------------------------------------------------------
# 五、跟踪器：卡尔曼滤波 + 速度估计
# ---------------------------------------------------------------------------

class BallTracker:
    """
    状态向量 X = [x, y, vx, vy]（像素、像素/帧），常速度模型：

        x' = x + vx,  y' = y + vy,  vx' = vx,  vy' = vy

    每帧两步：
        predict()  用上一步状态外推，得到"这帧球应该在哪"
        correct()  用检测结果修正外推值
    - 有检测：正常更新，位置和速度都被平滑；
    - 短时丢失（比如手挡住）：只 predict 不外推速度，最多撑 MAX_LOST_FRAMES 帧；
    - 速度再做一次指数平滑，避免箭头方向抖。
    """

    def __init__(self, fps, vel_alpha=0.55):
        self.fps = max(float(fps), 1.0)
        self.vel_alpha = vel_alpha      # 速度指数平滑系数，越大越跟手、越小越平滑

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
        self.center = None          # 当前帧球心（滤波后）
        self.prev_center = None     # 上一个有效检测帧的球心（用于算速度）
        self.vx = 0.0               # 平滑后的速度，单位 px/帧
        self.vy = 0.0
        self.accel = 0.0            # 速率变化率 px/s^2，仅用于显示"速度变化"

    # ---- 内部 ----
    def _init(self, center):
        self.kf.statePost = np.array(
            [[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True
        self.lost_frames = 0

    def _predict(self):
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    # ---- 对外 ----
    def predicted_center(self):
        """
        只根据当前状态外推一步，返回预测球心，**不修改滤波器状态**。
        注意不能直接调用 cv2.KalmanFilter.predict()——它会把 statePost 一起推进，
        一帧里调两次就等于外推了两步，速度会翻倍。
        """
        if not self.initialized:
            return None
        s = self.kf.statePost
        return (float(s[0, 0] + s[2, 0]), float(s[1, 0] + s[3, 0]))

    def update(self, measured, transform=None):
        """
        measured：本帧检测到的球心，None 表示本帧没检测到
        transform：上一帧 -> 本帧的相机运动变换，None 表示不补偿
        返回 (center, detected)
        """
        if not self.initialized:
            if measured is None:
                return None, False
            self._init(measured)
            self.center = measured
            self.prev_center = measured
            return measured, True

        if measured is not None:
            self._predict()                                  # 先外推，再修正
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

        # ---- 速度：用"补偿后的帧间位移"，而不是直接拿两帧球心相减 ----
        if detected and self.prev_center is not None:
            ref = apply_affine(transform, self.prev_center) if transform is not None \
                else self.prev_center
            dvx = center[0] - ref[0]
            dvy = center[1] - ref[1]

            speed_before = self.speed_px_s
            self.vx = self.vel_alpha * dvx + (1.0 - self.vel_alpha) * self.vx
            self.vy = self.vel_alpha * dvy + (1.0 - self.vel_alpha) * self.vy
            speed_after = self.speed_px_s

            # 加速度只做显示用，再平滑一次，否则数字跳得看不清
            dt = 1.0 / self.fps
            self.accel = 0.25 * ((speed_after - speed_before) / dt) + 0.75 * self.accel

        if detected:
            self.prev_center = center
        self.center = center
        return center, detected

    def _reset(self):
        self.initialized = False
        self.center = None
        self.prev_center = None
        self.vx = self.vy = 0.0
        self.accel = 0.0
        self.lost_frames = 0

    # ---- 供绘制使用的派生量 ----
    @property
    def speed_px_s(self):
        """速率，像素/秒。"""
        return math.hypot(self.vx, self.vy) * self.fps

    @property
    def direction(self):
        """单位方向向量 (dx, dy)；速度约为 0 时返回 (0, 0)。"""
        norm = math.hypot(self.vx, self.vy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (self.vx / norm, self.vy / norm)


# ---------------------------------------------------------------------------
# 六、可视化
# ---------------------------------------------------------------------------

class Visualizer:
    """把检测/跟踪结果画到画面上：箭头、变速轨迹、速度曲线、HUD。"""

    def __init__(self, trail_len=TRAIL_LEN, history_len=SPEED_HISTORY_LEN):
        self.trail = deque(maxlen=max(trail_len, 1))              # [(x, y, speed)]
        self.speeds = deque(maxlen=max(history_len, 2))           # 速度时间序列

    def push(self, center, speed):
        self.trail.append((int(center[0]), int(center[1]), speed))
        self.speeds.append(speed)

    def draw_trail(self, frame):
        """轨迹线按"当时的速率"着色，一眼能看出哪段快、哪段慢。"""
        pts = list(self.trail)
        for i in range(1, len(pts)):
            x0, y0, s0 = pts[i - 1]
            x1, y1, s1 = pts[i]
            color = speed_color(0.5 * (s0 + s1))
            cv2.line(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    @staticmethod
    def arrow_end(center, direction, speed):
        """按当前速率算出箭头终点，画箭头和写标注共用，保证两者对齐。"""
        length = clamp(speed * ARROW_GAIN, ARROW_MIN, ARROW_MAX)
        return (center[0] + direction[0] * length,
                center[1] + direction[1] * length)

    @classmethod
    def draw_arrow(cls, frame, center, direction, speed):
        """
        箭头起点=球心，长度∝速率，方向=速度方向。
        先画一层深色描边再画彩色，任何背景上都看得清。
        """
        if speed < SPEED_MIN_SHOW:
            return
        end = cls.arrow_end(center, direction, speed)
        start = (int(round(center[0])), int(round(center[1])))
        end_i = (int(round(end[0])), int(round(end[1])))

        color = speed_color(speed)
        cv2.arrowedLine(frame, start, end_i, (20, 20, 20), 6, cv2.LINE_AA, tipLength=0.24)
        cv2.arrowedLine(frame, start, end_i, color, 3, cv2.LINE_AA, tipLength=0.24)
        # 起点画个小圆点，强调"箭头的原点就是球心"
        cv2.circle(frame, start, 4, (0, 0, 255), -1, cv2.LINE_AA)

    @classmethod
    def draw_speed_label(cls, frame, center, direction, speed):
        """在箭头末端标出速率数值：带深色底框，压在轨迹线上也能看清。"""
        if speed < SPEED_MIN_SHOW:
            return
        text = "%.0f px/s" % speed
        scale, thickness = 0.6, 2
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        ex, ey = cls.arrow_end(center, direction, speed)
        nx, ny = -direction[1], direction[0]      # 箭头方向的法向量
        h, w = frame.shape[:2]
        tx = int(clamp(ex + nx * 22 + 6, 6, w - tw - 12))
        ty = int(clamp(ey + ny * 22 - 6, th + 8, h - 8))

        cv2.rectangle(frame, (tx - 5, ty - th - 6), (tx + tw + 5, ty + base),
                      (25, 25, 25), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    speed_color(speed), thickness, cv2.LINE_AA)

    @staticmethod
    def put_text(frame, text, org, scale=0.6, color=(255, 255, 255), thickness=1):
        """带黑色描边的白字，深色/浅色背景都能读。"""
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

    def draw_speed_curve(self, frame):
        """
        左下角画"速度-时间"曲线，直接对应作业要求的"速度变化"。
        纵轴自动缩放到历史最大值，横轴是最近 SPEED_HISTORY_LEN 帧。
        """
        speeds = list(self.speeds)
        if len(speeds) < 2:
            return
        if frame.shape[0] < 220 or frame.shape[1] < 360:
            return          # 分辨率太小时右下角放不下曲线，直接不画（不影响主功能）

        panel_w, panel_h = 320, 130
        x0 = 10
        y0 = frame.shape[0] - panel_h - 12
        roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        # 半透明黑底：先复制一份，再按权重叠加，比直接涂抹更不挡视线
        cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.0, 0.0, roi)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1),
                      (90, 90, 90), 1, cv2.LINE_AA)

        vmax = max(max(speeds), SPEED_VMAX * 0.25)      # 纵轴上限，至少 225 px/s
        plot_x, plot_y = x0 + 34, y0 + 10
        plot_w, plot_h = panel_w - 46, panel_h - 34

        # 网格与刻度
        for frac in (0.0, 0.5, 1.0):
            gy = int(plot_y + plot_h * (1.0 - frac))
            cv2.line(frame, (plot_x, gy), (plot_x + plot_w, gy), (70, 70, 70), 1, cv2.LINE_AA)
            self.put_text(frame, "%d" % int(vmax * frac), (x0 + 4, gy + 5),
                          scale=0.38, color=(200, 200, 200))
        self.put_text(frame, "speed px/s", (plot_x, y0 + panel_h - 4),
                      scale=0.4, color=(180, 180, 180))

        # 曲线：每个点用速度对应的颜色，段与段之间连线
        n = len(speeds)
        pts = []
        for i, s in enumerate(speeds):
            px = plot_x + int(plot_w * i / (n - 1))
            py = plot_y + int(plot_h * (1.0 - clamp(s / vmax, 0.0, 1.0)))
            pts.append((px, py, s))
        for i in range(1, n):
            cv2.line(frame, pts[i - 1][:2], pts[i][:2], speed_color(pts[i][2]), 2, cv2.LINE_AA)

        # 当前值：末尾画个点并标数字
        last_x, last_y, last_s = pts[-1]
        cv2.circle(frame, (last_x, last_y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        self.put_text(frame, "%.0f" % last_s, (min(last_x + 6, x0 + panel_w - 40), last_y + 5),
                      scale=0.45, color=speed_color(last_s))

    def draw_hud(self, frame, tracker, detected, frame_idx, total, compensated):
        state = "TRACKING" if detected else ("PREDICTING" if tracker.center else "LOST")
        lines = [
            "frame %d/%d" % (frame_idx, total),
            "state: %s" % state,
            "speed: %.0f px/s" % tracker.speed_px_s,
            "accel: %+.0f px/s^2" % tracker.accel,
            "cam-comp: %s" % ("ON" if compensated else "OFF"),
        ]
        for i, text in enumerate(lines):
            self.put_text(frame, text, (12, 26 + i * 26), scale=0.62)

    def draw_target(self, frame, center, radius, contour):
        """目标标记：外接圆 + 球心点 + 轮廓。"""
        c = (int(round(center[0])), int(round(center[1])))
        cv2.circle(frame, c, int(max(radius, 6)), (0, 255, 0), 2, cv2.LINE_AA)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)
        # 十字准星，自瞄里常用的瞄准标记
        cv2.line(frame, (c[0] - 9, c[1]), (c[0] + 9, c[1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (c[0], c[1] - 9), (c[0], c[1] + 9), (0, 255, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 七、主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="绿球识别与运动方向/速度标注（NUDT 自瞄/雷达 CV 作业）")
    parser.add_argument("--video", default=None, help="输入视频，默认用仓库根目录的 homework.mp4")
    parser.add_argument("--output", default=None, help="结果视频保存路径，如 out.mp4")
    parser.add_argument("--headless", action="store_true", help="不开窗口，只处理/保存（SSH、服务器用）")
    parser.add_argument("--compensate", action="store_true",
                       help="开启相机运动补偿：只看球相对桌面的运动（默认为图像坐标下的运动）")
    parser.add_argument("--show-mask", action="store_true", help="右上角显示绿色掩膜小窗，方便调阈值")
    parser.add_argument("--csv", default=None, help="把逐帧数据导出成 csv，便于写报告/画图")
    parser.add_argument("--scale", type=float, default=1.0, help="显示窗口缩放比例，如 0.75")
    parser.add_argument("--trail", type=int, default=TRAIL_LEN, help="轨迹保留帧数")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    video_path = args.video or find_default_video()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 打不开视频：%s" % video_path)
        print("        用 --video 指定视频路径，或把 homework.mp4 放到仓库根目录。")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] 视频: %s" % video_path)
    print("[INFO] 分辨率 %dx%d, %.2f fps, 共 %d 帧" % (width, height, fps, total))
    print("[INFO] 相机运动补偿: %s" % ("ON" if args.compensate else "OFF"))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("[ERROR] 无法创建输出视频：%s" % args.output)
            cap.release()
            return 1

    detector = GreenBallDetector()
    compensator = CamMotionCompensator() if args.compensate else None
    tracker = BallTracker(fps)
    viz = Visualizer(trail_len=args.trail)

    csv_rows = []
    prev_gray = None
    frame_idx = 0
    detected_frames = 0
    last_radius = 30.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) 相机运动补偿（可选）
        transform = None
        if compensator is not None:
            transform = compensator.estimate(prev_gray, gray, tracker.center, last_radius)

        # 2) 检测：用卡尔曼预测位置当 hint，多个绿色候选时保持连续性
        det = detector.detect(frame, tracker.predicted_center() or tracker.center)

        # 3) 卡尔曼更新，顺便算出速度
        center, detected = tracker.update(det["center"] if det else None, transform)
        if detected:
            detected_frames += 1
            if det is not None:
                # 画圆用"等效面积半径" sqrt(area/pi)，比最小外接圆更贴合球的实际边界
                last_radius = math.sqrt(det["area"] / math.pi)

        # 4) 可视化
        if center is not None:
            if det is not None and detected:
                viz.draw_target(frame, center, last_radius, det["contour"])

            viz.push(center, tracker.speed_px_s)
            viz.draw_trail(frame)
            viz.draw_arrow(frame, center, tracker.direction, tracker.speed_px_s)
            viz.draw_speed_label(frame, center, tracker.direction, tracker.speed_px_s)

        viz.draw_hud(frame, tracker, detected, frame_idx, total, compensator is not None)
        viz.draw_speed_curve(frame)

        if args.show_mask and width > 300:
            mask = detector.build_mask(frame)
            small = cv2.resize(mask, (256, 144), interpolation=cv2.INTER_NEAREST)
            frame[10:154, width - 266:width - 10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

        if args.csv:
            csv_rows.append([
                frame_idx, frame_idx / fps,
                "" if center is None else round(center[0], 2),
                "" if center is None else round(center[1], 2),
                round(tracker.speed_px_s, 2), round(tracker.accel, 2),
                round(last_radius, 2), int(detected),
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

        prev_gray = gray
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if args.csv and csv_rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            writer_csv = csv.writer(fp)
            writer_csv.writerow(["frame", "time_s", "x", "y",
                                 "speed_px_s", "accel_px_s2", "radius_px", "detected"])
            writer_csv.writerows(csv_rows)
        print("[INFO] 逐帧数据已保存：%s" % args.csv)

    print("[INFO] 处理完成：共 %d 帧，检测到目标 %d 帧 (%.1f%%)" % (
        frame_idx, detected_frames,
        100.0 * detected_frames / max(frame_idx, 1)))
    if args.output:
        print("[INFO] 结果视频：%s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
