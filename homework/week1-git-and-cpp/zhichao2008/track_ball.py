#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_ball.py —— 绿球检测、运动方向与速度可视化

功能：
    1. 读取脚本所在目录中的 homework.mp4；
    2. 基于 HSV 颜色空间分割绿色区域，识别画面中的绿色小球；
    3. 用连续帧约束和形状评分排除绿色圆柱等干扰物；
    4. 在球上绘制方向箭头：箭头方向 = 球的运动方向，
       箭头长度与球在图像中的运动速度成正比；
    5. 绘制运动轨迹、速度数值和状态信息；
    6. 将结果逐帧写入同目录下的 output_video.mp4。

运行方式（在本文件所在目录打开终端）：
    python track_ball.py
    如果系统 Python 没有安装 OpenCV，可使用当前目录已有的虚拟环境：
    ./venv/bin/python track_ball.py

依赖：
    opencv-python, numpy

说明：
    - 视频中同时存在绿色圆柱和绿色小球。圆柱偏长，小球接近圆形，
      因此程序先按“长宽比 + 圆度”给绿色轮廓打分，再利用相邻帧之间
      的位置连续性和面积连续性锁定小球，避免误跟到圆柱。
    - 箭头长度表示“图像平面内的表观速度”。因为拍摄时相机也在移动，
      这个速度既包含球的真实运动，也包含相机运动造成的画面位移。
    - 箭头长度按固定比例换算，并设置上限，保证不同帧之间可以直观比较：
      球越快，箭头越长；球越慢，箭头越短。
    - OpenCV 的 putText 对中文字体支持不便，因此画面 HUD 使用英文；
      代码注释、终端输出均为中文。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# 一、路径与可调参数
# ============================================================================

# 使用 __file__ 定位脚本所在目录，这样无论从哪个目录执行脚本，
# 都能正确找到 homework.mp4 并生成 output_video.mp4。
BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = BASE_DIR / "homework.mp4"
OUTPUT_VIDEO = BASE_DIR / "output_video.mp4"

# HSV 绿色阈值。OpenCV 中 H 的范围是 0~179，不是 0~360。
# H≈35~90 覆盖黄绿到青绿；S 下限用于排除灰白色；V 下限用于排除暗部。
# 如果实际视频偏暗或偏黄，可优先调整 H 的上下限。
HSV_LOWER = np.array([35, 70, 40], dtype=np.uint8)
HSV_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# 过滤过小的绿色噪点。当前视频中球的最小面积约 1 万像素，
# 220 像素的下限足以过滤地面反光、线缆等零碎区域。
MIN_CONTOUR_AREA = 220
MIN_BOX_SIDE = 10

# 形状评分的参数。
# ASPECT_LOG_SIGMA 越大，对长宽比的容忍度越高；
# 小球接近 1:1，而绿色圆柱通常明显“高瘦”，长宽比会偏离 1。
ASPECT_LOG_SIGMA = 0.30

# 相邻帧跟踪评分的权重：
# SHAPE_WEIGHT   —— 轮廓是否像球；
# MOTION_WEIGHT  —— 是否靠近根据上一帧速度预测的位置；
# AREA_WEIGHT    —— 面积是否与上一帧接近（避免突然换成另一个绿物体）。
SHAPE_WEIGHT = 1.4
MOTION_WEIGHT = 2.2
AREA_WEIGHT = 0.7

# 轨迹与中心平滑参数。
# CENTER_ALPHA 越大，框越贴近检测结果但抖动越明显；越小则越平滑。
CENTER_ALPHA = 0.55
# 用于预测下一帧位置的速度平滑系数（单位：像素/帧）。
MOTION_ALPHA = 0.35
# 用于绘制箭头的速度平滑系数（越小，箭头方向/长度越稳定）。
ARROW_VELOCITY_ALPHA = 0.30
# 轨迹保留的帧数。
TRAIL_LENGTH = 24
# 最多容忍连续多少帧丢失目标；丢失期间用上一帧速度外推位置。
MAX_MISSED_FRAMES = 15

# 箭头长度换算：
# 长度（像素） = 速度（像素/秒） × ARROW_SECONDS。
# 取 0.18 秒，相当于把箭头画成球“未来约 0.18 秒内会走出的距离”。
ARROW_SECONDS = 0.18
# 箭头最长像素数，防止高速时箭头过长、完全冲出画面。
MAX_ARROW_LENGTH = 220.0
# 低于该长度时箭头画出来只是一个小点，反而干扰观察，因此不绘制。
MIN_ARROW_LENGTH = 10.0
# 对应上一行长度阈值的最低速度。低于该速度视为“近似静止”。
MIN_SPEED_FOR_ARROW = MIN_ARROW_LENGTH / ARROW_SECONDS

# 画面配色（OpenCV 使用 BGR 顺序）。
COLOR_ARROW = (0, 235, 255)      # 亮黄色，和绿色球、木色桌面都有明显对比
COLOR_TARGET = (255, 255, 255)   # 目标圆环：白色
COLOR_TARGET_DARK = (20, 20, 20) # 目标圆环描边：深灰
COLOR_TRAIL_START = (0, 80, 255) # 轨迹起点：橙黄色
COLOR_TRAIL_END = (0, 255, 255)  # 轨迹终点：亮黄色
COLOR_TEXT = (245, 245, 245)
COLOR_OK = (80, 255, 120)
COLOR_WARN = (0, 180, 255)


# ============================================================================
# 二、检测结果的数据结构
# ============================================================================


@dataclass
class GreenCandidate:
    """描述一个绿色连通区域（候选绿球）的几何特征。"""

    center: np.ndarray             # 轮廓质心坐标 (x, y)，浮点数
    area: float                    # 轮廓面积（像素）
    bbox: tuple[int, int, int, int]  # 外接矩形 (x, y, w, h)
    aspect_ratio: float            # 宽 / 高
    circularity: float             # 圆度 4πA/P²，完美圆约为 1
    fill_ratio: float              # 轮廓面积 / 外接矩形面积
    shape_score: float             # 综合形状评分，越大越像球


# ============================================================================
# 三、绿色区域分割与候选轮廓提取
# ============================================================================


def build_green_mask(frame: np.ndarray) -> np.ndarray:
    """
    将 BGR 帧转换为 HSV，并提取绿色区域掩膜。

    形态学操作：
        - 开运算（先腐蚀后膨胀）：去掉桌面上的小噪点；
        - 闭运算（先膨胀后腐蚀）：填补球体高光或阴影造成的空洞。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 只保留 H、S、V 都在绿色范围内的像素，得到二值图。
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # 使用椭圆核比矩形核更符合球体的自然边缘。
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def find_green_candidates(frame: np.ndarray) -> list[GreenCandidate]:
    """
    查找当前帧中所有足够大的绿色轮廓，并计算“像球程度”的几何评分。

    形状评分由三部分组成：
        1. 长宽比评分：|log(w/h)| 越小，越接近正方形；
        2. 圆度：4πA/P²，圆越标准越接近 1；
        3. 填充率：轮廓面积 / 外接矩形面积，球通常较饱满。

    注意：单帧形状评分只能用于“初始化”和“辅助选择”。当圆柱被画面
    边缘截断时，它的长宽比也会接近 1，所以最终还需要帧间连续性约束。
    """
    mask = build_green_mask(frame)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[GreenCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            continue

        # 计算轮廓质心。相比外接矩形中心，质心对部分遮挡更稳健。
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # 圆度 = 4π × 面积 / 周长²。理论上圆为 1，实际分割结果通常小于 1。
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        )
        circularity = min(circularity, 1.0)

        # 用对数比描述长宽比，使 0.5 和 2.0 得到同样大小的惩罚。
        aspect_ratio = w / float(h)
        aspect_score = math.exp(
            -((math.log(max(aspect_ratio, 1e-6)) / ASPECT_LOG_SIGMA) ** 2)
        )

        fill_ratio = area / float(w * h)
        shape_score = aspect_score + 0.5 * circularity + 0.1 * fill_ratio

        candidates.append(
            GreenCandidate(
                center=np.array([cx, cy], dtype=np.float32),
                area=area,
                bbox=(x, y, w, h),
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                fill_ratio=fill_ratio,
                shape_score=shape_score,
            )
        )

    return candidates


# ============================================================================
# 四、带连续性约束的绿球跟踪器
# ============================================================================


class GreenBallTracker:
    """
    在连续的绿色候选中锁定同一个绿球。

    第一帧：
        选择“形状最像球”的候选作为初始目标。

    后续帧：
        对每个候选计算综合分数：
            - shape_score：形状得分；
            - continuity：与“上一帧中心 + 速度外推位置”的距离得分；
            - area_similarity：与上一帧面积的相似度。
        取综合分数最高者。连续性权重较大，因此即使绿色圆柱被画面裁切
        后形状变圆，也不会轻易把目标切换过去。

    丢失处理：
        连续若干帧找不到绿色候选时，用上一帧速度外推球的位置，
        并让速度逐渐衰减；超过 MAX_MISSED_FRAMES 后重新初始化。
    """

    def __init__(self) -> None:
        self.last_candidate: GreenCandidate | None = None
        self.last_center: np.ndarray | None = None

        # 原始中心的帧间位移平滑值，单位：像素/帧，用于位置预测。
        self.motion = np.zeros(2, dtype=np.float32)
        # 用于绘制箭头的平滑速度，单位：像素/帧。
        self.arrow_velocity = np.zeros(2, dtype=np.float32)
        # 绘制时使用的平滑中心，减少阈值分割带来的轻微抖动。
        self.smooth_center: np.ndarray | None = None
        # 绘制圆环/箭头起点时使用的平滑半径。
        self.display_radius = 20.0

        self.missed_frames = 0
        self.initialized = False

    def reset(self) -> None:
        """清空跟踪状态。超过最大丢失帧数后调用。"""
        self.last_candidate = None
        self.last_center = None
        self.motion = np.zeros(2, dtype=np.float32)
        self.arrow_velocity = np.zeros(2, dtype=np.float32)
        self.smooth_center = None
        self.display_radius = 20.0
        self.missed_frames = 0
        self.initialized = False

    @staticmethod
    def _predict_position(
        last_center: np.ndarray, motion: np.ndarray
    ) -> np.ndarray:
        """
        根据上一帧中心和帧间速度预测当前帧位置。

        这里只使用 0.8 倍速度，是为了减弱“突然加速/急停”时预测过冲，
        让预测点略靠近上一帧，从而提高候选匹配的容错性。
        """
        return last_center + 0.8 * motion

    def _accept(
        self, candidate: GreenCandidate, *, reacquired: bool = False
    ) -> GreenCandidate:
        """接受当前候选，并更新中心、速度、半径等跟踪状态。"""
        center = candidate.center.astype(np.float32)
        previous_center = self.last_center

        # 更新位置预测使用的速度（单位：像素/帧）。
        if previous_center is not None:
            frame_delta = center - previous_center
            if self.initialized and not reacquired:
                self.motion = (
                    (1.0 - MOTION_ALPHA) * self.motion + MOTION_ALPHA * frame_delta
                )
            else:
                # 刚初始化或重新捕获时没有可靠速度，直接置零更安全。
                self.motion = np.zeros(2, dtype=np.float32)
        else:
            self.motion = np.zeros(2, dtype=np.float32)

        # 平滑中心。第一帧或重新捕获时直接使用检测中心，避免从旧位置漂移。
        if self.smooth_center is None or reacquired:
            self.smooth_center = center.copy()
            self.arrow_velocity = np.zeros(2, dtype=np.float32)
        else:
            previous_smooth = self.smooth_center.copy()
            self.smooth_center = (
                CENTER_ALPHA * center + (1.0 - CENTER_ALPHA) * previous_smooth
            )

            # 箭头速度用“检测中心位移”计算，再作指数平滑。
            # 使用检测中心而不是完全平滑中心，能让速度变化响应更快。
            raw_delta = center - previous_center
            self.arrow_velocity = (
                (1.0 - ARROW_VELOCITY_ALPHA) * self.arrow_velocity
                + ARROW_VELOCITY_ALPHA * raw_delta
            )

        # 半径也做平滑。圆柱被裁切时外接框会突然变化，平滑可减少圆环跳动。
        detected_radius = 0.5 * max(candidate.bbox[2], candidate.bbox[3])
        if not self.initialized or reacquired:
            self.display_radius = detected_radius
        else:
            self.display_radius = (
                0.65 * detected_radius + 0.35 * self.display_radius
            )

        self.last_candidate = candidate
        self.last_center = center
        self.missed_frames = 0
        self.initialized = True
        return candidate

    def _handle_miss(self) -> None:
        """当前帧没有检测到绿色候选时的处理逻辑。"""
        self.missed_frames += 1

        # 如果连续丢失太久，清空状态；下一次出现绿色候选时重新初始化，
        # 避免用已经失效的速度一直外推。
        if self.missed_frames > MAX_MISSED_FRAMES:
            self.reset()
            return

        # 短时间丢失时，使用上一帧速度外推中心；速度按 0.85 倍衰减。
        if self.last_center is not None:
            self.last_center = self.last_center + self.motion
            if self.smooth_center is not None:
                self.smooth_center = self.smooth_center + self.motion
            self.motion *= 0.85
            self.arrow_velocity *= 0.85

    def update(self, candidates: list[GreenCandidate]) -> GreenCandidate | None:
        """
        输入当前帧的所有候选轮廓，返回选中的绿球候选。

        返回 None 表示当前帧没有检测到目标；此时 smooth_center 可能仍然
        保存着外推位置，但调用方不应绘制“已确认”的目标框和方向箭头。
        """
        if not candidates:
            self._handle_miss()
            return None

        # 第一帧没有历史信息，只能依靠形状评分选取最像球的轮廓。
        if not self.initialized or self.last_candidate is None:
            best = max(candidates, key=lambda item: item.shape_score)
            return self._accept(best)

        # 根据上一帧的位置与速度，预测目标在当前帧的大致位置。
        predicted_center = self._predict_position(self.last_center, self.motion)

        # 空间搜索尺度随球体大小自适应：球越大，允许的帧间位移越大。
        last_equivalent_radius = math.sqrt(self.last_candidate.area / math.pi)
        spatial_scale = max(80.0, 2.5 * last_equivalent_radius + 15.0)

        best_candidate: GreenCandidate | None = None
        best_total_score = -float("inf")

        for candidate in candidates:
            distance = float(
                np.linalg.norm(candidate.center - predicted_center)
            )
            # 距离越小，连续性得分越接近 1。使用高斯形式，避免硬阈值造成跳变。
            continuity_score = math.exp(
                -0.5 * (distance / spatial_scale) ** 2
            )

            # 面积连续性：当前面积 / 上一帧面积。
            # 使用对数比可以同时公平地处理“变大”和“变小”。
            area_ratio = candidate.area / max(self.last_candidate.area, 1.0)
            area_similarity = math.exp(
                -0.5 * (math.log(max(area_ratio, 1e-6)) / 0.45) ** 2
            )

            total_score = (
                SHAPE_WEIGHT * candidate.shape_score
                + MOTION_WEIGHT * continuity_score
                + AREA_WEIGHT * area_similarity
            )
            if total_score > best_total_score:
                best_total_score = total_score
                best_candidate = candidate

        # 理论上 candidates 非空，best_candidate 一定存在；保留兜底判断。
        if best_candidate is None:
            self._handle_miss()
            return None

        # 如果此前刚发生过丢失，重新捕获时重置速度，避免箭头方向突然跳变。
        reacquired = self.missed_frames > 0
        return self._accept(best_candidate, reacquired=reacquired)

    def velocity_px_per_second(self, fps: float) -> np.ndarray:
        """返回平滑后的速度向量，单位：像素/秒。"""
        return self.arrow_velocity.astype(np.float32) * float(fps)


# ============================================================================
# 五、绘制函数
# ============================================================================


def draw_trail(frame: np.ndarray, trail: deque[np.ndarray]) -> None:
    """
    绘制历史运动轨迹。

    越靠近当前帧的线段越亮、越粗，形成简单的“渐隐”效果；
    轨迹同时能让观察者更直观地看出球从哪里来、往哪里去。
    """
    if len(trail) < 2:
        return

    points = list(trail)
    segment_count = len(points) - 1

    for index in range(1, len(points)):
        t = index / float(segment_count)
        # BGR 颜色从橙色逐渐过渡到亮黄色。
        color = (
            0,
            int(COLOR_TRAIL_START[1] * (1 - t) + COLOR_TRAIL_END[1] * t),
            int(COLOR_TRAIL_START[2] * (1 - t) + COLOR_TRAIL_END[2] * t),
        )
        thickness = 1 + int(2 * t)
        p1 = tuple(np.round(points[index - 1]).astype(int))
        p2 = tuple(np.round(points[index]).astype(int))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_target(
    frame: np.ndarray, center: np.ndarray, radius: float
) -> None:
    """在绿球外侧绘制双色圆环和十字准星。"""
    center_point = tuple(np.round(center).astype(int))
    circle_radius = max(8, int(round(radius + 7)))

    # 先画一圈粗深色描边，再画白色细线，确保在浅色木桌和绿球上都可见。
    cv2.circle(
        frame, center_point, circle_radius, COLOR_TARGET_DARK, 5, cv2.LINE_AA
    )
    cv2.circle(frame, center_point, circle_radius, COLOR_TARGET, 2, cv2.LINE_AA)

    # 十字准星帮助确认球心位置。
    arm = 8
    cv2.line(
        frame,
        (center_point[0] - arm, center_point[1]),
        (center_point[0] + arm, center_point[1]),
        COLOR_TARGET,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (center_point[0], center_point[1] - arm),
        (center_point[0], center_point[1] + arm),
        COLOR_TARGET,
        2,
        cv2.LINE_AA,
    )


def _max_drawable_distance(
    start: tuple[float, float],
    direction: tuple[float, float],
    width: int,
    height: int,
    margin: int = 8,
) -> float:
    """
    计算从 start 沿 direction 出发，在碰到画面边界前最多能画多长。

    这可以避免球靠近画面边缘时，箭头和箭头头部被裁掉而看不见。
    注意：这只是在极少数贴边帧上为了“保持可见”做的裁剪，
    常规情况下箭头长度仍严格按照速度比例计算。
    """
    sx, sy = start
    dx, dy = direction
    limits: list[float] = []

    # 分别计算沿 x、y 方向到达四条边界所需的距离。
    if dx > 1e-6:
        limits.append((width - 1 - margin - sx) / dx)
    elif dx < -1e-6:
        limits.append((margin - sx) / dx)

    if dy > 1e-6:
        limits.append((height - 1 - margin - sy) / dy)
    elif dy < -1e-6:
        limits.append((margin - sy) / dy)

    positive_limits = [value for value in limits if value > 0.0]
    return min(positive_limits) if positive_limits else 0.0


def draw_direction_arrow(
    frame: np.ndarray,
    center: np.ndarray,
    velocity_px_per_second: np.ndarray,
    ball_radius: float,
) -> None:
    """
    绘制运动方向箭头。

    设计要点：
        1. 方向：取平滑速度向量的方向；
        2. 长度：length = speed × ARROW_SECONDS，再限制最大值；
        3. 起点：通常从球体边缘外开始；若球贴着画面边缘、按原长度放不下，
           则把起点向球心收缩，尽量让完整箭头留在画面内；
        4. 对比：用粗黑箭头打底，再叠加亮黄色箭头，任何背景上都清楚。

    因为长度和速度是线性比例关系（上限以内），所以比较不同帧的
    箭头长短，就能看出球在该时刻运动得快还是慢。
    """
    speed = float(np.linalg.norm(velocity_px_per_second))
    if speed < MIN_SPEED_FOR_ARROW:
        return

    direction = velocity_px_per_second / speed
    dx, dy = float(direction[0]), float(direction[1])

    expected_length = min(speed * ARROW_SECONDS, MAX_ARROW_LENGTH)
    height, width = frame.shape[:2]

    # 先计算从球心出发、沿运动方向到画面边界还剩多少空间。
    available_from_center = _max_drawable_distance(
        (float(center[0]), float(center[1])), (dx, dy), width, height
    )

    # 默认起点在球体外侧；若剩余空间不足以容纳预期箭头，
    # 则将起点朝球心方向回缩（最多回缩到球心）。
    # 这样在画面边缘仍能保留一支方向正确、清晰可见的箭头。
    preferred_offset = ball_radius + 3.0
    start_offset = min(
        preferred_offset,
        max(0.0, available_from_center - expected_length),
    )
    start_x = float(center[0] + dx * start_offset)
    start_y = float(center[1] + dy * start_offset)

    # 保险起见，再从实际起点计算一次可绘制距离。
    available = _max_drawable_distance(
        (start_x, start_y), (dx, dy), width, height
    )
    draw_length = min(expected_length, available)
    if draw_length < MIN_ARROW_LENGTH:
        return

    start = (int(round(start_x)), int(round(start_y)))
    end = (
        int(round(start_x + dx * draw_length)),
        int(round(start_y + dy * draw_length)),
    )

    # 粗黑描边 + 细亮黄色箭头，保证在木色、白色、绿色背景上都醒目。
    cv2.arrowedLine(
        frame,
        start,
        end,
        (10, 10, 10),
        8,
        cv2.LINE_AA,
        tipLength=0.28,
    )
    cv2.arrowedLine(
        frame,
        start,
        end,
        COLOR_ARROW,
        4,
        cv2.LINE_AA,
        tipLength=0.28,
    )


def draw_hud(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    fps: float,
    speed_px_s: float,
    tracking: bool,
) -> None:
    """
    在左上角绘制半透明信息面板。

    显示内容：
        - 当前帧号；
        - 目标状态（追踪中 / 搜索中）；
        - 当前平滑速度；
        - 箭头长度的换算规则，便于理解箭头含义。
    """
    panel_x1, panel_y1 = 14, 14
    panel_x2, panel_y2 = 418, 154

    # addWeighted 实现半透明面板，不会完全遮住背景物体。
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (25, 25, 25), -1
    )
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(
        frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (175, 175, 175), 1
    )

    # 标题。
    cv2.putText(
        frame,
        "Green Ball Tracker",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    # 状态文字放在面板右上角。
    status_text = "TRACKING" if tracking else "SEARCHING"
    status_color = COLOR_OK if tracking else COLOR_WARN
    status_size = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )[0]
    cv2.putText(
        frame,
        status_text,
        (panel_x2 - status_size[0] - 12, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # 帧号与 FPS。
    frame_text = f"Frame: {frame_index:03d}/{total_frames:03d}    FPS: {fps:.1f}"
    cv2.putText(
        frame,
        frame_text,
        (28, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    # 速度数值。使用平滑速度，避免数字快速跳动。
    cv2.putText(
        frame,
        f"Speed: {speed_px_s:6.1f} px/s",
        (28, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        COLOR_ARROW,
        2,
        cv2.LINE_AA,
    )

    # 解释箭头长度与速度的比例关系。
    cv2.putText(
        frame,
        f"Arrow length = speed x {ARROW_SECONDS:.2f}s",
        (28, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "longer arrow  ->  faster ball",
        (28, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


# ============================================================================
# 六、主流程
# ============================================================================


def main() -> int:
    """读取视频、逐帧检测/绘制、写出结果视频。"""
    if not INPUT_VIDEO.exists():
        print(f"[错误] 找不到输入视频：{INPUT_VIDEO}")
        print("       请把 homework.mp4 放在 track_ball.py 同一目录下。")
        return 1

    # 打开视频。
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        print(f"[错误] OpenCV 无法打开视频：{INPUT_VIDEO}")
        print("       请检查文件是否损坏，以及 OpenCV 是否支持该编码格式。")
        return 1

    # 读取视频参数：帧率、分辨率、总帧数。
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        # 某些文件的 FPS 元数据异常，使用 30 作为兜底值。
        print("[提示] 视频 FPS 信息异常，按 30 FPS 处理。")
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        print("[错误] 无法获取视频分辨率。")
        return 1

    # MP4V 是 OpenCV 内置支持的常见 MP4 编码，兼容性较好。
    # 如果希望得到 H.264 文件，可在写出后用 ffmpeg 再转码。
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, fps, (width, height), isColor=True
    )
    if not writer.isOpened():
        cap.release()
        print(f"[错误] 无法创建输出视频：{OUTPUT_VIDEO}")
        print("       请检查目录写权限，或系统 OpenCV 是否支持 mp4v 编码。")
        return 1

    print("=" * 66)
    print("绿球追踪开始")
    print(f"输入视频：{INPUT_VIDEO}")
    print(f"输出视频：{OUTPUT_VIDEO}")
    print(f"分辨率：{width} x {height}，FPS：{fps:.2f}，总帧数：{total_frames}")
    print("=" * 66)

    tracker = GreenBallTracker()
    trail: deque[np.ndarray] = deque(maxlen=TRAIL_LENGTH)

    processed_frames = 0
    tracked_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed_frames += 1

            # 步骤 1：寻找所有绿色候选。
            candidates = find_green_candidates(frame)

            # 步骤 2：结合形状、位置连续性、面积连续性锁定绿球。
            selected = tracker.update(candidates)

            # 步骤 3：计算平滑速度。只有确认检测到目标时才显示速度；
            # 丢失时速度置 0，避免把外推速度误当成真实速度。
            if selected is not None:
                tracked_frames += 1
                velocity_px_s = tracker.velocity_px_per_second(fps)
                speed_px_s = float(np.linalg.norm(velocity_px_s))

                if tracker.smooth_center is not None:
                    trail.append(tracker.smooth_center.copy())
                    draw_target(
                        frame,
                        tracker.smooth_center,
                        tracker.display_radius,
                    )
                    draw_direction_arrow(
                        frame,
                        tracker.smooth_center,
                        velocity_px_s,
                        tracker.display_radius,
                    )
            else:
                speed_px_s = 0.0

            # 步骤 4：绘制历史轨迹和 HUD 信息。
            draw_trail(frame, trail)
            draw_hud(
                frame,
                processed_frames,
                total_frames,
                fps,
                speed_px_s,
                tracking=selected is not None,
            )

            # 步骤 5：写入输出视频。
            writer.write(frame)

            # 在终端打印进度，每 30 帧刷新一次，避免刷屏。
            if processed_frames % 30 == 0 or processed_frames == total_frames:
                percent = (
                    100.0 * processed_frames / total_frames
                    if total_frames > 0
                    else 0.0
                )
                print(
                    f"\r处理进度：{processed_frames}/{total_frames} "
                    f"({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
    finally:
        # 无论循环正常结束还是中途异常，都要释放资源。
        cap.release()
        writer.release()

    print()  # 结束进度行
    print("=" * 66)

    if processed_frames == 0:
        print("[错误] 没有从视频中读到任何帧，未生成有效输出。")
        return 1

    print("处理完成！")
    print(f"共处理 {processed_frames} 帧，其中 {tracked_frames} 帧检测到绿球。")
    print(f"结果已保存：{OUTPUT_VIDEO}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
