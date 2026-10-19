# -*- coding: utf-8 -*-
"""
绿球识别与运动标注 —— Week1 作业（修复版）

任务：识别视频 homework.mp4 中的绿色小球，画圆框标位置，
      用箭头标注运动方向和速度（箭头方向 = 运动方向，长度 ∝ 速度大小）。
修复：1）突变检测死锁——连续多帧突变时接受新速度并重置，避免追踪无法恢复；
      2）丢帧速度错误——记录帧索引，按实际时间间隔算速度，不再假设固定帧 gap。

核心思路：
    1. HSV 颜色分割：BGR 转 HSV，按绿色色相提取绿色区域。
    2. 形态学去噪：开运算去小噪点，闭运算补小黑洞。
    3. 轮廓筛选：按面积、圆度（4πA/P²）、长宽比，把绿球和绿色圆柱区分开。
    4. 实际时间速度：用当前位置和上一帧有效位置的实际帧差算速度，丢帧也正确。
    5. 指数平滑：新速度 = α×瞬时速度 + (1−α)×上一帧平滑速度，避免箭头抖。
    6. 自适应突变检测：预测本帧位置，偏差过大时判定突变；单帧突变缓慢更新，
       连续多帧突变则接受新速度并重置（防止急掉头后追踪死锁）。
    7. 绘制标注：用外接圆实际半径画绿色圆框；红色箭头起点是球心，终点是
       球心 + 平滑速度向量 × 缩放系数。

运行（在仓库根目录 jlcv_practice/ 下）：
    python3 homework/week1-git-and-cpp/SWJ/green_ball_tracker.py
按 q 键退出。依赖：pip install opencv-python numpy
"""

import cv2
import numpy as np
import math
import os

# ----------------------------------------------------------------------
# 0. 路径：自动定位到仓库根目录的 homework.mp4
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VIDEO_PATH = os.path.join(REPO_ROOT, "homework.mp4")

# ----------------------------------------------------------------------
# 1. 颜色分割参数
# ----------------------------------------------------------------------
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)

# ----------------------------------------------------------------------
# 2. 轮廓筛选参数（把绿球和绿色圆柱区分开）
# ----------------------------------------------------------------------
AREA_MIN = 5000.0      # 小于此面积的轮廓当作噪声丢弃
AREA_MAX = 25000.0     # 大于此面积当作圆柱丢弃（放宽到 25000 避免漏检大球）
CIRC_MIN = 0.60        # 圆度 = 4π × 面积 / 周长²，正圆等于 1，越不规则越小
ASPECT_MIN = 0.80      # 长宽比下限
ASPECT_MAX = 1.40      # 长宽比上限（贴边时球被切掉一部分会变扁）

# ----------------------------------------------------------------------
# 3. 平滑与突变检测参数
# ----------------------------------------------------------------------
ALPHA_NORMAL = 0.5     # 正常时的指数平滑系数，越大越灵敏但越抖
ALPHA_BURST = 0.15     # 突变时的缓慢更新系数，用于急掉头的平滑过渡
BURST_LIMIT = 3        # 连续多少帧突变后重置速度，避免追踪死锁
DISP_THRESHOLD = 40.0  # 突变检测阈值（像素），实际与预测偏差超过则判定突变
ARROW_SCALE = 0.15     # 箭头像素长度 = 速度 × 此系数，越小箭头越短
ARROW_MAX_LEN = 100    # 箭头最大长度（像素），防止速度突变导致箭头飞出画面
MIN_SPEED = 50.0       # 低于此速度不画箭头，避免球静止时箭头乱抖


def find_green_ball(frame):
    """在当前帧中找出绿球，返回 (cx, cy, radius)，找不到返回 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN or area > AREA_MAX:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        if circularity < CIRC_MIN or aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        if circularity > best_circularity:
            best_circularity = circularity
            best = (cx, cy, radius)
    return best


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频：{VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"视频 FPS = {fps:.1f}，按 q 键退出")

    prev_center = None      # 上一帧有效球心
    prev_frame_idx = None   # 上一帧有效帧号
    smooth_vx = 0.0         # 平滑速度 x 分量
    smooth_vy = 0.0         # 平滑速度 y 分量
    burst_count = 0         # 连续突变计数器
    frame_idx = 0           # 当前帧号

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = find_green_ball(frame)

        if result is not None:
            cx, cy, radius = result

            if prev_center is not None:
                # 用实际帧差算瞬时速度，丢帧时 dt_frames > 1，速度仍正确
                dt_frames = frame_idx - prev_frame_idx
                if dt_frames > 0:
                    instant_vx = (cx - prev_center[0]) * fps / dt_frames
                    instant_vy = (cy - prev_center[1]) * fps / dt_frames
                else:
                    instant_vx = 0.0
                    instant_vy = 0.0

                # 自适应突变检测：用上一帧平滑速度预测本帧位置（按实际时间）
                pred_cx = prev_center[0] + smooth_vx * dt_frames / fps
                pred_cy = prev_center[1] + smooth_vy * dt_frames / fps
                err = math.hypot(cx - pred_cx, cy - pred_cy)

                if err <= DISP_THRESHOLD:
                    burst_count = 0
                    smooth_vx = ALPHA_NORMAL * instant_vx + (1.0 - ALPHA_NORMAL) * smooth_vx
                    smooth_vy = ALPHA_NORMAL * instant_vy + (1.0 - ALPHA_NORMAL) * smooth_vy
                else:
                    burst_count += 1
                    if burst_count >= BURST_LIMIT:
                        # 连续多帧突变：接受新速度并重置，避免急掉头后追踪死锁
                        smooth_vx = instant_vx
                        smooth_vy = instant_vy
                        burst_count = 0
                    else:
                        # 单帧突变（可能急掉头）：缓慢更新，避免箭头卡住
                        smooth_vx = ALPHA_BURST * instant_vx + (1.0 - ALPHA_BURST) * smooth_vx
                        smooth_vy = ALPHA_BURST * instant_vy + (1.0 - ALPHA_BURST) * smooth_vy

            prev_center = (cx, cy)
            prev_frame_idx = frame_idx
        # 丢帧时：prev_center / prev_frame_idx 不更新，下次捕获时 dt_frames 会反映实际间隔

        # ---------------- 绘制标注 ----------------
        if result is not None:
            cx, cy, radius = result
            cx, cy = int(cx), int(cy)
            r = max(int(radius), 10)
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)   # 绿色圆框
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 质心点

            speed = math.hypot(smooth_vx, smooth_vy)
            if speed > MIN_SPEED:
                length = speed * ARROW_SCALE
                if length > ARROW_MAX_LEN:
                    length = ARROW_MAX_LEN
                dx = smooth_vx / speed * length
                dy = smooth_vy / speed * length
                end_x = int(cx + dx)
                end_y = int(cy + dy)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 3)

            cv2.putText(frame, f"{speed:.0f} px/s", (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Green Ball Tracking", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
