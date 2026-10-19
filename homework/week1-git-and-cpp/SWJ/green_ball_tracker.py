# -*- coding: utf-8 -*-
"""
绿球识别与运动标注 —— Week1 作业（改进版）

任务：识别视频 homework.mp4 中的绿色小球，画圆框标位置，用箭头标注运动方向和速度（箭头方向 = 运动方向，长度 ∝ 速度大小）。
改进点：指数平滑速度 + 自适应突变检测 + 隔帧速度估计，箭头更平滑不抖动。

核心思路：
    1. HSV 颜色分割：BGR 转 HSV，按绿色色相提取绿色区域。
    2. 形态学去噪：开运算去小噪点，闭运算补小黑洞。
    3. 轮廓筛选：按面积、圆度（4πA/P²）、长宽比，把绿球和绿色圆柱区分开。
    4. 隔帧瞬时速度：用当前帧和若干帧前的位置差算速度，比单帧差分更稳。
    5. 指数平滑：新速度 = α×瞬时速度 + (1−α)×上一帧平滑速度，避免箭头抖。
    6. 自适应突变检测：用上一帧平滑速度预测本帧位置，偏差过大时判定突变；单帧突变时缓慢更新（可能是急掉头），连续多帧突变才冻结速度（防止误检导致箭头乱飞）。
    7. 绘制标注：用外接圆实际半径画绿色圆框；红色箭头起点是球心，终点是球心 + 平滑速度向量 × 缩放系数。

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
BURST_LIMIT = 3        # 连续多少帧突变才冻结速度，单帧突变不冻结
DISP_THRESHOLD = 40.0  # 突变检测阈值（像素），实际与预测偏差超过则判定突变
VELOCITY_GAP = 3       # 速度估计隔开多少帧，越大对单帧噪声越不敏感
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

    center_history = []      # 保存最近若干帧的球心，用于隔帧算速度
    prev_center = None       # 上一帧有效球心，用于突变预测
    smooth_vx = 0.0          # 平滑速度 x 分量
    smooth_vy = 0.0          # 平滑速度 y 分量
    burst_count = 0          # 连续突变计数器

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = find_green_ball(frame)

        if result is not None:
            cx, cy, radius = result

            # 记录位置历史，保留 VELOCITY_GAP+1 个点
            center_history.append((cx, cy))
            if len(center_history) > VELOCITY_GAP + 1:
                center_history.pop(0)

            # ---------------- 隔帧瞬时速度 ----------------
            # 用当前位置和 VELOCITY_GAP 帧前的位置做差分，除以时间间隔
            if len(center_history) > VELOCITY_GAP:
                cx_old, cy_old = center_history[-(VELOCITY_GAP + 1)]
                instant_vx = (cx - cx_old) * fps / VELOCITY_GAP
                instant_vy = (cy - cy_old) * fps / VELOCITY_GAP
            elif prev_center is not None:
                # 历史不够，退回到相邻两帧差分
                instant_vx = (cx - prev_center[0]) * fps
                instant_vy = (cy - prev_center[1]) * fps
            else:
                instant_vx = 0.0
                instant_vy = 0.0

            # ---------------- 自适应突变检测 ----------------
            if prev_center is not None:
                # 用上一帧平滑速度预测本帧位置
                pred_cx = prev_center[0] + smooth_vx / fps
                pred_cy = prev_center[1] + smooth_vy / fps
                err = math.hypot(cx - pred_cx, cy - pred_cy)

                if err <= DISP_THRESHOLD:
                    # 正常帧：重置突变计数，做正常指数平滑
                    burst_count = 0
                    smooth_vx = ALPHA_NORMAL * instant_vx + (1.0 - ALPHA_NORMAL) * smooth_vx
                    smooth_vy = ALPHA_NORMAL * instant_vy + (1.0 - ALPHA_NORMAL) * smooth_vy
                else:
                    # 突变帧：累计计数
                    burst_count += 1
                    if burst_count >= BURST_LIMIT:
                        # 连续多帧突变，判定为误检，冻结平滑速度不更新
                        pass
                    else:
                        # 单帧突变（可能是急掉头）：用小系数缓慢更新，避免箭头卡住
                        smooth_vx = ALPHA_BURST * instant_vx + (1.0 - ALPHA_BURST) * smooth_vx
                        smooth_vy = ALPHA_BURST * instant_vy + (1.0 - ALPHA_BURST) * smooth_vy

            prev_center = (cx, cy)
        # 没检测到球时：所有状态都不更新，保持上一帧

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
