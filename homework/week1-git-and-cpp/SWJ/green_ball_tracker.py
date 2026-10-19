# -*- coding: utf-8 -*-
"""
绿球识别与运动标注 —— Week1 作业

任务：识别视频 homework.mp4 中的绿色小球，画圆框标位置，
      用箭头标注运动方向和速度（箭头方向 = 运动方向，长度 ∝ 速度大小）。
核心思路（逻辑简单，便于逐行理解）：
    1. HSV 颜色分割：BGR 转 HSV，按绿色色相提取绿色区域，对光照不敏感。
    2. 形态学去噪：开运算去小噪点，闭运算补小黑洞，得到干净掩模。
    3. 轮廓筛选：按面积、圆度（4πA/P²）、长宽比，把绿球和绿色圆柱区分开。
    4. 跳变保护：某帧位置和上一帧相差过大则判定为误检，避免箭头乱飞。
    5. 平滑速度：最近若干帧首尾位移除以时间，比单帧差分更平滑，箭头不抖。
    6. 绘制标注：用外接圆实际半径画能包住整个球的绿色圆框；红色箭头起点是球心，终点是球心 + 速度向量 × 缩放系数。
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
# 往上溯 4 层目录到达仓库根目录 jlcv_practice/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VIDEO_PATH = os.path.join(REPO_ROOT, "homework.mp4")

# ----------------------------------------------------------------------
# 1. 颜色分割参数（HSV 绿色范围，可根据视频光照微调）
# ----------------------------------------------------------------------
# OpenCV 的 HSV 中 H 范围 0~179，绿色大约落在 35~90 之间。
GREEN_LOWER = np.array([35, 40, 40])   # HSV 下限 (H, S, V)
GREEN_UPPER = np.array([90, 255, 255]) # HSV 上限 (H, S, V)
KERNEL = np.ones((5, 5), np.uint8)     # 5×5 结构元素，用于开/闭运算

# ----------------------------------------------------------------------
# 2. 轮廓筛选参数（把绿球和绿色圆柱区分开）
# ----------------------------------------------------------------------
# 绿球面积约 9000~16000，靠近镜头时会变大；圆柱面积通常更大（>20000）。
AREA_MIN = 5000.0      # 小于此面积的轮廓当作噪声丢弃
AREA_MAX = 25000.0     # 大于此面积当作圆柱丢弃（放宽到 25000 避免漏检大球）
# 圆度 = 4π × 面积 / 周长²，正圆等于 1，越不规则越小。绿球≈0.75，拖影/圆柱≈0.5~0.6。
CIRC_MIN = 0.60
# 长宽比 = 宽 / 高，正圆等于 1。贴边时球被切掉一部分会变扁，放宽上限避免漏检。
ASPECT_MIN = 0.80
ASPECT_MAX = 1.40

# ----------------------------------------------------------------------
# 3. 平滑与绘制参数
# ----------------------------------------------------------------------
SMOOTH_WINDOW = 5      # 最近几帧做平均速度，越大越平滑但越迟钝
ARROW_SCALE = 0.15     # 箭头像素长度 = 速度(px/s) × 此系数，越小箭头越短
ARROW_MAX_LEN = 100    # 箭头最大长度（像素），防止速度突变导致箭头飞出画面
MIN_SPEED = 50.0       # 低于此速度不画箭头，避免球静止时箭头乱抖
MAX_DISP = 120.0       # 跳变保护：相邻两帧最大允许位移（像素），超过则判定为误检


def find_green_ball(frame):
    """在当前帧中找出绿球，返回 (cx, cy, radius)，找不到返回 None。"""
    # BGR → HSV，便于按颜色色相提取绿色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 做掩膜：HSV 落在绿色范围内的像素设为白，其余设为黑
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    # 开运算去小噪点，闭运算补小黑洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    # 找所有轮廓，逐个判断是否像“球”
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = -1.0
    for cnt in contours:
        # 按面积粗筛，排除太小的噪点和太大的圆柱
        area = cv2.contourArea(cnt)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        # 圆度，越接近 1 越像正圆
        circularity = 4.0 * math.pi * area / (peri * peri)

        # 长宽比，排除被边缘切得很扁或本身细长的物体
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h

        # 既要圆，长宽比也要接近 1，才当作绿球；排除细长的绿色圆柱
        if circularity < CIRC_MIN or aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue

        # 用一阶矩计算质心，比外接圆圆心更准确
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 外接圆半径，用来画能包住整个球的圆框
        (_, _), radius = cv2.minEnclosingCircle(cnt)

        # 选圆度最高的候选作为绿球
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

    centers_history = []   # 最近若干帧的中心坐标，用于算平滑速度
    prev_center = None     # 上一帧通过检查的中心坐标，用于跳变保护

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = find_green_ball(frame)

        # ---------------- 跳变保护 + 记录历史 ----------------
        if result is not None:
            cx, cy, radius = result
            if prev_center is not None:
                disp = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if disp > MAX_DISP:
                    # 位移太大，判定为误检：本帧不记录，也不更新 prev_center，
                    # 下一帧还能和上一次有效位置比较，避免连锁误判
                    result = None
                else:
                    centers_history.append((cx, cy))
                    if len(centers_history) > SMOOTH_WINDOW:
                        centers_history.pop(0)
                    prev_center = (cx, cy)
            else:
                centers_history.append((cx, cy))
                prev_center = (cx, cy)

        # ---------------- 计算平滑速度：history 首尾位移 / 时间 ----------------
        vx = vy = 0.0
        speed = 0.0
        if len(centers_history) >= 2:
            x0, y0 = centers_history[0]
            x1, y1 = centers_history[-1]
            dt = (len(centers_history) - 1) / fps   # 首尾之间经过了多少秒
            if dt > 0:
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                speed = math.hypot(vx, vy)          # 速度大小，单位像素/秒

        # ---------------- 绘制标注 ----------------
        if result is not None:
            cx, cy, radius = result
            cx, cy = int(cx), int(cy)
            r = max(int(radius), 10)                 # 用外接圆实际半径，能包住整个球
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)   # 绿色圆框
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 质心点

            if speed > MIN_SPEED:
                # 箭头长度 = 速度 × 缩放系数，并做上限截断
                length = speed * ARROW_SCALE
                if length > ARROW_MAX_LEN:
                    length = ARROW_MAX_LEN
                # 单位速度方向 × 长度 = 箭头在屏幕上的偏移量
                dx = vx / speed * length
                dy = vy / speed * length
                end_x = int(cx + dx)
                end_y = int(cy + dy)
                # 红色箭头：起点是球心，方向是运动方向，长度代表速度大小
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 3)

            # 在球旁边显示当前速度数值
            cv2.putText(frame, f"{speed:.0f} px/s", (cx + 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Green Ball Tracking", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
