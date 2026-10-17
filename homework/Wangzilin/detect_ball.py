# -*- coding: utf-8 -*-
"""
RoboMaster 视觉自瞄培训 · 大作业
任务：识别视频中的绿色小球，用圆形目标框标出它、用箭头标注运动方向、并显示像素速度。

作者：王梓霖 (Wangzilin)
使用的 AI 工具：OpenCode（辅助编写与调试，算法逻辑已逐行理解）

核心思路（在老师 PPT 第 13~18 页的基础上做了两处改进）：
    1. 颜色分割 + 形态学，得到干净的绿色区域          （PPT 第 14、15 页）
    2. 轮廓筛选：面积 + 圆度 4πA/P² + 长宽比           （PPT 第 16 页）
       —— 用来把「绿球」和「绿色圆柱」区分开
    3. 跨帧追踪：速度预测 + 距离门控 + 丢帧外推         （PPT 第 17 页）
       改进 A：不采用「贴边就全局放宽阈值」（实测会把绿色圆柱也放进来），
               而是「严格优先，只有丢帧时才在预测位置附近放宽阈值」，
               丢帧数从 13 帧降到 0 帧。
       改进 B：预测与观测加权融合（alpha-beta 滤波，即卡尔曼的简化思想，PPT 第 19 页）。
               球被画面边缘切掉时，观测到的只是半个球、质心会偏向画面内侧，
               此时降低对观测的信任、多依赖速度预测，红圈就不会"缩"进去。
       改进 C：贴边时再结合"可见面积的变化趋势"判断球是在离开还是回到画面。
               只看单帧面积会被噪声骗，所以用"当前面积 vs 5 次观测前的面积"。
               球正在回到画面（面积变大）时提高增益，避免速度估计带着旧方向
               "冲过头"、在折返处出现一段抖动。
    4. 速度用 px/s 表示                                （PPT 第 18 页）

运行方式（在仓库根目录 jlcv_practice/ 下）：
    python homework/Wangzilin/detect_ball.py
预览窗口中按 q 键退出。
"""

import math
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------
# 0. 路径
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_PATH = os.path.join(REPO_ROOT, "homework.mp4")

# ----------------------------------------------------------------------
# 1. 检测参数
# ----------------------------------------------------------------------
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])

# --- 严格阈值：正常检测使用，精确但要求形状规整 ---
AREA_MIN = 7000.0
AREA_MAX = 32000.0
CIRC_MIN = 0.68
ASPECT_MIN = 0.85
ASPECT_MAX = 1.15

# --- 宽松阈值：只在"预测位置附近"兜底、以及贴边形变时使用 ---
RELAX_AREA_MIN = 4000.0
RELAX_AREA_MAX = 40000.0
RELAX_CIRC_MIN = 0.45
RELAX_ASPECT_MIN = 0.40
RELAX_ASPECT_MAX = 1.80

# ----------------------------------------------------------------------
# 2. 追踪参数
# ----------------------------------------------------------------------
GATE_DIST = 120.0        # 正常门控：候选离预测点超过该距离就拒绝
GATE_DIST_TIGHT = 90.0   # 兜底门控：放宽阈值时用更严的距离限制
MAX_LOST = 8             # 丢帧后最多外推 8 帧（PPT 第 17 页）
EDGE_MARGIN = 8          # 外接矩形距画面边缘小于该像素数即视为"贴边"

# ----------------------------------------------------------------------
# 3. 融合参数（alpha-beta 滤波，卡尔曼的简化版，PPT 第 19 页）
#    位置 = 预测位置 + alpha × (观测位置 - 预测位置)
#    alpha 越大越相信观测；越小越相信预测。
# ----------------------------------------------------------------------
ALPHA_NORMAL = 0.75      # 观测可信（形状规整、不在边缘）
ALPHA_EDGE = 0.25        # 观测不可信（贴边被切 / 形状变形）
BETA_NORMAL = 0.35       # 速度修正系数（观测可信）
BETA_EDGE = 0.10         # 速度修正系数（观测不可信）

# --- 贴边时的"重新回到画面"判定（改进 C）---
#     贴边时压低增益会让速度更新变慢，球折返时预测会"带着旧速度冲过头"，
#     表现为一圈明显的抖动。用"可见面积的变化趋势"判断球是在离开还是在回来：
#     面积变大 = 球正在回到画面 -> 提高增益，让速度及时跟上折返。
ALPHA_REENTER = 0.60     # 贴边 + 可见面积正在变大
BETA_REENTER = 0.30
AREA_TREND_SPAN = 5      # 用"当前面积 vs 5 次观测前的面积"判断趋势（抗单帧噪声）
RADIUS_STEP = 5.0        # 半径每帧最多变化多少像素，避免贴边恢复那一帧"突然跳大"

# ----------------------------------------------------------------------
# 4. 绘制参数
# ----------------------------------------------------------------------
ARROW_SCALE = 4.0
ARROW_MIN_SPEED = 60.0
COLOR_ACTIVE = (0, 0, 255)
COLOR_LOST = (160, 160, 160)


def new_state():
    """新建空的跟踪状态。center 为 None 表示还没锁定目标。"""
    return {"center": None, "radius": 0.0, "velocity": (0.0, 0.0), "lost": 0,
            "areas": []}


def detect_candidates(hsv, kernel):
    """找出这一帧里所有"可能是绿球"的候选，每个候选带 strict / near_edge 标记。"""
    img_h, img_w = hsv.shape[:2]

    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < RELAX_AREA_MIN or area > RELAX_AREA_MAX:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh

        if circularity < RELAX_CIRC_MIN:
            continue
        if aspect < RELAX_ASPECT_MIN or aspect > RELAX_ASPECT_MAX:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        (_, _), radius = cv2.minEnclosingCircle(cnt)

        near_edge = (bx <= EDGE_MARGIN or by <= EDGE_MARGIN or
                     bx + bw >= img_w - EDGE_MARGIN or
                     by + bh >= img_h - EDGE_MARGIN)

        is_strict = (AREA_MIN <= area <= AREA_MAX and
                     circularity >= CIRC_MIN and
                     ASPECT_MIN <= aspect <= ASPECT_MAX)

        candidates.append({
            "center": (cx, cy),
            "radius": radius,
            "score": circularity * area,
            "area": area,
            "strict": is_strict,
            "near_edge": near_edge,
        })
    return candidates


def process_frame(frame, state, fps, kernel):
    """处理一帧：检测 -> 追踪(预测+融合) -> 绘制。state 为跨帧可变字典。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    candidates = detect_candidates(hsv, kernel)

    if state["center"] is None:
        # ---------- 初始化：只允许用严格候选，避免锁到圆柱 ----------
        strict_all = [c for c in candidates if c["strict"]]
        if strict_all:
            best = max(strict_all, key=lambda c: c["score"])
            state["center"] = (float(best["center"][0]), float(best["center"][1]))
            state["radius"] = float(best["radius"])
            state["velocity"] = (0.0, 0.0)
            state["lost"] = 0
    else:
        # ---------- ① 预测：用上一帧速度推断本帧球的位置 ----------
        pred_x = state["center"][0] + state["velocity"][0]
        pred_y = state["center"][1] + state["velocity"][1]

        def dist_to_pred(c):
            return math.hypot(c["center"][0] - pred_x, c["center"][1] - pred_y)

        # ---------- ② 严格优先，找不到再在预测附近用宽松候选兜底 ----------
        strict_near = [c for c in candidates if c["strict"] and dist_to_pred(c) <= GATE_DIST]
        if strict_near:
            chosen = min(strict_near, key=dist_to_pred)
        else:
            relaxed_near = [c for c in candidates if dist_to_pred(c) <= GATE_DIST_TIGHT]
            chosen = min(relaxed_near, key=dist_to_pred) if relaxed_near else None

        if chosen is not None:
            # ---------- ③ 面积趋势：判断球在"离开画面"还是"回到画面" ----------
            #     单帧比较会被噪声骗（面积忽大忽小），所以比较"当前 vs 5 次观测前"
            areas = state["areas"]
            areas.append(chosen["area"])
            if len(areas) > AREA_TREND_SPAN + 1:
                areas.pop(0)
            area_trend = areas[-1] - areas[0] if len(areas) > AREA_TREND_SPAN else 0.0

            # ---------- ④ 融合：按可信度加权"预测"和"观测" ----------
            reliable = chosen["strict"] and not chosen["near_edge"]
            if reliable:
                alpha, beta = ALPHA_NORMAL, BETA_NORMAL
            elif area_trend > 0:
                # 贴边但可见面积在变大 -> 球正在回到画面：观测重新可信，速度要及时跟上
                alpha, beta = ALPHA_REENTER, BETA_REENTER
            else:
                # 贴边且面积在缩小 -> 球正在离开画面：观测被切得太厉害，主要靠预测
                alpha, beta = ALPHA_EDGE, BETA_EDGE

            rx = chosen["center"][0] - pred_x   # 观测与预测的偏差
            ry = chosen["center"][1] - pred_y

            state["center"] = (pred_x + alpha * rx, pred_y + alpha * ry)
            state["velocity"] = (state["velocity"][0] + beta * rx,
                                 state["velocity"][1] + beta * ry)

            if reliable:
                # 半径限速变化，避免贴边恢复那一帧"突然变大"
                delta = float(chosen["radius"]) - state["radius"]
                delta = max(-RADIUS_STEP, min(RADIUS_STEP, delta))
                state["radius"] += delta
            # 贴边时半径会被低估，不更新（沿用上一次的值）
            state["lost"] = 0
        else:
            # ---------- ⑤ 丢帧外推：跑出画面就收手 ----------
            h, w = frame.shape[:2]
            if pred_x < 0 or pred_y < 0 or pred_x > w or pred_y > h:
                state.update(new_state())
            else:
                state["center"] = (pred_x, pred_y)
                state["lost"] += 1
                if state["lost"] > MAX_LOST:
                    state.update(new_state())

    # ---------------- 绘制 ----------------
    if state["center"] is not None:
        cx, cy = int(state["center"][0]), int(state["center"][1])
        r = max(int(state["radius"]), 10)
        vx, vy = state["velocity"]
        speed = math.hypot(vx, vy) * fps
        color = COLOR_ACTIVE if state["lost"] == 0 else COLOR_LOST

        cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        if speed > ARROW_MIN_SPEED:
            cv2.arrowedLine(frame, (cx, cy),
                            (int(cx + vx * ARROW_SCALE), int(cy + vy * ARROW_SCALE)),
                            color, 3, tipLength=0.3)

        cv2.putText(frame, f"{speed:.0f} px/s", (cx + r + 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "Green Ball Tracking (press q to quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频文件：{VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kernel = np.ones((5, 5), np.uint8)
    state = new_state()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(frame, state, fps, kernel)
        cv2.imshow("Green Ball Tracking", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
