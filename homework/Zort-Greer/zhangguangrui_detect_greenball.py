#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
  绿球识别 + 运动方向/速度箭头标注  （学习注释版）
============================================================

【任务】
  识别视频里的绿色球体，用箭头标出它的运动方向和速度：
    - 箭头方向  = 小球运动方向
    - 箭头长度  = 速度大小（越长越快）
    - 附加数字  = 速度大小（px/s）

【这个视频的难点】
  画面里有两个绿色物体：
    1. 绿色【球体】—— 圆的，是我们真正要跟踪的目标
    2. 绿色【圆柱体】—— 细长，是干扰物
  如果只是“找最大的绿色区域”，会错误地锁在更大的圆柱体上。

【核心思路：用“圆度”区分球和圆柱】
  一个物体在图像里投影是不是“圆”，可以用下面这个量衡量：

        圆度 = 轮廓面积 / 最小外接圆面积

   - 完美的圆 → 圆度 ≈ 1
   - 细长的矩形/圆柱 → 圆度明显 < 1
  本视频里：球 ≈ 0.82~0.91，圆柱 ≈ 0.45~0.71，
  所以用一个阈值 0.80 就能把两者干净分开。

【整体流程（对应下面每个函数）】
  1. 逐帧读视频，转 HSV 颜色空间，扣出绿色区域          -> detect_ball
  2. 形态学去噪（开运算 + 闭运算）                        -> detect_ball
  3. 找轮廓，算圆度，只保留“圆度达标”的那个球            -> detect_ball
  4. 缺失帧线性插值 + 滑动平均平滑轨迹                    -> interpolate_and_smooth
  5. 对平滑轨迹做差分求速度（方向 + 大小）                -> main
  6. 按速度画箭头 + 数字，写出结果视频                    -> main

【运行方法】
  python detect_green_ball.py --video homework.mp4 --output result.mp4
"""

import argparse
import numpy as np
import cv2


# ------------------------------------------------------------------
# 可调参数：换视频/换场景时主要改这里
# ------------------------------------------------------------------
MIN_AREA = 500      # 最小面积（像素²）：比这小的绿色区域当作噪声忽略
MIN_ROUND = 0.80    # 圆度阈值：>= 判定为球；< 判定为干扰物或“检测不到”
SMOOTH_WIN = 5      # 轨迹平滑的滑动窗口大小（帧数）
GREEN_LOWER = (35, 40, 40)    # 绿色 HSV 下界 (H, S, V)
GREEN_UPPER = (85, 255, 255)  # 绿色 HSV 上界 (H, S, V)


def roundness(contour):
    """计算一个轮廓的“圆度”：面积 / 最小外接圆面积。

    返回 0~1 之间的值，越接近 1 越像圆。
    - 球（圆的）  -> 接近 0.85 以上
    - 圆柱（细长）-> 明显低于 0.8
    """
    area = cv2.contourArea(contour)          # 轮廓包围的真实面积
    (_, _), r = cv2.minEnclosingCircle(contour)  # 能框住该轮廓的最小圆半径
    if r <= 0:
        return 0.0
    return area / (np.pi * r * r)            # 面积 / 圆面积


def detect_ball(video, lower, upper):
    """第一遍扫描：逐帧检测绿球质心，返回每帧的 (cx, cy, r) 或 None。

    返回：
      pts  : list，长度 = 帧数，每个元素是 (质心x, 质心y, 半径) 或 None
      fps  : 视频帧率
      (w,h): 画面宽高

    这一步只做“检测”，不画任何东西；画箭头在第二遍扫描（main）里做。
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("无法打开视频: %s" % video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- 1. BGR -> HSV -------------------------------------------------
        # 为什么要 HSV 而不是直接用 RGB？
        # RGB 里“绿色”是 R、G、B 三个数的组合，受光照/阴影影响大；
        # HSV 把“颜色(色相 H)”“鲜艳程度(饱和度 S)”“亮度(V)”分开，
        # 这样用 H 一个维度就能锁定“绿”，对亮度变化更鲁棒。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 2. 阈值分割：把“绿色像素”置 255，其余置 0 ---------------------
        # inRange(hsv, 下界, 上界) 返回一个二值图 mask。
        # H 在 [35,85] 是绿色区段；S、V 都 >=40 是排除太暗/太灰的像素。
        mask = cv2.inRange(hsv, lower, upper)

        # --- 3. 形态学去噪 -------------------------------------------------
        # 开运算(先腐蚀后膨胀)：去掉孤立的小噪点
        # 闭运算(先膨胀后腐蚀)：填上物体内部的小空洞，让轮廓更完整
        # 核大小(5,5)/(9,9) 是按“球半径约 60~90 像素”调的，球变小需调小。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # --- 4. 找轮廓 -----------------------------------------------------
        # findContours 返回所有连通白色区域的外轮廓。
        # RETR_EXTERNAL 只取最外层轮廓（不要内部的洞）。
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 5. 用圆度挑出“球” ---------------------------------------------
        # 遍历所有绿色轮廓，取“圆度最高”的那个；
        # 只有它圆度 >= MIN_ROUND 才认定是球，否则记 None（表示这一帧没找到球）。
        best = None
        best_round = -1.0
        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            rd = roundness(c)
            if rd > best_round:
                best_round = rd
                # 用图像矩求质心：m00=面积，m10/m00=质心x，m01/m00=质心y
                M = cv2.moments(c)
                if M["m00"] > 1:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = 0.0, 0.0
                (_, _), r = cv2.minEnclosingCircle(c)
                best = (cx, cy, r)

        pts.append(best if best_round >= MIN_ROUND else None)

    cap.release()
    return pts, fps, (w, h)


def interpolate_and_smooth(pts, win=SMOOTH_WIN):
    """第二道处理：补缺失帧 + 平滑轨迹。

    为什么要这一步？
    - 球贴屏幕边缘被裁切时（弹跳到顶端/底端瞬间），圆度会骤降，
      导致那几帧被 detect_ball 判为 None。用相邻帧线性插值补上。
    - 检测质心存在逐帧抖动，直接差分求速度会很“毛糙”，
      先做滑动平均，速度向量才平滑。

    返回平滑后的 (sx, sy, sr)，分别是质心x、质心y、半径的数组。
    """
    n = len(pts)
    # 把 None 换成 NaN，便于 numpy 处理
    cx = np.array([p[0] if p else np.nan for p in pts], dtype=float)
    cy = np.array([p[1] if p else np.nan for p in pts], dtype=float)
    r = np.array([p[2] if p else np.nan for p in pts], dtype=float)

    idx = np.arange(n)
    valid = ~np.isnan(cx)                       # 哪些帧是有效检测
    if valid.sum() < 2:
        raise SystemExit("检测到的绿球帧数过少，请检查绿色 HSV 范围")

    # 线性插值：把 NaN 位置用两侧有效值连成直线填上
    cx = np.interp(idx, idx[valid], cx[valid])
    cy = np.interp(idx, idx[valid], cy[valid])
    r = np.interp(idx, idx[valid], r[valid])

    # 滑动平均：win 帧取均值，消除抖动
    # 关键：np.pad(..., mode='edge') 用边缘值复制来填充，
    # 而不是 np.convolve(mode='same') 默认的 0 填充。
    # 若用 0 填充，首尾几帧会被拉向坐标原点(左上角)，速度会算错。
    pad = win // 2
    kernel = np.ones(win) / win
    sx = np.convolve(np.pad(cx, pad, mode="edge"), kernel, mode="valid")
    sy = np.convolve(np.pad(cy, pad, mode="edge"), kernel, mode="valid")
    sr = np.convolve(np.pad(r, pad, mode="edge"), kernel, mode="valid")
    return sx, sy, sr


def main():
    # --- 命令行参数 -------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="homework.mp4", help="输入视频路径")
    ap.add_argument("--output", default="result.mp4", help="输出视频路径")
    ap.add_argument("--max-len", type=int, default=150, help="箭头最大长度(px)")
    ap.add_argument("--min-len", type=int, default=15, help="箭头最小长度(px)")
    args = ap.parse_args()

    # --- 第一遍：检测 -----------------------------------------------------
    pts, fps, (w, h) = detect_ball(args.video, GREEN_LOWER, GREEN_UPPER)
    n = len(pts)
    hit = sum(1 for p in pts if p)
    print("视频: %d 帧, %.0f fps, %dx%d, 检出绿球 %d 帧 (缺失 %d 帧)"
          % (n, fps, w, h, hit, n - hit))

    # --- 插值 + 平滑 ------------------------------------------------------
    sx, sy, sr = interpolate_and_smooth(pts)

    # --- 求速度 -----------------------------------------------------------
    # np.gradient 对数组做“中心差分”，相当于每帧的位移/时间 = 速度。
    # 位移单位是像素/帧，乘 fps 变成像素/秒 (px/s)。
    vx = np.gradient(sx) * fps
    vy = np.gradient(sy) * fps
    speed = np.hypot(vx, vy)          # 速度大小 = sqrt(vx^2 + vy^2)

    # 箭头长度映射：用速度的 95 分位做归一化基准。
    # 这样“越长越快”是相对的、且保证箭头不会无限长越出画面。
    p95 = max(float(np.percentile(speed, 95)), 1e-6)

    # --- 第二遍：重新读视频，逐帧画箭头，写结果 ---------------------------
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # mp4v 编码器
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cx, cy = int(round(sx[i])), int(round(sy[i]))
        rad = int(round(sr[i]))

        # 黄色圆圈标出检出的球（便于肉眼确认没锁到圆柱上）
        cv2.circle(frame, (cx, cy), rad, (0, 255, 255), 2)

        # 速度向量 -> 箭头
        v = np.array([vx[i], vy[i]])
        sp = float(np.hypot(v[0], v[1]))
        if sp > 1e-6:   # 速度几乎为 0 时不画箭头（方向无意义）
            u = v / sp                        # 单位方向向量
            L = int(np.clip(sp / p95 * args.max_len, args.min_len, args.max_len))
            tip = (cx + int(round(u[0] * L)), cy + int(round(u[1] * L)))
            # arrowedLine: 起点(球心) -> 终点(球心+方向*长度)，红色箭头
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 0, 255), 3, cv2.LINE_AA, 0, 0.25)

        # 球心小圆点（红色）
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 速度数字：先画黑色粗字(描边)，再画白色细字，保证任何背景都清晰
        label = "%.0f px/s" % sp
        org = (cx + 15, cy - 15)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        i += 1

    cap.release()
    out.release()
    print("已写出: %s" % args.output)


if __name__ == "__main__":
    main()
