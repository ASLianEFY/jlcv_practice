#!/usr/bin/env python3
"""绿色小球跟踪：更紧的红圈锁定 + 洋红箭头表示运动，速度固定显示在左上角"""
import cv2, numpy as np, sys, os

src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/homework.mp4")
out = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/Downloads/homework_tracked.mp4")

cap = cv2.VideoCapture(src)
if not cap.isOpened():
    sys.exit(f"打不开视频: {src}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
if not writer.isOpened():
    sys.exit(f"无法创建输出视频: {out}")

LO = np.array([35, 80, 50])
HI = np.array([85, 255, 255])
K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

last_center, last_r = None, 0.0
prev_center = None
vel = (0.0, 0.0)
lost = 0

MAX_LOST        = 15    # 超过这么多帧没检测到，彻底放弃
LOST_TOLERANCE  = 2     # 连续丢失 <= 2 帧仍显示红圈，用速度外推位置
POS_EMA         = 0.88 # 位置平滑：新检测权重（越大跟得越紧）
ARROW_SCALE     = 3.0
ARROW_MAX       = 140
SPEED_MIN       = 1.5
VEL_EMA         = 0.5
TEXT_X, TEXT_Y  = 20, 35

while True:
    ok, frame = cap.read()
    if not ok:
        break

    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LO, HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 预测位置：有历史时用“上次位置 + 速度”作为这一帧球该在的地方
    pred = None
    if last_center is not None:
        pred = (last_center[0] + vel[0], last_center[1] + vel[1])

    best = None
    best_score = -1.0
    best_xy, best_r = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 200:
            continue
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circ = 4.0 * np.pi * a / (p * p)
        if circ < 0.68:
            continue
        (_, _), re = cv2.minEnclosingCircle(c)
        if re < 1:
            continue
        fill = a / (np.pi * re * re)
        if fill < 0.65:
            continue

        # 圆心用质心，比最小外接圆稳；半径用面积反推
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        mx = M["m10"] / M["m00"]
        my = M["m01"] / M["m00"]
        ra = np.sqrt(a / np.pi)

        score = circ * a
        if pred is not None:
            dist = np.hypot(mx - pred[0], my - pred[1])
            # 离预测位置越远，分数打折越多，但不会直接排除
            score = score / (1.0 + dist / max(last_r, 20.0))

        if score > best_score:
            best = c
            best_score = score
            best_xy = (mx, my)
            best_r = ra

    if best is not None:
        x, y = best_xy
        r = best_r
        if last_center is None:
            last_center, last_r = (x, y), r
            prev_center = last_center
            vel = (0.0, 0.0)
        else:
            ax, ay = last_center
            last_center = (ax * (1 - POS_EMA) + x * POS_EMA,
                           ay * (1 - POS_EMA) + y * POS_EMA)
            last_r = last_r * (1 - POS_EMA) + r * POS_EMA
            if prev_center is None:
                prev_center = last_center
                vel = (0.0, 0.0)
            else:
                nvx = last_center[0] - prev_center[0]
                nvy = last_center[1] - prev_center[1]
                vel = (vel[0] * (1 - VEL_EMA) + nvx * VEL_EMA,
                       vel[1] * (1 - VEL_EMA) + nvy * VEL_EMA)
                prev_center = last_center
        lost = 0
        color = (0, 0, 255)
    else:
        lost += 1
        if last_center is not None and lost <= MAX_LOST:
            # 用速度外推位置，圈继续往前跟着球走
            last_center = (last_center[0] + vel[0], last_center[1] + vel[1])
            if prev_center is not None:
                prev_center = (prev_center[0] + vel[0], prev_center[1] + vel[1])
            vel = (vel[0] * 0.9, vel[1] * 0.9)
            # 短暂丢失仍显示红圈，超过容忍帧数才变橙色
            color = (0, 0, 255) if lost <= LOST_TOLERANCE else (0, 165, 255)
        else:
            last_center = None
            prev_center = None
            vel = (0.0, 0.0)
            color = (0, 165, 255)

    speed = float(np.hypot(vel[0], vel[1]))

    if last_center is not None:
        cx, cy = int(round(last_center[0])), int(round(last_center[1]))
        cv2.circle(frame, (cx, cy), int(last_r), color, 2)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        if speed > SPEED_MIN:
            L = min(speed * ARROW_SCALE, ARROW_MAX)
            ux, uy = vel[0] / speed, vel[1] / speed
            tip = (int(round(cx + ux * L)), int(round(cy + uy * L)))
            cv2.arrowedLine(frame, (cx, cy), tip, (255, 0, 255), 2, tipLength=0.25)

    text = f"v = {speed:.1f} px/f   ({speed*fps:.0f} px/s)"
    cv2.putText(frame, text, (TEXT_X, TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (TEXT_X, TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"已保存: {out}")
