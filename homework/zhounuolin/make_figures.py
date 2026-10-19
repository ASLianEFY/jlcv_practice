# -*- coding: utf-8 -*-
"""
生成 README.md 里用的说明图（不是作业主程序，只是"出图脚本"）。

用法（在 homework/zhounuolin/ 目录下）：
    python3 make_figures.py

生成的图：
    docs/figure_compare.jpg     绿球 vs 绿色圆柱 的圆度对比（说明检测原理）
    docs/figure_result_*.jpg    主程序真实输出视频里的几帧（说明最终效果）
"""

import os
import subprocess
import sys

import cv2
import numpy as np

import green_ball_tracker as T

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
VIDEO = T.find_default_video()


def label(img, text, org, color=(255, 255, 255), scale=0.6):
    """带深色底框的文字（Hershey 字体不支持中文，图内文字统一用 ASCII）。"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    # 底框，保证文字在任何背景上都读得清
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base), (25, 25, 25), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def make_compare_figure(frame_index=0):
    """
    左图：原始帧上标出"通过筛选的绿球"和"被圆度筛掉的绿色圆柱"；
    右图：绿色掩膜。两者并排，一眼看出为什么只靠颜色阈值不行。
    """
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[WARN] 读不到视频，跳过对比图")
        return False

    detector = T.GreenBallDetector()
    mask = detector.build_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = frame.copy()
    ball = None
    cyl = None
    for cnt in contours:
        area, radius, center, circ = detector.contour_features(cnt)
        if not (T.MIN_AREA <= area <= T.MAX_AREA):
            continue
        if circ >= T.MIN_CIRCULARITY:
            ball = (center, radius, circ)
            cv2.circle(vis, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)
            label(vis, "BALL circ=%.3f -> KEEP" % circ,
                  (int(center[0]) - 90, int(center[1] - radius) - 12), (0, 255, 0))
        else:
            if cyl is None or area > cyl[2]:
                cyl = (center, radius, area, circ)
            cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2, cv2.LINE_AA)
            label(vis, "CYLINDER circ=%.3f -> REJECT" % circ,
                  (int(center[0]) - 90, int(center[1] - radius) - 12), (0, 0, 255))

    label(vis, "circularity = contour_area / min_enclosing_circle_area", (14, 26), (255, 255, 0))
    label(vis, "threshold = %.2f  (findContours -> filter -> keep the ball)" % T.MIN_CIRCULARITY,
          (14, 52), (255, 255, 0))

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    label(mask_bgr, "HSV green mask (inRange + open + close)", (14, 26), (0, 255, 0))

    out = np.hstack([vis, mask_bgr])
    path = os.path.join(DOCS, "figure_compare.jpg")
    cv2.imwrite(path, out)
    print("[OK] %s  (ball circ=%.3f, cylinder circ=%s)" % (
        path, ball[2] if ball else -1, "%.3f" % cyl[3] if cyl else "n/a"))
    return True


def make_result_figures(frames=(60, 150, 455)):
    """跑一遍主程序（无窗口），再从输出视频里截几张图，保证图就是程序真实输出。"""
    tmp_video = os.path.join(DOCS, "_tmp_result.mp4")
    cmd = [sys.executable, os.path.join(HERE, "green_ball_tracker.py"),
           "--video", VIDEO, "--headless", "--output", tmp_video]
    print("[INFO] 生成结果视频：%s" % " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)

    cap = cv2.VideoCapture(tmp_video)
    want = set(frames)
    idx = 0
    saved = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            path = os.path.join(DOCS, "figure_result_%03d.jpg" % idx)
            cv2.imwrite(path, frame)
            saved.append(path)
        idx += 1
    cap.release()
    if os.path.isfile(tmp_video):
        os.remove(tmp_video)
    for path in saved:
        print("[OK] %s" % path)
    return len(saved) > 0


def make_compensate_figure(frame_index=320):
    """相机补偿模式的结果图：同一帧图像位移大、补偿后位移很小。"""
    tmp_video = os.path.join(DOCS, "_tmp_comp.mp4")
    cmd = [sys.executable, os.path.join(HERE, "green_ball_tracker.py"),
           "--video", VIDEO, "--headless", "--compensate", "--output", tmp_video]
    print("[INFO] 生成补偿模式视频：%s" % " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)

    cap = cv2.VideoCapture(tmp_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    os.path.isfile(tmp_video) and os.remove(tmp_video)
    if not ok:
        return False
    path = os.path.join(DOCS, "figure_compensate.jpg")
    cv2.imwrite(path, frame)
    print("[OK] %s" % path)
    return True


def main():
    os.makedirs(DOCS, exist_ok=True)
    if not os.path.isfile(VIDEO):
        print("[ERROR] 找不到视频：%s" % VIDEO)
        return 1
    make_compare_figure()
    make_result_figures()
    make_compensate_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
