#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 README 用的说明图（docs/*.jpg）。

    python3 make_figures.py

产物：
    docs/figure_compare.jpg   绿球 vs 绿色圆柱的圆度对比（含掩膜）
    docs/figure_result_XXX.jpg 结果视频抽帧
"""
import math
import os

import cv2
import numpy as np

from green_ball_tracker import (GreenBallDetector, Visualizer, BallTracker,
                                MIN_CIRCULARITY, find_default_video)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def make_compare(video_path):
    """找一帧同时包含绿球和绿色圆柱的图，左右拼对比。"""
    det = GreenBallDetector()
    cap = cv2.VideoCapture(video_path)
    best_frame, best_cands = None, []
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        mask = det.build_mask(f)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            a, r, ct, ci = det.contour_features(c)
            if a >= 3000:
                cands.append((c, a, r, ct, ci))
        # 找到一帧既有球(圆度>阈值)又有圆柱(圆度<阈值)的
        has_ball = any(c[4] >= MIN_CIRCULARITY for c in cands)
        has_cyl = any(c[4] < MIN_CIRCULARITY for c in cands)
        if has_ball and has_cyl:
            best_frame, best_cands = f.copy(), cands
            break
        idx += 1
    cap.release()

    if best_frame is None:
        print("[WARN] 没找到同时含球和圆柱的帧，跳过对比图")
        return

    left = best_frame.copy()
    for c, a, r, ct, ci in best_cands:
        is_ball = ci >= MIN_CIRCULARITY
        color = (0, 255, 0) if is_ball else (0, 0, 255)
        cv2.drawContours(left, [c], -1, color, 2, cv2.LINE_AA)
        (cx, cy), rr = cv2.minEnclosingCircle(c)
        cv2.circle(left, (int(cx), int(cy)), int(rr), color, 1, cv2.LINE_AA)
        label = "circ=%.3f" % ci
        org = (int(cx) - 60, int(cy) - int(rr) - 10)
        cv2.putText(left, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(left, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    mask = det.build_mask(best_frame)
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h = left.shape[0]
    left = cv2.resize(left, (int(left.shape[1] * 0.75), int(h * 0.75)))
    right = cv2.resize(right, (int(right.shape[1] * 0.75), int(right.shape[0] * 0.75)))
    combine = np.hstack([left, right])

    cv2.putText(combine, "green ball vs green cylinder (circularity)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(combine, "green ball vs green cylinder (circularity)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    path = os.path.join(OUT_DIR, "figure_compare.jpg")
    cv2.imwrite(path, combine, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("[INFO] %s" % path)


def make_results(video_path, frame_ids=(60, 150, 300, 455)):
    """跑一遍完整流程，抽指定帧存图。"""
    det = GreenBallDetector()
    tracker = BallTracker(30.0)
    viz = Visualizer()
    cap = cv2.VideoCapture(video_path)
    want = set(frame_ids)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = det.detect(frame, tracker.predicted_center() or tracker.center)
        center, detected = tracker.update(d["center"] if d else None)
        radius = math.sqrt(d["area"] / math.pi) if d is not None else 0
        if center is not None:
            if d is not None and detected:
                viz.draw_target(frame, center, radius, d["contour"])
            viz.push(center, tracker.speed_px_s)
            viz.draw_trail(frame)
            viz.draw_arrow(frame, center, tracker.direction, tracker.speed_px_s)
            viz.draw_speed_label(frame, center, tracker.direction, tracker.speed_px_s)
        viz.draw_hud(frame, tracker, detected, idx, 473)
        viz.draw_speed_curve(frame)
        if idx in want:
            path = os.path.join(OUT_DIR, "figure_result_%03d.jpg" % idx)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print("[INFO] %s" % path)
        idx += 1
    cap.release()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    vp = find_default_video()
    print("[INFO] 视频:", vp)
    make_compare(vp)
    make_results(vp)
    print("[INFO] 全部出图完成")
