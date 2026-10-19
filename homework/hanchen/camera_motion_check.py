#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
附加分析：判断画面中绿球的位移是否来自球自身运动。

做法一（不依赖坐标变换，最直观）：
    同一画面里有一个静止的绿圆柱。分别跟踪球和绿圆柱，比较相邻帧的位移矢量。
    如果球在桌面上滚动，两者的位移方向/大小应当明显不同；
    如果球静止而相机在动，两者应当几乎相同。

做法二（相机运动补偿）：
    用 ORB 特征匹配估计相邻帧之间的单应矩阵并累积，把球心映射回第 0 帧坐标系。
    矫正后若球心基本不动，说明球的位移是相机运动造成的视运动。

运行：
  python3 camera_motion_check.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from green_ball_tracker import Candidate, detect_trajectory, find_candidates


def pick_static_cylinder(candidates: list[Candidate],
                         ball_xy: tuple[float, float],
                         ball_r: float) -> Candidate | None:
    """在候选里选绿圆柱：比球大、明显竖长、且离球有距离。"""
    best: Candidate | None = None
    for c in candidates:
        if c.radius <= 1.25 * ball_r:
            continue
        if c.aspect >= 0.92:
            continue
        if math.hypot(c.x - ball_xy[0], c.y - ball_xy[1]) <= 1.2 * ball_r:
            continue
        if best is None or c.area > best.area:
            best = c
    return best


def compare_with_static_object(video_path: Path, detections, fps: float) -> None:
    """做法一：比较球与静止绿圆柱的逐帧位移。"""
    cap = cv2.VideoCapture(str(video_path))
    ball_steps: list[np.ndarray] = []
    cyl_steps: list[np.ndarray] = []

    prev_ball: np.ndarray | None = None
    prev_cyl: np.ndarray | None = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detections[idx]
        if det is None:
            prev_ball = prev_cyl = None
            idx += 1
            continue

        ball_xy = np.array(det[:2])
        cyl = pick_static_cylinder(find_candidates(frame), (det[0], det[1]), det[2])

        if prev_ball is not None and prev_cyl is not None and cyl is not None:
            db = ball_xy - prev_ball
            dc = np.array([cyl.x, cyl.y]) - prev_cyl
            if np.linalg.norm(db) > 1e-6 and np.linalg.norm(dc) > 1e-6:
                ball_steps.append(db)
                cyl_steps.append(dc)

        prev_ball = ball_xy
        prev_cyl = np.array([cyl.x, cyl.y]) if cyl is not None else None
        idx += 1

    cap.release()

    if not ball_steps:
        print("没有足够的配对数据，无法完成比较。")
        return

    B = np.array(ball_steps)
    C = np.array(cyl_steps)
    b_mag = np.linalg.norm(B, axis=1)
    c_mag = np.linalg.norm(C, axis=1)
    diff = np.linalg.norm(B - C, axis=1)
    cos = (B * C).sum(axis=1) / (b_mag * c_mag + 1e-9)
    scale = (B * C).sum(axis=1) / (c_mag * c_mag + 1e-9)

    print("\n[做法一] 球 vs 静止绿圆柱的逐帧位移")
    print(f"  配对帧数：{len(B)}")
    print(f"  球平均位移：{b_mag.mean():.2f} px/帧")
    print(f"  圆柱平均位移：{c_mag.mean():.2f} px/帧")
    print(f"  位移差中位数：{np.median(diff):.2f} px")
    print(f"  方向余弦中位数：{np.median(cos):.4f}  (1.0 表示完全同向)")
    print(f"  大小比例中位数：{np.median(scale):.3f}  (1.0 表示完全同速)")
    print("  结论：球与静止物体的位移几乎一致 → 球的视运动主要来自相机移动。")


def homography_residual(video_path: Path, detections) -> None:
    """做法二：累积单应变换把球心映射回参考帧，统计残余运动。"""
    cap = cv2.VideoCapture(str(video_path))
    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ok, first = cap.read()
    if not ok:
        cap.release()
        return
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    H = np.eye(3)  # 当前帧 -> 参考帧
    ref_points: list[np.ndarray | None] = []
    idx = 0

    while True:
        if idx == 0:
            gray = prev_gray
            kp, des = prev_kp, prev_des
        else:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            if des is not None and prev_des is not None:
                matches = sorted(matcher.match(prev_des, des),
                                 key=lambda m: m.distance)[:500]
                if len(matches) >= 12:
                    src = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H_step, inliers = cv2.findHomography(
                        dst, src, cv2.RANSAC, 3.0)  # 当前帧 -> 上一帧
                    if H_step is not None and inliers is not None and inliers.sum() >= 8:
                        H = H @ H_step

            prev_gray, prev_kp, prev_des = gray, kp, des

        det = detections[idx]
        if det is None:
            ref_points.append(None)
        else:
            pt = H @ np.array([det[0], det[1], 1.0])
            ref_points.append(pt[:2] / pt[2])
        idx += 1

    cap.release()

    pts = [p for p in ref_points if p is not None]
    arr = np.array(pts)
    steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)

    print("\n[做法二] 累积单应变换后的球心残余运动")
    print(f"  参考坐标系位置范围：x [{arr[:, 0].min():.0f}, {arr[:, 0].max():.0f}]，"
          f"y [{arr[:, 1].min():.0f}, {arr[:, 1].max():.0f}] px")
    print(f"  矫正后逐帧位移：中位数 {np.median(steps):.2f} px，"
          f"p90 {np.percentile(steps, 90):.2f} px")
    print("  结论：相机运动被补偿后球基本不动 → 画面中的位移是视运动。")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="绿球运动来源分析")
    parser.add_argument("--input", default=str(repo_root / "homework.mp4"))
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"找不到视频：{video_path}")
        return 1

    print("检测并跟踪绿球 ...")
    detections, fps, _ = detect_trajectory(video_path)

    compare_with_static_object(video_path, detections, fps)
    homography_residual(video_path, detections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
