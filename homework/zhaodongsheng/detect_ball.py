"""
绿球识别与运动追踪
====================

任务：
    识别视频中的绿色小球，并用箭头标注其运动方向和速度变化。

标注规则（来自作业要求）：
    - 箭头方向  = 小球的运动方向
    - 箭头长度  = 速度大小（越长越快，越短越慢）

实现思路（五步）：
    1. 逐帧读取视频
    2. 颜色分割：把图像转到 HSV 空间，用绿色阈值区间提取绿色区域
    3. 去噪：形态学开/闭运算，去掉细小的噪声点
    4. 找球：找轮廓 -> 面积过滤(去噪点) -> 圆度过滤(排除圆柱体侧面等非圆形干扰)
    5. 追踪：比较相邻两帧的球心位移，画箭头(方向=位移方向，长度正比于速度)

依赖：
    pip install opencv-python numpy

运行：
    python detect_ball.py                         # 默认读 homework.mp4，窗口实时显示
    python detect_ball.py -i 视频.mp4 -o out.mp4  # 指定输入，并保存处理结果
"""

import argparse
import os
import cv2
import numpy as np

# 脚本所在目录（用于自动定位视频，避免相对路径依赖"当前工作目录"）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def default_video():
    """自动寻找仓库根目录下的 homework.mp4。"""
    candidates = [
        os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "homework.mp4")),
        os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "homework.mp4")),
        os.path.normpath(os.path.join(SCRIPT_DIR, "homework.mp4")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description="识别绿球并用箭头标注运动方向与速度")
    p.add_argument("-i", "--input", default=None,
                   help="输入视频路径（默认自动定位 homework.mp4）")
    p.add_argument("-o", "--output", default=None,
                   help="可选：保存处理后的视频（如 out.mp4）")
    return p.parse_args()


def detect_ball(frame):
    """在一帧图像里找到绿球，返回球心 (cx, cy)；找不到返回 None。"""
    # 1. 颜色分割：BGR -> HSV，绿色区间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])    # 绿色下限（H 40~85 对应绿色）
    upper_green = np.array([85, 255, 255])  # 绿色上限
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. 形态学去噪：开运算去小噪点，闭运算填小空洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:                     # 面积太小 -> 噪声，跳过
            continue

        # 圆度过滤：轮廓面积 / 最小外接圆面积。
        # 完美圆形 -> 接近 1；圆柱体侧面、长条 -> 明显小于 1。
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius ** 2
        circularity = area / circle_area if circle_area > 0 else 0
        if circularity < 0.7:              # 不够圆 -> 不是球，跳过
            continue

        # 在通过圆度门槛的候选中，选圆度最高的那个。
        # 球永远比圆柱体更圆，这样即使圆柱体圆度短暂越过 0.7 也不会误选。
        if circularity > best_circularity:
            best_circularity = circularity
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best = (cx, cy)

    return best


def main():
    args = parse_args()
    video_path = args.input or default_video()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 输出视频写入器（可选）
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    prev_center = None
    scale = 3.0   # 箭头放大倍数：把像素级位移放大，让箭头更直观

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center = detect_ball(frame)

        if center is not None:
            # 圈出球
            cv2.circle(frame, center, 8, (0, 0, 255), 2)

            # 与上一帧比较，算运动方向和速度
            if prev_center is not None:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                speed = float(np.hypot(dx, dy))   # 位移大小即本帧速度

                if speed > 2.0:                    # 太慢就不画箭头（避免抖动）
                    # 箭头：从当前球心出发，方向=(dx,dy)，长度正比于速度
                    tip = (int(center[0] + dx * scale),
                           int(center[1] + dy * scale))
                    cv2.arrowedLine(frame, center, tip,
                                    (0, 0, 255), 3, tipLength=0.25)

                    # 显示速度数值
                    cv2.putText(frame, f"Speed: {speed:.1f}",
                                (center[0] + 12, center[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

            prev_center = center

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Green Ball Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
