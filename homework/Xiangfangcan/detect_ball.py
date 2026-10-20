import cv2
import numpy as np


def main():
    # 读取视频文件
    video_path = "../../../homework.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    prev_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 绿色的 HSV 范围 (如果追踪不到圆球，可以微调这里的数值)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_center = None

        # === 核心修改部分 开始 ===
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 1. 面积太小直接跳过 (过滤噪点)
            if area < 100:
                continue

            # 2. 形状过滤：计算圆度
            # 获取最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius**2)

            # 计算圆度 (轮廓面积 / 外接圆面积)
            # 完美圆形的圆度接近 1.0，圆柱体侧面会拉低这个值
            circularity = area / circle_area if circle_area > 0 else 0

            # 只处理圆度大于 0.7 的轮廓 (说明它是个比较圆的球)
            if circularity > 0.7:
                # 如果找到了圆球，计算它的质心
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_center = (cx, cy)
                    # 找到目标后直接 break 退出循环，防止被后面更大的圆柱体覆盖
                    break
        # === 核心修改部分 结束 ===

        # 绘制结果
        if current_center is not None:
            cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

            if prev_center is not None:
                dx = current_center[0] - prev_center[0]
                dy = current_center[1] - prev_center[1]

                speed = np.sqrt(dx**2 + dy**2)

                # 移动速度阈值，太小就不画箭头了
                if speed > 2.0:
                    scale_factor = 3.0  # 箭头放大倍数
                    end_point = (
                        int(current_center[0] + dx * scale_factor),
                        int(current_center[1] + dy * scale_factor),
                    )

                    # 画箭头
                    cv2.arrowedLine(
                        frame, prev_center, end_point, (0, 0, 255), 2, tipLength=0.3
                    )

                    # 显示速度
                    cv2.putText(
                        frame,
                        f"Speed: {speed:.1f}",
                        (current_center[0] + 10, current_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            prev_center = current_center

        cv2.imshow("Green Ball Tracking", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
