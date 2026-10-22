from pathlib import Path
import cv2
import numpy as np
import math

# ================= 配置参数 =================
VIDEO_PATH = str(Path(__file__).resolve().parents[2] / "homework.mp4")        # 你的视频文件名
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result.mp4")     # 输出视频文件名

# 绿色的 HSV 颜色范围（如果你的视频里球颜色偏黄或偏暗，可能需要微调这两个值）
lower_green = np.array([35, 43, 46])
upper_green = np.array([85, 255, 255])

# 速度放大系数：因为视频里每帧移动的像素很小，乘以一个系数让箭头变长
SPEED_SCALE = 8.0 
# ============================================

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：找不到视频文件 {VIDEO_PATH}，请检查名字是否一致！")
    exit()

# 获取视频的帧率和尺寸，用于保存视频
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 记录上一帧小球的位置
prev_x, prev_y = None, None

print("开始处理视频，按 'q' 键可以退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break # 视频结束

    # 转换为 HSV 颜色空间，更容易识别颜色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建绿色掩膜（只保留绿色部分）
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 进行形态学操作，去除噪点（比如阴影造成的干扰）
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_found = False
    curr_x, curr_y = None, None

    if contours:
        # 找到面积最大的轮廓（假设球是画面里最大的绿色物体之一）
        # 但如果圆柱体比球大，我们需要通过“圆度”来过滤
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: # 忽略太小的噪点
                continue

            # 计算圆度 (Circularity = 4 * pi * Area / Perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            # 圆的圆度接近 1，圆柱体的圆度会低很多
            # 假设圆度 > 0.7 才是我们要找的球
            if circularity > 0.7:
                # 找到球了！获取外接圆
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                curr_x, curr_y = int(x), int(y)
                ball_found = True
                
                # 画一个圈把小球的轮廓标出来（可选，为了好看）
                cv2.circle(frame, (curr_x, curr_y), int(radius), (0, 255, 255), 2)
                break # 找到球就跳出循环

    # 如果这一帧找到了球，并且上一帧也有位置，就计算速度和方向画箭头
    if ball_found and prev_x is not None and prev_y is not None:
        # 计算位移 (dx, dy)
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        # 计算速度（移动距离）
        speed = math.sqrt(dx**2 + dy**2)

        # 如果移动太小（比如球停住了），就不画箭头
        if speed > 1.0: 
            # 计算箭头的终点
            arrow_end_x = int(curr_x + dx * SPEED_SCALE)
            arrow_end_y = int(curr_y + dy * SPEED_SCALE)

            # 画箭头 (参数：图像, 起点, 终点, 颜色(BGR), 线宽, 箭头尖端长度, 尖端角度)
            # 颜色设为红色 (0, 0, 255) 以便和绿球对比明显
            cv2.arrowedLine(frame, (curr_x, curr_y), (arrow_end_x, arrow_end_y), (0, 0, 255), 3, tipLength=0.3)
            
            # 在箭头旁边写上速度数值
            cv2.putText(frame, f"Speed: {speed:.1f}", (curr_x + 10, curr_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 更新上一帧的位置
    if ball_found:
        prev_x, prev_y = curr_x, curr_y
    else:
        # 如果这一帧没找到球，重置位置，避免下次找到球时画出跨越屏幕的离谱箭头
        prev_x, prev_y = None, None

    # 写入输出视频
    out.write(frame)

    # 实时显示画面
    cv2.imshow('Green Ball Tracking', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"处理完成！请查看生成的 {OUTPUT_PATH} 文件。")