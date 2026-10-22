from pathlib import Path
import cv2
import numpy as np

# 视频绝对路径
VIDEO_PATH = str(Path(__file__).resolve().parents[2] / "homework.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("打不开视频，请检查路径！")
    exit()

prev_center = None 
lost_count = 0 # 把计数器放到外面

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 颜色提取（HSV空间）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 70, 40])
    upper_green = np.array([100, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # 建议核不要太大，16有点夸张了，如果卡顿可以改回 9 或 11
    kernel = np.ones((16, 16), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 2. 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_center = None 
    best_ball_area = 0

    # 3. 遍历轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤掉极小的噪点
        if area > 500: 
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # 核心判断
                    if circularity > 0.75 and aspect_ratio < 2.5:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            can_update = True 
                            if prev_center is not None:
                                dx = cX - prev_center[0]
                                dy = cY - prev_center[1]
                                distance = np.sqrt(dx**2 + dy**2)
                                
                                # 物理防瞬移过滤
                                if distance > 100: 
                                    can_update = False

                            if can_update and area > best_ball_area:
                                best_ball_area = area
                                ball_center = (cX, cY)
        
    # 画图逻辑
    if ball_center is not None:
        lost_count = 0 # 找到了就清零
        
        # 【修复核心】利用上一帧位置做“惯性平滑”
        if prev_center is not None:
            display_x = int(prev_center[0] * 0.5 + ball_center[0] * 0.5)
            display_y = int(prev_center[1] * 0.5 + ball_center[1] * 0.5)
            display_center = (display_x, display_y)
        else:
            display_center = ball_center
            
        # 【修复】画圈用 display_center，而不是 ball_center，这样就不闪了
        cv2.circle(frame, display_center, 40, (0, 0, 255), 3)
        
        if prev_center is not None:
            dx = ball_center[0] - prev_center[0]
            dy = ball_center[1] - prev_center[1]
            speed = np.sqrt(dx**2 + dy**2)
            
            if speed > 1: 
                draw_end_x = int(ball_center[0] + dx * 3)
                draw_end_y = int(ball_center[1] + dy * 3)
                
                cv2.arrowedLine(frame, prev_center, (draw_end_x, draw_end_y), (0, 255, 255), 4, tipLength=0.3)
                cv2.putText(frame, f"v={speed:.1f}", (ball_center[0]+15, ball_center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        prev_center = ball_center
    else:
        lost_count += 1
        # 只有连续丢失超过 5 帧，才彻底忘记
        if lost_count < 5 and prev_center is not None:
            cv2.circle(frame, prev_center, 40, (0, 0, 255), 1) # 画个细空心圈占位
        else:
            prev_center = None
        
    cv2.imshow('Green Ball Tracking', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
