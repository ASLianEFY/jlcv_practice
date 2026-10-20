import cv2
import numpy as np
import math
from pathlib import Path

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

# ================== 1. 参数设置 ==================
video_path = str(Path(__file__).resolve().parents[2] / "homework.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 错误：无法打开视频文件！")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
greenLower = (35, 120, 80) 
greenUpper = (85, 255, 255)

prev_center = None
prev_area = None  
lost_count = 0  # 新增：计数器，记录跟丢了多少帧
speed_scale = 2.0  

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = None

# ================== 2. 主循环 ==================
while True:
    print("正在处理第", cap.get(cv2.CAP_PROP_POS_FRAMES), "帧...") # 加上这行
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter('output_new.avi', fourcc, fps, (w, h))

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    current_radius = 0
    current_area = 0

    valid_candidates = []

    if len(cnts) > 0:
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100: 
                continue
                
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            
            # 【改动1】放宽形状限制：允许它变成半圆、长条（被挡住一半）
            if 0.7 < aspect_ratio < 1.6: 
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                
                # 【改动2】放宽圆度限制
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                if circularity > 0.3: 
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        valid_candidates.append((cx, cy, w, h, area))

    # ================== 核心：防瞬移 + 容忍残缺 ==================
    if len(valid_candidates) > 0:
        if prev_center is not None and prev_area is not None:
            valid_candidates.sort(key=lambda item: (item[0] - prev_center[0])**2 + (item[1] - prev_center[1])**2)
            
            for cand in valid_candidates:
                cx, cy, w, h, area = cand
                dist_to_prev = math.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)
                
                # 【改动3】偏心的面积检测：
                # 允许面积变小（因为被挡住了，area < prev_area 是正常的）
                # 严格限制面积变大（如果突然变大 1.5 倍，说明那是圆柱体）
                # 顺便也限制一下不能缩得太离谱（小于原来的 10% 就是噪点）
                is_area_valid = (prev_area * 0.1 < area < prev_area * 1.5)
                
                # 允许距离稍微放宽一点，因为球残缺时质心会偏移
                if dist_to_prev < 50 and is_area_valid:
                    center = (cx, cy)
                    current_radius = int((w + h) / 4)
                    current_area = area
                    break 
            
        else:
            # 重新找回目标时，找最像球的（面积最大）
            best_match = max(valid_candidates, key=lambda item: item[4])
            center = (best_match[0], best_match[1])
            current_radius = int((best_match[2] + best_match[3]) / 4)
            current_area = best_match[4]

    # 画圈
    if center is not None:
        cv2.circle(frame, center, current_radius, (0, 255, 0), 2)

    # 画箭头
    if center is not None and prev_center is not None:
        dX = center[0] - prev_center[0]
        dY = center[1] - prev_center[1]
        dist = np.sqrt(dX**2 + dY**2)

        if dist > 2: 
            end_pt = (int(center[0] + dX * speed_scale),
                      int(center[1] + dY * speed_scale))
            
            cv2.arrowedLine(frame, center, end_pt, (0, 0, 255), 3, tipLength=0.4)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    # ================== 【改动4】记忆保鲜机制 ==================
    if center is not None:
        prev_center = center
        prev_area = current_area 
        lost_count = 0 # 找到了，计数器清零
    else:
        lost_count += 1
        # 即使这一帧没找到，也不要立刻清空记忆。允许它“瞎” 5 帧。
        # 只要 5 帧之内球再次出现，就能连上；超过 5 帧还没找到，才认定彻底丢了。
        if lost_count > 5:
            prev_center = None 
            prev_area = None

    # 显示与保存
    cv2.imshow('Green Ball Tracking', frame)
    out.write(frame)
    cv2.imshow('Mask Debug', mask) 
    
    delay = int(1000 / fps) if fps > 0 else 30  
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

