import cv2
import numpy as np

def is_ball_robust(contour, hsv_frame, mask):
    """
    综合判断轮廓是否为球体。
    1. 几何特征：长宽比 + 填充率 (排除侧放的圆柱)
    2. 颜色特征：HSV 通道内的颜色方差 (排除有纹理/标签/明显反光的物体)
    """
    # --- 1. 基础几何筛选 (快速排除明显的长条圆柱) ---
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0 or w == 0: return False
    
    # 面积太小直接忽略
    area = cv2.contourArea(contour)
    if area < 200: return False

    aspect_ratio = float(w) / h
    # 允许一定的透视变形，但太极端的不要
    if aspect_ratio < 0.7 or aspect_ratio > 1.4: 
        return False

    # 填充率检查 (圆形的理论填充率是 0.785)
    extent = float(area) / (w * h)
    if extent < 0.6: 
        return False

    # --- 2. 颜色/纹理一致性检查 (核心：区分正对圆柱和球) ---
    # 创建该轮廓的掩膜
    cnt_mask = np.zeros(hsv_frame.shape[:2], np.uint8)
    cv2.drawContours(cnt_mask, [contour], -1, 255, -1)

    # 提取该区域内的 HSV 像素
    mean_val, std_val = cv2.meanStdDev(hsv_frame, mask=cnt_mask)
    
    # H (色调), S (饱和度), V (亮度) 的标准差
    std_h = std_val[0][0]
    std_s = std_val[1][0]
    std_v = std_val[2][0]

    # 逻辑分析：
    # 球体通常是纯色且光照平滑过渡的，std_v (亮度标准差) 会维持在一个中等且稳定的范围。
    # 圆柱体正对时：
    #   - 如果有标签/文字：std_h (色调标准差) 会非常大。
    #   - 如果是光滑塑料：会出现高光带或阴影带，导致 std_v 异常高。
    if std_h > 15: 
        return False
        
    if std_v > 60 and std_s < 30:
        return False

    return True

def track_green_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        return

    prev_center = None
    smoothing_factor = 0.5
    smoothed_velocity = np.array([0, 0], dtype=np.float32)

    print("正在处理... 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 颜色分割
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # 形态学优化
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 2. 查找并筛选轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_center = None
        candidate_list = []
        
        for cnt in contours:
            if is_ball_robust(cnt, hsv_frame, mask):
                # 收集所有通过形状+颜色筛选的候选
                area = cv2.contourArea(cnt)
                candidate_list.append({"contour":cnt, "area":area})

        best_contour = None
        if len(candidate_list) > 0:
            # =========核心改动：候选里面选面积最小的，就是小球=========
            min_item = min(candidate_list, key=lambda x:x["area"])
            best_contour = min_item["contour"]

        if best_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            current_center = (int(x), int(y))
            radius = int(radius)
            
            # 绘制确认后的球
            cv2.circle(frame, current_center, radius, (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (current_center[0]-20, current_center[1]-radius-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3. 计算速度并画箭头
        if prev_center and current_center:
            velocity_vector = np.array(current_center) - np.array(prev_center)
            smoothed_velocity = smoothing_factor * velocity_vector + (1 - smoothing_factor) * smoothed_velocity
            
            start_point = current_center
            arrow_length_scale = 5 
            end_point = (int(start_point[0] + smoothed_velocity[0] * arrow_length_scale),
                         int(start_point[1] + smoothed_velocity[1] * arrow_length_scale))
            
            cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)

        prev_center = current_center
        cv2.imshow('Green Ball Tracking', frame)
        cv2.imshow("mask", mask) # 增加mask窗口，方便调试
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_green_ball("homework.mp4")
