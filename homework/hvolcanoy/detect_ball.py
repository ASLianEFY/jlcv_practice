import cv2
import numpy as np

cap = cv2.VideoCapture("homework.mp4")

# 绿色HSV阈值，可以后续微调
lower_green = np.array([35, 40, 40])
upper_green = np.array([77, 255, 255])

# 上一帧小球坐标
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)

            if prev_x is not None:
                # 画速度箭头（红色）
                cv2.line(frame, (prev_x, prev_y), (cx, cy), (0,0,255), 2)
            prev_x, prev_y = cx, cy

    cv2.imshow("track", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()