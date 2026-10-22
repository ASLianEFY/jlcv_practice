#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main(){ 
    // 读取视频。如果视频没拷过来，直接改成 cap(0) 就能用你电脑的摄像头拍绿东西测试
    VideoCapture cap("homework.mp4"); 
    if (!cap.isOpened()) {
        cout << "无法打开视频 homework.mp4" << endl;
        return -1;
    }

    Mat frame, hsv, mask;
    Point2f prev_center(-1, -1); 
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; 

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. 颜色空间转换 BGR -> HSV
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Scalar lower_green(35, 43, 46);
        Scalar upper_green(85, 255, 255);
        inRange(hsv, lower_green, upper_green, mask);

        // 2. 形态学操作（去噪点）
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 3. 找轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2f current_center(-1, -1);
        double max_area = 0;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 1000) continue; // 忽略太小的噪点

            // 4. 筛选圆度和长宽比（区分小球和圆柱）
            double perimeter = arcLength(contours[i], true);
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);
            Rect rect = boundingRect(contours[i]);
            double aspect_ratio = (double)rect.width / rect.height;

// 判断目标是否碰到了画面的边缘（上下左右留出10个像素的容差）
bool is_edge = (rect.x <= 10) || (rect.y <= 10) || 
               (rect.x + rect.width >= frame.cols - 10) || 
               (rect.y + rect.height >= frame.rows - 10);

// 如果碰到了边缘，我们就放宽条件（圆度>0.4，长宽比>0.4即可）
// 如果没碰边缘，就依然用你刚才设的严格条件（0.7, 0.8, 1.2）

// 统一判断是不是目标
bool is_target = false;
if (is_edge) {
    if (circularity > 0.4 && aspect_ratio > 0.4 && aspect_ratio < 2.5) {
        is_target = true;
    }
} else {
    if (circularity > 0.7 && aspect_ratio > 0.8 && aspect_ratio < 1.2) {
        is_target = true;
    }
}
// 如果是目标，进行位置距离防骗
if (is_target) {
    Moments m = moments(contours[i]);
    if (m.m00 != 0) {
        Point2f candidate_center(m.m10 / m.m00, m.m01 / m.m00);
        
        // 【防骗机制】：如果上一帧看到了球，且当前目标跳跃太大（超过150像素），直接拉黑！
        if (prev_center.x != -1) {
            double dist = sqrt(pow(candidate_center.x - prev_center.x, 2) + pow(candidate_center.y - prev_center.y, 2));
            if (dist > 150) {
                continue; 
            }
        }
        // 选面积最大的
        if (area > max_area) {
            max_area = area;
            current_center = candidate_center;
        }
     }
  }
} 

        // 5. 画圈、算速度、画箭头
        if (current_center.x != -1) {
            float radius = sqrt(max_area / CV_PI);
            circle(frame, current_center, radius, Scalar(0, 255, 255), 3); // 画黄圈
            if (prev_center.x != -1) {
                double dx = current_center.x - prev_center.x;
                double dy = current_center.y - prev_center.y;
                double distance = sqrt(dx * dx + dy * dy);
                double speed = distance * fps;

                //引入放大系数3.0,计算出新起点
                Point2f arrow_end(current_center.x + dx * 3.0, current_center.y + dy * 3.0);
                //从球心画出去，这样箭头不会把球挡住
                arrowedLine(frame, current_center, arrow_end, Scalar(255, 0, 0), 3); // 画蓝箭头
                string text = "speed: " + to_string((int)speed) + " px/s";
                putText(frame, text, current_center + Point2f(20, -20), 
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
            }
            prev_center = current_center;
        } else {
            prev_center = Point2f(-1, -1);
        }

        // 窗口名字改成了 liguanguan
        imshow("liguanguan Homework", frame);
        if (waitKey(30) == 27) break; // 按 ESC 退出
    }
    return 0;
}


