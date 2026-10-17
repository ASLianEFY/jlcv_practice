#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// 把阈值设为全局变量，方便滑动条修改
int hmin = 35, hmax = 85;
int smin = 80, smax = 255;
int vmin = 80, vmax = 255;

int main() {
    // 1. 打开视频
    VideoCapture cap("homework.mp4"); 
    if (!cap.isOpened()) {
        cout << "视频打不开，请检查名字是不是 homework.mp4" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    // 准备保存输出的视频
    VideoWriter out("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    Mat frame, hsv, mask; 
    
    int last_x = -1, last_y = -1; // 上一帧坐标
    
    // 创建一个窗口，并添加滑动条用于实时调参
    namedWindow("Tracking", WINDOW_AUTOSIZE);
    createTrackbar("H Min", "Tracking", &hmin, 180);
    createTrackbar("H Max", "Tracking", &hmax, 180);
    createTrackbar("S Min", "Tracking", &smin, 255);
    createTrackbar("V Min", "Tracking", &vmin, 255);

    // 2. 循环读取视频
    while (true) {
        cap >> frame; 
        if (frame.empty()) break; 

        // 3. 转 HSV 并利用滑动条的值进行二值化
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);

        // 4. 形态学去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 5. 找轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int best_x = -1, best_y = -1; 
        int best_radius = 0;
        double max_area = 0; 

        // 6. 遍历筛选绿球
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 100) continue; 

            double perimeter = arcLength(contours[i], true);
            double circularity = 4 * 3.14159 * area / (perimeter * perimeter);
            if (circularity < 0.65) continue; 

            Rect rect = boundingRect(contours[i]);
            double ratio = (double)rect.width / rect.height;
            if (ratio < 0.7 || ratio > 1.3) continue;

            if (area > max_area) {
                max_area = area;
                Moments m = moments(contours[i]);
                best_x = m.m10 / m.m00;
                best_y = m.m01 / m.m00;
                best_radius = (rect.width + rect.height) / 4; // 算出球的大致半径
            }
        }

        // 7. 绘制与速度计算
        if (best_x != -1 && best_y != -1) {
            Point center(best_x, best_y);

            // 画贴合球大小的黄色外框和红色中心点
            circle(frame, center, best_radius + 5, Scalar(0, 255, 255), 2);
            circle(frame, center, 3, Scalar(0, 0, 255), -1);

            if (last_x != -1 && last_y != -1) {
                int dx = best_x - last_x;
                int dy = best_y - last_y;
                double dist = sqrt(dx * dx + dy * dy);
                double speed = dist * fps; 

                // 优化：如果球动了，才画箭头
                if (dist > 1.0) {
                    double ux = dx / dist; 
                    double uy = dy / dist; 
                    
                    // 动态箭头长度：速度越快，箭头越长 (限制在 30~120 之间)
                    int arrow_length = (int)(speed * 0.2);
                    if (arrow_length < 30) arrow_length = 30;
                    if (arrow_length > 120) arrow_length = 120;

                    Point end_point(best_x + ux * arrow_length, best_y + uy * arrow_length); 
                    arrowedLine(frame, center, end_point, Scalar(255, 0, 0), 3); // 加粗箭头更好看

                    // 画文字底板，防看不清
                    rectangle(frame, Point(10, 10), Point(250, 50), Scalar(0, 0, 0), -1);
                    
                    char text[50];
                    sprintf(text, "speed: %.1f px/s", speed);
                    putText(frame, text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
                }
            }
            last_x = best_x;
            last_y = best_y;
        }

        // 8. 显示
        out.write(frame);
        imshow("Tracking", frame);
        
        // 按 ESC 退出
        if (waitKey(1000 / (int)fps) == 27) break;
    }

    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}