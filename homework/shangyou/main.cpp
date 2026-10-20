#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace cv;
using namespace std;

int main() {
    unsetenv("GTK_PATH");
    unsetenv("GIO_MODULE_DIR");

    // ==================== 参数配置区 ====================
    const string VIDEO_PATH  = "homework.mp4";
    const string OUTPUT_PATH = "output.mp4";

    // 绿色阈值
    const Scalar LOWER_GREEN(35, 70, 50);
    const Scalar UPPER_GREEN(90, 255, 255);

    // 过滤参数 (调参核心区)
    const double MIN_AREA = 200.0;      // 最小面积
    const double MAX_AREA = 30000.0;    // 最大面积（如果圆柱特别大，把这个值调小）
    const double CIRCULARITY_THRESH = 0.70; // 圆形度 (越接近1越圆)
    const double MAX_ASPECT_RATIO = 1.4;    // 长宽比 (球体在 1.0~1.4 之间)
    const double FILL_RATIO_MIN = 0.65;     // 外接矩形填充度 (球在 0.78 左右)

    // 平滑系数
    const double POS_ALPHA = 0.6;
    const double VEL_ALPHA = 0.4;

    // 红色箭头参数
    const double ARROW_SCALE     = 0.20;
    const double MAX_ARROW_LEN   = 250.0;
    const double MIN_ARROW_LEN   = 8.0;
    const Scalar ARROW_COLOR(0, 0, 255);
    const int    ARROW_THICKNESS = 3;
    // ====================================================

    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "[错误] 无法打开视频: " << VIDEO_PATH << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 1e-3 || fps > 240) fps = 30.0;
    int W = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter writer(OUTPUT_PATH, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(W, H));
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    Point2f smoothPos(0, 0);
    Point2f smoothVel(0, 0);
    bool   hasPos    = false;
    double prevT     = -1.0;
    int    frameIdx  = 0;
    int    lostCnt   = 0;

    Mat frame, hsv, mask;
    vector<vector<Point>> contours;

    int wait_time = (int)(1000.0 / fps);
    if (wait_time <= 0) wait_time = 1;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        frameIdx++;

        double t = cap.get(CAP_PROP_POS_MSEC) / 1000.0;
        if (t <= 0) t = frameIdx / fps;
        double dt = (prevT >= 0) ? (t - prevT) : (1.0 / fps);
        if (dt <= 1e-6) dt = 1.0 / fps;
        prevT = t;

        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, LOWER_GREEN, UPPER_GREEN, mask);
        morphologyEx(mask, mask, MORPH_OPEN,  kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        contours.clear();
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool found = false;
        Point2f center(0, 0);
        float radius = 0.f;

        if (!contours.empty()) {
            int bestIdx = -1;
            double bestScore = -1; // 评分最高者胜出

            for (size_t i = 0; i < contours.size(); i++) {
                double a = contourArea(contours[i]);
                if (a < MIN_AREA || a > MAX_AREA) continue; // 面积过滤

                double p = arcLength(contours[i], true);
                if (p == 0) continue;

                // 1. 圆形度
                double circularity = 4.0 * CV_PI * a / (p * p);
                if (circularity < CIRCULARITY_THRESH) continue;

                // 2. 长宽比
                Rect rect = boundingRect(contours[i]);
                double w = rect.width, h = rect.height;
                double aspect_ratio = (w > h) ? (w / h) : (h / w);
                if (aspect_ratio > MAX_ASPECT_RATIO) continue;

                // 3. 填充度 (完美圆是 π/4 ≈ 0.785)
                double fill_ratio = a / (w * h);
                if (fill_ratio < FILL_RATIO_MIN) continue;

                // 【核心拦截】如果已经锁定目标（hasPos为true），则距离锁定点太远的轮廓直接无视
                Moments M = moments(contours[i]);
                if (M.m00 > 0) {
                    Point2f currentCenter((float)(M.m10 / M.m00), (float)(M.m01 / M.m00));
                    if (hasPos) {
                        double dist = norm(currentCenter - smoothPos);
                        if (dist > 150.0) continue; // 150像素外的干扰物（如圆柱）直接丢弃
                    }

                    // 打分：圆形度越接近1、面积越大，分数越高
                    double score = circularity * a;
                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx = (int)i;
                        center = currentCenter;
                        Point2f tmpC; float tmpR;
                        minEnclosingCircle(contours[bestIdx], tmpC, tmpR);
                        radius = tmpR;
                    }
                }
            }

            if (bestIdx >= 0) found = true;
        }

        // ================== 跟踪与计算速度 ==================
        if (found) {
            lostCnt = 0;
            if (!hasPos) {
                smoothPos = center;
                smoothVel = Point2f(0, 0);
                hasPos = true;
            } else {
                Point2f newPos = POS_ALPHA * center + (1 - POS_ALPHA) * smoothPos;
                Point2f instVel((newPos.x - smoothPos.x) / (float)dt, (newPos.y - smoothPos.y) / (float)dt);
                smoothVel = VEL_ALPHA * instVel + (1 - VEL_ALPHA) * smoothVel;
                smoothPos = newPos;
            }

            Point p((int)round(smoothPos.x), (int)round(smoothPos.y));

            // 画出绿球中心与轮廓
            circle(frame, p, max((int)radius, 5), Scalar(0, 255, 0), 2);
            circle(frame, p, 3, Scalar(0, 255, 255), -1);

            // 画红箭头
            float speed = sqrt(smoothVel.x * smoothVel.x + smoothVel.y * smoothVel.y);
            if (speed > 1e-6f) {
                double arrowLen = min((double)speed * ARROW_SCALE, MAX_ARROW_LEN);
                if (arrowLen >= MIN_ARROW_LEN) {
                    Point2f dir(smoothVel.x / speed, smoothVel.y / speed);
                    Point2f tipF = smoothPos + dir * (float)arrowLen;
                    Point tip((int)round(tipF.x), (int)round(tipF.y));

                    arrowedLine(frame, p, tip, ARROW_COLOR, ARROW_THICKNESS, LINE_AA, 0, 0.25);

                    char buf[64];
                    snprintf(buf, sizeof(buf), "%.0f px/s", speed);
                    putText(frame, buf, Point(tip.x + 5, tip.y - 5), FONT_HERSHEY_SIMPLEX, 0.55, ARROW_COLOR, 2, LINE_AA);
                }
            }

            char info[64];
            snprintf(info, sizeof(info), "Speed: %.1f px/s", speed);
            putText(frame, info, Point(15, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);

        } else {
            lostCnt++;
            if (lostCnt > 5) {
                hasPos = false;
                smoothVel = Point2f(0, 0);
            }
        }

        writer.write(frame);
        imshow("Green Ball Tracking", frame);

        int key = waitKey(wait_time) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();
    cout << "[完成] 已保存到: " << OUTPUT_PATH << endl;
    return 0;
}
