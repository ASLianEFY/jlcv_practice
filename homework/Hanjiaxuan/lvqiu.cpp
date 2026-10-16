#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <deque>
#include <cstdio>

// 【本代码使用了 AI 辅助生成与优化，工具名称：Kimi】—— 群公告要求，提交时切勿删除这行

using namespace cv;
using namespace std;

// ---- HSV 阈值（针对绿色），用滑动条实时可调，应对光照变化 ----
int h_min = 35, h_max = 85;
int s_min = 80, s_max = 255;
int v_min = 80, v_max = 255;

// ---- 追踪相关参数 ----
const double EMA_ALPHA   = 0.30;  // 速度指数平滑系数：越小显示越稳
const int    MAX_LOST    = 30;    // v4：连续丢失多少帧后判定彻底丢失（配合失锁刹车，可以更耐心）
const int    GATE_RADIUS = 90;    // 门限半径(px)：只有落在预测点附近的目标才被采信，防止乱跳
const size_t TRAJ_LEN    = 30;    // 轨迹长度（帧）
const double H_FOV       = 60.0;  // 假设的水平视场角(度)，演示用；实际应由相机标定得到
const double V_FOV       = 40.0;  // 假设的垂直视场角(度)

int main() {
    // 1. 打开视频 (请确保 homework.mp4 与可执行文件在同一目录下)
    VideoCapture cap("homework.mp4");
    if (!cap.isOpened()) {
        cerr << "错误：无法打开视频文件，请检查路径和文件名！" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 1e-6) fps = 30.0;   // 部分视频读不到fps，给个兜底值
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    // 保存处理后的演示视频
    VideoWriter out("output.mp4", VideoWriter::fourcc('m','p','4','v'), fps, Size(w, h));
    if (!out.isOpened()) {
        cerr << "错误：无法创建输出视频！" << endl;
        return -1;
    }

    // 2. 窗口与滑动条
    namedWindow("frame", WINDOW_AUTOSIZE);
    createTrackbar("H_min", "frame", &h_min, 180);
    createTrackbar("H_max", "frame", &h_max, 180);
    createTrackbar("S_min", "frame", &s_min, 255);
    createTrackbar("S_max", "frame", &s_max, 255);
    createTrackbar("V_min", "frame", &v_min, 255);
    createTrackbar("V_max", "frame", &v_max, 255);

    // 3. 卡尔曼滤波器：状态向量 [x, y, vx, vy]^T，观测向量 [x, y]^T
    //    模型假设：匀速运动（速度不变），用预测值在目标短暂丢失时维持追踪
    KalmanFilter KF(4, 2, 0);
    double dt = 1.0 / fps;
    KF.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, (float)dt, 0,
        0, 1, 0, (float)dt,
        0, 0, 1,       0,
        0, 0, 0,       1);
    setIdentity(KF.measurementMatrix);                       // 只观测位置
    setIdentity(KF.processNoiseCov,   Scalar::all(1e-2));    // 过程噪声：对模型的信任程度
    setIdentity(KF.measurementNoiseCov, Scalar::all(1.0));   // 观测噪声：对检测的信任程度
    setIdentity(KF.errorCovPost,      Scalar::all(1.0));

    bool   kalman_inited = false;  // 滤波器是否已初始化
    int    lost = 0;               // 连续丢失帧数
    double vel_smooth = 0.0;       // 平滑后的速度 (px/s)
    deque<Point> traj;             // 目标轨迹
    Point  last_seen(-1, -1);      // v4：上次成功检测的位置，重捕时优先锁定它附近的目标

    Mat frame, hsv, mask;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 4. BGR -> HSV，阈值分割出绿色区域
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), mask);

        // 5. 形态学去噪：开运算去噪点，闭运算补空洞
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN,  kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 6. 轮廓检测 + 筛选，收集所有候选球心
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Point>   cands;      // 候选球心
        vector<double>  circs;      // 候选对应的圆度
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 150 || area > 19950) continue;    // 面积过滤（上限按本视频分辨率调到 30 万，改小会全程检测不到）

            double perimeter = arcLength(contours[i], true);
            if (perimeter < 1e-6) continue;               // 修正：避免圆度计算除零

            // 圆度 = 4*pi*A/P^2，圆为1，圆柱侧面投影(长方形)远小于1 -> 用来排除绿色圆柱
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);

            Rect rect = boundingRect(contours[i]);
            double aspect = (double)rect.width / rect.height;

            // 比参考代码略宽松：运动模糊会让轮廓变形，阈值卡太死容易频繁丢目标
            // v4：圆度/实心度进一步放宽（画面边缘畸变+快速晃动会让轮廓变形），圆柱顶面仍由长宽比把关
            if (circularity > 0.58 && aspect > 0.70 && aspect < 1.35) {
                vector<Point> hull;
                convexHull(contours[i], hull);
                double hullArea = contourArea(hull);
                double solidity = (hullArea > 1e-6) ? (area / hullArea) : 0;
                if (solidity <= 0.78) continue;     // 凹缺/弯月形等异形直接排除

                Moments M = moments(contours[i]);
                // 修正v2：加大括号，保证 cands 和 circs 永远同步入队
                if (M.m00 != 0) {
                    cands.push_back(Point((int)(M.m10 / M.m00), (int)(M.m01 / M.m00)));
                    circs.push_back(circularity);
                }
            }
        }

        // 7. 卡尔曼预测一步（如果滤波器已启动）
        Point predPt(-1, -1);
        if (kalman_inited) {
            Mat pred = KF.predict();
            predPt = Point((int)pred.at<float>(0), (int)pred.at<float>(1));
        }

        // 8. 数据关联：优先采信预测点门限内的候选，防止错误匹配导致速度突变
        Point measure(-1, -1);
        if (!cands.empty()) {
            if (kalman_inited) {
                // v4：门限随连续丢失帧数线性放大——丢得越久允许匹配越远，
                //     防止预测点飘走后球明明被检测到却被门限拒收、永远锁不回来
                double gate = GATE_RADIUS * (1.0 + lost);
                double best_d = 1e9;
                for (size_t i = 0; i < cands.size(); i++) {
                    double d = norm(cands[i] - predPt);
                    if (d < gate && d < best_d) { best_d = d; measure = cands[i]; }
                }
                // 门限内没有候选 -> 宁可认为丢失，也不强行匹配远处的干扰（参考代码的隐患）
            } else {
                // v4：重捕初始化优先选离上次命中点最近的候选，避免误锁静止的绿色圆柱顶面
                if (last_seen.x >= 0) {
                    double best_d = 1e9;
                    for (size_t i = 0; i < cands.size(); i++) {
                        double d = norm(cands[i] - last_seen);
                        if (d < 250 && d < best_d) { best_d = d; measure = cands[i]; }
                    }
                }
                if (measure.x < 0) {
                    // 上次位置附近没有候选（首次启动或目标真的移走了）：选圆度最高的
                    size_t best_i = 0;
                    for (size_t i = 1; i < circs.size(); i++)
                        if (circs[i] > circs[best_i]) best_i = i;
                    measure = cands[best_i];
                }
            }
        }

        // 9. 根据关联结果更新滤波器与轨迹
        if (measure.x >= 0) {
            if (!kalman_inited) {
                // 初始化：位置为观测值，速度先设为 0
                KF.statePost = (Mat_<float>(4, 1) << (float)measure.x, (float)measure.y, 0.f, 0.f);
                setIdentity(KF.errorCovPost, Scalar::all(1.0));  // 修正v2：重捕后协方差也重置，避免沿用旧估计导致首帧抖动
                kalman_inited = true;
                vel_smooth = 0.0;
            } else {
                Mat z = (Mat_<float>(2, 1) << (float)measure.x, (float)measure.y);
                KF.correct(z);
            }
            lost = 0;
            last_seen = measure;    // v4：记录最近命中位置

            // 速度取滤波器内部估计 (px/s)，比“位移*fps”的瞬时速度稳定得多
            double vx = KF.statePost.at<float>(2);
            double vy = KF.statePost.at<float>(3);
            double speed = sqrt(vx * vx + vy * vy);

            // 指数滑动平均(EMA)平滑显示速度，避免单帧抖动刷屏
            vel_smooth = EMA_ALPHA * speed + (1.0 - EMA_ALPHA) * vel_smooth;

            // v3：轨迹画滤波后的位置而非原始检测点，单帧检测抖动不会传导到轨迹上
            Point drawPt((int)KF.statePost.at<float>(0), (int)KF.statePost.at<float>(1));
            traj.push_back(drawPt);
            if (traj.size() > TRAJ_LEN) traj.pop_front();

            // 画出识别结果：质心绿点 + 外圈圆框
            circle(frame, measure, 5, Scalar(0, 255, 0), -1);
            circle(frame, measure, 25, Scalar(0, 255, 0), 2);

            // 雷达/自瞄输出：把像素坐标换算成相对相机中心的偏航/俯仰角
            // 修正：改用卡尔曼滤波后的位置，与速度的口径一致，显示更稳定
            double fx = KF.statePost.at<float>(0);
            double fy = KF.statePost.at<float>(1);
            double yaw   = ((fx - w / 2.0) / (w / 2.0)) * (H_FOV / 2.0);
            double pitch = -((fy - h / 2.0) / (h / 2.0)) * (V_FOV / 2.0); // 图像y轴向下，故取负
            char text[128];

            // 运动方向箭头：方向取平滑速度，长度随速度变化（12~140px）
            // 修正：增加 speed > 1e-6 判断，防止球急停时 vx/speed 除零产生 NaN
            // v3：去掉 vel_smooth 下限，只要球在动就画箭头（低速时箭头短而已）
            if (speed > 1e-6) {
                double ux = vx / speed, uy = vy / speed;
                int arrow_len = (int)max(12.0, min(140.0, vel_smooth * 0.12));  // v4：比例放大，快慢一目了然
                Point end(measure.x + (int)(ux * arrow_len), measure.y + (int)(uy * arrow_len));
                arrowedLine(frame, measure, end, Scalar(0, 0, 255), 2, LINE_AA, 0, 0.3);
            }

            snprintf(text, sizeof(text), "v=%.1f px/s  yaw=%.1f deg  pitch=%.1f deg", vel_smooth, yaw, pitch);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        } else {
            lost++;
            if (kalman_inited && lost <= MAX_LOST) {
                // 短暂丢失：用预测点维持轨迹与显示，等待目标重新出现
                vel_smooth *= 0.9;  // 速度逐渐衰减，避免重新捕获时突变
                // v4：滤波器内部速度也刹车，预测点只在原地附近衰减晃动，不再直线飞出画面
                KF.statePost.at<float>(2) *= 0.85f;
                KF.statePost.at<float>(3) *= 0.85f;
                if (predPt.x >= 0) {
                    // v3：预测点最多只画进轨迹前 5 帧，防止长时间丢失时轨迹无限往外飘
                    if (lost < 5) {
                        traj.push_back(predPt);
                        if (traj.size() > TRAJ_LEN) traj.pop_front();
                    }
                    circle(frame, predPt, 5, Scalar(0, 255, 255), -1); // 黄色=预测点
                }
                putText(frame, "Predicting...", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            } else {
                // 丢失太久：重置整个追踪状态（修正参考代码丢球重捕时速度爆冲的问题）
                kalman_inited = false;
                vel_smooth = 0.0;
                traj.clear();
                putText(frame, "Target Lost", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
        }

        // 10. 画轨迹折线
        for (size_t i = 1; i < traj.size(); i++)
            line(frame, traj[i - 1], traj[i], Scalar(255, 0, 0), 2);

        out.write(frame);
        imshow("frame", frame);
        // imshow("mask", mask);   // 调 HSV 阈值时可打开，观察二值化效果

        // 修正：waitKey 返回值与 0xFF 掩码后再比较，避免个别平台高位带标志位导致 ESC 失灵
        if ((waitKey(max(1, (int)(1000 / fps))) & 0xFF) == 27) break;  // ESC 退出
    }

    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}