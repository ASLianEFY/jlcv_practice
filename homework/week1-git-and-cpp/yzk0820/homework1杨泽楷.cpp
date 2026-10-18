// green_ball_tracker.cpp
// 功能：识别视频中的绿球，并用箭头标注其运动方向和速度
//       箭头方向 = 绿球运动方向，箭头长度 ∝ 速度大小
//       形状筛选：圆度C、长宽比r、面积A
//       跨帧追踪：速度预测 + 距离门控 + 半径门控 + 综合评分
// 依赖：OpenCV 3.x / 4.x
// 编译：g++ green_ball_tracker.cpp -o tracker `pkg-config --cflags --libs opencv4`
// 运行：./tracker homework.mp4      # 或 ./tracker 0 打开摄像头

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

// ------------------------------------------------------------------
// 追踪状态（单目标追踪）
// ------------------------------------------------------------------
struct TrackState {
    bool        active     = false;
    cv::Point2f lastPos    {0.0f, 0.0f};   // 上一帧球心
    cv::Point2f vel        {0.0f, 0.0f};   // 平滑后的速度（像素/帧）
    float       lastRadius = 0.0f;         // 上一帧球半径
    int         lostFrames = 0;            // 连续丢失帧数
};

// 单个几何合格候选
struct Candidate {
    std::vector<cv::Point> contour;
    cv::Point2f center     {0.0f, 0.0f};
    float       radius     = 0.0f;
    double      area       = 0.0;
    double      circularity= 0.0;
    double      aspect     = 1.0;
    double      shapeScore = 0.0;
    double      score      = 0.0;   // 综合评分
};

// ------------------------------------------------------------------
// 对最近若干帧的球心做最小二乘线性拟合（备用 / 调试用）
// ------------------------------------------------------------------
static cv::Point2f estimateVelocity(const std::deque<cv::Point2f>& pts)
{
    const int n = static_cast<int>(pts.size());
    if (n < 2) return cv::Point2f(0.0f, 0.0f);

    double sx = 0, sy = 0, st = 0, stt = 0, sxt = 0, syt = 0;
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        sx  += pts[i].x;  sy  += pts[i].y;
        st  += t;         stt += t * t;
        sxt += t * pts[i].x;  syt += t * pts[i].y;
    }
    const double denom = n * stt - st * st;
    if (std::abs(denom) < 1e-9) return cv::Point2f(0.0f, 0.0f);

    const double vx = (n * sxt - st * sx) / denom;
    const double vy = (n * syt - st * sy) / denom;
    return cv::Point2f(static_cast<float>(vx), static_cast<float>(vy));
}

int main(int argc, char** argv)
{
    // ---------------- 1. 打开视频源 ----------------
    const std::string source = (argc > 1) ? argv[1] : "homework.mp4";
    cv::VideoCapture cap;
    if (source == "0" || source == "camera")
        cap.open(0);
    else
        cap.open(source);

    if (!cap.isOpened()) {
        std::cerr << "无法打开视频源: " << source << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 30.0;

    // ---------------- 2. 可调参数 ----------------
    // 绿色 HSV 阈值（OpenCV 中 H 范围 0~180）
    const cv::Scalar HSV_LO(35,  80,  60);
    const cv::Scalar HSV_HI(85, 255, 255);

    // 箭头 / 速度显示
    const int    HIST_LEN      = 8;
    const double ARROW_SCALE   = 3.0;
    const double MAX_ARROW_LEN = 250.0;
    const double MIN_SPEED     = 0.25;

    // 几何筛选（区分绿色圆珠 / 绿色小球 / 其它绿色物体）
    const double MIN_AREA        = 120.0;
    const double MAX_AREA_RATIO  = 0.25;
    const double MIN_CIRCULARITY = 0.70;
    const double MIN_ASPECT      = 0.75;
    const double MAX_ASPECT      = 1.33;

    // 跨帧追踪参数
    const float  GATE_RADIUS_K  = 3.0f;   // 距离门控：允许最大距离 = K * 上一帧半径
    const float  MIN_GATE       = 30.0f;  // 距离门控的最小值（像素），防止半径小时过严
    const float  RADIUS_TOL     = 0.50f;  // 半径门控：|r_now - r_prev| / r_prev 上限
    const int    MAX_LOST       = 10;     // 连续丢失多少帧后放弃追踪
    const double SHAPE_WEIGHT   = 0.40;   // 综合评分中形状分权重
    const double TRACK_WEIGHT   = 0.60;   // 综合评分中追踪分权重
    const float  VEL_SMOOTH     = 0.60f;  // 速度平滑系数（0=不平滑，1=完全用瞬时速度）

    const cv::Scalar COLOR_BALL      (0, 255, 0);    // 选中绿球：绿
    const cv::Scalar COLOR_ARROW     (0, 0, 255);    // 速度箭头：红
    const cv::Scalar COLOR_TEXT      (0, 255, 255);  // 文本：黄
    const cv::Scalar COLOR_SHAPE_REJ (128, 128, 128); // 形状不合格：灰
    const cv::Scalar COLOR_GATE_REJ  (0, 165, 255);  // 门控剔除：橙
    const cv::Scalar COLOR_PREDICT   (255, 0, 255);  // 预测点：品红

    std::deque<cv::Point2f> history;
    TrackState track;
    cv::Mat frame, hsv, mask;
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // ---------------- 3. 逐帧处理 ----------------
    while (cap.read(frame)) {
        if (frame.empty()) break;

        const double frameArea = static_cast<double>(frame.rows) * frame.cols;
        const double MAX_AREA  = frameArea * MAX_AREA_RATIO;

        // ---- 3.1 颜色分割 ----
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, HSV_LO, HSV_HI, mask);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // ---- 3.2 形状筛选：面积 / 圆度 / 长宽比 ----
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Candidate> cands;
        for (const auto& c : contours) {
            const double area = cv::contourArea(c);
            if (area < MIN_AREA || area > MAX_AREA) {
                std::vector<std::vector<cv::Point>> tmp{ c };
                cv::drawContours(frame, tmp, -1, COLOR_SHAPE_REJ, 1);
                continue;
            }
            const double peri = cv::arcLength(c, true);
            if (peri < 1e-6) continue;

            const double circularity = 4.0 * CV_PI * area / (peri * peri);
            const cv::Rect br = cv::boundingRect(c);
            if (br.height <= 0) continue;
            const double aspect = static_cast<double>(br.width) / br.height;

            const bool isRound  = circularity >= MIN_CIRCULARITY;
            const bool isSquare = aspect >= MIN_ASPECT && aspect <= MAX_ASPECT;
            if (!(isRound && isSquare)) {
                std::vector<std::vector<cv::Point>> tmp{ c };
                cv::drawContours(frame, tmp, -1, COLOR_SHAPE_REJ, 1);
                continue;
            }

            Candidate cd;
            cd.contour     = c;
            cd.area        = area;
            cd.circularity = circularity;
            cd.aspect      = aspect;
            cv::minEnclosingCircle(c, cd.center, cd.radius);

            // 形状评分：圆度越高越好，长宽比越接近 1 越好
            cd.shapeScore = 0.5 * circularity
                          + 0.5 * (1.0 - std::min(1.0, std::abs(aspect - 1.0)));
            cands.push_back(cd);
        }

        // ---- 3.3 跨帧追踪：位置预测 + 距离/半径门控 + 综合评分 ----
        bool        found       = false;
        Candidate   best;
        cv::Point2f predicted(0.0f, 0.0f);
        float       gateDist   = 0.0f;

        if (!cands.empty()) {
            if (track.active) {
                // 基于上一帧速度的位置预测
                predicted = track.lastPos + track.vel;
                // 距离门控阈值：随球半径自适应，但有下限
                gateDist  = std::max(GATE_RADIUS_K * track.lastRadius, MIN_GATE);

                // 画出预测点便于观察
                cv::circle(frame,
                           cv::Point(cvRound(predicted.x), cvRound(predicted.y)),
                           4, COLOR_PREDICT, -1);

                // 画出候选门控圆
                cv::circle(frame,
                           cv::Point(cvRound(predicted.x), cvRound(predicted.y)),
                           static_cast<int>(gateDist), COLOR_PREDICT, 1);
            }

            int bestIdx = -1;
            for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
                Candidate& cd = cands[i];
                double total = cd.shapeScore;

                if (track.active) {
                    // --- 距离门控 ---
                    const double dist = cv::norm(cd.center - predicted);
                    if (dist > gateDist) {
                        std::vector<std::vector<cv::Point>> tmp{ cd.contour };
                        cv::drawContours(frame, tmp, -1, COLOR_GATE_REJ, 1);
                        continue;
                    }
                    // --- 半径门控 ---
                    if (track.lastRadius > 1e-3f) {
                        const float rr =
                            std::abs(cd.radius - track.lastRadius) / track.lastRadius;
                        if (rr > RADIUS_TOL) {
                            std::vector<std::vector<cv::Point>> tmp{ cd.contour };
                            cv::drawContours(frame, tmp, -1, COLOR_GATE_REJ, 1);
                            continue;
                        }
                    }
                    // --- 追踪评分：越接近预测点越高 ---
                    const double trackScore = 1.0 - dist / gateDist;
                    total = SHAPE_WEIGHT * cd.shapeScore
                          + TRACK_WEIGHT * trackScore;
                }

                cd.score = total;
                if (bestIdx < 0 || total > cands[bestIdx].score)
                    bestIdx = i;
            }

            if (bestIdx >= 0) {
                found = true;
                best  = cands[bestIdx];
            }
        }

        // ---- 3.4 更新追踪状态 / 绘制 ----
        if (found) {
            // 速度更新：瞬时位移做平滑
            if (track.active) {
                const cv::Point2f instVel = best.center - track.lastPos;
                track.vel = track.vel * (1.0f - VEL_SMOOTH) + instVel * VEL_SMOOTH;
            } else {
                track.vel = cv::Point2f(0.0f, 0.0f);
            }
            track.active     = true;
            track.lastPos    = best.center;
            track.lastRadius = best.radius;
            track.lostFrames = 0;

            history.push_back(best.center);
            while (static_cast<int>(history.size()) > HIST_LEN)
                history.pop_front();

            // 画球
            cv::circle(frame, best.center,
                       static_cast<int>(std::max(best.radius, 3.0f)),
                       COLOR_BALL, 2);
            cv::circle(frame, best.center, 2, COLOR_BALL, -1);

            // 速度
            const double speed = std::sqrt(
                static_cast<double>(track.vel.x) * track.vel.x +
                static_cast<double>(track.vel.y) * track.vel.y);

            // 画运动方向 / 速度箭头
            if (speed > MIN_SPEED) {
                double len = speed * ARROW_SCALE;
                if (len > MAX_ARROW_LEN) len = MAX_ARROW_LEN;

                const cv::Point2f dir(
                    static_cast<float>(track.vel.x / speed),
                    static_cast<float>(track.vel.y / speed));
                const cv::Point2f tip(
                    static_cast<float>(best.center.x + dir.x * len),
                    static_cast<float>(best.center.y + dir.y * len));

                cv::arrowedLine(frame,
                    cv::Point(cvRound(best.center.x), cvRound(best.center.y)),
                    cv::Point(cvRound(tip.x),        cvRound(tip.y)),
                    COLOR_ARROW, 3, cv::LINE_AA, 0, 0.25);
            }

            // 状态信息
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "speed: %.2f px/frame (%.1f px/s)",
                          speed, speed * fps);
            cv::putText(frame, buf, cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv::LINE_AA);

            std::snprintf(buf, sizeof(buf),
                          "C=%.2f  r=%.2f  A=%.0f  score=%.2f",
                          best.circularity, best.aspect, best.area, best.score);
            cv::putText(frame, buf, cv::Point(20, 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv::LINE_AA);
        } else {
            // 丢失处理
            if (track.active) {
                track.lostFrames++;
                if (track.lostFrames > MAX_LOST) {
                    track.active = false;
                    track.vel    = cv::Point2f(0.0f, 0.0f);
                    history.clear();
                }
            }
            cv::putText(frame, "green ball lost", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        // ---- 3.5 显示 ----
        cv::imshow("Green Ball Tracker", frame);
        cv::imshow("mask", mask);

        const int key = cv::waitKey(1);
        if (key == 27) break;   // ESC 退出
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
