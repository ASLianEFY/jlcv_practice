// 绿球检测与运动箭头标注
// 编译：g++ -O2 trace_ball.cpp -o trace_ball $(pkg-config --cflags --libs opencv4)
// 运行：./trace_ball [输入视频] [输出视频]   （默认 homework.mp4 -> output.mp4）

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct BallCandidate {
    cv::Point2f center;
    double roundness;
};

// 提取所有“像绿球”的轮廓。绿色圆柱的圆度更低、外接矩形更瘦长，
// 大部分情况下会被直接过滤掉；偶尔两者都在候选里时由跟踪器区分。
std::vector<BallCandidate> findGreenBalls(const cv::Mat& frame)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35, 60, 60), cv::Scalar(90, 255, 255), mask);

    const cv::Mat open_kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    const cv::Mat close_kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BallCandidate> balls;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 1500.0 || area > 40000.0) {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }
        const double aspect = static_cast<double>(box.width) / box.height;
        if (aspect < 0.78 || aspect > 1.45) {
            continue;  // 圆柱外接矩形的宽高比约为 0.6~0.7
        }

        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) {
            continue;
        }
        const double circularity =
            4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.76) {
            continue;  // 球的圆度约为 0.85，圆柱约为 0.74
        }

        const cv::Moments moments = cv::moments(contour);
        if (moments.m00 <= 0.0) {
            continue;
        }
        const cv::Point2f center(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00)
        );
        balls.push_back({center, circularity - 0.5 * std::abs(aspect - 1.0)});
    }
    return balls;
}

// 四状态卡尔曼滤波：[x, y, vx, vy]，匀速模型 + 加速度过程噪声。
class BallTracker {
public:
    BallTracker()
        : filter_(4, 2, 0, CV_32F), initialized_(false), lost_frames_(0)
    {
        filter_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        filter_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        // 过程噪声取大一些，使滤波器能跟上手持相机的快速运动。
        const float q = 20.0f;
        filter_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
            q / 4, 0, q / 2, 0,
            0, q / 4, 0, q / 2,
            q / 2, 0, q, 0,
            0, q / 2, 0, q);
        cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(4.0));
    }

    // 本帧是否成功关联到绿球。
    bool update(const std::vector<BallCandidate>& candidates)
    {
        if (!initialized_) {
            if (candidates.empty()) {
                return false;
            }
            const BallCandidate& best = *std::max_element(
                candidates.begin(), candidates.end(),
                [](const BallCandidate& a, const BallCandidate& b) {
                    return a.roundness < b.roundness;
                });
            filter_.statePost = (cv::Mat_<float>(4, 1) <<
                best.center.x, best.center.y, 0.0f, 0.0f);
            filter_.errorCovPost = (cv::Mat_<float>(4, 4) <<
                100, 0, 0, 0,
                0, 100, 0, 0,
                0, 0, 900, 0,
                0, 0, 0, 900);
            initialized_ = true;
            return true;
        }

        const cv::Mat prediction = filter_.predict();
        const cv::Point2f predicted(
            prediction.at<float>(0), prediction.at<float>(1)
        );

        const BallCandidate* match = nullptr;
        double best_distance = 1e9;
        for (const auto& candidate : candidates) {
            const double distance = cv::norm(candidate.center - predicted);
            if (distance < best_distance) {
                best_distance = distance;
                match = &candidate;
            }
        }

        // 球每帧最多移动约 30 像素，而绿色圆柱与球至少相距 180 像素，
        // 80 像素的关联门限足以区分两者。
        if (match != nullptr && best_distance <= 80.0) {
            const cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
                match->center.x, match->center.y);
            filter_.correct(measurement);
            lost_frames_ = 0;
            return true;
        }

        // 本帧没有观测：沿用预测值；连续丢失较久才重新初始化。
        filter_.statePost = filter_.statePre;
        filter_.errorCovPost = filter_.errorCovPre;
        if (++lost_frames_ > 10) {
            initialized_ = false;
            lost_frames_ = 0;
        }
        return false;
    }

    cv::Point2f position() const
    {
        return cv::Point2f(
            filter_.statePost.at<float>(0),
            filter_.statePost.at<float>(1)
        );
    }

    cv::Point2f velocity() const
    {
        return cv::Point2f(
            filter_.statePost.at<float>(2),
            filter_.statePost.at<float>(3)
        );
    }

private:
    cv::KalmanFilter filter_;
    bool initialized_;
    int lost_frames_;
};

}  // namespace

int main(int argc, char** argv)
{
    const std::string input_path = argc > 1 ? argv[1] : "homework.mp4";
    const std::string output_path = argc > 2 ? argv[2] : "output.mp4";

    const double arrow_scale = 5.0;       // 每 1 像素/帧 对应的箭头长度
    const double max_arrow_length = 160.0;

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "错误：无法打开输入视频 " << input_path << std::endl;
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 30.0;
    }

    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
        std::cerr << "错误：输入视频中没有可读取的帧。" << std::endl;
        return 1;
    }

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        frame.size()
    );
    if (!writer.isOpened()) {
        std::cerr << "错误：无法创建输出视频 " << output_path << std::endl;
        return 1;
    }

    BallTracker tracker;
    int frame_count = 0;
    int tracked_count = 0;

    do {
        const std::vector<BallCandidate> candidates = findGreenBalls(frame);
        const bool tracked = tracker.update(candidates);
        if (tracked) {
            ++tracked_count;
        }

        if (tracked) {
            const cv::Point2f center = tracker.position();
            const cv::Point2f velocity = tracker.velocity();
            const double speed = cv::norm(velocity);

            cv::circle(
                frame, center, 5, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA
            );

            // 箭头方向 = 运动方向；长度与速度成正比。
            if (speed > 1.0) {
                const double length =
                    std::min(arrow_scale * speed, max_arrow_length);
                const cv::Point2f direction =
                    velocity * (1.0f / static_cast<float>(speed));
                const cv::Point2f tip =
                    center + direction * static_cast<float>(length);
                cv::arrowedLine(
                    frame, center, tip, cv::Scalar(0, 0, 255),
                    3, cv::LINE_AA, 0, 0.25
                );
            }
        }

        writer.write(frame);
        ++frame_count;
    } while (capture.read(frame) && !frame.empty());

    std::cout << "处理完成：" << frame_count << " 帧，跟踪到绿球 "
              << tracked_count << " 帧，输出文件：" << output_path << std::endl;
    return 0;
}
