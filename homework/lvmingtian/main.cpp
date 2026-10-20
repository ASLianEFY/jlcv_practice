#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Detection {
    cv::Point2f center;
    cv::Rect bounds;
    double score;
};

std::vector<Detection> findGreenBalls(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35, 55, 45), cv::Scalar(95, 255, 255), mask);

    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Detection> detections;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 1000.0) {
            continue;
        }

        const cv::Rect bounds = cv::boundingRect(contour);
        const double short_side = std::min(bounds.width, bounds.height);
        const double long_side = std::max(bounds.width, bounds.height);
        const double aspect_ratio = short_side / long_side;
        const double perimeter = cv::arcLength(contour, true);
        if (aspect_ratio < 0.68 || perimeter <= 0.0) {
            continue;
        }

        const double circularity =
            4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.62) {
            continue;
        }

        const cv::Moments moments = cv::moments(contour);
        if (moments.m00 == 0.0) {
            continue;
        }

        const cv::Point2f center(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00));
        detections.push_back(
            {center, bounds, area * circularity * circularity});
    }

    return detections;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path input_path =
        argc > 1 ? argv[1] : "homework.mp4";
    const std::filesystem::path output_path =
        argc > 2 ? argv[2] : "homework/lvmingtian/result.mp4";

    cv::VideoCapture input(input_path.string());
    if (!input.isOpened()) {
        std::cerr << "无法打开输入视频: " << input_path << '\n';
        return 1;
    }

    const double fps = input.get(cv::CAP_PROP_FPS);
    const int frame_count =
        static_cast<int>(input.get(cv::CAP_PROP_FRAME_COUNT));
    const cv::Size frame_size(
        static_cast<int>(input.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(input.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (fps <= 0.0 || frame_count <= 0 ||
        frame_size.width <= 0 || frame_size.height <= 0) {
        std::cerr << "视频信息无效，无法处理。\n";
        return 1;
    }

    std::error_code error;
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path(), error);
        if (error) {
            std::cerr << "无法创建输出目录: " << error.message() << '\n';
            return 1;
        }
    }

    cv::VideoWriter output(
        output_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
        frame_size);
    if (!output.isOpened()) {
        std::cerr << "无法创建输出视频: " << output_path << '\n';
        return 1;
    }

    std::cout << "分辨率: " << frame_size.width << " x " << frame_size.height
              << "\n帧率: " << std::fixed << std::setprecision(2) << fps
              << " FPS\n总帧数: " << frame_count
              << "\n时长: " << frame_count / fps << " 秒\n";

    constexpr int velocity_window = 5;
    std::deque<std::pair<int, cv::Point2f>> history;
    cv::Point2f previous_center;
    bool has_previous_center = false;
    int detected_frames = 0;
    cv::Mat frame;

    for (int frame_index = 0; input.read(frame); ++frame_index) {
        const std::vector<Detection> detections = findGreenBalls(frame);
        const Detection* best = nullptr;
        double best_score = -1.0;

        for (const Detection& detection : detections) {
            double score = detection.score;
            if (has_previous_center) {
                const double distance =
                    cv::norm(detection.center - previous_center);
                if (distance > 180.0) {
                    continue;
                }
                score /= 1.0 + distance / 90.0;
            }
            if (score > best_score) {
                best = &detection;
                best_score = score;
            }
        }

        if (best != nullptr) {
            ++detected_frames;
            previous_center = best->center;
            has_previous_center = true;
            history.emplace_back(frame_index, best->center);
            while (!history.empty() &&
                   frame_index - history.front().first > velocity_window) {
                history.pop_front();
            }

            cv::rectangle(frame, best->bounds, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, best->center, 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, "Green ball", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 255, 0), 2);

            if (!history.empty() &&
                frame_index - history.front().first >= velocity_window) {
                const cv::Point2f displacement =
                    best->center - history.front().second;
                const double distance = cv::norm(displacement);
                if (distance > 1.0) {
                    const double speed = distance * fps / velocity_window;
                    const double arrow_length =
                        std::clamp(speed * 0.16, 15.0, 120.0);
                    const cv::Point2f arrow_end =
                        best->center +
                        displacement * static_cast<float>(arrow_length / distance);
                    cv::arrowedLine(frame, best->center, arrow_end,
                                    cv::Scalar(0, 0, 255), 4, cv::LINE_AA, 0,
                                    0.25);

                    std::string speed_text =
                        "Speed: " + std::to_string(static_cast<int>(speed)) +
                        " px/s";
                    cv::putText(frame, speed_text, cv::Point(20, 75),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                cv::Scalar(0, 0, 255), 2);
                }
            }
        } else {
            cv::putText(frame, "Green ball not detected", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2);
        }

        output.write(frame);
    }

    input.release();
    output.release();
    if (detected_frames == 0) {
        std::cerr << "处理完成，但没有检测到绿球，请检查 HSV 阈值。\n";
        return 1;
    }

    std::cout << "检测到绿球的帧数: " << detected_frames << " / "
              << frame_count << "\n输出视频: " << output_path << '\n';
    return 0;
}
