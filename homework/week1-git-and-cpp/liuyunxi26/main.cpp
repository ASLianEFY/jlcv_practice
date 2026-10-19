#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

struct BallCandidate {
    cv::Point2f center;
    double area;
    double circularity;
};

static std::vector<BallCandidate> findGreenBalls(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::Mat mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(35, 40, 35), cv::Scalar(90, 255, 255), mask);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BallCandidate> candidates;
    const double minArea = std::max(40.0, frame.total() * 0.00005);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < minArea) {
            continue;
        }

        const cv::Rect bounds = cv::boundingRect(contour);
        const double aspectRatio = static_cast<double>(bounds.width) / bounds.height;
        const double perimeter = cv::arcLength(contour, true);
        const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter + 1e-6);
        if (aspectRatio < 0.65 || aspectRatio > 1.45 || circularity < 0.42) {
            continue;
        }

        const cv::Moments moments = cv::moments(contour);
        if (moments.m00 <= 0.0) {
            continue;
        }
        candidates.push_back({
            cv::Point2f(static_cast<float>(moments.m10 / moments.m00),
                        static_cast<float>(moments.m01 / moments.m00)),
            area,
            circularity
        });
    }
    return candidates;
}

int main(int argc, char** argv) {
    const std::string inputPath = argc > 1 ? argv[1] : "../../../homework.mp4";
    const std::string outputPath = argc > 2 ? argv[2] : "output.mp4";

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "无法打开输入视频: " << inputPath << '\n';
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 30.0;
    }
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width <= 0 || height <= 0) {
        std::cerr << "无法读取视频尺寸。\n";
        return 1;
    }

    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height)
    );
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频: " << outputPath
                  << "。请检查 OpenCV 的视频编码器支持和输出路径。\n";
        return 1;
    }

    std::optional<cv::Point2f> lastCenter;
    std::optional<cv::Point2f> previousCenter;
    int lastDetectionFrame = -1;
    int previousDetectionFrame = -1;
    double lastArea = 0.0;
    int missedFrames = 0;
    int frameIndex = 0;
    const double frameDiagonal = std::hypot(width, height);
    const int historyWindow = std::max(1, cvRound(fps * 0.15));
    std::deque<std::pair<int, cv::Point2f>> history;

    cv::Mat frame;
    while (capture.read(frame)) {
        const std::vector<BallCandidate> candidates = findGreenBalls(frame);
        int selectedIndex = -1;

        if (!candidates.empty() && lastCenter && missedFrames <= 8) {
            cv::Point2f velocity(0.0f, 0.0f);
            if (previousCenter && lastDetectionFrame > previousDetectionFrame) {
                velocity = (*lastCenter - *previousCenter) *
                    (1.0f / static_cast<float>(lastDetectionFrame - previousDetectionFrame));
            }
            const int elapsedFrames = frameIndex - lastDetectionFrame;
            const cv::Point2f predicted = *lastCenter + velocity *
                static_cast<float>(std::min(elapsedFrames, 5));
            const double maxDistance = std::max(80.0, frameDiagonal * 0.18 * elapsedFrames);
            double bestScore = std::numeric_limits<double>::infinity();

            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                const double distance = cv::norm(candidates[i].center - predicted);
                if (distance > maxDistance) {
                    continue;
                }
                const double areaPenalty = 18.0 * std::abs(
                    std::log(std::max(candidates[i].area, 1.0) / std::max(lastArea, 1.0)));
                const double score = distance + areaPenalty + 30.0 * (1.0 - candidates[i].circularity);
                if (score < bestScore) {
                    bestScore = score;
                    selectedIndex = i;
                }
            }
        } else if (!candidates.empty()) {
            selectedIndex = static_cast<int>(std::distance(
                candidates.begin(),
                std::max_element(candidates.begin(), candidates.end(),
                    [](const BallCandidate& left, const BallCandidate& right) {
                        return left.area * left.circularity < right.area * right.circularity;
                    })
            ));
        }

        std::optional<cv::Point2f> trackedCenter;
        if (selectedIndex >= 0) {
            const cv::Point2f detectedCenter = candidates[selectedIndex].center;
            cv::Point2f filteredCenter = detectedCenter;
            if (lastCenter && missedFrames <= 8) {
                filteredCenter = *lastCenter * 0.35f + detectedCenter * 0.65f;
            }

            previousCenter = lastCenter;
            previousDetectionFrame = lastDetectionFrame;
            lastCenter = filteredCenter;
            lastDetectionFrame = frameIndex;
            lastArea = candidates[selectedIndex].area;
            missedFrames = 0;
            trackedCenter = filteredCenter;
            history.emplace_back(frameIndex, filteredCenter);
        } else {
            ++missedFrames;
            if (missedFrames > 8) {
                lastCenter.reset();
                previousCenter.reset();
                lastArea = 0.0;
                history.clear();
            }
        }

        while (!history.empty() && frameIndex - history.front().first > fps * 0.5) {
            history.pop_front();
        }

        double speed = 0.0;
        cv::Point2f direction(0.0f, 0.0f);
        if (trackedCenter && history.size() >= 2) {
            const auto& current = history.back();
            const std::pair<int, cv::Point2f>* earlier = nullptr;
            for (const auto& sample : history) {
                if (current.first - sample.first >= historyWindow) {
                    earlier = &sample;
                } else {
                    break;
                }
            }
            if (!earlier && history.front().first < current.first) {
                earlier = &history.front();
            }
            if (earlier) {
                const cv::Point2f displacement = current.second - earlier->second;
                const double distance = cv::norm(displacement);
                const double elapsedSeconds = (current.first - earlier->first) / fps;
                if (distance > 0.5 && elapsedSeconds > 0.0) {
                    speed = distance / elapsedSeconds;
                    direction = displacement * static_cast<float>(1.0 / distance);
                }
            }
        }

        if (trackedCenter) {
            const cv::Point center(cvRound(trackedCenter->x), cvRound(trackedCenter->y));
            cv::circle(frame, center, 8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            if (speed > 0.0) {
                const int arrowLength = std::clamp(cvRound(speed * 0.14), 8, 120);
                const cv::Point end(
                    cvRound(center.x + direction.x * arrowLength),
                    cvRound(center.y + direction.y * arrowLength)
                );
                cv::arrowedLine(frame, center, end, cv::Scalar(0, 70, 255),
                                3, cv::LINE_AA, 0, 0.25);
            }
        }

        std::ostringstream status;
        status << std::fixed << std::setprecision(1) << "Speed: " << speed << " px/s";
        cv::putText(frame, trackedCenter ? status.str() : "Green ball: not detected",
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        writer.write(frame);
        cv::imshow("Green Ball Tracking", frame);
        const int key = cv::waitKey(std::max(1, cvRound(1000.0 / fps))) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }
        ++frameIndex;
    }

    capture.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "处理结果已保存到: " << outputPath << '\n';
    return 0;
}