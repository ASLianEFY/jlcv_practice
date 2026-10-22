#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

struct Ball {
    cv::Point2f center;
    float radius;
    double area;
};

static void drawSpeedOverlay(cv::Mat& frame, double speed, bool tracking,
                             int missedFrames) {
    const cv::Rect panel(18, 18, 350, 104);
    cv::Mat overlay = frame(panel);
    const cv::Mat background = overlay.clone();
    cv::rectangle(overlay, cv::Rect(0, 0, panel.width, panel.height),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(overlay, 0.78, background, 0.22, 0.0, overlay);
    cv::rectangle(frame, panel, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    const char* status = !tracking ? "GREEN BALL: SEARCHING"
                                   : missedFrames == 0 ? "GREEN BALL: TRACKING"
                                                       : "GREEN BALL: PREDICTING";
    cv::putText(frame, status,
                {32, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(100, 255, 100), 2, cv::LINE_AA);
    cv::putText(frame, "Speed:", {32, 98}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    const std::string value = tracking
        ? cv::format("%.0f px/s", speed)
        : "-- px/s";
    cv::putText(frame, value, {155, 98}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(60, 100, 255), 2, cv::LINE_AA);
}

static bool findBall(const cv::Mat& frame, const cv::Point2f& prediction,
                     bool tracking, double expectedArea, double maxDistance,
                     Ball& best) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(30, 60, 20), cv::Scalar(95, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestScore = 0.0;
    bool found = false;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 1500 || area > frame.total() * 0.15)
            continue;

        const double perimeter = cv::arcLength(contour, true);
        const double circularity =
            perimeter > 0.0 ? 4.0 * CV_PI * area / (perimeter * perimeter) : 0.0;
        const cv::Rect bounds = cv::boundingRect(contour);
        const double aspect = static_cast<double>(
            std::min(bounds.width, bounds.height)) /
            std::max(bounds.width, bounds.height);
        if (circularity < 0.68 || aspect < 0.80)
            continue;

        cv::Point2f center;
        float radius = 0.0f;
        cv::minEnclosingCircle(contour, center, radius);
        const double distance = tracking ? cv::norm(center - prediction) : 0.0;
        if (tracking && distance > maxDistance)
            continue;
        const double sizePenalty =
            tracking ? std::exp(-0.35 * std::abs(std::log(area / expectedArea))) : 1.0;
        const double score = circularity * aspect * sizePenalty /
                             (1.0 + distance / 180.0);
        if (score > bestScore) {
            bestScore = score;
            best = {center, radius, area};
            found = true;
        }
    }
    return found;
}

int main() {
    cv::VideoCapture cap("homework.mp4");
    if (!cap.isOpened()) {
        std::cerr << "无法打开 homework.mp4\n";
        return 1;
    }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const cv::Size frameSize(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (!std::isfinite(fps) || fps <= 0 || frameSize.empty()) {
        std::cerr << "输入视频的帧率或尺寸无效\n";
        return 1;
    }

    cv::VideoWriter writer("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frameSize);
    if (!writer.isOpened()) {
        std::cerr << "无法创建 result.mp4\n";
        return 1;
    }

    cv::KalmanFilter kalman(4, 2, 0, CV_32F);
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1.0f / fps, 0, 0, 1, 0, 1.0f / fps, 0, 0, 1, 0, 0, 0, 0, 1);
    kalman.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    kalman.measurementMatrix.at<float>(0, 0) = 1.0f;
    kalman.measurementMatrix.at<float>(1, 1) = 1.0f;
    const float dt = static_cast<float>(1.0 / fps);
    const float accelerationVariance = 90000.0f;
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    kalman.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
    for (const auto& axes : {std::pair<int, int>{0, 2}, {1, 3}}) {
        const int position = axes.first;
        const int velocity = axes.second;
        kalman.processNoiseCov.at<float>(position, position) =
            accelerationVariance * dt4 * 0.25f;
        kalman.processNoiseCov.at<float>(position, velocity) =
            accelerationVariance * dt3 * 0.5f;
        kalman.processNoiseCov.at<float>(velocity, position) =
            accelerationVariance * dt3 * 0.5f;
        kalman.processNoiseCov.at<float>(velocity, velocity) =
            accelerationVariance * dt2;
    }
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(16.0));

    bool tracking = false;
    cv::Point2f smoothedVelocity(0.0f, 0.0f);
    double expectedArea = 0.0;
    double displayedSpeed = 0.0;
    float trackedRadius = 0.0f;
    int missedFrames = 0;
    int maxMissedFrames = 0;
    int frames = 0;
    int detections = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        ++frames;
        cv::Point2f prediction;
        if (tracking) {
            const cv::Mat predicted = kalman.predict();
            prediction = {predicted.at<float>(0), predicted.at<float>(1)};
        }

        Ball ball{};
        const double predictedSpeed = tracking
            ? cv::norm(cv::Point2f(kalman.statePre.at<float>(2),
                                   kalman.statePre.at<float>(3)))
            : 0.0;
        const double associationRadius =
            std::min(180.0, std::max(60.0, predictedSpeed / fps * 3.0 +
                                               trackedRadius * 1.5));
        const bool detected = findBall(frame, prediction, tracking, expectedArea,
                                       associationRadius, ball);
        if (!tracking && detected) {
            kalman.statePost = (cv::Mat_<float>(4, 1) <<
                ball.center.x, ball.center.y, 0.0f, 0.0f);
            kalman.errorCovPost = cv::Mat::zeros(4, 4, CV_32F);
            kalman.errorCovPost.at<float>(0, 0) = 16.0f;
            kalman.errorCovPost.at<float>(1, 1) = 16.0f;
            kalman.errorCovPost.at<float>(2, 2) = 250000.0f;
            kalman.errorCovPost.at<float>(3, 3) = 250000.0f;
            tracking = true;
            smoothedVelocity = {0.0f, 0.0f};
            expectedArea = ball.area;
            trackedRadius = ball.radius;
            displayedSpeed = 0.0;
            cv::circle(frame, ball.center, cvRound(ball.radius),
                       cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            ++detections;
        } else if (tracking) {
            if (detected) {
                const cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
                    ball.center.x, ball.center.y);
                kalman.correct(measurement);
                expectedArea = 0.9 * expectedArea + 0.1 * ball.area;
                trackedRadius = 0.85f * trackedRadius + 0.15f * ball.radius;
                missedFrames = 0;
                ++detections;
            } else {
                kalman.statePre.copyTo(kalman.statePost);
                kalman.errorCovPre.copyTo(kalman.errorCovPost);
                ++missedFrames;
                maxMissedFrames = std::max(maxMissedFrames, missedFrames);
            }

            const cv::Point2f center(
                kalman.statePost.at<float>(0), kalman.statePost.at<float>(1));
            const cv::Point2f velocity(
                kalman.statePost.at<float>(2), kalman.statePost.at<float>(3));
            smoothedVelocity = 0.88f * smoothedVelocity + 0.12f * velocity;
            displayedSpeed = 0.88 * displayedSpeed +
                             0.12 * cv::norm(smoothedVelocity);
            if (center.x >= 0 && center.y >= 0 &&
                center.x < frame.cols && center.y < frame.rows) {
                cv::circle(frame, center, cvRound(trackedRadius),
                           cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            const double speed = cv::norm(smoothedVelocity);
            const double length = std::min(100.0, displayedSpeed * 0.2);
            if (length > 1.0 && center.x >= 0 && center.y >= 0 &&
                center.x < frame.cols && center.y < frame.rows &&
                speed > 1.0) {
                cv::Point start(cvRound(center.x), cvRound(center.y));
                const cv::Point2f direction =
                    smoothedVelocity / static_cast<float>(speed);
                cv::Point end(
                    cvRound(center.x + direction.x * length),
                    cvRound(center.y + direction.y * length));
                if (cv::clipLine(frame.size(), start, end))
                    cv::arrowedLine(frame, start, end, cv::Scalar(0, 0, 255),
                                    2, cv::LINE_AA, 0, 0.3);
            }
        }
        drawSpeedOverlay(frame, displayedSpeed, tracking, missedFrames);
        writer.write(frame);
    }

    writer.release();
    cap.release();
    if (frames == 0) {
        std::cerr << "输入视频没有可读取的帧\n";
        return 1;
    }
    std::cout << "已逐帧处理 " << frames << " 帧，FPS=" << fps
              << "，检测 " << detections << " 帧，最长连续预测 "
              << maxMissedFrames << " 帧；输出 result.mp4\n";
    return 0;
}
