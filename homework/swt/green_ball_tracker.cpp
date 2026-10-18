#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

namespace {

bool findGreenBall(const cv::Mat& frame, cv::Point& center, double& area) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35, 70, 50), cv::Scalar(90, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const std::vector<cv::Point>* largestContour = nullptr;
    area = 0.0;
    for (const auto& contour : contours) {
        const double contourArea = cv::contourArea(contour);
        if (contourArea > area) {
            area = contourArea;
            largestContour = &contour;
        }
    }

    if (largestContour == nullptr || area < 25.0) {
        return false;
    }

    const cv::Moments moments = cv::moments(*largestContour);
    if (moments.m00 == 0.0) {
        return false;
    }

    center = cv::Point(static_cast<int>(moments.m10 / moments.m00),
                       static_cast<int>(moments.m01 / moments.m00));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string inputPath = argc > 1 ? argv[1] : "homework.mp4";
    const std::string outputPath = argc > 2 ? argv[2] : "homework/swt/annotated.mp4";

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "Error: cannot open input video: " << inputPath << '\n';
        return 1;
    }

    const double sourceFps = capture.get(cv::CAP_PROP_FPS);
    const double fps = sourceFps > 0.0 ? sourceFps : 30.0;
    const int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "Error: cannot create output video: " << outputPath << '\n';
        return 1;
    }

    cv::Point previousCenter;
    bool hasPreviousCenter = false;
    int framesSinceDetection = 0;
    int detectedFrames = 0;
    int frameCount = 0;
    cv::Mat frame;

    while (capture.read(frame)) {
        ++frameCount;
        ++framesSinceDetection;

        cv::Point center;
        double ballArea = 0.0;
        if (findGreenBall(frame, center, ballArea)) {
            ++detectedFrames;
            cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), cv::FILLED);

            if (hasPreviousCenter) {
                const cv::Point2d displacement(center.x - previousCenter.x,
                                               center.y - previousCenter.y);
                const double distance = cv::norm(displacement);
                const double elapsedSeconds = framesSinceDetection / fps;
                const double speed = elapsedSeconds > 0.0 ? distance / elapsedSeconds : 0.0;

                if (distance > 1.0) {
                    const double arrowLength = std::clamp(speed * 0.15, 8.0, 140.0);
                    const cv::Point tip(
                        center.x + static_cast<int>(std::lround(displacement.x / distance * arrowLength)),
                        center.y + static_cast<int>(std::lround(displacement.y / distance * arrowLength)));
                    cv::arrowedLine(frame, center, tip, cv::Scalar(0, 0, 255), 3,
                                    cv::LINE_AA, 0, 0.25);
                }

                cv::putText(frame, cv::format("Speed: %.1f px/s", speed),
                            cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            } else {
                cv::putText(frame, "Tracking green ball...", cv::Point(20, 35),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255),
                            2, cv::LINE_AA);
            }

            cv::putText(frame, cv::format("Area: %.0f px", ballArea),
                        cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            previousCenter = center;
            hasPreviousCenter = true;
            framesSinceDetection = 0;
        } else {
            hasPreviousCenter = false;
            cv::putText(frame, "Green ball not found", cv::Point(20, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255),
                        2, cv::LINE_AA);
        }

        writer.write(frame);
    }

    capture.release();
    writer.release();

    if (frameCount == 0) {
        std::cerr << "Error: input video contains no readable frames.\n";
        return 1;
    }

    std::cout << "Processed " << frameCount << " frames; green ball detected in "
              << detectedFrames << " frames.\n"
              << "Annotated video saved to: " << outputPath << '\n';
    return 0;
}