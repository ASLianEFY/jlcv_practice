#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    const std::string inputPath = "homework.mp4";
    const std::string outputPath = "homework_annotated.mp4";

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "无法打开视频文件: " << inputPath << std::endl;
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        fps = 30.0;
    }

    const int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频文件: " << outputPath << std::endl;
        return 1;
    }

    cv::Point previousCenter;
    bool hasPreviousCenter = false;
    cv::Mat frame;

    while (capture.read(frame)) {
        cv::Mat hsv;
        cv::Mat greenMask;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(35, 70, 50), cv::Scalar(90, 255, 255), greenMask);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(greenMask, greenMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(greenMask, greenMask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(greenMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int largestContourIndex = -1;
        double largestArea = 100.0;
        for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
            const double area = cv::contourArea(contours[i]);
            if (area > largestArea) {
                largestArea = area;
                largestContourIndex = i;
            }
        }

        if (largestContourIndex >= 0) {
            const cv::Moments moments = cv::moments(contours[largestContourIndex]);
            if (moments.m00 != 0.0) {
                const cv::Point center(
                    static_cast<int>(moments.m10 / moments.m00),
                    static_cast<int>(moments.m01 / moments.m00));

                cv::drawContours(frame, contours, largestContourIndex, cv::Scalar(0, 255, 0), 2);
                cv::circle(frame, center, 4, cv::Scalar(0, 0, 255), cv::FILLED);

                if (hasPreviousCenter) {
                    const cv::Point displacement = center - previousCenter;
                    const double distance = cv::norm(displacement);
                    const double speed = distance * fps;

                    if (distance > 0.0) {
                        const double arrowLength = std::min(speed * 0.15, 160.0);
                        const cv::Point arrowEnd(
                            center.x + static_cast<int>(std::round(displacement.x * arrowLength / distance)),
                            center.y + static_cast<int>(std::round(displacement.y * arrowLength / distance)));
                        cv::arrowedLine(frame, center, arrowEnd, cv::Scalar(0, 0, 255), 3, cv::LINE_AA, 0, 0.25);
                    }

                    cv::putText(
                        frame,
                        "Speed: " + std::to_string(static_cast<int>(std::round(speed))) + " px/s",
                        cv::Point(20, 35),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(0, 0, 255),
                        2,
                        cv::LINE_AA);
                }

                previousCenter = center;
                hasPreviousCenter = true;
            }
        } else {
            hasPreviousCenter = false;
        }

        writer.write(frame);
        cv::imshow("Green Ball Motion", frame);
        const int key = cv::waitKey(static_cast<int>(1000.0 / fps));
        if (key == 'q' || key == 27) {
            break;
        }
    }

    capture.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "标注视频已保存到: " << outputPath << std::endl;
    return 0;
}