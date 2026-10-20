// ============================================================
// 绿球追踪作业：检测视频中的绿色球，并画出运动方向和速度变化。
//
// 小白阅读说明：
// 1. 程序从命令行读取视频文件，例如：./green_ball ball.mp4
// 2. HSV 是一种描述颜色的方式。我们把画面转换成 HSV，
//    再用 inRange 找出绿色范围内的像素。
// 3. 找到绿色区域后，程序选取最大的轮廓作为绿球，
//    用它的中心点代表绿球的位置。
// 4. 把当前中心和上一帧中心连起来，就能得到运动方向；
//    两个中心之间的距离除以经过的帧数，就是像素/帧速度。
// 5. 每段新速度会和上一段速度比较，并标注加速、减速或近似不变。
// 6. 处理结果会显示在窗口中，并保存为 annotated_output.mp4。
//
// 编译示例（需要已安装 OpenCV）：
// g++ main.cpp -o green_ball $(pkg-config --cflags --libs opencv4)
// 运行示例：
// ./green_ball 输入视频.mp4
// 也可以指定输出文件：
// ./green_ball 输入视频.mp4 输出视频.mp4
// ============================================================

#include <opencv2/opencv.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    // 检查是否提供了输入视频路径。
    if (argc < 2) {
        std::cerr << "用法: " << argv[0]
                  << " <输入视频> [输出视频]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath =
        (argc >= 3) ? argv[2] : "annotated_output.mp4";

    // 打开视频；如果文件不存在或格式不支持，就清楚地提示错误。
    cv::VideoCapture video(inputPath);
    if (!video.isOpened()) {
        std::cerr << "错误：无法打开视频文件 " << inputPath << '\n';
        return 1;
    }

    const int frameWidth =
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight =
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 有些视频没有正确记录帧率，这时使用常见的 30 帧/秒作为输出帧率。
    double fps = video.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
        fps = 30.0;
    }

    // 创建输出视频。mp4v 是常见的 MP4 编码格式。
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "错误：无法创建输出视频 " << outputPath
                  << "。请检查文件夹权限或输出文件名。\n";
        return 1;
    }

    // 保存上一段运动的中心、帧编号和速度，用来和当前运动作比较。
    cv::Point previousCenter;
    int previousFrameNumber = -1;
    double previousSpeed = -1.0;
    int frameNumber = 0;

    cv::Mat frame;
    while (video.read(frame)) {
        // HSV 颜色空间更适合按颜色范围找物体。
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 绿色的 HSV 范围。不同光线下可能需要稍微调整这些数字。
        cv::Mat greenMask;
        cv::inRange(
            hsv,
            cv::Scalar(35, 70, 50),
            cv::Scalar(85, 255, 255),
            greenMask);

        // 去掉小噪点，并填补绿色区域中细小的空隙。
        const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(greenMask, greenMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(greenMask, greenMask, cv::MORPH_CLOSE, kernel);

        // 轮廓代表图像中连在一起的区域；最大的绿色区域作为球的候选。
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            greenMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int largestContourIndex = -1;
        double largestArea = 100.0;  // 小于此面积的区域视为噪点。
        for (std::size_t i = 0; i < contours.size(); ++i) {
            const double area = cv::contourArea(contours[i]);
            if (area > largestArea) {
                largestArea = area;
                largestContourIndex = static_cast<int>(i);
            }
        }

        if (largestContourIndex >= 0) {
            const std::vector<cv::Point>& ballContour =
                contours[largestContourIndex];
            const cv::Moments moments = cv::moments(ballContour);

            // moments.m00 是轮廓面积；确认它不为零后计算中心坐标。
            if (moments.m00 > 0.0) {
                const cv::Point center(
                    static_cast<int>(moments.m10 / moments.m00),
                    static_cast<int>(moments.m01 / moments.m00));

                cv::drawContours(
                    frame,
                    contours,
                    largestContourIndex,
                    cv::Scalar(0, 255, 0),
                    2);
                cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);
                cv::putText(
                    frame,
                    "Green ball",
                    cv::Point(center.x + 10, center.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 0, 255),
                    2);

                if (previousFrameNumber >= 0) {
                    // 用帧编号差计算速度，即使中间有几帧没检测到球也适用。
                    const int frameGap = frameNumber - previousFrameNumber;
                    const double distance = cv::norm(center - previousCenter);
                    const double speed = distance / frameGap;

                    // 箭头从上一位置指向当前位置，长度和移动距离一致。
                    cv::arrowedLine(
                        frame,
                        previousCenter,
                        center,
                        cv::Scalar(255, 0, 0),
                        3,
                        cv::LINE_AA,
                        0,
                        0.2);

                    std::ostringstream speedLabel;
                    speedLabel << std::fixed << std::setprecision(1)
                               << "Speed: " << speed << " px/frame";
                    cv::putText(
                        frame,
                        speedLabel.str(),
                        cv::Point(20, 35),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 0, 0),
                        2);

                    if (previousSpeed >= 0.0) {
                        const double speedDifference = speed - previousSpeed;
                        const char* change = "Speed: nearly unchanged";
                        if (speedDifference > 1.0) {
                            change = "Speeding up";
                        } else if (speedDifference < -1.0) {
                            change = "Slowing down";
                        }
                        cv::putText(
                            frame,
                            change,
                            cv::Point(20, 65),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            cv::Scalar(255, 0, 0),
                            2);
                    } else {
                        cv::putText(
                            frame,
                            "Speed change: measuring",
                            cv::Point(20, 65),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            cv::Scalar(255, 0, 0),
                            2);
                    }

                    previousSpeed = speed;
                }

                previousCenter = center;
                previousFrameNumber = frameNumber;
            }
        } else {
            cv::putText(
                frame,
                "Green ball not found",
                cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 255),
                2);
        }

        // 保存标注后的画面，同时在屏幕上播放。
        writer.write(frame);
        cv::imshow("Green Ball Tracking", frame);

        // 按 Esc 键提前结束视频处理。
        if (cv::waitKey(1) == 27) {
            break;
        }

        ++frameNumber;
    }

    video.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "处理完成，标注视频已保存到: " << outputPath << '\n';
    return 0;
}