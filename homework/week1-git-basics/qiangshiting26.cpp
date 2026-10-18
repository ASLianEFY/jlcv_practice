#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    string inputFile = argc > 1 ? argv[1] : "homework.mp4";
    string outputFile = argc > 2 ? argv[2] : "green_ball_tracking.avi";

    if (argc == 1) {
        ifstream inputTest(inputFile);
        if (!inputTest) {
            inputFile = "../../homework.mp4";
        }
    }
    VideoCapture video(inputFile);
    if (!video.isOpened()) {
        cout << "打不开视频。请把 homework.mp4 放到当前文件夹再试一次。\n";
        return 1;
    }

    double fps = video.get(CAP_PROP_FPS);
    if (fps <= 0) {
        fps = 30;
    }

    VideoWriter result;
    Point previousCenter;
    bool hasPreviousCenter = false;
    int frameCount = 0;
    int foundCount = 0;
    Mat frame;

    cout << "视频打开了，正在找绿球，请稍等...\n";
    while (video.read(frame)) {
        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(35, 60, 40), Scalar(95, 255, 255), mask);
        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point center;
        double largestScore = 0;
        bool foundBall = false;

        for (const vector<Point>& contour : contours) {
            double area = contourArea(contour);
            if (area < 500 || area > frame.total() * 0.04) {
                continue;
            }

            Rect box = boundingRect(contour);
            double shape = double(box.width) / box.height;
            double perimeter = arcLength(contour, true);
            double roundness = 4 * CV_PI * area / (perimeter * perimeter);

            // 圆球接近圆形；细长的绿色圆柱会被跳过。
            if (shape < 0.75 || shape > 1.33 || roundness < 0.55) {
                continue;
            }

            double score = area * roundness;
            Moments m = moments(contour);
            if (score > largestScore && m.m00 > 0) {
                center = Point(int(m.m10 / m.m00), int(m.m01 / m.m00));
                largestScore = score;
                foundBall = true;
            }
        }

        if (foundBall) {
            foundCount++;
            circle(frame, center, 6, Scalar(0, 255, 255), -1);
            putText(frame, "Green ball", center + Point(12, -12),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

            if (hasPreviousCenter) {
                Point movement = center - previousCenter;
                double distance = norm(movement);
                if (distance > 0) {
                    double speed = distance * fps;
                    double arrowLength = min(speed * 0.2, 120.0);
                    Point end = center + movement * (arrowLength / distance);
                    arrowedLine(frame, center, end, Scalar(0, 0, 255), 3);
                    putText(frame, "Speed: " + to_string(int(speed)) + " px/s",
                            center + Point(12, 18), FONT_HERSHEY_SIMPLEX, 0.6,
                            Scalar(0, 0, 255), 2);
                }
            }

            previousCenter = center;
            hasPreviousCenter = true;
        }

        if (!result.isOpened()) {
            result.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        fps, frame.size());
            if (!result.isOpened()) {
                cout << "保存失败。请检查文件夹是否可以写入。\n";
                return 1;
            }
        }

        result.write(frame);
        frameCount++;
    }

    if (frameCount == 0) {
        cout << "没有读到视频画面，请检查视频文件是否正常。\n";
        return 1;
    }

    cout << "处理完成！共检查 " << frameCount << " 帧。\n"
         << "在 " << foundCount << " 帧里找到了绿球。\n"
         << "标注后的视频已保存为：" << outputFile << "\n"
         << "箭头指向运动方向；箭头越长，表示速度越快。\n";
    return 0;
}
