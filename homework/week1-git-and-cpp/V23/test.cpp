#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    // --- 1. Open video ---
    string videoPath = "your_video.mp4";   // <-- replace with actual path
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: cannot open video: " << videoPath << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.1) fps = 30.0;  // fallback

    // --- 2. Green HSV range (tune if needed) ---
    Scalar lowerGreen(35, 60, 60);
    Scalar upperGreen(85, 255, 255);

    // --- 3. Video writer setup ---
    string outputPath = "homework/name/output_with_arrows.avi";
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    double outFps = fps;
    Size frameSize;
    VideoWriter writer;
    bool writerReady = false;

    Point prevCenter(-1, -1);
    Mat frame;

    while (cap.read(frame)) {
        if (!writerReady) {
            frameSize = frame.size();
            writer.open(outputPath, fourcc, outFps, frameSize);
            if (!writer.isOpened()) {
                cerr << "Error: cannot open VideoWriter" << endl;
                return -1;
            }
            writerReady = true;
        }

        // --- 4. Detect green ball ---
        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, lowerGreen, upperGreen, mask);

        // Morphological cleanup
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point currentCenter(-1, -1);
        double maxArea = 0;
        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area > 300 && area > maxArea) {  // filter noise by area
                maxArea = area;
                Moments m = moments(c);
                if (m.m00 != 0) {
                    currentCenter = Point((int)(m.m10 / m.m00), (int)(m.m01 / m.m00));
                }
            }
        }

        // --- 5. Draw detection circle ---
        if (currentCenter.x >= 0) {
            circle(frame, currentCenter, 8, Scalar(0, 255, 0), 2);
            circle(frame, currentCenter, 3, Scalar(0, 0, 255), -1);
        }

        // --- 6. Draw velocity arrow ---
        if (prevCenter.x >= 0 && currentCenter.x >= 0) {
            double dx = currentCenter.x - prevCenter.x;
            double dy = currentCenter.y - prevCenter.y;
            double speed = sqrt(dx * dx + dy * dy);

            if (speed > 2.0) {  // avoid drawing for near-stationary frames
                // Arrow from previous center to current center.
                // Length = displacement = speed (since fps is constant).
                arrowedLine(frame, prevCenter, currentCenter,
                            Scalar(0, 255, 255), 2, LINE_AA, 0, 0.3);

                // Optional: label speed value
                string label = "v=" + to_string((int)speed) + " px/f";
                putText(frame, label, currentCenter + Point(10, -10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }
        }

        // Update previous center
        if (currentCenter.x >= 0) {
            prevCenter = currentCenter;
        }

        // --- 7. Output ---
        writer.write(frame);
        imshow("Green Ball with Velocity Arrow", frame);
        if (waitKey(1) == 27) break;  // ESC to quit
    }

    cap.release();
    if (writerReady) writer.release();
    destroyAllWindows();
    return 0;
}
