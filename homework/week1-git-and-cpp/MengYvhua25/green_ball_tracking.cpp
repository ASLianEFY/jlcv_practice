// green_ball_tracking.cpp
// Build: g++ -std=c++17 green_ball_tracking.cpp -o green_ball_tracking `pkg-config --cflags --libs opencv4`
// Run:   ./green_ball_tracking input.mp4 output.mp4
//
// Identity principle:
// 1) A candidate must be green AND approximately round (aspect ratio + circularity).
// 2) Keep the same object across frames by choosing the candidate nearest the motion-predicted
//    position, with shape/size as secondary evidence.
// 3) Estimate velocity from a centered temporal difference, smooth position noise, and draw
//    the arrow from the tracked ball center (never from a generic green-region centroid).

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

struct Candidate {
    cv::Point2f center;
    double area, ratio, circularity, shapeCost;
};

static std::vector<Candidate> detectGreenBall(const cv::Mat& frame) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(35, 55, 35), cv::Scalar(95, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Candidate> out;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 3500.0) continue;
        cv::Rect box = cv::boundingRect(contour);
        double ratio = static_cast<double>(box.width) / std::max(1, box.height);
        double perimeter = cv::arcLength(contour, true);
        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter + 1e-6);
        // A tall cylinder fails this gate even though it shares the ball's green color.
        if (ratio < 0.76 || ratio > 1.28 || circularity < 0.64) continue;
        cv::Moments m = cv::moments(contour);
        if (m.m00 <= 0) continue;
        cv::Point2f center(static_cast<float>(m.m10 / m.m00),
                           static_cast<float>(m.m01 / m.m00));
        double shapeCost = std::abs(std::log(std::max(area, 1.0) / 11500.0))
                         + 1.4 * std::abs(1.0 - ratio)
                         + 1.2 * std::max(0.0, 0.72 - circularity);
        out.push_back({center, area, ratio, circularity, shapeCost});
    }
    return out;
}

static double sqDist(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f d = a - b;
    return d.dot(d);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.mp4 output.mp4\n";
        return 2;
    }
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input video: " << argv[1] << "\n";
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::vector<cv::Mat> frames;
    std::vector<std::vector<Candidate>> allCandidates;
    cv::Mat frame;
    while (cap.read(frame)) {
        frames.push_back(frame.clone());
        allCandidates.push_back(detectGreenBall(frame));
    }
    cap.release();
    if (frames.empty()) {
        std::cerr << "Input video has no readable frames.\n";
        return 1;
    }

    // Track identity using shape gates plus a short-horizon constant-velocity prediction.
    std::vector<std::optional<cv::Point2f>> track(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& cs = allCandidates[i];
        if (cs.empty()) continue;
        int prevI = -1, prevPrevI = -1;
        for (int j = static_cast<int>(i) - 1; j >= 0 && j >= static_cast<int>(i) - 4; --j)
            if (track[j].has_value()) { prevI = j; break; }
        if (prevI < 0) {
            auto best = std::min_element(cs.begin(), cs.end(),
                [](const Candidate& a, const Candidate& b) { return a.shapeCost < b.shapeCost; });
            track[i] = best->center;
            continue;
        }
        for (int j = prevI - 1; j >= 0; --j)
            if (track[j].has_value()) { prevPrevI = j; break; }

        cv::Point2f velocity(0, 0);
        if (prevPrevI >= 0)
            velocity = (*track[prevI] - *track[prevPrevI]) *
                       (1.0f / std::max(1, prevI - prevPrevI));
        cv::Point2f prediction = *track[prevI] +
            velocity * static_cast<float>(std::min<int>(static_cast<int>(i) - prevI, 4));

        double bestCost = std::numeric_limits<double>::infinity();
        int bestIdx = -1;
        for (int k = 0; k < static_cast<int>(cs.size()); ++k) {
            double d2 = sqDist(cs[k].center, prediction);
            double cost = cs[k].shapeCost + 0.006 * std::sqrt(d2) + 0.000015 * d2;
            if (cost < bestCost) { bestCost = cost; bestIdx = k; }
        }
        track[i] = cs[bestIdx].center;
    }

    // Interpolate only tiny gaps (<= 5 frames), not long losses that could cause identity switches.
    for (size_t i = 0; i < track.size(); ++i) {
        if (track[i].has_value()) continue;
        int left = -1, right = -1;
        for (int j = static_cast<int>(i)-1; j >= 0; --j)
            if (track[j].has_value()) { left = j; break; }
        for (size_t j = i+1; j < track.size(); ++j)
            if (track[j].has_value()) { right = static_cast<int>(j); break; }
        if (left >= 0 && right >= 0 && right-left <= 5) {
            float a = static_cast<float>(static_cast<int>(i)-left) / (right-left);
            track[i] = *track[left] * (1.0f-a) + *track[right] * a;
        }
    }

    const int smoothRadius = std::max(2, static_cast<int>(fps * 0.08));
    std::vector<std::optional<cv::Point2f>> smooth(track.size());
    for (int i = 0; i < static_cast<int>(track.size()); ++i) {
        cv::Point2f sum(0, 0); int count = 0;
        for (int j = std::max(0, i-smoothRadius);
             j <= std::min(static_cast<int>(track.size())-1, i+smoothRadius); ++j) {
            if (track[j]) { sum += *track[j]; ++count; }
        }
        if (count) smooth[i] = sum * (1.0f / count);
    }

    std::vector<cv::Point2f> direction(track.size(), cv::Point2f(0,0));
    std::vector<double> speed(track.size(), 0.0);
    const int halfWindow = std::max(2, static_cast<int>(fps * 0.16));
    for (int i = 0; i < static_cast<int>(smooth.size()); ++i) {
        int a = std::max(0, i-halfWindow), b = std::min(static_cast<int>(smooth.size())-1, i+halfWindow);
        if (!smooth[a] || !smooth[b] || a == b) continue;
        cv::Point2f delta = *smooth[b] - *smooth[a];
        double distance = cv::norm(delta);
        if (distance > 1e-6) direction[i] = delta * static_cast<float>(1.0 / distance);
        speed[i] = distance / ((b-a) / fps); // apparent image-plane pixels per second
    }

    cv::VideoWriter writer(argv[2], cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Cannot create output video: " << argv[2] << "\n";
        return 1;
    }
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::Mat out = frames[i].clone();
        if (smooth[i]) {
            cv::Point center(cvRound(smooth[i]->x), cvRound(smooth[i]->y));
            cv::circle(out, center, 8, cv::Scalar(0,255,255), 2, cv::LINE_AA);
            int len = std::clamp(static_cast<int>(20 + speed[i] * 0.14), 20, 125);
            if (speed[i] < 8) len = 18;
            cv::Point end(cvRound(center.x + direction[i].x * len),
                          cvRound(center.y + direction[i].y * len));
            cv::arrowedLine(out, center, end, cv::Scalar(0,70,255), 5, cv::LINE_AA, 0, 0.28);
        }
        cv::Mat overlay = out.clone();
        cv::rectangle(overlay, cv::Rect(16,16,349,77), cv::Scalar(25,25,25), cv::FILLED);
        cv::addWeighted(overlay, 0.60, out, 0.40, 0, out);
        cv::putText(out, "Green ball: direction / speed", cv::Point(28,47),
                    cv::FONT_HERSHEY_SIMPLEX, 0.66, cv::Scalar(255,255,255), 2, cv::LINE_AA);
        cv::putText(out, "Longer arrow = faster", cv::Point(28,77),
                    cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(255,255,255), 2, cv::LINE_AA);
        writer.write(out);
    }
    writer.release();
    std::cout << "Wrote " << argv[2] << " (" << frames.size() << " frames, " << fps << " fps)\n";
    return 0;
}