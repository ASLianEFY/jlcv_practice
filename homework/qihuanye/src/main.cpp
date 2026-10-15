#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

namespace
{

// 绿球候选目标，保存颜色分割后单个轮廓的几何信息。
struct BallCandidate
{
  cv::Point2f center;
  float radius = 0.0F;
  double area = 0.0;
  double circularity = 0.0;
  double aspect_ratio = 0.0;
  cv::Rect bounding_box;
};

// 绿球跟踪状态，保存当前帧定位结果以及相邻帧速度信息。
struct BallTrack
{
  cv::Point2f center;
  cv::Point2f velocity;
  float radius = 0.0F;
  double speed = 0.0;
  cv::Rect bounding_box;
  int lost_frames = 0;
  bool initialized = false;
};

constexpr int kArrowColorBlue = 255;
constexpr int kArrowColorGreen = 64;
constexpr int kArrowColorRed = 64;
constexpr int kMaxLostFrames = 8;

bool isNearFrameBorder(const cv::Rect & bounding_box, const cv::Size & frame_size, int margin = 8)
{
  return bounding_box.x <= margin ||
         bounding_box.y <= margin ||
         bounding_box.x + bounding_box.width >= frame_size.width - margin ||
         bounding_box.y + bounding_box.height >= frame_size.height - margin;
}

cv::Point2f clampPointToFrame(const cv::Point2f & point, const cv::Size & frame_size)
{
  const float x = std::clamp(point.x, 0.0F, static_cast<float>(frame_size.width - 1));
  const float y = std::clamp(point.y, 0.0F, static_cast<float>(frame_size.height - 1));
  return cv::Point2f(x, y);
}

// 在 HSV 空间提取绿色区域，并按面积、圆度、长宽比筛出更像球体的候选目标。
std::vector<BallCandidate> detectBallCandidates(const cv::Mat & frame)
{
  cv::Mat hsv_frame;
  cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

  cv::Mat mask;
  // 当前视频里的绿球和绿色圆柱颜色接近，先用较宽的绿色阈值保留目标区域。
  cv::inRange(hsv_frame, cv::Scalar(35, 45, 40), cv::Scalar(95, 255, 255), mask);

  // 开闭运算用于去除零散噪声，并填补球体区域内部的小空洞。
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<BallCandidate> candidates;
  candidates.reserve(contours.size());

  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    // 贴边时绿球会因为裁切和透视变化变大，因此上限留得比中间帧更宽。
    if (area < 7000.0 || area > 32000.0) {
      continue;
    }

    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) {
      continue;
    }

    const cv::Rect bounding_box = cv::boundingRect(contour);
    if (bounding_box.height <= 0) {
      continue;
    }

    const double aspect_ratio =
      static_cast<double>(bounding_box.width) / static_cast<double>(bounding_box.height);
    const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    const bool near_frame_border = isNearFrameBorder(bounding_box, frame.size());

    // 绿色圆柱的投影通常更“瘦长”，这里利用圆度和长宽比把它与绿球区分开。
    if (!near_frame_border && (circularity < 0.68 || aspect_ratio < 0.85 || aspect_ratio > 1.20)) {
      continue;
    }

    // 靠近边缘时球体轮廓可能被裁切，因此放宽几何约束，但仍保留最低可信度限制。
    if (near_frame_border && (circularity < 0.45 || aspect_ratio < 0.55 || aspect_ratio > 1.45)) {
      continue;
    }

    cv::Point2f center;
    float radius = 0.0F;
    cv::minEnclosingCircle(contour, center, radius);

    candidates.push_back(BallCandidate{
      center,
      radius,
      area,
      circularity,
      aspect_ratio,
      bounding_box
    });
  }

  return candidates;
}

// 为每个候选目标计算形状分数，分数越高表示越接近“球”的外观特征。
double computeShapeScore(const BallCandidate & candidate)
{
  const double circularity_score = std::clamp((candidate.circularity - 0.68) / 0.12, 0.0, 1.0);
  const double aspect_score =
    std::clamp(1.0 - std::abs(candidate.aspect_ratio - 1.0) / 0.20, 0.0, 1.0);
  const double area_score = std::clamp(1.0 - std::abs(candidate.area - 12000.0) / 10000.0, 0.0, 1.0);
  return 0.45 * circularity_score + 0.35 * aspect_score + 0.20 * area_score;
}

std::optional<BallCandidate> chooseBestCandidate(
  const std::vector<BallCandidate> & candidates, const BallTrack & previous_track)
{
  if (candidates.empty()) {
    return std::nullopt;
  }

  double best_score = -std::numeric_limits<double>::infinity();
  const BallCandidate * best_candidate = nullptr;

  for (const auto & candidate : candidates) {
    double score = computeShapeScore(candidate);

    if (previous_track.initialized) {
      // 结合上一帧速度做一个简单的位置预测，优先选择运动连续的目标。
      const cv::Point2f predicted_center = previous_track.center + previous_track.velocity;
      const double distance = cv::norm(candidate.center - predicted_center);
      const double max_allowed_distance =
        std::max(previous_track.radius * 1.8, previous_track.speed * 3.5 + 35.0);

      // 若候选目标与预测位置相差过大，说明大概率已经跳到了别的绿色物体上，直接拒绝。
      if (distance > max_allowed_distance) {
        continue;
      }

      score -= distance * 0.01;

      // 半径变化过大通常意味着跟丢或误匹配，因此增加惩罚项。
      const double radius_gap = std::abs(candidate.radius - previous_track.radius);
      if (radius_gap > previous_track.radius * 0.45) {
        continue;
      }
      score -= radius_gap * 0.02;
    }

    if (score > best_score) {
      best_score = score;
      best_candidate = &candidate;
    }
  }

  if (best_candidate == nullptr) {
    return std::nullopt;
  }

  return *best_candidate;
}

// 生成速度状态文本，用于展示当前速度以及加减速趋势。
std::string buildStatusText(double speed_pixels_per_second, double speed_delta_pixels_per_second)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << "speed: " << speed_pixels_per_second << " px/s";

  if (std::abs(speed_delta_pixels_per_second) < 1e-3) {
    stream << "  stable";
  } else if (speed_delta_pixels_per_second > 0.0) {
    stream << "  accelerating";
  } else {
    stream << "  decelerating";
  }

  return stream.str();
}

// 在当前帧绘制检测圆、速度箭头和状态文本。
void drawTrackOverlay(
  cv::Mat & frame, const BallTrack & current_track, const BallTrack & previous_track,
  double fps, int frame_index, bool is_predicted)
{
  cv::circle(frame, current_track.center, static_cast<int>(std::round(current_track.radius)),
    cv::Scalar(0, 255, 255), 2);
  cv::circle(frame, current_track.center, 4, cv::Scalar(0, 0, 255), -1);

  const double speed_pixels_per_second = current_track.speed * fps;
  const double previous_speed_pixels_per_second = previous_track.speed * fps;
  const double speed_delta_pixels_per_second =
    speed_pixels_per_second - previous_speed_pixels_per_second;

  cv::Point2f arrow_vector = current_track.velocity;
  if (cv::norm(arrow_vector) < 1.0F) {
    arrow_vector = cv::Point2f(30.0F, 0.0F);
  }

  // 箭头长度随速度变化，既体现方向也体现快慢，但限制上下界避免观感失衡。
  const double arrow_scale = std::clamp(current_track.speed * 4.5, 25.0, 140.0);
  const double velocity_norm = cv::norm(arrow_vector);
  const cv::Point2f arrow_direction = arrow_vector * static_cast<float>(arrow_scale / velocity_norm);
  const cv::Point2f arrow_end = current_track.center + arrow_direction;

  cv::arrowedLine(
    frame, current_track.center, arrow_end,
    cv::Scalar(kArrowColorBlue, kArrowColorGreen, kArrowColorRed), 3, cv::LINE_AA, 0, 0.28);

  const cv::Point info_anchor(
    std::max(20, current_track.bounding_box.x - 20),
    std::max(40, current_track.bounding_box.y - 16));

  cv::putText(
    frame, is_predicted ? "green ball predicted" : "green ball", info_anchor,
    cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
  cv::putText(
    frame, buildStatusText(speed_pixels_per_second, speed_delta_pixels_per_second),
    info_anchor + cv::Point(0, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::putText(
    frame, "frame: " + std::to_string(frame_index), cv::Point(20, 36),
    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

}  // namespace

int main(int argc, char ** argv)
{
  // 从 homework/qihuanye 目录运行，读取仓库根目录的视频，结果写入当前目录。
  const std::filesystem::path default_input = "../../homework.mp4";
  const std::filesystem::path default_output = "homework-annotated.mp4";

  const std::filesystem::path input_path = argc > 1 ? argv[1] : default_input;
  const std::filesystem::path output_path = argc > 2 ? argv[2] : default_output;

  cv::VideoCapture capture(input_path.string());
  if (!capture.isOpened()) {
    std::cerr << "Failed to open input video: " << input_path << std::endl;
    return 1;
  }

  const double fps = capture.get(cv::CAP_PROP_FPS);
  const int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  cv::VideoWriter writer(
    output_path.string(),
    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps > 0.0 ? fps : 30.0,
    cv::Size(frame_width, frame_height));
  if (!writer.isOpened()) {
    std::cerr << "Failed to create output video: " << output_path << std::endl;
    return 1;
  }

  BallTrack previous_track;
  int frame_index = 0;

  // 逐帧处理视频：检测绿球、更新速度并写入标注结果。
  for (cv::Mat frame; capture.read(frame); ++frame_index) {
    const auto candidates = detectBallCandidates(frame);
    const auto selected_candidate = chooseBestCandidate(candidates, previous_track);

    if (selected_candidate.has_value()) {
      BallTrack current_track;
      current_track.center = selected_candidate->center;
      current_track.radius = selected_candidate->radius;
      current_track.initialized = true;
      current_track.bounding_box = selected_candidate->bounding_box;
      current_track.lost_frames = 0;

      if (previous_track.initialized) {
        current_track.velocity = current_track.center - previous_track.center;
        current_track.speed = cv::norm(current_track.velocity);
      } else {
        current_track.velocity = cv::Point2f(0.0F, 0.0F);
        current_track.speed = 0.0;
      }

      drawTrackOverlay(frame, current_track, previous_track, fps, frame_index, false);
      previous_track = current_track;
    } else if (previous_track.initialized && previous_track.lost_frames < kMaxLostFrames) {
      // 短时间丢失时使用上一帧速度做位置外推，避免目标贴边时直接中断显示。
      BallTrack predicted_track = previous_track;
      predicted_track.center = clampPointToFrame(
        previous_track.center + previous_track.velocity, frame.size());
      predicted_track.bounding_box.x =
        static_cast<int>(std::round(predicted_track.center.x - predicted_track.radius));
      predicted_track.bounding_box.y =
        static_cast<int>(std::round(predicted_track.center.y - predicted_track.radius));
      predicted_track.bounding_box.width = static_cast<int>(std::round(predicted_track.radius * 2.0F));
      predicted_track.bounding_box.height = static_cast<int>(std::round(predicted_track.radius * 2.0F));
      predicted_track.lost_frames = previous_track.lost_frames + 1;

      drawTrackOverlay(frame, predicted_track, previous_track, fps, frame_index, true);
      previous_track = predicted_track;
    } else {
      // 当前帧未找到可信目标时保留提示，方便后续调参排查。
      cv::putText(
        frame, "green ball lost", cv::Point(20, 36), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      previous_track = BallTrack{};
    }

    writer.write(frame);
  }

  std::cout << "Annotated video written to: " << output_path << std::endl;
  return 0;
}
