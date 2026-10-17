// Mortis/src/main.cpp
// 绿球识别与运动追踪（Week1 作业）
//
// 目标：从视频中检测绿色小球，用圆形框标出位置，用箭头表示运动方向，
//       箭头长度表示速度大小，并标注球的大小。
//
// 处理流水线（每帧执行）：
//   1. 绿色分割：BGR -> HSV，阈值提取绿色区域，开+闭运算去噪
//   2. 椭圆拟合：对每个绿色轮廓 fitEllipse，得到圆心和长短轴
//   3. 形状筛选：用椭圆短轴/长轴（圆度）区分绿球（≈1）与绿色圆柱（细长）
//   4. 跨帧匹配：圆度打分 + 卡尔曼滤波预测位置，抑制目标跳变
//   5. 可视化：检测圆 + 固定红色速度箭头 + 球大小/速度文本

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>  // cv::KalmanFilter 的声明

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// 当前作业视频的检测与绘制参数
// ---------------------------------------------------------------------------

// 绿色 HSV 范围（8 位 HSV 图像中，OpenCV 的 H 通常为 [0,179]；范围需按实际视频调试）
const cv::Scalar kLowerGreen(35, 45, 40);
const cv::Scalar kUpperGreen(95, 255, 255);

// 面积下限：仅用于滤除细小噪点，真正的球/圆柱区分交给椭圆圆度
constexpr double kMinArea = 2000.0;

// 椭圆拟合的圆度阈值（短轴/长轴）：正圆≈1，细长≈0，用于区分绿球与绿色圆柱
constexpr double kMinEllipseRoundness = 0.80;
constexpr double kTypicalBallRadius = 60.0;
constexpr double kRadiusTolerance = 40.0;

// 跨帧匹配惩罚与门限
constexpr double kDistancePenalty = 0.02;
constexpr double kRadiusPenalty = 0.03;
constexpr double kMaxRadiusChangeRatio = 0.50;

// 速度箭头：长度 = clamp(速度 × kArrowScale, kArrowMinLen, kArrowMaxLen)
constexpr double kArrowScale = 4.5;
constexpr double kArrowMinLen = 25.0;
constexpr double kArrowMaxLen = 140.0;

// 速度低于此阈值时不画箭头，避免静止目标仍显示一个虚假的方向
constexpr double kMinArrowSpeed = 1.0;

// 卡尔曼滤波噪声参数。数值越大，表示越相信模型/测量中的不确定性。
// 加速度过程噪声越大：越愿意快速改变速度，转向时跟得更快，但也可能更抖。
constexpr float kKalmanAccelerationNoise = 1.0F;
// 检测中心点的测量噪声方差（像素^2）；这里假设检测中心的标准差约为 4 像素。
// 调大该值会更信任运动模型、降低检测抖动；调小则更贴近每帧检测位置。
constexpr float kKalmanMeasurementNoiseVariance = 16.0F;

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

// 单个绿色轮廓的几何信息
struct BallCandidate {
  cv::Point2f center;
  float radius = 0.0F;      // 椭圆长短轴平均值的一半（像素）
  double roundness = 0.0;   // 椭圆拟合短轴/长轴，1 表示正圆
};

// 当前帧的跟踪状态
struct BallTrack {
  cv::Point2f center;
  cv::Point2f velocity;         // 卡尔曼滤波估计的速度（像素/帧）
  float radius = 0.0F;
  double speed = 0.0;           // 卡尔曼滤波估计的速度大小（像素/帧）
};

// 匀速卡尔曼滤波：[x, y, vx, vy]，观测为 [x, y]，时间单位为帧。
// 初始化后，每帧先 predict()，再用检测位置 correct()；漏检时重置。
class BallKalmanTracker {
public:
  BallKalmanTracker() : filter_(4, 2, 0, CV_32F)
  {
    // 状态转移矩阵 A：x' = x + vx、y' = y + vy，速度暂按匀速保持。
    filter_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
      1.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 1.0F, 0.0F, 1.0F,
      0.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 0.0F, 1.0F);

    // 观测矩阵 H：检测器只提供 x、y，不直接提供 vx、vy。
    filter_.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    filter_.measurementMatrix.at<float>(0, 0) = 1.0F;
    filter_.measurementMatrix.at<float>(1, 1) = 1.0F;

    // 过程噪声协方差 Q：用白噪声加速度模型描述目标运动并非完全匀速。
    // dt = 1 帧时，单个坐标对应的矩阵项为 [dt^4/4, dt^3/2; dt^3/2, dt^2]。
    const float q = kKalmanAccelerationNoise;
    filter_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
      0.25F * q, 0.0F,       0.5F * q, 0.0F,
      0.0F,       0.25F * q, 0.0F,       0.5F * q,
      0.5F * q,   0.0F,       q,          0.0F,
      0.0F,       0.5F * q,   0.0F,       q);

    // 测量噪声协方差 R：检测中心可能因轮廓边缘变化而有几像素抖动。
    filter_.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
      kKalmanMeasurementNoiseVariance, 0.0F,
      0.0F, kKalmanMeasurementNoiseVariance);
  }

  bool isInitialized() const { return initialized_; }

  // 用首次检测结果初始化位置；初始速度设为 0，因为单个观测无法推断运动方向。
  void initialize(const cv::Point2f & center)
  {
    filter_.statePost = cv::Mat::zeros(4, 1, CV_32F);
    filter_.statePost.at<float>(0, 0) = center.x;
    filter_.statePost.at<float>(1, 0) = center.y;

    // 初始位置的不确定性设为 25（标准差约 5 像素），速度不确定性设为 100。
    filter_.errorCovPost = cv::Mat::zeros(4, 4, CV_32F);
    filter_.errorCovPost.at<float>(0, 0) = 25.0F;
    filter_.errorCovPost.at<float>(1, 1) = 25.0F;
    filter_.errorCovPost.at<float>(2, 2) = 100.0F;
    filter_.errorCovPost.at<float>(3, 3) = 100.0F;

    initialized_ = true;
  }

  // 预测当前帧的位置，供候选匹配和后续测量校正使用。
  cv::Point2f predict()
  {
    const cv::Mat prediction = filter_.predict();
    return cv::Point2f(prediction.at<float>(0, 0), prediction.at<float>(1, 0));
  }

  // 有真实检测时，用检测到的中心修正当前帧的预测结果。
  cv::Point2f correct(const cv::Point2f & measured_center)
  {
    if (!initialized_) {
      initialize(measured_center);
      return measured_center;
    }

    const cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
      measured_center.x, measured_center.y);
    const cv::Mat estimate = filter_.correct(measurement);
    return cv::Point2f(estimate.at<float>(0, 0), estimate.at<float>(1, 0));
  }

  // 返回当前滤波状态里的速度估计（像素/帧）。
  cv::Point2f velocity() const
  {
    if (!initialized_) {
      return cv::Point2f();
    }
    const cv::Mat & state = filter_.statePost;
    return cv::Point2f(state.at<float>(2, 0), state.at<float>(3, 0));
  }

  void reset()
  {
    initialized_ = false;
  }

private:
  cv::KalmanFilter filter_;
  bool initialized_ = false;
};

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

// 检测绿球候选：HSV 分割 -> 椭圆拟合 -> 圆度筛选
std::vector<BallCandidate> detectBallCandidates(const cv::Mat & frame)
{
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask;
  cv::inRange(hsv, kLowerGreen, kUpperGreen, mask);

  // 开运算去噪点，闭运算填空洞。结构元素固定，因此只创建一次并复用。
  static const cv::Mat kernel =
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<BallCandidate> candidates;
  candidates.reserve(contours.size());

  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < kMinArea || contour.size() < 5) {
      continue;
    }

    // fitEllipse 对轮廓做最小二乘椭圆拟合，同时给出圆心、长短轴与朝向。
    const cv::RotatedRect ellipse = cv::fitEllipse(contour);
    const float major = std::max(ellipse.size.width, ellipse.size.height);
    const float minor = std::min(ellipse.size.width, ellipse.size.height);
    const double roundness = (major > 0.0F) ? static_cast<double>(minor) / major : 0.0;

    // 绿色圆柱是细长目标，椭圆短轴/长轴会明显小于正圆，这里直接排除。
    if (roundness < kMinEllipseRoundness) {
      continue;
    }

    const cv::Point2f center = ellipse.center;
    const float radius = 0.25F * (major + minor);  // 长短轴平均值的一半 ≈ 圆半径

    candidates.push_back(BallCandidate{center, radius, roundness});
  }

  return candidates;
}

// 形状打分：椭圆圆度为主，半径接近典型球尺寸为辅。
double computeShapeScore(const BallCandidate & c)
{
  const double roundness_score = std::clamp(
    (c.roundness - kMinEllipseRoundness) / (1.0 - kMinEllipseRoundness), 0.0, 1.0);
  const double radius_score = std::clamp(
    1.0 - std::abs(c.radius - kTypicalBallRadius) / kRadiusTolerance, 0.0, 1.0);
  return 0.7 * roundness_score + 0.3 * radius_score;
}

// 从候选中选出最可能的目标：形状分 + 卡尔曼预测位置的连续性。
// predicted_center 是当前帧的先验预测位置；尚未初始化滤波器时为空。
std::optional<BallCandidate> chooseBestCandidate(
  const std::vector<BallCandidate> & candidates, const BallTrack & previous,
  const std::optional<cv::Point2f> & predicted_center)
{
  if (candidates.empty()) {
    return std::nullopt;
  }

  double best_score = -std::numeric_limits<double>::infinity();
  const BallCandidate * best = nullptr;

  for (const auto & c : candidates) {
    double score = computeShapeScore(c);

    if (predicted_center.has_value()) {
      const double distance = cv::norm(c.center - *predicted_center);
      const double gate = std::max(previous.radius * 2.0, previous.speed * 3.0 + 40.0);

      // 距离预测位置太远，可能是误选了另一块绿色区域。
      if (distance > gate) {
        continue;
      }
      score -= distance * kDistancePenalty;

      // 半径突变过大 => 误匹配，拒绝
      const double radius_gap = std::abs(c.radius - previous.radius);
      if (radius_gap > previous.radius * kMaxRadiusChangeRatio) {
        continue;
      }
      score -= radius_gap * kRadiusPenalty;
    }

    if (score > best_score) {
      best_score = score;
      best = &c;
    }
  }

  if (best == nullptr) {
    return std::nullopt;
  }
  return *best;
}

// 所有文字均为英文，使用 OpenCV 内置字体；org 为文本左上角。
void drawText(cv::Mat & frame, const std::string & text, cv::Point org,
  int font_height, cv::Scalar color, int thickness)
{
  cv::putText(frame, text, org + cv::Point(0, font_height), cv::FONT_HERSHEY_DUPLEX,
    static_cast<double>(font_height) / 28.0, color, thickness, cv::LINE_AA);
}

// 绘制检测圆、速度箭头和状态文本
void drawTrackOverlay(cv::Mat & frame, const BallTrack & current, double fps, int frame_index)
{
  cv::circle(frame, current.center, static_cast<int>(std::round(current.radius)),
    cv::Scalar(0, 255, 255), 2);
  cv::circle(frame, current.center, 4, cv::Scalar(0, 0, 255), -1);

  const double speed_pps = current.speed * fps;

  // 只有速度和方向都可信时才画箭头；
  const cv::Point2f arrow = current.velocity;
  const double arrow_norm = cv::norm(arrow);
  if (current.speed >= kMinArrowSpeed && arrow_norm > 1e-6) {
    const double len = std::clamp(current.speed * kArrowScale, kArrowMinLen, kArrowMaxLen);
    const cv::Point2f dir = arrow * static_cast<float>(len / arrow_norm);
    const cv::Point2f end = current.center + dir;
    cv::arrowedLine(frame, current.center, end, cv::Scalar(0, 0, 255),
      3, cv::LINE_AA, 0, 0.28);
  }

  // 文字固定绘制在左上角，避免遮挡运动中的小球。
  constexpr int kTextLeft = 20;
  constexpr int kTextLine = 40;
  int text_y = 32;

  drawText(frame, "frame: " + std::to_string(frame_index),
    cv::Point(kTextLeft, text_y), 22, cv::Scalar(255, 255, 255), 2);
  text_y += kTextLine;

  std::ostringstream status;
  status << std::fixed << std::setprecision(1)
         << "speed: " << speed_pps << " px/s";
  drawText(frame, status.str(), cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 255), 2);
  text_y += kTextLine;

  // 球的大小标注（图像中的像素半径，不是现实世界长度）
  const std::string size_text =
    "r: " + std::to_string(static_cast<int>(std::round(current.radius))) + " px";
  drawText(frame, size_text, cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 0), 2);
}

}  // namespace

int main()
{
  // 从仓库根目录运行，仅处理本次作业的视频。
  const std::string input_path = "homework.mp4";
  const std::string output_path = "homework/Mortis/output.mp4";

  cv::VideoCapture capture(input_path);
  if (!capture.isOpened()) {
    std::cerr << "无法打开输入视频: " << input_path << std::endl;
    return 1;
  }

  const double fps = capture.get(cv::CAP_PROP_FPS);
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

  cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps, cv::Size(width, height));
  if (!writer.isOpened()) {
    std::cerr << "无法创建输出视频: " << output_path << std::endl;
    return 1;
  }

  BallTrack previous;
  BallKalmanTracker kalman_tracker;
  int frame_index = 0;

  for (cv::Mat frame; capture.read(frame); ++frame_index) {
    // 先预测当前帧的小球位置，再用这个位置辅助候选匹配。
    std::optional<cv::Point2f> predicted_center;
    if (kalman_tracker.isInitialized()) {
      predicted_center = kalman_tracker.predict();
    }

    const auto candidates = detectBallCandidates(frame);
    const auto selected = chooseBestCandidate(candidates, previous, predicted_center);

    if (selected.has_value()) {
      BallTrack current;
      current.radius = selected->radius;

      current.center = kalman_tracker.correct(selected->center);
      current.velocity = kalman_tracker.velocity();
      current.speed = cv::norm(current.velocity);

      drawTrackOverlay(frame, current, fps, frame_index);
      previous = current;
    } else {
      // 漏检时不绘制预测目标，重置后由下一次检测重新初始化。
      kalman_tracker.reset();
      drawText(frame, "green ball lost", cv::Point(20, 36), 28, cv::Scalar(0, 0, 255), 2);
      previous = BallTrack{};
    }

    writer.write(frame);
  }

  std::cout << "已写入标注视频: " << output_path
            << "（处理帧数: " << frame_index << ")" << std::endl;
  return 0;
}
