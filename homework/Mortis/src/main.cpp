// first-homework/Mortis/src/main.cpp
// 绿球识别与运动追踪（Week1 作业）
//
// 目标：从视频中检测绿色小球，用圆形框标出位置，用箭头表示运动方向，
//       箭头长度表示速度大小、颜色随速度变化，并标注球的大小。
//
// 处理流水线（每帧执行）：
//   1. 绿色分割：BGR -> HSV，阈值提取绿色区域，开+闭运算去噪
//   2. 椭圆拟合：对每个绿色轮廓 fitEllipse，直接得到圆心、长短轴与朝向
//   3. 形状筛选：用椭圆短轴/长轴（圆度）区分绿球（≈1）与绿色圆柱（细长）
//   4. 跨帧匹配：圆度打分 + 卡尔曼滤波预测位置，抑制目标跳变
//   5. 可视化：检测圆 + 速度箭头(随速度变色) + 球大小/速度文本 + 运动轨迹

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>  // cv::KalmanFilter 的声明
#include <opencv2/freetype.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// 可调参数（换视频时主要改这里）
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

// 短时遮挡：连续丢失帧数不超过该值时用速度外推，否则重置
constexpr int kMaxLostFrames = 8;

// 速度箭头：长度 = clamp(速度 × kArrowScale, kArrowMinLen, kArrowMaxLen)
constexpr double kArrowScale = 4.5;
constexpr double kArrowMinLen = 25.0;
constexpr double kArrowMaxLen = 140.0;

// 运动轨迹：保留最近 N 帧位置绘制"彗尾"
constexpr int kMaxTrail = 30;

// 速度变色参考值（像素/帧）：速度达到该值时箭头显示为红色
constexpr double kFastSpeedRef = 30.0;

// 速度低于此阈值时不画箭头，避免静止目标仍显示一个虚假的方向
constexpr double kMinArrowSpeed = 1.0;

// 小于该速度变化量时，状态文本显示 stable，以减少轻微抖动造成的闪烁
constexpr double kStableSpeedDeltaPps = 2.0;

// 元数据无法提供有效 FPS 时采用的显示/输出假设值；此时速度只是近似值
constexpr double kDefaultFps = 30.0;

// 卡尔曼滤波噪声参数。数值越大，表示越相信模型/测量中的不确定性。
// 加速度过程噪声越大：越愿意快速改变速度，转向时跟得更快，但也可能更抖。
constexpr float kKalmanAccelerationNoise = 1.0F;
// 检测中心点的测量噪声方差（像素^2）；这里假设检测中心的标准差约为 4 像素。
// 调大该值会更信任运动模型、降低检测抖动；调小则更贴近每帧检测位置。
constexpr float kKalmanMeasurementNoiseVariance = 16.0F;

// 黑体（SimHei）候选路径：优先 Windows 字体（WSL 可访问），失败回退到内置字体
const char * const kFontPaths[] = {
  "/mnt/c/Windows/Fonts/simhei.ttf",
  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
};

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

// 单个绿色轮廓的几何信息
struct BallCandidate {
  cv::Point2f center;
  float radius = 0.0F;      // 霍夫圆检测的半径（像素）
  double roundness = 0.0;   // 椭圆拟合短轴/长轴，1 表示正圆
  cv::Rect bounding_box;
};

// 当前帧的跟踪状态
struct BallTrack {
  cv::Point2f center;
  cv::Point2f velocity;         // 卡尔曼滤波估计的速度（像素/帧）
  cv::Point2f smooth_velocity;  // 保留原字段名以兼容绘制代码；现在同样保存滤波后的速度
  float radius = 0.0F;
  double speed = 0.0;           // 卡尔曼滤波估计的速度大小（像素/帧）
  double confidence = 0.0;      // 检测置信度（0~1），来自形状打分
  cv::Rect bounding_box;
  int lost_frames = 0;
  bool initialized = false;
};

// ---------------------------------------------------------------------------
// 卡尔曼滤波器：使用匀速模型估计小球中心位置和速度
// ---------------------------------------------------------------------------
// 状态向量 x = [位置 x, 位置 y, 速度 vx, 速度 vy]^T，共 4 个状态量。
// 观测向量 z = [检测到的中心 x, 检测到的中心 y]^T，只直接观测位置。
//
// 每帧执行顺序：
//   1. predict()：根据上一时刻的位置和速度预测当前帧的位置；
//   2. 若检测到小球，correct() 用实际检测中心修正预测；
//   3. 若本帧没有检测到小球，acceptPrediction() 将预测结果作为新状态，
//      下一帧就能从本帧预测位置继续预测，而不是反复从最后一次检测位置出发。
//
// 当前按“每帧”为时间单位，因此速度单位是像素/帧。假设相邻帧之间短时间内近似匀速；
// 目标突然转向时，过程噪声会允许滤波器逐步修正速度。
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
    // 这能让速度在连续观测到转向或加速时逐步发生变化。
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

    initialized_ = false;
    prediction_pending_ = false;
  }

  bool isInitialized() const { return initialized_; }

  // 用首次检测结果初始化位置；初始速度设为 0，因为单个观测无法推断运动方向。
  void initialize(const cv::Point2f & center)
  {
    filter_.statePost = cv::Mat::zeros(4, 1, CV_32F);
    filter_.statePost.at<float>(0, 0) = center.x;
    filter_.statePost.at<float>(1, 0) = center.y;
    filter_.statePost.at<float>(2, 0) = 0.0F;
    filter_.statePost.at<float>(3, 0) = 0.0F;

    // 初始位置的不确定性设为 25（标准差约 5 像素），速度不确定性设为 100。
    // 较大的速度初始方差可让后续几帧的观测快速建立合理速度估计。
    filter_.errorCovPost = cv::Mat::zeros(4, 4, CV_32F);
    filter_.errorCovPost.at<float>(0, 0) = 25.0F;
    filter_.errorCovPost.at<float>(1, 1) = 25.0F;
    filter_.errorCovPost.at<float>(2, 2) = 100.0F;
    filter_.errorCovPost.at<float>(3, 3) = 100.0F;

    initialized_ = true;
    prediction_pending_ = false;
  }

  // 预测当前帧状态。调用后必须接着调用 correct() 或 acceptPrediction()。
  cv::Point2f predict()
  {
    if (!initialized_) {
      return cv::Point2f();
    }

    // predict() 的结果放在 statePre 中，包含预测位置和预测速度。
    const cv::Mat prediction = filter_.predict();
    prediction_pending_ = true;
    return cv::Point2f(prediction.at<float>(0, 0), prediction.at<float>(1, 0));
  }

  // 有真实检测时，用检测到的中心修正当前帧的预测结果。
  cv::Point2f correct(const cv::Point2f & measured_center)
  {
    if (!initialized_) {
      initialize(measured_center);
      return measured_center;
    }

    // 正常流程会在本帧先调用 predict()，然后调用 correct()。
    // 此保护分支用于避免调用顺序意外时直接使用过期预测状态。
    if (!prediction_pending_) {
      (void)filter_.predict();
      prediction_pending_ = true;
    }

    const cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
      measured_center.x, measured_center.y);
    const cv::Mat estimate = filter_.correct(measurement);
    prediction_pending_ = false;
    return cv::Point2f(estimate.at<float>(0, 0), estimate.at<float>(1, 0));
  }

  // 本帧没有观测时，接受预测状态，以便下一帧从预测结果继续向前推算。
  void acceptPrediction()
  {
    if (!initialized_ || !prediction_pending_) {
      return;
    }
    filter_.statePre.copyTo(filter_.statePost);
    filter_.errorCovPre.copyTo(filter_.errorCovPost);
    prediction_pending_ = false;
  }

  // 返回当前滤波状态里的速度估计（像素/帧）。
  cv::Point2f velocity() const
  {
    if (!initialized_) {
      return cv::Point2f();
    }
    const cv::Mat & state = prediction_pending_ ? filter_.statePre : filter_.statePost;
    return cv::Point2f(state.at<float>(2, 0), state.at<float>(3, 0));
  }

  void reset()
  {
    initialized_ = false;
    prediction_pending_ = false;
  }

private:
  cv::KalmanFilter filter_;
  bool initialized_ = false;
  bool prediction_pending_ = false;
};

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

// 把点限制在画面内（外推时避免越界）
cv::Point2f clampPointToFrame(const cv::Point2f & p, const cv::Size & frame_size)
{
  const float x = std::clamp(p.x, 0.0F, static_cast<float>(frame_size.width - 1));
  const float y = std::clamp(p.y, 0.0F, static_cast<float>(frame_size.height - 1));
  return cv::Point2f(x, y);
}

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
    if (area < kMinArea) {
      continue;
    }
    if (contour.size() < 5) {
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

    const cv::Rect box(static_cast<int>(std::round(center.x - radius)),
                       static_cast<int>(std::round(center.y - radius)),
                       static_cast<int>(std::round(radius * 2.0F)),
                       static_cast<int>(std::round(radius * 2.0F)));

    candidates.push_back(BallCandidate{center, radius, roundness, box});
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
  const std::optional<cv::Point2f> & predicted_center, double * confidence = nullptr)
{
  if (candidates.empty()) {
    return std::nullopt;
  }

  double best_score = -std::numeric_limits<double>::infinity();
  const BallCandidate * best = nullptr;

  for (const auto & c : candidates) {
    double score = computeShapeScore(c);

    if (previous.initialized && predicted_center.has_value()) {
      // 用卡尔曼滤波器的先验预测位置计算距离，而不是简单地把上一帧位移加一次。
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
  if (confidence != nullptr) {
    *confidence = computeShapeScore(*best);
  }
  return *best;
}

// 生成速度状态文本（速度 + 加减速趋势）
std::string buildStatusText(double speed_pps, double delta_pps)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << "speed: " << speed_pps << " px/s";
  if (std::abs(delta_pps) < kStableSpeedDeltaPps) {
    ss << "  stable";
  } else if (delta_pps > 0.0) {
    ss << "  accelerating";
  } else {
    ss << "  decelerating";
  }
  return ss.str();
}

// 速度 → 箭头颜色：绿(慢) → 黄(中) → 红(快)
cv::Scalar speedToColor(double speed)
{
  const double t = std::clamp(speed / kFastSpeedRef, 0.0, 1.0);
  const int red = static_cast<int>(255.0 * std::min(1.0, 2.0 * t));
  const int green = static_cast<int>(255.0 * std::min(1.0, 2.0 * (1.0 - t)));
  return cv::Scalar(0, green, red);  // BGR
}

// 惰性加载黑体（SimHei）；成功返回字体对象，失败返回空指针（回退到内置字体）
cv::Ptr<cv::freetype::FreeType2> getFont()
{
  static const cv::Ptr<cv::freetype::FreeType2> font = []() {
    auto f = cv::freetype::createFreeType2();
    for (const char * path : kFontPaths) {
      std::error_code ec;
      if (std::filesystem::exists(path, ec) && !ec) {
        f->loadFontData(path, 0);
        return f;
      }
    }
    return cv::Ptr<cv::freetype::FreeType2>();
  }();
  return font;
}

// 统一文字绘制：优先黑体，失败回退 OpenCV 内置字体（org 均为文本左上角）
void drawText(cv::Mat & frame, const std::string & text, cv::Point org,
  int font_height, cv::Scalar color, int thickness)
{
  const auto font = getFont();
  if (font) {
    font->putText(frame, text, org, font_height, color, thickness, cv::LINE_AA, false);
  } else {
    const double scale = static_cast<double>(font_height) / 28.0;
    cv::putText(frame, text, org + cv::Point(0, font_height), cv::FONT_HERSHEY_DUPLEX,
      scale, color, thickness, cv::LINE_AA);
  }
}

// 绘制 YOLO 风格标签：目标框上方背景色块 + 「类名 置信度」白字
void drawYoloLabel(cv::Mat & frame, const std::string & label, const cv::Rect & box,
  cv::Scalar bg_color, int font_height)
{
  const auto font = getFont();
  const int thickness = -1;  // 实心填充

  cv::Size text_size;
  if (font) {
    int baseline = 0;
    text_size = font->getTextSize(label, font_height, thickness, &baseline);
  } else {
    int baseline = 0;
    text_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX,
      static_cast<double>(font_height) / 28.0, 2, &baseline);
  }

  const int pad_x = 8;
  const int pad_y = 4;
  cv::Point tl(box.x, box.y - text_size.height - pad_y * 2);
  if (tl.y < 0) {
    tl.y = box.y + 2;  // 贴顶边时放到框内，避免越界
  }
  cv::rectangle(frame, cv::Rect(tl.x, tl.y, text_size.width + pad_x * 2,
    text_size.height + pad_y * 2), bg_color, cv::FILLED);

  if (font) {
    font->putText(frame, label, cv::Point(tl.x + pad_x, tl.y + pad_y), font_height,
      cv::Scalar(255, 255, 255), thickness, cv::LINE_AA, false);
  } else {
    cv::putText(frame, label, cv::Point(tl.x + pad_x, tl.y + pad_y + text_size.height),
      cv::FONT_HERSHEY_DUPLEX, static_cast<double>(font_height) / 28.0,
      cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

// 绘制运动轨迹（彗尾）：旧点偏绿、新点偏红
void drawTrail(cv::Mat & frame, const std::deque<cv::Point2f> & trail)
{
  if (trail.size() < 2) {
    return;
  }
  for (size_t i = 1; i < trail.size(); ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(trail.size() - 1);
    const cv::Scalar color(0, static_cast<int>(255.0 * (1.0 - t)), static_cast<int>(255.0 * t));
    cv::line(frame, trail[i - 1], trail[i], color, 2, cv::LINE_AA);
  }
}

// 绘制检测圆、速度箭头和状态文本
void drawTrackOverlay(cv::Mat & frame, const BallTrack & current, const BallTrack & previous,
  double fps, int frame_index, bool predicted)
{
  cv::circle(frame, current.center, static_cast<int>(std::round(current.radius)),
    cv::Scalar(0, 255, 255), 2);
  cv::circle(frame, current.center, 4, cv::Scalar(0, 0, 255), -1);

  const double speed_pps = current.speed * fps;
  const double prev_speed_pps = previous.speed * fps;
  const double delta_pps = speed_pps - prev_speed_pps;

  // 只有速度和方向都可信时才画箭头；静止时不伪造一个向右的方向。
  const cv::Point2f arrow = current.smooth_velocity;
  const double arrow_norm = cv::norm(arrow);
  if (current.speed >= kMinArrowSpeed && arrow_norm > 1e-6) {
    const double len = std::clamp(current.speed * kArrowScale, kArrowMinLen, kArrowMaxLen);
    const cv::Point2f dir = arrow * static_cast<float>(len / arrow_norm);
    const cv::Point2f end = current.center + dir;
    cv::arrowedLine(frame, current.center, end, speedToColor(current.speed),
      3, cv::LINE_AA, 0, 0.28);
  }

  // YOLO 风格标签：类名 + 置信度，贴在目标框上方（外推时用灰色背景区分）。
  {
    std::ostringstream label_ss;
    label_ss << "green ball " << std::fixed << std::setprecision(2) << current.confidence;
    const cv::Scalar label_bg = predicted ? cv::Scalar(110, 110, 110) : cv::Scalar(0, 180, 0);
    drawYoloLabel(frame, label_ss.str(), current.bounding_box, label_bg, 24);
  }

  // 其余文字固定绘制在左上角，避免遮挡运动中的小球。
  constexpr int kTextLeft = 20;
  constexpr int kTextLine = 40;
  int text_y = 32;

  drawText(frame, "frame: " + std::to_string(frame_index),
    cv::Point(kTextLeft, text_y), 22, cv::Scalar(255, 255, 255), -1);
  text_y += kTextLine;

  // 外推期间的速度只是沿用上次观测值；重获目标的第一帧也不计算跨越丢失区间的瞬时速度。
  std::string status_text;
  if (predicted) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1)
       << "speed: ~" << speed_pps << " px/s  predicted";
    status_text = ss.str();
  } else if (previous.lost_frames > 0) {
    status_text = "speed: recalibrating";
  } else {
    status_text = buildStatusText(speed_pps, delta_pps);
  }
  drawText(frame, status_text, cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 255), -1);
  text_y += kTextLine;

  // 球的大小标注（图像中的像素半径，不是现实世界长度）
  const std::string size_text =
    "r: " + std::to_string(static_cast<int>(std::round(current.radius))) + " px";
  drawText(frame, size_text, cv::Point(kTextLeft, text_y), 24,
    cv::Scalar(255, 255, 0), -1);
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::filesystem::path default_input = "first-homework/first-homework.mp4";
  const std::filesystem::path default_output =
    "first-homework/Mortis/first-homework-annotated.mp4";

  const std::filesystem::path input_path = argc > 1 ? argv[1] : default_input;
  const std::filesystem::path output_path = argc > 2 ? argv[2] : default_output;

  // 防止输入、输出指向同一路径，避免打开写入器时覆盖输入视频。
  try {
    const auto absolute_input = std::filesystem::absolute(input_path).lexically_normal();
    const auto absolute_output = std::filesystem::absolute(output_path).lexically_normal();
    if (absolute_input == absolute_output) {
      std::cerr << "输入视频和输出视频不能是同一个路径。" << std::endl;
      return 1;
    }
  } catch (const std::filesystem::filesystem_error & e) {
    std::cerr << "检查文件路径失败: " << e.what() << std::endl;
    return 1;
  }

  cv::VideoCapture capture(input_path.string());
  if (!capture.isOpened()) {
    std::cerr << "无法打开输入视频: " << input_path << std::endl;
    return 1;
  }

  const double reported_fps = capture.get(cv::CAP_PROP_FPS);
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

  if (width <= 0 || height <= 0) {
    std::cerr << "输入视频的宽度或高度无效: " << width << "x" << height << std::endl;
    return 1;
  }

  const bool fps_is_valid = std::isfinite(reported_fps) && reported_fps > 0.0;
  const double fps = fps_is_valid ? reported_fps : kDefaultFps;
  if (!fps_is_valid) {
    std::cerr << "警告：无法从视频读取有效 FPS，将按 " << kDefaultFps
              << " FPS 处理；显示的 px/s 速度可能不准确。" << std::endl;
  }

  // 仅当路径包含目录时创建父目录；输出为当前目录下的文件名时不传空路径。
  const std::filesystem::path output_parent = output_path.parent_path();
  if (!output_parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(output_parent, ec);
    if (ec) {
      std::cerr << "无法创建输出目录: " << output_parent
                << "，原因: " << ec.message() << std::endl;
      return 1;
    }
  }

  cv::VideoWriter writer(output_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps, cv::Size(width, height));
  if (!writer.isOpened()) {
    std::cerr << "无法创建输出视频: " << output_path
              << "。请检查路径、写入权限以及当前 OpenCV 的视频编码器支持。" << std::endl;
    return 1;
  }

  BallTrack previous;
  BallKalmanTracker kalman_tracker;  // 负责跨帧融合检测位置、估计速度并预测短时运动
  std::deque<cv::Point2f> trail;
  int frame_index = 0;

  for (cv::Mat frame; capture.read(frame); ++frame_index) {
    if (frame.empty()) {
      std::cerr << "警告：第 " << frame_index << " 帧为空，停止处理。" << std::endl;
      break;
    }

    // 先预测当前帧的小球位置，再用这个位置辅助候选匹配。
    // 如果滤波器还没初始化（视频开始或跟踪超时重置），则本帧没有预测中心。
    std::optional<cv::Point2f> predicted_center;
    if (kalman_tracker.isInitialized()) {
      predicted_center = kalman_tracker.predict();
    }

    const auto candidates = detectBallCandidates(frame);
    double confidence = 0.0;
    const auto selected = chooseBestCandidate(candidates, previous, predicted_center, &confidence);

    if (selected.has_value()) {
      BallTrack current;
      current.radius = selected->radius;
      current.bounding_box = selected->bounding_box;
      current.confidence = confidence;
      current.initialized = true;
      current.lost_frames = 0;

      if (kalman_tracker.isInitialized()) {
        // 有检测结果：使用观测位置修正本帧预测，得到更平滑的位置与速度估计。
        current.center = kalman_tracker.correct(selected->center);
      } else {
        // 第一次发现目标（或长时间丢失后重置）时，以检测中心初始化滤波器。
        kalman_tracker.initialize(selected->center);
        current.center = selected->center;
      }

      // 圆心使用滤波后的坐标；标签框也同步平移，避免框和圆心明显错位。
      const cv::Point2f center_offset = current.center - selected->center;
      current.bounding_box.x += static_cast<int>(std::round(center_offset.x));
      current.bounding_box.y += static_cast<int>(std::round(center_offset.y));

      // 卡尔曼状态同时估计位置和速度，因此不再需要额外的 EMA 速度平滑。
      current.velocity = kalman_tracker.velocity();
      current.smooth_velocity = current.velocity;
      current.speed = cv::norm(current.velocity);

      trail.push_back(current.center);
      if (trail.size() > kMaxTrail) {
        trail.pop_front();
      }
      drawTrail(frame, trail);
      drawTrackOverlay(frame, current, previous, fps, frame_index, false);
      previous = current;
    } else if (previous.initialized && previous.lost_frames < kMaxLostFrames &&
               predicted_center.has_value()) {
      // 短时丢失：没有实际观测，接受卡尔曼预测状态并继续外推。
      // 画面中显示的是预测位置，并非这一帧真实检测到了小球。
      kalman_tracker.acceptPrediction();
      BallTrack predicted = previous;
      predicted.center = clampPointToFrame(*predicted_center, frame.size());
      predicted.velocity = kalman_tracker.velocity();
      predicted.smooth_velocity = predicted.velocity;
      predicted.speed = cv::norm(predicted.velocity);
      predicted.bounding_box.x = static_cast<int>(std::round(predicted.center.x - predicted.radius));
      predicted.bounding_box.y = static_cast<int>(std::round(predicted.center.y - predicted.radius));
      predicted.bounding_box.width = static_cast<int>(std::round(predicted.radius * 2.0F));
      predicted.bounding_box.height = static_cast<int>(std::round(predicted.radius * 2.0F));
      predicted.lost_frames = previous.lost_frames + 1;

      trail.push_back(predicted.center);
      if (trail.size() > kMaxTrail) {
        trail.pop_front();
      }
      drawTrail(frame, trail);
      drawTrackOverlay(frame, predicted, previous, fps, frame_index, true);
      previous = predicted;
    } else {
      // 连续丢失超过允许帧数，清空轨迹并重置滤波器，避免长期错误外推。
      trail.clear();
      kalman_tracker.reset();
      drawText(frame, "green ball lost", cv::Point(20, 36), 28, cv::Scalar(0, 0, 255), -1);
      previous = BallTrack{};
    }

    writer.write(frame);
  }

  if (frame_index == 0) {
    std::cerr << "输入视频没有可读取的帧，未生成有效输出。" << std::endl;
    return 1;
  }

  std::cout << "已写入标注视频: " << output_path
            << "（处理帧数: " << frame_index << ")" << std::endl;
  return 0;
}
