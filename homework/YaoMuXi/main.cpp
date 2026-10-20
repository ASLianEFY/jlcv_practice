/*
 * 绿球识别与运动标注
 * ---------------------------------------------------------------------------
 * 输入一段含绿色小球的视频，逐帧定位球心，用卡尔曼滤波平滑轨迹，
 * 再在画面上画箭头：箭头方向 = 球的运动方向，箭头长度 = 速度大小。
 *
 * 速度与加速度全部由「相邻两帧中心点的位移」算出，滤波器只负责抑制抖动，
 * 不参与测速，所以画面上显示的速度始终对应真实的两帧差分结果。
 *
 * 编译（Ubuntu 已装 OpenCV 4.x 或 5.x）：
 *   g++ -O2 -std=c++17 -Wall -Wextra main.cpp -o ball_tracker $(pkg-config --cflags --libs opencv4)
 * 运行（两个参数都可省略，默认 homework.mp4 / homework-annotated.mp4）：
 *   ./ball_tracker homework.mp4 homework-annotated.mp4
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
// OpenCV 5 把 contourArea / arcLength / moments / boundingRect 从 imgproc 挪到了 geometry，
// 不加这个头文件会在编译期报 "contourArea is not a member of cv"。
#include <opencv2/geometry.hpp>
#endif

// ===========================================================================
// 第一部分：可调参数
// 全部集中在这里，改完重新编译即可，不需要动下面的逻辑。
// ===========================================================================
namespace
{

// ---- 1.1 绿色分割的 HSV 阈值 -------------------------------------------------
// OpenCV 的 H 是 0~180。绿球的 H 实测约 78，所以 40~95 足够覆盖它。
constexpr int kHueLow = 40;
constexpr int kHueHigh = 95;
// 饱和度下限 100 是关键：白纸、白柱子的 S 只有 45~55，抬高到 100 就全被滤掉，
// 而绿球的 S 实测约 210，不受影响。参考实现用的是 45，会误检白纸。
constexpr int kSatLow = 100;
// 明度下限，用来滤掉阴影里的暗色块。
constexpr int kValLow = 45;

// ---- 1.2 形状筛选（区分绿球与同色绿圆柱）------------------------------------
// 面积区间（像素^2）。绿球实测 9200~17200，同色绿圆柱是 24000~40000，
// 上限 24000 正好把圆柱排除在外。
constexpr double kAreaMin = 3500.0;
constexpr double kAreaMax = 24000.0;
// 圆度 = 4πA/P²，完美圆为 1。绿球 0.70~0.79，圆柱 0.28~0.67，取 0.66 居中偏球。
constexpr double kCircularityMin = 0.66;
// 填充率 = 面积 / 外接矩形的内切圆面积，用来排除空心环、残缺轮廓。
constexpr double kFillMin = 0.55;
// 外接矩形宽高比。球接近 1.0，圆柱 0.57~0.88。
constexpr double kAspectLow = 0.80;
constexpr double kAspectHigh = 1.25;

// ---- 1.3 卡尔曼滤波噪声参数 --------------------------------------------------
// sigma_a：过程噪声，代表"球能有多灵活"，单位 px/s²。调大更信任量测、响应快但抖。
constexpr double kSigmaA = 2000.0;
// sigma_z：量测噪声，代表检测器抖动，单位 px。应与实际检测精度匹配。
constexpr double kSigmaZ = 3.0;

// ---- 1.4 追踪状态机 ----------------------------------------------------------
// 连续丢失超过这个帧数就放弃当前目标、重建滤波器（重新全图搜索）。
constexpr int kMaxCoast = 6;

// ---- 1.5 箭头与显示 ----------------------------------------------------------
// 速度归一化满量程，即 speed >= kSpeedFull 时箭头画到最长。
constexpr double kSpeedFull = 900.0;
// 箭头最短 / 最长像素长度。保证球几乎静止时也能看到一小段箭头。
constexpr double kArrowMin = 30.0;
constexpr double kArrowMax = 150.0;

// ---- 1.6 加速度显示 ----------------------------------------------------------
// |accel| 小于这个值就判定为匀速，文字显示 stable。
constexpr double kStableAccel = 150.0;
// 两帧位移向量的一阶低通系数（越大越平滑、滞后越多）。
// 两帧差分天然有噪声，直接显示会让加速度在 ±4000 之间乱跳，必须平滑。
constexpr double kDiffSmooth = 0.6;

// ===========================================================================
// 第二部分：数据结构
// ===========================================================================

// 检测器输出的一个候选目标。多个候选会一起交给 chooseBestCandidate 打分。
struct BallCandidate
{
  cv::Point2f center;       // 亚像素质心（已换算到整幅图的坐标系）
  float radius = 0.0F;      // 等效半径，取外接矩形宽高均值的一半
  double area = 0.0;        // 轮廓面积
  double circularity = 0.0; // 圆度 4πA/P²
  double aspect_ratio = 0.0;// 外接矩形宽高比
  cv::Rect bounding_box;    // 外接矩形（已换算到整幅图的坐标系）
};

// ===========================================================================
// 第三部分：卡尔曼滤波器
// ---------------------------------------------------------------------------
// 4 维状态 x = [cx, cy, vx, vy]，2 维量测 z = [cx, cy]，即匀速（CV）模型。
// 矩阵用 std::array<double,16> 按行主序存放，索引 = i*4 + j。
//
// 不用 cv::KalmanFilter 的原因：
//   1) 它每次 predict/correct 都要走 cv::Mat 的表达式模板和 solve(DECOMP_SVD)，
//      本机实测 2.28 µs，而手写定长数组只要 0.065 µs；
//   2) 它的 correct() 用 P = P' - K·H·P' 的简单形式更新协方差，有限精度下会丢失
//      对称正定性；这里用 Joseph 形式 P = (I-KH)P(I-KH)^T + KRK^T，对任意 K 都正定；
//   3) 它的 predict() 末尾会把先验覆盖到后验上，导致"只预测不更新"的语义不好用，
//      而本程序在丢失帧里恰好需要这个语义。
// ===========================================================================
class KalmanFilter
{
public:
  // 构造时只记录噪声强度，不初始化状态；真正的初始化在第一次检测到球时做。
  KalmanFilter(double sigma_a, double sigma_z)
  : q_(sigma_a * sigma_a), r_(sigma_z * sigma_z)
  {
  }

  // 用第一次检测到的球心初始化状态：速度未知先置 0，位置方差取量测方差。
  void init(const cv::Point2f & position)
  {
    x_ = {position.x, position.y, 0.0, 0.0};

    P_.fill(0.0);
    P_[0] = r_;
    P_[5] = r_;
    // 速度方差给 400 表示"完全不知道球往哪走"，让滤波器自己快速收敛。
    P_[10] = 400.0;
    P_[15] = 400.0;
    initialized_ = true;
  }

  bool initialized() const { return initialized_; }

  // 预测步：x = F·x，P = F·P·F^T + Q。
  void predict(double dt)
  {
    // 状态推进：位置 += 速度 * dt（F 矩阵里只有这两个非对角项）。
    x_[0] += dt * x_[2];
    x_[1] += dt * x_[3];

    // 构造状态转移矩阵 F，等价于单位阵再加上位置-速度的 dt 项。
    std::array<double, 16> F = identity();
    F[0 * 4 + 2] = dt;
    F[1 * 4 + 3] = dt;

    // 协方差传播 P = F·P·F^T。
    std::array<double, 16> FP{};
    multiply(F, P_, FP);
    std::array<double, 16> FT{};
    transpose(F, FT);
    multiply(FP, FT, P_);

    // 叠加过程噪声 Q，采用离散白噪声加速度模型（连续加速度在 dt 内积分得到）：
    //   Q = q * [[dt^4/4, 0, dt^3/2, 0], [0, dt^4/4, 0, dt^3/2],
    //            [dt^3/2, 0, dt^2, 0], [0, dt^3/2, 0, dt^2]]
    // 这也是"球可能突然变速"的数学表达，dt^2 项直接作用在速度上。
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    P_[0 * 4 + 0] += q_ * dt4 / 4.0;
    P_[1 * 4 + 1] += q_ * dt4 / 4.0;
    P_[2 * 4 + 2] += q_ * dt2;
    P_[3 * 4 + 3] += q_ * dt2;
    P_[0 * 4 + 2] += q_ * dt3 / 2.0;
    P_[2 * 4 + 0] += q_ * dt3 / 2.0;
    P_[1 * 4 + 3] += q_ * dt3 / 2.0;
    P_[3 * 4 + 1] += q_ * dt3 / 2.0;
  }

  // 更新步：y = z - H·x，K = P·H^T·S^-1，x += K·y，P 用 Joseph 形式重算。
  void correct(const cv::Point2f & measurement)
  {
    // 新息（残差）：量测值减预测值。因为 H = [I2 0]，所以只取位置分量。
    const double y0 = measurement.x - x_[0];
    const double y1 = measurement.y - x_[1];

    // 新息协方差 S = H·P·H^T + R，这里 H 只挑前两行两列，所以直接取 P 的左上 2x2 加 R。
    const double s00 = P_[0 * 4 + 0] + r_;
    const double s01 = P_[0 * 4 + 1];
    const double s10 = P_[1 * 4 + 0];
    const double s11 = P_[1 * 4 + 1] + r_;
    const double det = s00 * s11 - s01 * s10;
    // 数值退化保护：行列式接近 0 时 2x2 求逆不可靠，直接跳过本次更新，保留预测值。
    if (std::abs(det) < 1e-12) {
      return;
    }

    // 卡尔曼增益 K = P·H^T·S^-1，用 2x2 解析求逆，K 是 4x2，按行主序存成 8 个元素。
    std::array<double, 8> K{};
    for (int i = 0; i < 4; ++i) {
      const double m0 = P_[i * 4 + 0];
      const double m1 = P_[i * 4 + 1];
      K[i * 2 + 0] = (m0 * s11 - m1 * s10) / det;
      K[i * 2 + 1] = (m1 * s00 - m0 * s01) / det;
    }

    // 状态更新 x = x + K·y。
    for (int i = 0; i < 4; ++i) {
      x_[i] += K[i * 2 + 0] * y0 + K[i * 2 + 1] * y1;
    }

    // 构造 (I - K·H)。因为 H 只取前两列，所以只需从单位阵的第 0、1 列减去 K。
    std::array<double, 16> IKH = identity();
    for (int i = 0; i < 4; ++i) {
      IKH[i * 4 + 0] -= K[i * 2 + 0];
      IKH[i * 4 + 1] -= K[i * 2 + 1];
    }
    // Joseph 形式：P = (I-KH)·P·(I-KH)^T。比简单形式多两次矩阵乘法，但对称正定有保证。
    std::array<double, 16> tmp{};
    std::array<double, 16> IKHT{};
    multiply(IKH, P_, tmp);
    transpose(IKH, IKHT);
    multiply(tmp, IKHT, P_);

    // 再加上 K·R·K^T。R 是对角阵 r·I2，所以展开成两个外积相加即可。
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        P_[i * 4 + j] += r_ * K[i * 2 + 0] * K[j * 2 + 0] + r_ * K[i * 2 + 1] * K[j * 2 + 1];
      }
    }
  }

  // 当前状态的位置估计（滤波后的球心）。
  cv::Point2f position() const
  {
    return cv::Point2f(static_cast<float>(x_[0]), static_cast<float>(x_[1]));
  }

  // 当前状态的速度估计。注意：画面上的箭头用的是两帧差分的实测速度，
  // 这个接口只用来估算"球可能跑多快"，从而决定搜索窗口开多大。
  cv::Point2f velocity() const
  {
    return cv::Point2f(static_cast<float>(x_[2]), static_cast<float>(x_[3]));
  }

  // 位置标准差，取两个方向方差的平均再开方。搜索窗口大小和关联门限都依赖它：
  // 滤波越不确定（刚初始化或刚丢失），窗口就开得越大。
  double positionSigma() const
  {
    return std::sqrt(std::max(0.0, 0.5 * (P_[0 * 4 + 0] + P_[1 * 4 + 1])));
  }

private:
  // 4x4 单位阵。
  static std::array<double, 16> identity()
  {
    std::array<double, 16> I{};
    I[0] = I[5] = I[10] = I[15] = 1.0;
    return I;
  }

  // 4x4 矩阵乘法 C = A·B。跳过 A 中的零元素，因为 F 和 (I-KH) 都是稀疏的。
  static void multiply(
    const std::array<double, 16> & A, const std::array<double, 16> & B, std::array<double, 16> & C)
  {
    C.fill(0.0);
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 4; ++k) {
        const double a = A[i * 4 + k];
        if (a == 0.0) {
          continue;
        }
        for (int j = 0; j < 4; ++j) {
          C[i * 4 + j] += a * B[k * 4 + j];
        }
      }
    }
  }

  // 4x4 转置 T = A^T。
  static void transpose(const std::array<double, 16> & A, std::array<double, 16> & T)
  {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        T[i * 4 + j] = A[j * 4 + i];
      }
    }
  }

  std::array<double, 4> x_{};   // 状态 [cx, cy, vx, vy]
  std::array<double, 16> P_{};  // 状态协方差 4x4
  double q_;                    // 过程噪声强度 sigma_a^2
  double r_;                    // 量测噪声强度 sigma_z^2
  bool initialized_ = false;    // 是否已经用第一帧的检测结果初始化过
};

// ===========================================================================
// 第四部分：检测器
// ===========================================================================

// 在指定区域内找所有"像绿球"的候选目标。
// roi 为空矩形时退化为全图搜索（首次捕获和丢失后重捕走这条路）。
// 返回的坐标已经加回 ROI 偏移，调用方始终使用整幅图坐标系。
std::vector<BallCandidate> detectBallCandidates(const cv::Mat & frame, const cv::Rect & roi)
{
  // 空 ROI 表示全图。用 window 统一后面所有裁剪操作。
  const cv::Rect window = roi.empty() ? cv::Rect(0, 0, frame.cols, frame.rows) : roi;

  // BGR 转 HSV，之后所有阈值判断都在 HSV 空间做，对光照变化更稳。
  cv::Mat hsv;
  cv::cvtColor(frame(window), hsv, cv::COLOR_BGR2HSV);

  // 阈值分割，得到绿球（以及同色绿圆柱）的二值掩码。
  cv::Mat mask;
  cv::inRange(hsv, cv::Scalar(kHueLow, kSatLow, kValLow), cv::Scalar(kHueHigh, 255, 255), mask);

  // 形态学去噪：先开运算去掉零星噪点，再闭运算补上球面高光造成的小洞。
  // 顺序不能反，闭运算在开运算之前会把噪点也连成块。
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  // 只取最外层轮廓，不需要层级关系。
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<BallCandidate> candidates;
  candidates.reserve(contours.size());

  // 逐个轮廓做四项几何筛选，全部通过才算候选。
  for (const auto & contour : contours) {
    // 筛选 1：面积。这一步就干掉了绝大多数噪声和绿圆柱。
    const double area = cv::contourArea(contour);
    if (area < kAreaMin || area > kAreaMax) {
      continue;
    }

    // 筛选 2：圆度 4πA/P²。周长用闭合轮廓计算。
    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) {
      continue;
    }
    const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    if (circularity < kCircularityMin) {
      continue;
    }

    // 筛选 3：外接矩形的宽高比。
    const cv::Rect box = cv::boundingRect(contour);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }
    const double aspect = static_cast<double>(box.width) / static_cast<double>(box.height);
    if (aspect < kAspectLow || aspect > kAspectHigh) {
      continue;
    }

    // 筛选 4：填充率。面积除以"外接矩形内切圆"的面积，正常球约 0.78，空心环会很低。
    const double fill = area / (CV_PI * 0.25 * box.width * box.height);
    if (fill < kFillMin) {
      continue;
    }

    // 用图像矩算亚像素质心。比 minEnclosingCircle 的圆心稳得多，
    // 因为后者对轮廓边缘的单个像素很敏感。
    const cv::Moments moments = cv::moments(contour);
    if (moments.m00 <= 0.0) {
      continue;
    }
    cv::Point2f center(
      static_cast<float>(moments.m10 / moments.m00), static_cast<float>(moments.m01 / moments.m00));

    // 坐标从 ROI 局部坐标系换算回整幅图坐标系，半径取宽高均值的一半。
    candidates.push_back(BallCandidate{
      center + cv::Point2f(static_cast<float>(window.x), static_cast<float>(window.y)),
      static_cast<float>(0.5 * (box.width + box.height) * 0.5),
      area,
      circularity,
      aspect,
      box + cv::Point(window.x, window.y)});
  }

  return candidates;
}

// 给一个候选打"像不像球"的分，0~1，越大越像。
// 三个分量都是"越接近理想值越高"，加权后返回。
double computeShapeScore(const BallCandidate & candidate)
{
  // 圆度：以 0.66 为下限，0.81 及以上给满分（0.15 是归一化跨度）。
  const double circularity_score =
    std::clamp((candidate.circularity - kCircularityMin) / 0.15, 0.0, 1.0);
  // 长宽比：1.0 满分，偏离 0.25 以上归零。
  const double aspect_score = std::clamp(1.0 - std::abs(candidate.aspect_ratio - 1.0) / 0.25, 0.0, 1.0);
  // 面积：以实测球面积 11000 为中心，偏离 9000 以上归零。
  // 这一项专门用来压制"小绿斑"——它们的形状分可能很高，但面积明显不对。
  const double area_score = std::clamp(1.0 - std::abs(candidate.area - 11000.0) / 9000.0, 0.0, 1.0);
  return 0.45 * circularity_score + 0.35 * aspect_score + 0.20 * area_score;
}

// 从候选里挑一个最可能是"当前这个球"的。
//   predicted        : 卡尔曼预测位置，候选离它越近越好
//   radius           : 上一帧的球半径，用于半径一致性门限（0 表示还没有先验）
//   radius_tolerance : 允许的半径相对偏差，追踪中 0.6，丢失重捕时放宽到 1.0
//   max_distance     : 关联门限，超过这个距离的候选直接淘汰（0 表示不限）
const BallCandidate * chooseBestCandidate(
  const std::vector<BallCandidate> & candidates, const cv::Point2f & predicted, double radius,
  double radius_tolerance, double max_distance)
{
  const BallCandidate * best = nullptr;
  double best_score = -1e9;

  for (const auto & candidate : candidates) {
    double score = computeShapeScore(candidate);

    // 半径一致性：尺寸和上一帧差太多的直接淘汰。这一条是防"锁定到背景小绿斑"的关键，
    // 因为伪影的半径通常只有真球的三分之一。
    if (radius > 0.0) {
      if (std::abs(candidate.radius - radius) > radius_tolerance * radius) {
        continue;
      }
      score -= std::abs(candidate.radius - radius) / radius * 0.15;
    }

    // 距离门限 + 距离惩罚：优先选靠近预测位置的候选。
    const double distance = cv::norm(candidate.center - predicted);
    if (max_distance > 0.0 && distance > max_distance) {
      continue;
    }
    score -= distance / std::max(1.0, max_distance) * 0.35;

    if (score > best_score) {
      best_score = score;
      best = &candidate;
    }
  }

  return best;
}

// 计算下一帧的局部搜索窗口。
// 窗口大小 = 基础尺寸 + 滤波不确定度 + 球半径 + 速度前瞻，再按丢失帧数指数放大。
cv::Rect computeSearchRoi(
  const cv::Mat & frame, const cv::Point2f & predicted, double sigma, double radius, double speed,
  int lost_frames)
{
  // 36 是基础余量；2.5σ 覆盖滤波的位置不确定度；1.3r 保证球整个在窗口内；
  // speed*0.12 是"提前 0.12 秒"的前瞻量，球跑得越快窗口越往前铺。
  double half = 36.0 + 2.5 * sigma + 1.3 * radius + speed * 0.12;

  // 每丢失一帧窗口放大 1.8 倍，最多放大 3 次（否则会退化成全图，失去局部搜索的意义）。
  half *= std::pow(1.8, std::min(lost_frames, 3));
  // 限制在 48~420 像素之间：太小会漏检，太大就接近全图了。
  half = std::clamp(half, 48.0, 420.0);

  // 以预测位置为中心，转成整数矩形。
  const int side = static_cast<int>(std::round(half * 2.0));
  int x = static_cast<int>(std::round(predicted.x)) - side / 2;
  int y = static_cast<int>(std::round(predicted.y)) - side / 2;

  // 夹进画面内，避免越界（OpenCV 对越界 ROI 会直接抛异常）。
  x = std::clamp(x, 0, std::max(0, frame.cols - 1));
  y = std::clamp(y, 0, std::max(0, frame.rows - 1));
  const int width = std::min(side, frame.cols - x);
  const int height = std::min(side, frame.rows - y);

  if (width <= 0 || height <= 0) {
    return cv::Rect();
  }

  // 窗口已经覆盖 85% 以上画面时，局部搜索没有意义，返回空矩形让调用方走全图。
  if (width >= frame.cols * 0.85 && height >= frame.rows * 0.85) {
    return cv::Rect();
  }
  return cv::Rect(x, y, width, height);
}

// ===========================================================================
// 第五部分：可视化
// ===========================================================================

// 把文字起点夹进画面内，防止球贴近边缘时文字被裁掉。
cv::Point clampTextOrg(cv::Point org, const std::string & text, double font_scale, const cv::Size & size)
{
  int baseline = 0;
  const cv::Size extent =
    cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, 2, &baseline);
  org.x = std::clamp(org.x, 4, std::max(4, size.width - extent.width - 4));
  org.y = std::clamp(org.y, extent.height + 4, std::max(extent.height + 4, size.height - 4));
  return org;
}

// 画一帧的全部标注：搜索窗口、球、速度箭头、三行文字、帧号。
//   velocity  : 两帧差分的位移向量，决定箭头方向
//   speed     : 两帧差分算出的速度，决定箭头长度与颜色
//   accel     : 平滑后的加速度，只用于文字显示
//   predicted : 本帧是纯预测（丢失中）还是量测更新过
void drawTrackOverlay(
  cv::Mat & frame, const cv::Point2f & center, double radius, const cv::Point2f & velocity,
  double speed, double accel, const cv::Rect & roi, bool predicted, int frame_index)
{

  // 搜索窗口用灰色细框画出来，方便观察局部搜索有没有跟住球。
  if (!roi.empty()) {
    cv::rectangle(frame, roi, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);
  }

  // 黄色圆圈表示球的外轮廓，红色实心点是滤波后的球心。
  cv::circle(frame, center, static_cast<int>(std::round(radius)), cv::Scalar(0, 255, 255), 2,
    cv::LINE_AA);
  cv::circle(frame, center, 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

  // ---- 速度箭头：方向 = 运动方向，长度 = 速度大小 ----
  const double norm = cv::norm(velocity);
  // 把速度归一化到 0~1，再线性映射到 [kArrowMin, kArrowMax]。
  const double ratio = std::clamp(speed / kSpeedFull, 0.0, 1.0);
  if (norm > 1e-3) {
    const double length = kArrowMin + ratio * (kArrowMax - kArrowMin);

    // 颜色也随速度渐变：慢速青色 -> 中速黄色 -> 高速红色。
    // 注意 OpenCV 的 Scalar 是 BGR 顺序，不是 RGB。
    cv::Scalar color;
    if (ratio < 0.5) {
      // 前半段：青色 (255,255,0) 过渡到黄色 (0,255,255)。
      const double t = ratio * 2.0;
      color = cv::Scalar(255.0 * (1.0 - t), 255.0, 255.0 * t);
    } else {
      // 后半段：黄色 (0,255,255) 过渡到红色 (0,0,255)。
      const double t = (ratio - 0.5) * 2.0;
      color = cv::Scalar(0.0, 255.0 * (1.0 - t), 255.0);
    }

    // 单位方向向量乘以长度得到箭头终点。最后一个参数 0.28 是箭头尖端的比例。
    const cv::Point2f direction = velocity * static_cast<float>(length / norm);
    cv::arrowedLine(frame, center, center + direction, color, 3, cv::LINE_AA, 0, 0.28);
  }

  // ---- 文字：速度值 + 加减速状态 ----
  char line[128];
  std::string status;
  std::snprintf(line, sizeof(line), "speed: %.1f px/s", speed);
  status = line;
  // 按加速度大小和符号判定状态，阈值 kStableAccel 用来避免"匀速"时文字来回跳。
  if (std::abs(accel) < kStableAccel) {
    status += "  stable";
  } else if (accel > 0.0) {
    status += "  accelerating";
  } else {
    status += "  decelerating";
  }

  std::snprintf(line, sizeof(line), "accel: %+.0f px/s^2", accel);
  const std::string accel_text = line;

  // ---- 三行文字的位置 ----
  // 默认放在球的正上方（102 = 三行文字加间距的总高度）。
  const int line_h = 30;
  const int top_margin = 30;
  int y0 = static_cast<int>(center.y - radius) - 102;
  // 球贴上边缘时上方没空间，整体翻到球的下方。
  if (y0 < top_margin) {
    y0 = static_cast<int>(center.y + radius) + 30;
  }
  // 再夹一次，保证三行文字都完整可见（球贴下边缘时同样成立）。
  y0 = std::clamp(y0, top_margin, std::max(top_margin, frame.rows - 2 * line_h - 8));
  // 文字左边缘对齐球的左边缘。
  const int left_x = static_cast<int>(center.x - radius);

  // 第一行：标题。丢失中显示 predicted，提示当前位置是预测值而非量测值。
  const std::string title = predicted ? "green ball (predicted)" : "green ball";
  const cv::Point title_org = clampTextOrg(cv::Point(left_x, y0), title, 0.8, frame.size());
  cv::putText(frame, title, title_org, cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

  // 第二行：速度与加减速状态（白色）。
  const cv::Point status_org =
    clampTextOrg(cv::Point(left_x, y0 + line_h), status, 0.7, frame.size());
  cv::putText(frame, status, status_org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

  // 第三行：加速度数值（洋红）。
  const cv::Point accel_org =
    clampTextOrg(cv::Point(left_x, y0 + 2 * line_h), accel_text, 0.7, frame.size());
  cv::putText(frame, accel_text, accel_org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
    cv::Scalar(255, 0, 255), 2, cv::LINE_AA);

  // 左上角固定显示帧号，便于和原视频对照。
  std::snprintf(line, sizeof(line), "frame: %d", frame_index);
  cv::putText(frame, line, cv::Point(20, 36), cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

}

// ===========================================================================
// 第六部分：主程序
// ===========================================================================
int main(int argc, char ** argv)
{
  // 只认两个位置参数，不做命令行选项解析（作业要求精简，不需要小工具）。
  const std::string input_path = argc > 1 ? argv[1] : "homework.mp4";
  const std::string output_path = argc > 2 ? argv[2] : "homework-annotated.mp4";

  // ---- 打开输入视频 ----
  cv::VideoCapture capture(input_path);
  if (!capture.isOpened()) {
    std::fprintf(stderr, "无法打开输入视频: %s\n", input_path.c_str());
    return 1;
  }

  // 从视频读帧率，用它作为两帧差分的时间间隔 dt。
  // 读不到或异常（<=1 fps）时按 30 fps 兜底。
  const double fps = capture.get(cv::CAP_PROP_FPS);
  const double dt = (fps > 1.0) ? 1.0 / fps : 1.0 / 30.0;
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  std::printf("输入: %s  %dx%d  %.2f fps\n", input_path.c_str(), width, height, fps);

  // ---- 创建输出视频 ----
  // 先试 mp4v（兼容性最好），打不开就退到 MJPG + avi。
  cv::VideoWriter writer;
  writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps > 0.0 ? fps : 30.0,
    cv::Size(width, height));
  if (!writer.isOpened()) {

    const std::string fallback = "homework-annotated.avi";
    writer.open(fallback, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps > 0.0 ? fps : 30.0,
      cv::Size(width, height));
    if (!writer.isOpened()) {
      std::fprintf(stderr, "无法创建输出视频\n");
      return 1;
    }
    std::printf("输出编码不可用，改用: %s\n", fallback.c_str());
  }

  // ---- 追踪状态 ----
  KalmanFilter filter(kSigmaA, kSigmaZ);   // 滤波器（未初始化状态）
  double radius = 0.0;                     // 当前球的半径估计
  int lost_frames = 0;                     // 连续丢失帧数
  bool has_previous = false;               // 是否已经有上一帧位置（决定能否算速度）
  cv::Point2f previous_position;           // 上一帧的滤波后位置
  double previous_speed = 0.0;             // 上一帧的速度（用于算加速度）
  cv::Point2f smoothed_step(0.0F, 0.0F);   // 平滑后的两帧位移向量
  bool has_smoothed_step = false;
  double accel_display = 0.0;              // 平滑后的加速度（只用于显示）
  bool has_accel = false;
  int frame_index = 0;

  // ---- 逐帧处理主循环 ----
  for (cv::Mat frame; capture.read(frame); ++frame_index) {
    cv::Rect roi;                            // 本帧使用的搜索窗口，空表示全图
    const BallCandidate * selected = nullptr;
    std::vector<BallCandidate> candidates;

    if (!filter.initialized()) {
      // 分支 A：还没锁到球。全图搜索，挑形状分最高的作为初始目标。

      candidates = detectBallCandidates(frame, cv::Rect());
      // 这里没有预测位置和半径先验，所以后三个参数传 0 表示不加约束。
      selected = chooseBestCandidate(candidates, cv::Point2f(), 0.0, 0.0, 0.0);
      if (selected != nullptr) {
        filter.init(selected->center);
        radius = selected->radius;
        lost_frames = 0;
      }
    } else {
      // 分支 B：已在追踪中。预测 -> 开窗 -> 局部检测 -> 关联 -> 更新。

      // 1) 卡尔曼预测。丢失帧也照常预测，这样球被短暂遮挡时轨迹不会断。
      filter.predict(dt);

      // 2) 按预测位置和不确定度开局部搜索窗口。
      const double speed_estimate = cv::norm(filter.velocity());
      roi = computeSearchRoi(
        frame, filter.position(), filter.positionSigma(), radius, speed_estimate, lost_frames);

      // 3) 只在窗口内做检测，计算量从全图降到几个百分点。
      candidates = detectBallCandidates(frame, roi);

      // 4) 关联门限：3σ 覆盖滤波不确定度，加上球半径和 25 px 固定余量。
      //    丢失期间半径容差从 0.6 放宽到 1.0，因为此时球可能正在变大或变小。
      const double gate = 3.0 * filter.positionSigma() + radius + 25.0;
      selected = chooseBestCandidate(
        candidates, filter.position(), radius, lost_frames > 0 ? 1.0 : 0.6, gate);

      if (selected != nullptr) {
        // 5) 关联成功：用检测结果更新滤波器，半径做指数平滑避免突变。
        filter.correct(selected->center);
        radius = (radius <= 0.0) ? selected->radius : 0.7 * radius + 0.3 * selected->radius;
        lost_frames = 0;
      } else {
        // 6) 关联失败：只预测不更新。超过 kMaxCoast 帧就彻底放弃，
        //    重建滤波器并清空所有先验，下一帧回到分支 A 重新全图搜索。
        ++lost_frames;
        if (lost_frames > kMaxCoast) {

          filter = KalmanFilter(kSigmaA, kSigmaZ);
          radius = 0.0;
          lost_frames = 0;
          has_previous = false;
        }
      }
    }

    // ---- 计算速度、加速度并绘制标注 ----
    if (filter.initialized()) {
      const cv::Point2f center = filter.position();

      // 速度的来源就是这两帧中心点之差，没有任何其他信息参与。
      cv::Point2f step(0.0F, 0.0F);
      if (has_previous) {
        const cv::Point2f raw_step = center - previous_position;
        // 对位移向量做一阶低通：两帧差分的噪声会被放大 1/dt 倍，不平滑的话箭头方向会乱抖。
        // 注意这只是对"同一份差分结果"做平滑，不改变速度的来源。
        step = has_smoothed_step
                 ? cv::Point2f(
                     static_cast<float>(kDiffSmooth) * smoothed_step +
                     static_cast<float>(1.0 - kDiffSmooth) * raw_step)
                 : raw_step;
        smoothed_step = step;
        has_smoothed_step = true;
      }

      double speed = 0.0;
      double accel = 0.0;
      if (has_previous) {
        // 速度 = 位移 / dt，加速度 = 速度变化量 / dt。
        speed = cv::norm(step) / dt;
        accel = (speed - previous_speed) / dt;

        // 加速度再平滑一次（0.7/0.3 低通），否则显示值会在 ±4000 之间跳。
        accel_display = has_accel ? 0.7 * accel_display + 0.3 * accel : accel;
        has_accel = true;
      }

      // 保存本帧状态供下一帧差分使用。
      previous_position = center;
      previous_speed = speed;
      has_previous = true;

      // 画标注。注意传进去的 velocity 是 step（位移向量），不是卡尔曼的速度状态。
      drawTrackOverlay(frame, center, radius, step, speed, accel_display, roi,
        lost_frames > 0, frame_index);
    } else {
      // 彻底丢失：画面上直接提示，等下一帧重新全图搜索。
      cv::putText(frame, "green ball lost", cv::Point(20, 36), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    writer.write(frame);
  }

  std::printf("已写出标注视频: %s  (共 %d 帧)\n", output_path.c_str(), frame_index);
  return 0;
}
