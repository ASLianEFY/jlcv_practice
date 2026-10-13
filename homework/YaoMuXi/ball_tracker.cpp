// =============================================================================
//  RoboMaster 视觉组进阶作业 —— 绿球识别与运动矢量(箭头)标注
//  单文件版本：本文件包含全部实现（数据结构 / 检测器 / 卡尔曼滤波 /
//              追踪流水线 / 可视化标注 / 命令行 / 自检）
//
//  编译（Ubuntu）:
//     sudo apt install -y build-essential cmake libopencv-dev
//     mkdir -p build && cd build
//     cmake ..
//     make -j$(nproc)
//
//  运行:
//     ./rm_ball_tracker --video ~/Desktop/homework.mp4 --output out.avi --csv track.csv
//     ./rm_ball_tracker --selftest          # 卡尔曼滤波自检（不需要视频）
//     ./rm_ball_tracker --help
//
//  标注含义:
//     · 箭头方向 = 小球运动方向
//     · 箭头长度 = 速度大小（越长越快，越短越慢；按图像对角线归一化）
//     · 箭头颜色 = 速度大小（青 → 黄 → 红）
//     · 洋红虚线箭头 = 速度变化（沿运动方向=加速，反向=减速）
//
//  算法链路:  HSV 阈值分割 → 形态学 → 轮廓圆度/填充率/尺寸筛选
//             → 卡尔曼滤波(CV 模型 + 卡方机动检测 + 自适应 Q)
//             → 用预测位置与协方差动态开 ROI 局部搜索 → 矢量箭头标注
//  详细思路见 README.md
// =============================================================================

// ------------------------------- 头文件 -------------------------------------
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#if CV_VERSION_MAJOR >= 5
// OpenCV 5 把 contourArea / arcLength / minEnclosingCircle / boundingRect / moments
// 从 imgproc 移到了新的 geometry 模块。这里做版本兼容，
// 使同一份代码在 OpenCV 4.x（Ubuntu 的 libopencv-dev）与 5.x 上都能编译。
#include <opencv2/geometry.hpp>
#endif

#ifdef RMV_HAVE_HIGHGUI
#include <opencv2/highgui.hpp>
#endif


// ======================= 第一部分：数据结构与类声明 =======================
// =============================================================================
//  RoboMaster 视觉组进阶作业 —— 绿球识别与运动矢量(箭头)标注
//  本部分：数据结构 / 参数 / 检测器 / 卡尔曼滤波 / 追踪流水线 的声明
//
//  总体思路（对标 RM 视觉组自瞄/目标追踪的工程链路）:
//    1) 目标提取 : HSV 颜色阈值 + 形态学 + 轮廓圆度/填充率筛选  —— 不是帧差法
//    2) 状态估计 : 4 状态恒速(CV)卡尔曼滤波, 手写实现, 不依赖 OpenCV, 可单元测试
//    3) 局部搜索 : 用卡尔曼预测位置 + 位置不确定度动态开 ROI  —— 计算量降 1~2 个数量级
//    4) 鲁棒状态机: TRACKING / COASTING(惯性滑行) / LOST(全图重捕) + 数据关联门
//    5) 矢量标注 : 速度取自卡尔曼状态量(而非相邻帧帧差), 箭头长度正比于速度
//
//  编译环境: Ubuntu + CMake + OpenCV 4.x (见同目录 CMakeLists.txt)
// =============================================================================



namespace rmv {

// -----------------------------------------------------------------------------
// 基础类型
// -----------------------------------------------------------------------------

/// 二维向量（与 OpenCV 解耦，便于对滤波算法做单元测试）
struct Vec2d {
  double x{0.0};
  double y{0.0};
};

/// 追踪状态机
enum class TrackState {
  kInit,      ///< 尚未捕获目标（开机/复位后的初始态）
  kTracking,  ///< 本帧有量测更新，正常追踪
  kCoasting,  ///< 短暂丢失，仅靠预测惯性滑行，ROI 自动放大
  kLost       ///< 长时间丢失，退化为全图重新捕获
};

const char* toString(TrackState s);

/// 本帧搜索窗口信息（用于日志 / 可视化 / 性能统计）
struct RoiInfo {
  cv::Rect rect{};
  bool limited{false};  ///< false 表示本帧退化为全图搜索
  double half{0.0};     ///< 窗口半宽(px)，用于计算面积占比
};

/// 单帧检测结果
struct Detection {
  bool found{false};
  cv::Point2d center{0.0, 0.0};  ///< 全图坐标系下的亚像素质心
  double radius{0.0};            ///< 最小外接圆半径(px)
  double area{0.0};              ///< 轮廓面积(px^2)
  double circularity{0.0};       ///< 圆度 4*pi*A/P^2
  double fill{0.0};              ///< 填充率 A/(pi*r^2)
  double aspect{1.0};            ///< 外接矩形宽高比
  double score{0.0};             ///< 综合评分
};

/// 单帧追踪输出（可视化 / 决策 / 串口下发的统一数据结构）
struct TrackResult {
  TrackState state{TrackState::kInit};
  cv::Point2d position{0.0, 0.0};  ///< 卡尔曼滤波后位置(px)
  cv::Point2d velocity{0.0, 0.0};  ///< 卡尔曼滤波速度(px/s)
  double speed{0.0};               ///< |v| (px/s)
  double speed_display{0.0};       ///< 低通平滑后的显示速度(px/s)，用于箭头长度
  double speed_mps{0.0};           ///< 由已知球直径反推的物理速度(m/s)
  double speed_sigma{0.0};         ///< 速度标准差估计(px/s)，来自协方差阵
  double accel{0.0};               ///< 显示速度的变化率(px/s^2)
  double radius{0.0};              ///< 半径 EMA (px)
  double dt{0.0};                  ///< 本帧时间间隔(s)
  double mahalanobis{0.0};         ///< 新息马氏距离(机动检测统计量)
  bool maneuver{false};            ///< 本帧是否判定为机动
  bool measurement_used{false};    ///< 本帧是否用到量测
  int miss_count{0};               ///< 连续丢失帧数

  RoiInfo roi{};          ///< 本帧搜索窗口
  Detection detection{};  ///< 本帧检测结果

  // 位置协方差(px^2)，用于绘制不确定度椭圆
  double pxx{0.0};
  double pxy{0.0};
  double pyy{0.0};
};

// -----------------------------------------------------------------------------
// 参数（全部有合理默认值，可零配置直接运行；也可用 YAML 覆盖）
// -----------------------------------------------------------------------------

/// 绿球检测器参数
struct DetectorConfig {
  // HSV 阈值（OpenCV: H∈[0,179]，S/V∈[0,255]），需按现场光照标定
  int h_min{40};
  int h_max{85};
  int s_min{70};
  int s_max{255};
  int v_min{50};
  int v_max{255};

  int blur_ksize{5};   ///< 中值滤波核(奇数)，<3 表示关闭。中值滤波对椒盐噪声更稳
  int morph_open{3};   ///< 开运算核，0 关闭。先开后闭：去孤立噪点 + 补轮廓空洞
  int morph_close{5};  ///< 闭运算核，0 关闭

  double min_area{60.0};
  double max_area{200000.0};
  double min_radius{4.0};
  double max_radius{500.0};
  double min_circularity{0.45};  ///< 圆度下限，剔除长条形干扰
  double min_fill{0.55};         ///< 填充率下限，剔除空心/碎片
  double min_aspect{0.55};
  double max_aspect{1.80};

  double w_circularity{0.35};  ///< 评分权重：圆度
  double w_fill{0.25};         ///< 评分权重：填充率
  double w_size{0.20};         ///< 评分权重：相对尺寸(本帧最大候选为 1.0)
  double w_proximity{0.20};    ///< 评分权重：靠近 ROI 中心(预测位置)的程度
};

/// ROI 局部搜索参数
struct RoiConfig {
  double base_half{36.0};         ///< 基础半宽(px)
  double k_sigma{2.5};            ///< 位置不确定度系数(σ 的倍数)
  double k_radius{1.3};           ///< 目标半径系数
  double lookahead_time{0.12};    ///< 速度前瞻时间(s)：高速目标提前开窗
  double min_half{48.0};
  double max_half{420.0};
  double miss_grow{1.8};          ///< 丢失时窗口放大倍数
  double full_frame_ratio{0.85};  ///< 半宽超过短边该比例时直接退化为全图
};

/// 卡尔曼滤波 + 追踪状态机参数
struct TrackerConfig {
  double sigma_a{900.0};   ///< 过程噪声(加速度谱密度, px/s^2)：越大越信任量测、响应越快
  double sigma_z{3.5};     ///< 量测噪声标准差(px)：取检测质心的抖动幅度
  double gate_chi2{9.21};  ///< 机动检测门限(卡方分布 2 自由度 99% 分位)
  double q_boost{8.0};     ///< 判定机动时速度方差放大倍数(自适应 Q)
  double max_dt{0.2};      ///< 单步 dt 上限(掉帧保护)
  double init_pos_var{100.0};
  double init_vel_var{1.0e6};  ///< 初始速度方差，几乎“不知道”速度

  int max_miss_coast{3};    ///< 连续丢失超过该帧数 -> LOST
  int max_miss_lost{30};    ///< LOST 持续该帧数 -> 复位并重新全图捕获
  double relink_dist{120.0};///< LOST 重捕时允许的关联距离(px)

  double speed_tau{0.02};        ///< 显示速度一阶低通时间常数(s)，0 关闭
                                 ///< 注意: 低通会带来 tau*|dv/dt| 的滞后，实测速度误差
                                 ///< 与 tau 近似线性(≈20 + 640*tau px/s)，卡尔曼输出已足够
                                 ///< 平滑，这里只需极小值或直接置 0。
  double ball_diameter_m{0.065}; ///< 已知球直径(m)，用于由像素尺寸反推物理速度
  bool enable_speed_mps{true};
};

/// 应用总配置
struct AppConfig {
  DetectorConfig detector;
  RoiConfig roi;
  TrackerConfig tracker;
};

/// 从 YAML 文件读取参数（cv::FileStorage）。
/// 文件不存在或打开失败时保持默认值，返回 false。YAML 中布尔量请写 0/1。
bool loadConfig(const std::string& path, AppConfig& cfg);

// -----------------------------------------------------------------------------
// 卡尔曼滤波：4 状态恒速(Constant Velocity)模型
//   状态 x = [cx, cy, vx, vy]^T      量测 z = [cx, cy]^T
//   预测: x⁻ = F x ,  P⁻ = F P F^T + Q          (F = I + dt*[0 0 1 0; 0 0 0 1; 0...])
//   更新: y = z - H x⁻ , S = H P⁻ H^T + R , K = P⁻ H^T S⁻¹
//         x = x⁻ + K y , P = (I-KH) P⁻ (I-KH)^T + K R K^T   ← Joseph 形式，数值更稳
//   Q 取“离散白噪声加速度”模型:
//     Q = q * [dt^4/4  0  dt^3/2  0 ; 0  dt^4/4  0  dt^3/2 ;
//              dt^3/2  0  dt^2    0 ; 0  dt^3/2  0  dt^2  ] , q = sigma_a^2
// -----------------------------------------------------------------------------
class KalmanTracker {
 public:
  explicit KalmanTracker(const TrackerConfig& cfg = TrackerConfig());

  void clear();  ///< 复位为“未初始化”，下一帧将重新全图捕获
  bool initialized() const { return initialized_; }

  /// 用一次检测初始化位置（速度置 0，速度方差取很大值）
  void reset(double x, double y, double radius);

  /// 两点差分初始化速度（配合 reset 使用，解决“起步阶段速度未知”的问题）
  void setVelocity(double vx, double vy, double vel_var);

  /// 时间更新（无预测则惯性外推；丢失时只做这一步，协方差自然膨胀）
  void predict(double dt);

  /// 量测更新；若尚未初始化则等价于 reset
  void update(double zx, double zy, double radius);

  Vec2d position() const { return Vec2d{x_[0], x_[1]}; }
  Vec2d velocity() const { return Vec2d{x_[2], x_[3]}; }
  double speed() const { return std::hypot(x_[2], x_[3]); }
  double speedSigma() const;  ///< 速度标准差 sqrt((Pvvx + Pvvy)/2)
  double posSigma() const;    ///< 位置标准差 sqrt(max(Pxx, Pyy))，用于 ROI 开窗
  void positionCovariance(double& pxx, double& pxy, double& pyy) const;
  double radius() const { return radius_ema_; }
  double mahalanobis() const { return mahalanobis_; }
  bool maneuver() const { return maneuver_; }

  void inflateVelocity(double factor);  ///< 人工放大速度不确定度

 private:
  void symmetrize();  ///< 强制协方差阵对称，抑制浮点误差累积

  TrackerConfig cfg_;
  bool initialized_{false};
  double x_[4]{};   ///< 状态向量 [cx, cy, vx, vy]
  double P_[4][4]{};///< 状态协方差阵
  double radius_ema_{0.0};
  double mahalanobis_{0.0};
  bool maneuver_{false};
};

// -----------------------------------------------------------------------------
// 绿球检测器：HSV 阈值 + 形态学 + 轮廓圆度筛选（支持 ROI 局部搜索）
// -----------------------------------------------------------------------------
class BallDetector {
 public:
  explicit BallDetector(const DetectorConfig& cfg = DetectorConfig());

  /// 在 bgr 中搜索绿球。
  /// roi == nullptr 时全图搜索；否则只在窗口内做分割与轮廓分析（ROI 局部搜索）。
  /// 返回的坐标始终是全图坐标系，方便上层统一处理。
  ///
  /// expected_radius > 0 时启用"尺寸一致性门限"：只保留半径与期望值偏差不超过
  /// radius_tol(相对值) 的候选。这是抑制"背景小绿斑"(压缩伪影/草地/反光)的关键——
  /// 小圆斑的圆度往往比真球还高，必须靠尺寸先验才能区分。
  Detection detect(const cv::Mat& bgr, const cv::Rect* roi, double expected_radius = -1.0,
                   double radius_tol = 0.6);

  void setConfig(const DetectorConfig& cfg);
  const DetectorConfig& config() const { return cfg_; }
  const cv::Mat& lastMask() const { return mask_; }  ///< 最近一次的掩膜(ROI 局部坐标)

 private:
  DetectorConfig cfg_;
  cv::Mat hsv_, mask_;
  cv::Mat kernel_open_, kernel_close_;
};

// -----------------------------------------------------------------------------
// 追踪流水线：检测 + 卡尔曼 + ROI 局部搜索 + 状态机（上层只需调用 process）
// -----------------------------------------------------------------------------
class BallTrackerPipeline {
 public:
  explicit BallTrackerPipeline(const AppConfig& cfg = AppConfig());

  /// 处理一帧。dt 为本帧与上一帧的真实时间间隔(s)，由外部按数据源给出。
  TrackResult process(const cv::Mat& bgr, double dt);

  void reset();
  const AppConfig& config() const { return cfg_; }

 private:
  cv::Rect computeSearchRoi(const cv::Size& size) const;
  static cv::Rect growRect(const cv::Rect& r, double scale, const cv::Size& size);
  void fillOutput(TrackResult& out) const;

  AppConfig cfg_;
  BallDetector detector_;
  KalmanTracker tracker_;

  TrackState state_{TrackState::kInit};
  int miss_count_{0};
  int lost_frames_{0};

  bool need_velocity_seed_{false};  ///< 是否需要两点差分初始化速度
  bool has_prev_detection_{false};
  cv::Point2d prev_detection_{0.0, 0.0};

  double speed_display_{0.0};
  bool has_speed_display_{false};
  double last_speed_display_{0.0};
  double accel_display_{0.0};
  bool has_accel_{false};
};

}  // namespace rmv


// ======================= 第二部分：标注绘制器声明 =========================
// =============================================================================
//  可视化: 用箭头标注绿球的运动方向与速度大小（Visualizer 声明）
//
//  标注规则（对应作业要求）:
//    · 箭头方向 = 卡尔曼滤波输出的速度方向（即小球运动方向）
//    · 箭头长度 ∝ 速度大小: 越长越快, 越短越慢, 并按分辨率归一化
//    · 箭头颜色按速度映射(青 -> 黄 -> 红), 静止目标不画方向箭头
//    · 附加: 运动拖尾(按历史速度着色)、速度-时间曲线、速度变化率(加速度)箭头、
//            卡尔曼位置不确定度椭圆、ROI 搜索窗、HUD 信息面板
//
//  注意: cv::putText 使用 Hershey 矢量字体, 只支持 ASCII,
//        因此叠加在画面上的文字均为英文（需要中文可换 cv::freetype 模块）。
// =============================================================================




namespace rmv {

/// 可视化参数
struct VizConfig {
  bool draw_roi{true};           ///< 画出 ROI 搜索窗口
  bool draw_trail{true};         ///< 画出运动拖尾
  bool draw_arrow{true};         ///< 画出速度箭头（核心标注）
  bool draw_accel_arrow{true};   ///< 画出速度变化率(加速度)箭头
  bool draw_speed_chart{true};   ///< 画出速度-时间曲线
  bool draw_hud{true};           ///< 画出信息面板
  bool draw_legend{true};        ///< 画出速度色标/图例
  bool draw_uncertainty{true};   ///< 画出卡尔曼位置不确定度椭圆

  int trail_len{60};             ///< 拖尾长度(帧)

  double arrow_frac_max{0.15};   ///< 满量程速度对应的箭头长度 = 该比例 × 图像对角线
  double arrow_frac_min{0.02};   ///< 静止边界处的箭头长度(保证低速可见性)
  double speed_full_scale{900.0};///< 速度满量程(px/s)，决定配色与箭头长度映射
  double speed_dead_zone{20.0};  ///< 低于该速度视为静止(px/s)，不画方向箭头
  double accel_full_scale{4000.0};  ///< 加速度满量程(px/s^2)
  int arrow_thickness{3};
  double trail_thickness{2.0};

  int chart_width{320};
  int chart_height{120};
};

/// 标注绘制器：内部维护拖尾与速度历史，逐帧调用 draw()
class Visualizer {
 public:
  explicit Visualizer(const VizConfig& cfg = VizConfig());

  void reset();

  /// 在 frame 上原地绘制全部标注
  void draw(cv::Mat& frame, const TrackResult& tr);

  void setFps(double fps) { fps_ = fps; }
  void setFrameIndex(long idx) { frame_index_ = idx; }
  void setConfig(const VizConfig& cfg) { cfg_ = cfg; }
  const VizConfig& config() const { return cfg_; }

  /// 速度 -> 颜色 (BGR): 青(慢) -> 黄 -> 红(快)
  static cv::Scalar speedColor(double speed, double full_scale);

 private:
  void pushHistory(const TrackResult& tr);
  void drawRoi(cv::Mat& frame, const TrackResult& tr) const;
  void drawTrail(cv::Mat& frame) const;
  void drawUncertainty(cv::Mat& frame, const TrackResult& tr) const;
  void drawTargetMarker(cv::Mat& frame, const TrackResult& tr) const;
  void drawArrow(cv::Mat& frame, const TrackResult& tr) const;
  void drawAccelArrow(cv::Mat& frame, const TrackResult& tr) const;
  void drawHud(cv::Mat& frame, const TrackResult& tr) const;
  void drawSpeedChart(cv::Mat& frame) const;
  void drawLegend(cv::Mat& frame) const;

  VizConfig cfg_;
  std::deque<cv::Point2d> trail_;   ///< 拖尾位置
  std::deque<double> trail_speed_;  ///< 拖尾各点对应的速度(与 trail_ 同步裁剪)
  std::deque<double> speed_hist_;   ///< 速度历史(用于曲线, 保留更长)
  double fps_{0.0};
  long frame_index_{-1};
};

}  // namespace rmv


// ================= 第三部分：检测 / 卡尔曼 / 追踪流水线实现 ================
// =============================================================================
//  ball_tracker.cpp
//  实现: ① 参数加载  ② 卡尔曼滤波  ③ 绿球检测器(支持 ROI 局部搜索)
//        ④ 追踪流水线(ROI 开窗 + 数据关联 + 状态机)
//
//  说明: 本文件中所有算法的参数默认值均可零配置运行，
//        需要现场调参时再用 --config 传入 YAML。
// =============================================================================




namespace rmv {

// =============================================================================
// 一、小工具
// =============================================================================
const char* toString(TrackState s) {
  switch (s) {
    case TrackState::kInit:
      return "INIT";
    case TrackState::kTracking:
      return "TRACKING";
    case TrackState::kCoasting:
      return "COASTING";
    case TrackState::kLost:
      return "LOST";
  }
  return "UNKNOWN";
}

namespace {

// ---- cv::FileStorage 读取辅助：节点不存在时保留默认值 ----
void readInt(const cv::FileNode& n, const char* key, int& v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) {
    int t = v;
    c >> t;
    v = t;
  }
}

void readDouble(const cv::FileNode& n, const char* key, double& v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) {
    double t = v;
    c >> t;
    v = t;
  }
}

// YAML 中的布尔量请写 0 / 1（OpenCV 对 true/false 的解析不可靠）
void readBool01(const cv::FileNode& n, const char* key, bool& v) {
  const cv::FileNode c = n[key];
  if (!c.empty()) {
    int t = v ? 1 : 0;
    c >> t;
    v = (t != 0);
  }
}

// ---- 4x4 定长矩阵工具（行主序）----
constexpr int kN = 4;
using Mat4 = std::array<double, 16>;

inline double mget(const Mat4& m, int i, int j) {
  return m[static_cast<std::size_t>(i) * kN + j];
}
inline double& mref(Mat4& m, int i, int j) {
  return m[static_cast<std::size_t>(i) * kN + j];
}

Mat4 identity4() {
  Mat4 m{};
  for (int i = 0; i < kN; ++i) mref(m, i, i) = 1.0;
  return m;
}

Mat4 matMul(const Mat4& a, const Mat4& b) {
  Mat4 c{};
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      double s = 0.0;
      for (int k = 0; k < kN; ++k) s += mget(a, i, k) * mget(b, k, j);
      mref(c, i, j) = s;
    }
  }
  return c;
}

Mat4 matTranspose(const Mat4& a) {
  Mat4 c{};
  for (int i = 0; i < kN; ++i)
    for (int j = 0; j < kN; ++j) mref(c, i, j) = mget(a, j, i);
  return c;
}

/// 2x2 解析求逆，失败(奇异)返回 false
bool inv2(double a00, double a01, double a10, double a11, double& i00, double& i01, double& i10,
          double& i11) {
  const double det = a00 * a11 - a01 * a10;
  if (std::fabs(det) < 1e-12) return false;
  const double id = 1.0 / det;
  i00 = a11 * id;
  i01 = -a01 * id;
  i10 = -a10 * id;
  i11 = a00 * id;
  return true;
}

}  // namespace

// =============================================================================
// 二、参数加载（YAML，可选）
// =============================================================================
bool loadConfig(const std::string& path, AppConfig& cfg) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) return false;

  const cv::FileNode d = fs["detector"];
  if (!d.empty()) {
    readInt(d, "h_min", cfg.detector.h_min);
    readInt(d, "h_max", cfg.detector.h_max);
    readInt(d, "s_min", cfg.detector.s_min);
    readInt(d, "s_max", cfg.detector.s_max);
    readInt(d, "v_min", cfg.detector.v_min);
    readInt(d, "v_max", cfg.detector.v_max);
    readInt(d, "blur_ksize", cfg.detector.blur_ksize);
    readInt(d, "morph_open", cfg.detector.morph_open);
    readInt(d, "morph_close", cfg.detector.morph_close);
    readDouble(d, "min_area", cfg.detector.min_area);
    readDouble(d, "max_area", cfg.detector.max_area);
    readDouble(d, "min_radius", cfg.detector.min_radius);
    readDouble(d, "max_radius", cfg.detector.max_radius);
    readDouble(d, "min_circularity", cfg.detector.min_circularity);
    readDouble(d, "min_fill", cfg.detector.min_fill);
    readDouble(d, "min_aspect", cfg.detector.min_aspect);
    readDouble(d, "max_aspect", cfg.detector.max_aspect);
    readDouble(d, "w_circularity", cfg.detector.w_circularity);
    readDouble(d, "w_fill", cfg.detector.w_fill);
    readDouble(d, "w_size", cfg.detector.w_size);
    readDouble(d, "w_proximity", cfg.detector.w_proximity);
  }

  const cv::FileNode r = fs["roi"];
  if (!r.empty()) {
    readDouble(r, "base_half", cfg.roi.base_half);
    readDouble(r, "k_sigma", cfg.roi.k_sigma);
    readDouble(r, "k_radius", cfg.roi.k_radius);
    readDouble(r, "lookahead_time", cfg.roi.lookahead_time);
    readDouble(r, "min_half", cfg.roi.min_half);
    readDouble(r, "max_half", cfg.roi.max_half);
    readDouble(r, "miss_grow", cfg.roi.miss_grow);
    readDouble(r, "full_frame_ratio", cfg.roi.full_frame_ratio);
  }

  const cv::FileNode t = fs["tracker"];
  if (!t.empty()) {
    readDouble(t, "sigma_a", cfg.tracker.sigma_a);
    readDouble(t, "sigma_z", cfg.tracker.sigma_z);
    readDouble(t, "gate_chi2", cfg.tracker.gate_chi2);
    readDouble(t, "q_boost", cfg.tracker.q_boost);
    readDouble(t, "max_dt", cfg.tracker.max_dt);
    readDouble(t, "init_pos_var", cfg.tracker.init_pos_var);
    readDouble(t, "init_vel_var", cfg.tracker.init_vel_var);
    readInt(t, "max_miss_coast", cfg.tracker.max_miss_coast);
    readInt(t, "max_miss_lost", cfg.tracker.max_miss_lost);
    readDouble(t, "relink_dist", cfg.tracker.relink_dist);
    readDouble(t, "speed_tau", cfg.tracker.speed_tau);
    readDouble(t, "ball_diameter_m", cfg.tracker.ball_diameter_m);
    readBool01(t, "enable_speed_mps", cfg.tracker.enable_speed_mps);
  }

  fs.release();
  return true;
}

// =============================================================================
// 三、卡尔曼滤波（4 状态恒速模型）
// =============================================================================
KalmanTracker::KalmanTracker(const TrackerConfig& cfg) : cfg_(cfg) { clear(); }

void KalmanTracker::clear() {
  initialized_ = false;
  for (int i = 0; i < 4; ++i) {
    x_[i] = 0.0;
    for (int j = 0; j < 4; ++j) P_[i][j] = 0.0;
  }
  radius_ema_ = 0.0;
  mahalanobis_ = 0.0;
  maneuver_ = false;
}

void KalmanTracker::reset(double x, double y, double radius) {
  x_[0] = x;
  x_[1] = y;
  x_[2] = 0.0;
  x_[3] = 0.0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) P_[i][j] = 0.0;
  P_[0][0] = cfg_.init_pos_var;
  P_[1][1] = cfg_.init_pos_var;
  P_[2][2] = cfg_.init_vel_var;  // 速度完全未知 -> 大方差
  P_[3][3] = cfg_.init_vel_var;
  radius_ema_ = radius;
  mahalanobis_ = 0.0;
  maneuver_ = false;
  initialized_ = true;
}

void KalmanTracker::setVelocity(double vx, double vy, double vel_var) {
  if (!initialized_) return;
  x_[2] = vx;
  x_[3] = vy;
  if (vel_var > 0.0) {
    P_[2][2] = vel_var;
    P_[3][3] = vel_var;
  }
}

void KalmanTracker::predict(double dt) {
  if (!initialized_) return;
  dt = std::clamp(dt, 1e-4, cfg_.max_dt);

  // ---- 状态外推: x = F x  (CV 模型: 位置 += v*dt, 速度不变) ----
  x_[0] += dt * x_[2];
  x_[1] += dt * x_[3];

  // ---- 协方差外推: P = F P F^T + Q ----
  Mat4 F = identity4();
  mref(F, 0, 2) = dt;
  mref(F, 1, 3) = dt;

  Mat4 P{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) mref(P, i, j) = P_[i][j];

  Mat4 Pn = matMul(matMul(F, P), matTranspose(F));

  // 离散白噪声加速度模型: q = sigma_a^2
  const double q = cfg_.sigma_a * cfg_.sigma_a;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  const double q11 = dt4 * 0.25 * q;
  const double q13 = dt3 * 0.5 * q;
  const double q33 = dt2 * q;

  mref(Pn, 0, 0) += q11;
  mref(Pn, 0, 2) += q13;
  mref(Pn, 2, 0) += q13;
  mref(Pn, 2, 2) += q33;
  mref(Pn, 1, 1) += q11;
  mref(Pn, 1, 3) += q13;
  mref(Pn, 3, 1) += q13;
  mref(Pn, 3, 3) += q33;

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) P_[i][j] = mget(Pn, i, j);

  symmetrize();
}

void KalmanTracker::update(double zx, double zy, double radius) {
  if (!initialized_) {
    reset(zx, zy, radius);
    return;
  }

  const double r = cfg_.sigma_z * cfg_.sigma_z;

  // ---- S = H P H^T + R  (2x2) ----
  const double s00 = P_[0][0] + r;
  const double s01 = P_[0][1];
  const double s10 = P_[1][0];
  const double s11 = P_[1][1] + r;

  double i00 = 0.0, i01 = 0.0, i10 = 0.0, i11 = 0.0;
  if (!inv2(s00, s01, s10, s11, i00, i01, i10, i11)) {
    radius_ema_ = radius;  // 数值异常时至少保持几何量更新
    return;
  }

  // ---- 新息 y = z - H x⁻ 与马氏距离 d² = yᵀ S⁻¹ y ----
  const double yx = zx - x_[0];
  const double yy = zy - x_[1];
  const double d2 = yx * (i00 * yx + i01 * yy) + yy * (i10 * yx + i11 * yy);
  mahalanobis_ = std::sqrt(std::max(0.0, d2));

  // 机动检测: 卡方检验。量测突然远离预测(如急加速/急转弯) -> 判定机动
  maneuver_ = d2 > cfg_.gate_chi2;

  // ---- 增益 K = P H^T S⁻¹  (4x2) ----
  double K[4][2];
  for (int i = 0; i < 4; ++i) {
    const double p0 = P_[i][0];
    const double p1 = P_[i][1];
    K[i][0] = p0 * i00 + p1 * i10;
    K[i][1] = p0 * i01 + p1 * i11;
  }

  // ---- 状态更新 x = x⁻ + K y ----
  for (int i = 0; i < 4; ++i) x_[i] += K[i][0] * yx + K[i][1] * yy;

  // ---- 协方差更新(Joseph 形式): P = (I-KH) P (I-KH)^T + K R K^T ----
  Mat4 A = identity4();
  for (int i = 0; i < 4; ++i) {
    mref(A, i, 0) -= K[i][0];
    mref(A, i, 1) -= K[i][1];
  }

  Mat4 P{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) mref(P, i, j) = P_[i][j];

  Mat4 Pn = matMul(matMul(A, P), matTranspose(A));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double acc = 0.0;
      for (int k = 0; k < 2; ++k) acc += K[i][k] * r * K[j][k];
      mref(Pn, i, j) += acc;
    }
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) P_[i][j] = mget(Pn, i, j);

  symmetrize();

  // ---- 自适应 Q: 判定机动后立刻放大速度不确定度 ----
  // 作用有二: (1) 下一帧增益变大, 滤波器跟得上目标; (2) ROI 随 σ 自动放大
  if (maneuver_) {
    P_[2][2] *= cfg_.q_boost;
    P_[3][3] *= cfg_.q_boost;
  }

  // ---- 半径一阶低通(几何尺寸不需要卡尔曼, EMA 足够且稳定) ----
  const double a = 0.35;
  radius_ema_ = (radius_ema_ > 1e-6) ? (a * radius + (1.0 - a) * radius_ema_) : radius;
}

double KalmanTracker::speedSigma() const {
  if (!initialized_) return 0.0;
  return std::sqrt(std::max(0.0, 0.5 * (P_[2][2] + P_[3][3])));
}

double KalmanTracker::posSigma() const {
  if (!initialized_) return 0.0;
  return std::sqrt(std::max(0.0, std::max(P_[0][0], P_[1][1])));
}

void KalmanTracker::positionCovariance(double& pxx, double& pxy, double& pyy) const {
  pxx = P_[0][0];
  pxy = 0.5 * (P_[0][1] + P_[1][0]);
  pyy = P_[1][1];
}

void KalmanTracker::inflateVelocity(double factor) {
  if (!initialized_) return;
  P_[2][2] *= factor;
  P_[3][3] *= factor;
}

void KalmanTracker::symmetrize() {
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const double v = 0.5 * (P_[i][j] + P_[j][i]);
      P_[i][j] = v;
      P_[j][i] = v;
    }
  }
}

// =============================================================================
// 四、绿球检测器
// =============================================================================
BallDetector::BallDetector(const DetectorConfig& cfg) { setConfig(cfg); }

void BallDetector::setConfig(const DetectorConfig& cfg) {
  cfg_ = cfg;
  kernel_open_ = (cfg_.morph_open >= 2)
                     ? cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                 cv::Size(cfg_.morph_open, cfg_.morph_open))
                     : cv::Mat();
  kernel_close_ = (cfg_.morph_close >= 2)
                      ? cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                  cv::Size(cfg_.morph_close, cfg_.morph_close))
                      : cv::Mat();
}

Detection BallDetector::detect(const cv::Mat& bgr, const cv::Rect* roi, double expected_radius,
                               double radius_tol) {
  Detection best;
  if (bgr.empty() || bgr.type() != CV_8UC3) return best;

  // ---- 1. 确定搜索窗口：ROI 局部搜索 或 全图 ----
  cv::Rect win(0, 0, bgr.cols, bgr.rows);
  const bool limited = (roi != nullptr);
  if (limited) {
    win = (*roi) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (win.width < 4 || win.height < 4) return best;  // 窗口完全出界
  }
  const cv::Mat view = bgr(win);  // 子矩阵，不拷贝像素

  // ---- 2. 颜色空间转换 + 去噪 ----
  cv::cvtColor(view, hsv_, cv::COLOR_BGR2HSV);
  int blur = cfg_.blur_ksize;
  if (blur >= 3) {
    if (blur % 2 == 0) ++blur;  // 中值滤波核必须是奇数
    cv::medianBlur(hsv_, hsv_, blur);
  }

  // ---- 3. 颜色阈值分割（绿球：H 在绿色区间，S/V 足够高）----
  cv::inRange(hsv_, cv::Scalar(cfg_.h_min, cfg_.s_min, cfg_.v_min),
              cv::Scalar(cfg_.h_max, cfg_.s_max, cfg_.v_max), mask_);

  // ---- 4. 形态学：先开后闭 ----
  if (!kernel_open_.empty()) cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, kernel_open_);
  if (!kernel_close_.empty()) cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, kernel_close_);

  // ---- 5. 轮廓提取与几何筛选 ----
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const cv::Point2d win_center(win.x + 0.5 * win.width, win.y + 0.5 * win.height);
  const double win_diag = std::hypot(static_cast<double>(win.width), static_cast<double>(win.height));
  // 全图搜索时不启用“靠近中心”权重，否则会错误地偏向画面中心
  const double w_prox = limited ? cfg_.w_proximity : 0.0;
  const bool use_radius_gate = (expected_radius > 1e-3 && radius_tol > 0.0);

  // 先收集所有通过几何筛选的候选，再统一评分：
  // 尺寸先验需要知道"本帧最大的合法候选半径"，因此必须两遍扫描。
  struct Cand {
    cv::Point2d center;  ///< 全图坐标
    cv::Point2d local;   ///< 窗口局部坐标
    double radius{0.0};
    double area{0.0};
    double circularity{0.0};
    double fill{0.0};
    double aspect{1.0};
    double score{0.0};
  };
  std::vector<Cand> cands;
  cands.reserve(contours.size());
  double radius_max = 0.0;

  for (std::size_t ci = 0; ci < contours.size(); ++ci) {
    const std::vector<cv::Point>& c = contours[ci];

    const double area = cv::contourArea(c);
    if (area < cfg_.min_area || area > cfg_.max_area) continue;

    const double peri = cv::arcLength(c, true);
    if (peri < 1e-6) continue;

    // 圆度: 完美圆 = 1
    const double circularity = 4.0 * CV_PI * area / (peri * peri);
    if (circularity < cfg_.min_circularity) continue;

    cv::Point2f cc(0.f, 0.f);
    float rr = 0.f;
    cv::minEnclosingCircle(c, cc, rr);
    const double radius = static_cast<double>(rr);
    if (radius < cfg_.min_radius || radius > cfg_.max_radius) continue;

    // 填充率: 球在图像上的投影应几乎填满最小外接圆
    const double circle_area = CV_PI * radius * radius;
    const double fill = (circle_area > 1e-9) ? (area / circle_area) : 0.0;
    if (fill < cfg_.min_fill) continue;

    const cv::Rect bb = cv::boundingRect(c);
    if (bb.height < 1) continue;
    const double aspect = static_cast<double>(bb.width) / static_cast<double>(bb.height);
    if (aspect < cfg_.min_aspect || aspect > cfg_.max_aspect) continue;

    // ---- 尺寸一致性门限 ----
    // 背景里的"小绿斑"(视频压缩伪影、草地、反光)往往又圆又实, 圆度甚至高于真球,
    // 只靠形状无法剔除。这里用追踪中的半径 EMA 做尺寸约束, 把它们挡在门外。
    if (use_radius_gate) {
      const double dev = std::fabs(radius - expected_radius) / expected_radius;
      if (dev > radius_tol) continue;
    }

    // 亚像素质心: 图像矩比最小外接圆圆心更稳、抖动更小
    cv::Point2d center_local(cc.x, cc.y);
    const cv::Moments mo = cv::moments(c);
    if (mo.m00 > 1e-9) center_local = cv::Point2d(mo.m10 / mo.m00, mo.m01 / mo.m00);

    Cand cd;
    cd.local = center_local;
    cd.center = cv::Point2d(center_local.x + win.x, center_local.y + win.y);
    cd.radius = radius;
    cd.area = area;
    cd.circularity = circularity;
    cd.fill = fill;
    cd.aspect = aspect;
    cands.push_back(cd);
    radius_max = std::max(radius_max, radius);
  }

  // ---- 综合评分: 形状 + 尺寸先验 + 靠近预测位置 ----
  // 尺寸先验(相对本帧最大合法候选)非常重要: 全图重捕时没有位置先验,
  // 若只看圆度/填充率, 5px 的压缩伪影会击败 16px 的真球(实测过)。
  double best_score = -1.0e9;
  for (std::size_t k = 0; k < cands.size(); ++k) {
    const Cand& cd = cands[k];
    double prox = std::hypot(cd.local.x - win_center.x, cd.local.y - win_center.y);
    prox = std::min(1.0, prox / std::max(1.0, 0.5 * win_diag));
    const double size_prior = (radius_max > 1e-6) ? (cd.radius / radius_max) : 0.0;
    const double score = cfg_.w_circularity * cd.circularity + cfg_.w_fill * cd.fill +
                         cfg_.w_size * size_prior - w_prox * prox;

    if (score > best_score) {
      best_score = score;
      best.found = true;
      best.center = cd.center;
      best.radius = cd.radius;
      best.area = cd.area;
      best.circularity = cd.circularity;
      best.fill = cd.fill;
      best.aspect = cd.aspect;
      best.score = score;
    }
  }

  return best;
}

// =============================================================================
// 五、追踪流水线
// =============================================================================
BallTrackerPipeline::BallTrackerPipeline(const AppConfig& cfg)
    : cfg_(cfg), detector_(cfg.detector), tracker_(cfg.tracker) {}

void BallTrackerPipeline::reset() {
  tracker_.clear();
  state_ = TrackState::kInit;
  miss_count_ = 0;
  lost_frames_ = 0;
  need_velocity_seed_ = false;
  has_prev_detection_ = false;
  speed_display_ = 0.0;
  has_speed_display_ = false;
  last_speed_display_ = 0.0;
  accel_display_ = 0.0;
  has_accel_ = false;
}

/// 根据卡尔曼预测结果动态开窗：
///   half = base + k_sigma*σ_pos + k_radius*r + |v|*lookahead
/// 物理含义：越不确定、目标越大、速度越快 -> 窗口越大；丢失越久 -> 指数放大。
cv::Rect BallTrackerPipeline::computeSearchRoi(const cv::Size& size) const {
  const Vec2d v = tracker_.velocity();
  const double speed = std::hypot(v.x, v.y);

  double half = cfg_.roi.base_half + cfg_.roi.k_sigma * tracker_.posSigma() +
                cfg_.roi.k_radius * std::max(4.0, tracker_.radius()) +
                speed * cfg_.roi.lookahead_time;

  if (miss_count_ > 0) {
    half *= std::pow(cfg_.roi.miss_grow, static_cast<double>(std::min(miss_count_, 3)));
  }
  half = std::clamp(half, cfg_.roi.min_half, cfg_.roi.max_half);

  // 窗口已经接近整幅图像时，直接全图（避免无意义的裁剪开销）
  if (half > cfg_.roi.full_frame_ratio * 0.5 * std::min(size.width, size.height)) {
    return cv::Rect(0, 0, size.width, size.height);
  }

  const Vec2d p = tracker_.position();
  const cv::Rect r(cvRound(p.x - half), cvRound(p.y - half), cvRound(2.0 * half),
                   cvRound(2.0 * half));
  return r & cv::Rect(0, 0, size.width, size.height);
}

cv::Rect BallTrackerPipeline::growRect(const cv::Rect& r, double scale, const cv::Size& size) {
  const double cx = r.x + 0.5 * r.width;
  const double cy = r.y + 0.5 * r.height;
  const double hw = 0.5 * r.width * scale;
  const double hh = 0.5 * r.height * scale;
  const cv::Rect g(cvRound(cx - hw), cvRound(cy - hh), cvRound(2.0 * hw), cvRound(2.0 * hh));
  return g & cv::Rect(0, 0, size.width, size.height);
}

void BallTrackerPipeline::fillOutput(TrackResult& out) const {
  const Vec2d p = tracker_.position();
  const Vec2d v = tracker_.velocity();
  out.position = cv::Point2d(p.x, p.y);
  out.velocity = cv::Point2d(v.x, v.y);
  out.speed = tracker_.speed();
  out.speed_sigma = tracker_.speedSigma();
  out.radius = tracker_.radius();
  out.mahalanobis = tracker_.mahalanobis();
  out.maneuver = tracker_.maneuver();
  tracker_.positionCovariance(out.pxx, out.pxy, out.pyy);
}

TrackResult BallTrackerPipeline::process(const cv::Mat& bgr, double dt) {
  TrackResult out;
  out.dt = std::clamp(dt, 1e-4, cfg_.tracker.max_dt);
  if (bgr.empty()) {
    out.state = state_;
    return out;
  }

  // ---------------------------------------------------------------------------
  // 阶段 0: 尚未捕获 -> 全图搜索
  // ---------------------------------------------------------------------------
  if (!tracker_.initialized()) {
    const Detection det = detector_.detect(bgr, nullptr);
    out.detection = det;
    out.roi.rect = cv::Rect(0, 0, bgr.cols, bgr.rows);
    out.roi.limited = false;
    out.roi.half = 0.5 * std::max(bgr.cols, bgr.rows);

    if (!det.found) {
      state_ = TrackState::kLost;
      ++lost_frames_;
      out.state = state_;
      out.miss_count = lost_frames_;
      return out;
    }

    tracker_.reset(det.center.x, det.center.y, det.radius);
    prev_detection_ = det.center;
    has_prev_detection_ = true;
    need_velocity_seed_ = true;  // 下一帧用两点差分给出初速度
    miss_count_ = 0;
    lost_frames_ = 0;
    state_ = TrackState::kTracking;

    out.state = state_;
    out.measurement_used = true;
    out.miss_count = 0;
    fillOutput(out);
    return out;
  }

  // ---------------------------------------------------------------------------
  // 阶段 1: 时间更新（先预测，后观测；丢失时只做这一步 -> 惯性外推）
  // ---------------------------------------------------------------------------
  tracker_.predict(out.dt);

  // ---------------------------------------------------------------------------
  // 阶段 2: ROI 局部搜索  —— 只在预测位置附近做分割，计算量大幅下降
  //   TRACKING / COASTING : 局部窗口（COASTING 时窗口随 miss 次数自动放大）
  //   LOST               : 退化为全图搜索，重新捕获目标
  // ---------------------------------------------------------------------------
  Detection det;
  cv::Rect roi;
  const bool local = (state_ == TrackState::kTracking || state_ == TrackState::kCoasting);

  if (local) {
    roi = computeSearchRoi(bgr.size());
    // 带上追踪中的半径作为尺寸先验: 背景小绿斑(压缩伪影/反光)圆度更高但尺寸不符
    det = detector_.detect(bgr, &roi, tracker_.radius(), 0.6);
    if (!det.found && state_ == TrackState::kTracking) {
      // 第一次没找到：放大窗口再试一次（目标可能刚好跑出窗口边界）
      roi = growRect(roi, cfg_.roi.miss_grow, bgr.size());
      det = detector_.detect(bgr, &roi, tracker_.radius(), 0.6);
    }
  } else {
    roi = cv::Rect(0, 0, bgr.cols, bgr.rows);
    // LOST 全图重捕: 尺寸门限放宽; 若滤波器已复位(radius=0)则自动关闭该门限
    det = detector_.detect(bgr, nullptr, tracker_.radius(), 1.0);
  }

  out.roi.rect = roi;
  out.roi.limited = local;
  out.roi.half = local ? 0.5 * std::max(roi.width, roi.height)
                       : 0.5 * std::max(bgr.cols, bgr.rows);
  out.detection = det;

  // ---------------------------------------------------------------------------
  // 阶段 3: 数据关联 + 量测更新
  // ---------------------------------------------------------------------------
  bool used = false;

  if (det.found) {
    const Vec2d p = tracker_.position();
    const double dist = std::hypot(det.center.x - p.x, det.center.y - p.y);
    // 关联门: 位置不确定度 + 目标尺寸 + 固定裕量
    const double gate = 3.0 * tracker_.posSigma() + tracker_.radius() + 25.0;

    if (state_ == TrackState::kLost && dist > std::max(gate, cfg_.tracker.relink_dist)) {
      // 检出位置离原目标太远 -> 判定为“新目标”，直接复位重建航迹
      tracker_.reset(det.center.x, det.center.y, det.radius);
      need_velocity_seed_ = true;
      prev_detection_ = det.center;  // 下一帧用本帧位置做两点差分，避免用到旧航迹的位置
      has_prev_detection_ = true;
      has_speed_display_ = false;  // 显示速度/加速度重新收敛
      has_accel_ = false;
      miss_count_ = 0;
      lost_frames_ = 0;
      state_ = TrackState::kTracking;
    } else {
      if (need_velocity_seed_ && has_prev_detection_) {
        // 两点差分初始化速度：解决起步阶段“速度未知”导致的收敛慢
        const double dx = det.center.x - prev_detection_.x;
        const double dy = det.center.y - prev_detection_.y;
        if (out.dt > 1e-3) {
          const double var =
              2.0 * cfg_.tracker.sigma_z * cfg_.tracker.sigma_z / (out.dt * out.dt);
          tracker_.setVelocity(dx / out.dt, dy / out.dt, var);
        }
        need_velocity_seed_ = false;
      }

      tracker_.update(det.center.x, det.center.y, det.radius);
      prev_detection_ = det.center;
      has_prev_detection_ = true;
      miss_count_ = 0;
      lost_frames_ = 0;
      state_ = TrackState::kTracking;
    }
    used = true;
  } else {
    ++miss_count_;
    ++lost_frames_;
    if (miss_count_ > cfg_.tracker.max_miss_lost) {
      // 彻底丢失: 复位滤波器，下一帧重新全图捕获
      tracker_.clear();
      state_ = TrackState::kLost;
      need_velocity_seed_ = false;
      has_prev_detection_ = false;
      miss_count_ = 0;
      lost_frames_ = 0;
    } else {
      state_ = (miss_count_ > cfg_.tracker.max_miss_coast) ? TrackState::kLost
                                                           : TrackState::kCoasting;
    }
  }

  out.measurement_used = used;
  out.miss_count = miss_count_;
  out.state = state_;

  // ---------------------------------------------------------------------------
  // 阶段 4: 输出（含显示速度低通、加速度估计、物理速度换算）
  // ---------------------------------------------------------------------------
  if (!tracker_.initialized()) {
    speed_display_ = 0.0;
    has_speed_display_ = false;
    last_speed_display_ = 0.0;
    accel_display_ = 0.0;
    has_accel_ = false;
    return out;
  }

  fillOutput(out);

  // 显示速度一阶低通: 让箭头长度平滑，避免抖动（滤波输出已较平滑，此处再抑制显示跳变）
  const double v = tracker_.speed();
  if (!has_speed_display_) {
    speed_display_ = v;
  } else if (cfg_.tracker.speed_tau > 1e-6) {
    const double a = out.dt / (cfg_.tracker.speed_tau + out.dt);
    speed_display_ += a * (v - speed_display_);
  } else {
    speed_display_ = v;
  }

  // 速度变化率(加速度)：直接对应“速度变化”这一标注需求
  if (has_speed_display_ && out.dt > 1e-4) {
    const double raw = (speed_display_ - last_speed_display_) / out.dt;
    accel_display_ = has_accel_ ? (0.7 * accel_display_ + 0.3 * raw) : raw;
    has_accel_ = true;
  }
  last_speed_display_ = speed_display_;
  has_speed_display_ = true;

  out.speed_display = speed_display_;
  out.accel = accel_display_;

  // 物理速度换算（小孔成像，已知球直径即可，无需标定内参）:
  //   Z = f * D_m / d_px   =>   米/像素 = D_m / d_px ，d_px = 2r
  if (cfg_.tracker.enable_speed_mps && out.radius > 1e-3) {
    out.speed_mps = out.speed * cfg_.tracker.ball_diameter_m / (2.0 * out.radius);
  }

  return out;
}

}  // namespace rmv


// ======================= 第四部分：标注绘制实现 ===========================
// =============================================================================
//  visualizer.cpp
//  标注绘制实现：速度箭头(核心) + 拖尾 + 速度曲线 + 加速度箭头 + HUD + 图例
// =============================================================================



namespace rmv {

namespace {

constexpr int kMaxHistory = 400;  ///< 速度曲线保留的最大历史帧数

/// 半透明底板
void drawPanel(cv::Mat& frame, const cv::Rect& r, double alpha) {
  const cv::Rect rc = r & cv::Rect(0, 0, frame.cols, frame.rows);
  if (rc.width <= 0 || rc.height <= 0) return;
  cv::Mat roi = frame(rc);
  cv::Mat tint(roi.size(), roi.type(), cv::Scalar(15, 15, 15));
  cv::addWeighted(roi, 1.0 - alpha, tint, alpha, 0.0, roi);
}

/// 带黑色描边的文字，保证在任何背景上都清晰
void putTextShadow(cv::Mat& frame, const std::string& text, const cv::Point& org, double scale,
                   const cv::Scalar& color, int thickness = 1) {
  cv::putText(frame, text, org + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
  cv::putText(frame, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

/// 把文字起点夹到画面内，避免目标贴边时标注文字被裁掉
cv::Point clampTextOrg(const cv::Mat& frame, const cv::Point& org, const std::string& text,
                       double scale) {
  int baseline = 0;
  const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
  cv::Point p = org;
  p.x = std::max(2, std::min(p.x, frame.cols - sz.width - 2));
  p.y = std::max(sz.height + 2, std::min(p.y, frame.rows - 2));
  return p;
}

}  // namespace

Visualizer::Visualizer(const VizConfig& cfg) : cfg_(cfg) {}

void Visualizer::reset() {
  trail_.clear();
  trail_speed_.clear();
  speed_hist_.clear();
}

cv::Scalar Visualizer::speedColor(double speed, double full_scale) {
  double t = (full_scale > 1e-9) ? (speed / full_scale) : 0.0;
  t = std::clamp(t, 0.0, 1.0);
  // BGR: 青(255,255,0) -> 黄(0,255,255) -> 红(0,0,255)
  if (t < 0.5) {
    const double u = t / 0.5;
    return cv::Scalar(255.0 * (1.0 - u), 255.0, 255.0 * u);
  }
  const double u = (t - 0.5) / 0.5;
  return cv::Scalar(0.0, 255.0 * (1.0 - u), 255.0);
}

void Visualizer::pushHistory(const TrackResult& tr) {
  const bool alive = (tr.state == TrackState::kTracking || tr.state == TrackState::kCoasting);
  if (!alive) {
    // 长时间丢失时断掉拖尾，避免重捕后出现一条跨越画面的假轨迹
    if (tr.state == TrackState::kLost) {
      trail_.clear();
      trail_speed_.clear();
    }
    return;
  }
  trail_.push_back(tr.position);
  trail_speed_.push_back(tr.speed_display);
  speed_hist_.push_back(tr.speed_display);

  while (static_cast<int>(trail_.size()) > std::max(2, cfg_.trail_len)) {
    trail_.pop_front();
    trail_speed_.pop_front();
  }
  while (static_cast<int>(speed_hist_.size()) > kMaxHistory) speed_hist_.pop_front();
}

void Visualizer::draw(cv::Mat& frame, const TrackResult& tr) {
  if (frame.empty()) return;

  pushHistory(tr);

  if (cfg_.draw_roi) drawRoi(frame, tr);
  if (cfg_.draw_trail) drawTrail(frame);
  if (cfg_.draw_uncertainty) drawUncertainty(frame, tr);
  drawTargetMarker(frame, tr);
  if (cfg_.draw_arrow) drawArrow(frame, tr);
  if (cfg_.draw_accel_arrow) drawAccelArrow(frame, tr);
  if (cfg_.draw_hud) drawHud(frame, tr);
  if (cfg_.draw_speed_chart) drawSpeedChart(frame);
  if (cfg_.draw_legend) drawLegend(frame);
}

// -----------------------------------------------------------------------------
// ROI 搜索窗口
// -----------------------------------------------------------------------------
void Visualizer::drawRoi(cv::Mat& frame, const TrackResult& tr) const {
  if (!tr.roi.limited) return;  // 全图搜索时没必要画框
  const cv::Rect rc = tr.roi.rect & cv::Rect(0, 0, frame.cols, frame.rows);
  if (rc.width < 2 || rc.height < 2) return;
  cv::rectangle(frame, rc, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);
  putTextShadow(frame, "ROI", cv::Point(rc.x + 4, rc.y + 14), 0.42, cv::Scalar(160, 160, 160), 1);
}

// -----------------------------------------------------------------------------
// 运动拖尾：颜色同样编码速度，直观展示“哪里快、哪里慢”
// -----------------------------------------------------------------------------
void Visualizer::drawTrail(cv::Mat& frame) const {
  if (trail_.size() < 2) return;
  const int n = static_cast<int>(trail_.size());
  for (int i = 1; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    const double thickness = cfg_.trail_thickness * (0.35 + 0.65 * t);
    const double sp = trail_speed_[static_cast<std::size_t>(i)];
    cv::line(frame, trail_[static_cast<std::size_t>(i - 1)],
             trail_[static_cast<std::size_t>(i)], speedColor(sp, cfg_.speed_full_scale),
             std::max(1, cvRound(thickness)), cv::LINE_AA);
  }
}

// -----------------------------------------------------------------------------
// 卡尔曼位置不确定度椭圆 (2σ)：直接解释“为什么丢失时 ROI 会自动变大”
// -----------------------------------------------------------------------------
void Visualizer::drawUncertainty(cv::Mat& frame, const TrackResult& tr) const {
  if (tr.state == TrackState::kInit) return;
  const double a = tr.pxx;
  const double b = tr.pxy;
  const double c = tr.pyy;
  if (a <= 0.0 || c <= 0.0) return;

  const double diff = a - c;
  const double disc = std::sqrt(std::max(0.0, diff * diff + 4.0 * b * b));
  const double l1 = std::max(1e-6, 0.5 * (a + c + disc));  // 大特征值
  const double l2 = std::max(1e-6, 0.5 * (a + c - disc));  // 小特征值
  const double angle = 0.5 * std::atan2(2.0 * b, diff) * 180.0 / CV_PI;

  const double k = 2.0;  // 2σ 置信椭圆
  const cv::Size axes(cvRound(k * std::sqrt(l1)), cvRound(k * std::sqrt(l2)));
  if (axes.width < 2 || axes.height < 2) return;

  const cv::Point center(cvRound(tr.position.x), cvRound(tr.position.y));
  cv::ellipse(frame, center, axes, angle, 0.0, 360.0, cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
}

// -----------------------------------------------------------------------------
// 目标标记：原始量测(绿圈) vs 滤波估计(白色十字)
// -----------------------------------------------------------------------------
void Visualizer::drawTargetMarker(cv::Mat& frame, const TrackResult& tr) const {
  if (tr.detection.found) {
    const cv::Point c(cvRound(tr.detection.center.x), cvRound(tr.detection.center.y));
    cv::circle(frame, c, std::max(1, cvRound(tr.detection.radius)), cv::Scalar(0, 255, 0), 2,
               cv::LINE_AA);
  }

  if (tr.state == TrackState::kInit) return;

  const cv::Point p(cvRound(tr.position.x), cvRound(tr.position.y));
  const int arm = 7;
  cv::line(frame, cv::Point(p.x - arm, p.y), cv::Point(p.x + arm, p.y), cv::Scalar(255, 255, 255),
           1, cv::LINE_AA);
  cv::line(frame, cv::Point(p.x, p.y - arm), cv::Point(p.x, p.y + arm), cv::Scalar(255, 255, 255),
           1, cv::LINE_AA);
  cv::circle(frame, p, 3, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);

  // 在目标附近标出速度数值(px/s 与 m/s)
  std::string txt = cv::format("%.0f px/s", tr.speed_display);
  if (tr.speed_mps > 0.001) txt += cv::format("  %.2f m/s", tr.speed_mps);
  const double scale = 0.55;
  const cv::Point label_org(p.x + cvRound(tr.radius) + 10, p.y - cvRound(tr.radius) - 6);
  putTextShadow(frame, txt, clampTextOrg(frame, label_org, txt, scale), scale,
                speedColor(tr.speed_display, cfg_.speed_full_scale), 1);
}

// -----------------------------------------------------------------------------
// 核心标注：速度箭头
//   方向 = 速度方向；长度 = |v| / 满量程 * 最大长度，并按图像对角线归一化
// -----------------------------------------------------------------------------
void Visualizer::drawArrow(cv::Mat& frame, const TrackResult& tr) const {
  if (tr.state == TrackState::kInit) return;
  const double speed = tr.speed_display;
  const double v_dead = cfg_.speed_dead_zone;
  if (speed < v_dead) return;  // 基本静止 -> 不画方向

  const double diag =
      std::hypot(static_cast<double>(frame.cols), static_cast<double>(frame.rows));
  const double len_min = cfg_.arrow_frac_min * diag;  // 静止边界处的可见长度
  const double len_max = cfg_.arrow_frac_max * diag;  // 满量程速度对应的长度

  // 仿射映射: L(v_dead) = len_min, L(v_full) = len_max
  // 严格单调递增 -> “箭头越长速度越快”始终成立，同时保证低速时箭头仍然可见
  const double v_full = std::max(v_dead + 1.0, cfg_.speed_full_scale);
  const double t = std::clamp((speed - v_dead) / (v_full - v_dead), 0.0, 1.0);
  const double len = len_min + t * (len_max - len_min);

  const double n = std::hypot(tr.velocity.x, tr.velocity.y);
  if (n < 1e-6) return;
  const double ux = tr.velocity.x / n;
  const double uy = tr.velocity.y / n;

  const cv::Point p0(cvRound(tr.position.x), cvRound(tr.position.y));
  const cv::Point p1(cvRound(tr.position.x + ux * len), cvRound(tr.position.y + uy * len));

  cv::arrowedLine(frame, p0, p1, speedColor(speed, cfg_.speed_full_scale),
                  cfg_.arrow_thickness, cv::LINE_AA, 0, 0.32);
}

// -----------------------------------------------------------------------------
// 速度变化率(加速度)箭头：洋红色虚线
//   沿速度方向 => 加速；反速度方向 => 减速
// -----------------------------------------------------------------------------
void Visualizer::drawAccelArrow(cv::Mat& frame, const TrackResult& tr) const {
  if (tr.state == TrackState::kInit) return;
  if (tr.speed < cfg_.speed_dead_zone) return;

  const double a = tr.accel;
  if (std::fabs(a) < 0.08 * cfg_.accel_full_scale) return;

  const double n = std::hypot(tr.velocity.x, tr.velocity.y);
  if (n < 1e-6) return;
  double ux = tr.velocity.x / n;
  double uy = tr.velocity.y / n;
  if (a < 0.0) {  // 减速：反向
    ux = -ux;
    uy = -uy;
  }

  const double diag =
      std::hypot(static_cast<double>(frame.cols), static_cast<double>(frame.rows));
  const double len_max = 0.5 * cfg_.arrow_frac_max * diag;
  const double len =
      std::clamp(std::fabs(a) / std::max(1.0, cfg_.accel_full_scale), 0.0, 1.0) * len_max;
  if (len < 8.0) return;

  const cv::Point2d p0(tr.position.x, tr.position.y);
  const int segs = std::max(3, cvRound(len / 10.0));
  for (int i = 0; i < segs; i += 2) {
    const double t0 = static_cast<double>(i) / segs;
    const double t1 = std::min(1.0, static_cast<double>(i + 1) / segs);
    cv::line(frame, cv::Point(cvRound(p0.x + ux * len * t0), cvRound(p0.y + uy * len * t0)),
             cv::Point(cvRound(p0.x + ux * len * t1), cvRound(p0.y + uy * len * t1)),
             cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
  }

  const cv::Point tip(cvRound(p0.x + ux * len), cvRound(p0.y + uy * len));
  const cv::Point tail(cvRound(p0.x + ux * len * 0.82), cvRound(p0.y + uy * len * 0.82));
  cv::arrowedLine(frame, tail, tip, cv::Scalar(255, 0, 255), 2, cv::LINE_AA, 0, 0.45);
  const std::string atxt = (a > 0.0 ? "a: speed up" : "a: slow down");
  putTextShadow(frame, atxt, clampTextOrg(frame, tip + cv::Point(8, -8), atxt, 0.5), 0.5,
                cv::Scalar(255, 0, 255), 1);
}

// -----------------------------------------------------------------------------
// HUD 信息面板
// -----------------------------------------------------------------------------
void Visualizer::drawHud(cv::Mat& frame, const TrackResult& tr) const {
  const double roi_ratio =
      tr.roi.limited
          ? std::min(1.0, (2.0 * tr.roi.half) * (2.0 * tr.roi.half) /
                              (static_cast<double>(frame.cols) * frame.rows))
          : 1.0;

  std::vector<std::string> lines;
  lines.push_back(cv::format("[STATE] %-9s  frame %ld  fps %.1f", toString(tr.state), frame_index_,
                             fps_));
  lines.push_back(cv::format("|v|     %8.1f px/s", tr.speed_display));
  if (tr.speed_mps > 0.001) lines.push_back(cv::format("        %8.2f m/s", tr.speed_mps));
  lines.push_back(cv::format("vx/vy   %8.1f / %.1f px/s", tr.velocity.x, tr.velocity.y));
  lines.push_back(cv::format("a       %8.1f px/s^2  %s", tr.accel,
                             tr.accel > 0.0 ? "(speed up)" : "(slow down)"));
  lines.push_back(
      cv::format("sigma   %8.1f px/s   d2 %.2f%s", tr.speed_sigma, tr.mahalanobis,
                 tr.maneuver ? "  [MANEUVER]" : ""));
  lines.push_back(cv::format("ROI     %4dx%-4d  (%.1f%% of frame)", tr.roi.rect.width,
                             tr.roi.rect.height, 100.0 * roi_ratio));
  lines.push_back(cv::format("miss    %8d   radius %.1f px", tr.miss_count, tr.radius));

  const int line_h = 22;
  const int pad = 10;
  int text_w = 0;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    int baseline = 0;
    const cv::Size sz = cv::getTextSize(lines[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    text_w = std::max(text_w, sz.width);
  }
  const cv::Rect panel(pad, pad, std::min(frame.cols - 2 * pad, text_w + 2 * pad),
                       static_cast<int>(lines.size()) * line_h + pad);
  drawPanel(frame, panel, 0.55);
  cv::rectangle(frame, panel, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

  const cv::Scalar state_color = (tr.state == TrackState::kTracking) ? cv::Scalar(0, 255, 0)
                                 : (tr.state == TrackState::kCoasting) ? cv::Scalar(0, 255, 255)
                                                                       : cv::Scalar(0, 0, 255);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const cv::Point org(panel.x + pad, panel.y + pad + static_cast<int>(i) * line_h + 6);
    const cv::Scalar color = (i == 0) ? state_color : cv::Scalar(235, 235, 235);
    cv::putText(frame, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
  }
}

// -----------------------------------------------------------------------------
// 速度-时间曲线（右下角），直接体现“速度变化”
// -----------------------------------------------------------------------------
void Visualizer::drawSpeedChart(cv::Mat& frame) const {
  const int w = std::min(cfg_.chart_width, frame.cols / 2);
  const int h = std::min(cfg_.chart_height, frame.rows / 3);
  if (w < 80 || h < 60) return;

  const int margin = 16;
  const cv::Rect panel(frame.cols - w - margin, frame.rows - h - margin, w, h);
  drawPanel(frame, panel, 0.6);
  cv::rectangle(frame, panel, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

  const int top = panel.y + 22;
  const int bottom = panel.y + panel.height - 16;
  const int plot_h = std::max(1, bottom - top);

  // 网格
  for (int i = 1; i <= 3; ++i) {
    const int y = top + plot_h * i / 4;
    cv::line(frame, cv::Point(panel.x + 1, y), cv::Point(panel.x + panel.width - 2, y),
             cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
  }

  putTextShadow(frame, "speed (px/s)", cv::Point(panel.x + 8, panel.y + 16), 0.42,
                cv::Scalar(200, 200, 200), 1);
  putTextShadow(frame, cv::format("%.0f", cfg_.speed_full_scale),
                cv::Point(panel.x + 8, top + 12), 0.38, cv::Scalar(170, 170, 170), 1);
  putTextShadow(frame, "0", cv::Point(panel.x + 8, bottom + 12), 0.38, cv::Scalar(170, 170, 170),
                1);

  const int n = static_cast<int>(speed_hist_.size());
  if (n < 2) return;

  const int show = std::min(n, panel.width - 2);
  // 速度 -> 屏幕 y
  const double full = std::max(1.0, cfg_.speed_full_scale);
  for (int i = 1; i < show; ++i) {
    const double s0 = speed_hist_[static_cast<std::size_t>(n - show + i - 1)];
    const double s1 = speed_hist_[static_cast<std::size_t>(n - show + i)];
    const int y0 = bottom - cvRound(std::clamp(s0 / full, 0.0, 1.0) * plot_h);
    const int y1 = bottom - cvRound(std::clamp(s1 / full, 0.0, 1.0) * plot_h);
    cv::line(frame, cv::Point(panel.x + i, y0), cv::Point(panel.x + i + 1, y1),
             speedColor(s1, cfg_.speed_full_scale), 2, cv::LINE_AA);
  }

  // 当前值
  const double cur = speed_hist_.back();
  const int yc = bottom - cvRound(std::clamp(cur / full, 0.0, 1.0) * plot_h);
  cv::circle(frame, cv::Point(panel.x + show, yc), 3, cv::Scalar(255, 255, 255), cv::FILLED,
             cv::LINE_AA);
}

// -----------------------------------------------------------------------------
// 图例：速度色标 + 箭头长度规则说明
// -----------------------------------------------------------------------------
void Visualizer::drawLegend(cv::Mat& frame) const {
  const int bar_w = 170;
  const int bar_h = 12;
  if (frame.cols < bar_w + 160) return;

  const int x = frame.cols - bar_w - 20;
  const int y = 32;
  const cv::Rect panel(x - 10, y - 28, bar_w + 20, bar_h + 64);
  drawPanel(frame, panel, 0.55);
  cv::rectangle(frame, panel, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

  putTextShadow(frame, "speed color / arrow length", cv::Point(x, y - 10), 0.42,
                cv::Scalar(220, 220, 220), 1);

  for (int i = 0; i < bar_w; ++i) {
    const double s = cfg_.speed_full_scale * static_cast<double>(i) / (bar_w - 1);
    cv::line(frame, cv::Point(x + i, y), cv::Point(x + i, y + bar_h),
             speedColor(s, cfg_.speed_full_scale), 1);
  }
  cv::rectangle(frame, cv::Rect(x, y, bar_w, bar_h), cv::Scalar(120, 120, 120), 1, cv::LINE_AA);

  putTextShadow(frame, "slow", cv::Point(x, y + bar_h + 14), 0.4, cv::Scalar(200, 200, 200), 1);
  putTextShadow(frame, cv::format("fast %.0f px/s", cfg_.speed_full_scale),
                cv::Point(x + bar_w - 96, y + bar_h + 14), 0.4, cv::Scalar(200, 200, 200), 1);
  putTextShadow(frame, "longer arrow = faster", cv::Point(x, y + bar_h + 30), 0.4,
                cv::Scalar(200, 200, 200), 1);
}

}  // namespace rmv


// ===================== 第五部分：自检（卡尔曼单元测试） ====================
// =============================================================================
//  自检（卡尔曼滤波单元测试，离线，不需要视频/摄像头）
//
//  运行方式:  ./rm_ball_tracker --selftest
//
//  验证内容:
//    1) 匀速运动下, 滤波后的位置误差应明显小于原始检测(量测)误差
//       —— 这正是“用卡尔曼滤波而不是帧差/直接用检测值”的价值所在
//    2) 速度估计应收敛到真值
//    3) 目标机动(突然变向)时应被卡方机动检测捕捉到(自适应 Q 生效)
//    4) 丢失量测时只做预测: 位置不确定度应随时间增大(ROI 因此自动放大)
// =============================================================================

namespace {

int g_failed = 0;

void report(bool ok, const char* name, double value, double limit) {
  std::printf("  [%s] %-52s value=%9.3f  limit=%9.3f\n", ok ? "PASS" : "FAIL", name, value,
              limit);
  if (!ok) ++g_failed;
}

}  // namespace

int runSelfTest() {
  std::printf("==================== KalmanTracker unit test ====================\n");

  const double dt = 1.0 / 60.0;
  const double vx_true = 300.0;   // px/s
  const double vy_true = -150.0;  // px/s

  rmv::TrackerConfig cfg;  // 使用与主程序一致的默认参数
  rmv::KalmanTracker kf(cfg);

  std::mt19937 gen(20240521u);
  std::normal_distribution<double> noise(0.0, cfg.sigma_z);

  double x = 320.0;
  double y = 240.0;
  const double radius = 10.0;

  kf.reset(x, y, radius);

  double sum_err_meas = 0.0;
  double sum_err_kf = 0.0;
  double max_err_kf = 0.0;
  int cnt = 0;
  bool maneuver_seen = false;

  for (int i = 0; i < 400; ++i) {
    // ---- 目标做匀速直线运动 ----
    x += vx_true * dt;
    y += vy_true * dt;

    if (i == 300) {
      // ---- 制造一次机动: 速度突然反向(模拟球被弹回/急转弯) ----
      // 通过直接跳变真值位置模拟, 检验机动检测是否触发
      x += 120.0;
      y -= 80.0;
    }

    kf.predict(dt);

    const double zx = x + noise(gen);
    const double zy = y + noise(gen);
    kf.update(zx, zy, radius);

    if (kf.maneuver()) maneuver_seen = true;

    if (i > 50 && i < 300) {  // 跳过收敛期, 且避开机动段
      const double err_meas = std::hypot(zx - x, zy - y);
      const double err_kf = std::hypot(kf.position().x - x, kf.position().y - y);
      sum_err_meas += err_meas;
      sum_err_kf += err_kf;
      if (err_kf > max_err_kf) max_err_kf = err_kf;
      ++cnt;
    }
  }

  std::printf("\n[1] 位置估计精度 (匀速段, 共 %d 帧)\n", cnt);
  const double mean_err_meas = sum_err_meas / std::max(1, cnt);
  const double mean_err_kf = sum_err_kf / std::max(1, cnt);
  std::printf("      量测(直接用检测值) 平均误差 = %.3f px\n", mean_err_meas);
  std::printf("      卡尔曼滤波后      平均误差 = %.3f px  (最大 %.3f px)\n", mean_err_kf,
              max_err_kf);
  std::printf("      误差下降 = %.1f%%\n",
              100.0 * (1.0 - mean_err_kf / std::max(1e-9, mean_err_meas)));
  report(mean_err_kf < mean_err_meas, "滤波后位置误差 < 量测误差", mean_err_kf, mean_err_meas);
  report(mean_err_kf < 0.8 * mean_err_meas, "位置误差至少下降 20%", mean_err_kf,
         0.8 * mean_err_meas);

  std::printf("\n[2] 速度估计收敛性\n");
  const double verr = std::hypot(kf.velocity().x - vx_true, kf.velocity().y - vy_true);
  std::printf("      估计速度 = (%.1f, %.1f) px/s, 真值 = (%.1f, %.1f) px/s\n", kf.velocity().x,
              kf.velocity().y, vx_true, vy_true);
  report(verr < 60.0, "速度估计误差(机动后仍) < 60 px/s", verr, 60.0);

  std::printf("\n[3] 机动检测 (卡方检验 + 自适应 Q)\n");
  report(maneuver_seen, "机动事件被检测到", maneuver_seen ? 1.0 : 0.0, 1.0);

  std::printf("\n[4] 丢失量测时不确定度增长 (纯预测 -> ROI 自动放大)\n");
  const double sigma_before = kf.posSigma();
  for (int i = 0; i < 10; ++i) kf.predict(dt);  // 只预测, 不更新
  const double sigma_after = kf.posSigma();
  std::printf("      10 帧纯预测: sigma %.2f px -> %.2f px\n", sigma_before, sigma_after);
  report(sigma_after > sigma_before, "位置不确定度随时间增大", sigma_after, sigma_before);

  std::printf("\n[5] 速度不确定度 (用于速度置信判断)\n");
  std::printf("      speed = %.1f px/s, sigma_v = %.1f px/s\n", kf.speed(), kf.speedSigma());
  report(kf.speedSigma() > 0.0, "速度标准差可估计", kf.speedSigma(), 0.0);

  std::printf("\n=============================== 结果 ===============================\n");
  if (g_failed == 0) {
    std::printf("全部用例通过 ✔\n");
  } else {
    std::printf("有 %d 个用例失败 ✘\n", g_failed);
  }
  std::printf("===================================================================\n");
  return g_failed == 0 ? 0 : 1;
}


// ======================= 第六部分：命令行与主程序 =========================
// =============================================================================
//  main.cpp 内的命令行与主循环（本文件第六部分）
//
//  用法示例（Ubuntu，在 build/ 目录下运行）:
//    ./rm_ball_tracker --video ~/Desktop/homework.mp4 --output out.avi --csv track.csv
//    ./rm_ball_tracker --video ~/Desktop/homework.mp4 --show 1
//    ./rm_ball_tracker --camera 0 --output cam.avi            # 摄像头实时
//    ./rm_ball_tracker --video ~/Desktop/homework.mp4 --hsv 35,90,60,40   # 现场调阈值
//    ./rm_ball_tracker --config ../params.yaml --show 1
//    ./rm_ball_tracker --selftest                             # 卡尔曼滤波自检
//    ./rm_ball_tracker --help
//
//  参数写法同时支持 "--key value" 与 "--key=value" 两种形式。
//  （不使用 cv::CommandLineParser：OpenCV 5 起它不再接受空格分隔的写法，
//    会把值丢掉、字符串变成 "true"；自己解析可保证 4.x/5.x 行为一致。）
//
//  输出的箭头含义:
//    · 箭头的方向 = 小球当前运动方向
//    · 箭头的长度 = 速度大小 (越长越快, 越短越慢)
//    · 箭头颜色   = 速度大小 (青 -> 黄 -> 红, 见图例)
//    · 洋红虚线箭头表示"速度变化": 沿运动方向=加速, 反向=减速
// =============================================================================


#ifdef RMV_HAVE_HIGHGUI
#endif


/// 卡尔曼滤波自检（实现在本文件后面的"自检"小节，用 --selftest 调用，不需要视频输入）
int runSelfTest();

namespace {

// =============================================================================
// 极简命令行解析器
//   · 支持 --key value / --key=value / 短别名 (-v 1 或 -v=1)
//   · 负数不会被误判成选项名 (--camera -1 可用)
//   · 无值开关按 1 处理 (--show 等价于 --show 1)
// =============================================================================
class ArgParser {
 public:
  ArgParser(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.size() < 2 || a[0] != '-') {
        positional_.push_back(a);
        continue;
      }
      std::string key;
      std::string val;
      const std::size_t eq = a.find('=');
      if (eq != std::string::npos) {
        key = a.substr(0, eq);
        val = a.substr(eq + 1);
      } else {
        key = a;
        if (i + 1 < argc && !looksLikeOption(argv[i + 1])) val = argv[++i];
      }
      std::size_t p = 0;
      while (p < key.size() && key[p] == '-') ++p;
      key = canonical(key.substr(p));
      if (!isKnown(key)) {  // 拼错的参数要报出来, 否则会静默使用默认值(很难排查)
        unknown_.push_back(a);
        continue;
      }
      values_[key] = val;
    }
  }

  bool has(const std::string& key) const { return values_.find(key) != values_.end(); }

  std::string getString(const std::string& key, const std::string& def = std::string()) const {
    const std::map<std::string, std::string>::const_iterator it = values_.find(key);
    if (it == values_.end()) return def;
    return it->second;
  }

  /// 无值开关 -> 1
  double getDouble(const std::string& key, double def) const {
    const std::map<std::string, std::string>::const_iterator it = values_.find(key);
    if (it == values_.end()) return def;
    if (it->second.empty()) return 1.0;
    return std::atof(it->second.c_str());
  }

  /// 无值开关 -> 1
  int getInt(const std::string& key, int def) const {
    const std::map<std::string, std::string>::const_iterator it = values_.find(key);
    if (it == values_.end()) return def;
    if (it->second.empty()) return 1;
    return std::atoi(it->second.c_str());
  }

  const std::vector<std::string>& unknown() const { return unknown_; }
  const std::vector<std::string>& positional() const { return positional_; }

 private:
  /// 负数(-1, -0.5)应被当作"值"而不是"选项"
  static bool looksLikeOption(const std::string& s) {
    if (s.size() < 2 || s[0] != '-') return false;
    const char c = s[1];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return false;
    return true;
  }

  static std::string canonical(const std::string& k) {
    if (k == "h" || k == "help") return "help";
    if (k == "v") return "video";
    if (k == "o") return "output";
    if (k == "c") return "camera";
    if (k == "q") return "quiet";
    return k;
  }

  /// 合法参数名白名单(用于识别拼错的参数)
  static bool isKnown(const std::string& k) {
    static const char* kKnown[] = {"help",      "video",    "camera",       "config",
                                   "output",    "csv",      "show",         "scale",
                                   "max-frames", "start-frame", "fourcc",   "hsv",
                                   "speed-scale", "arrow-frac", "ball-diameter", "trail-len",
                                   "chart",     "trail",    "ellipse",      "roi-box",
                                   "hud",       "legend",   "quiet",        "selftest",
                                   "play-fps"};
    for (std::size_t i = 0; i < sizeof(kKnown) / sizeof(kKnown[0]); ++i) {
      if (k == kKnown[i]) return true;
    }
    return false;
  }

  std::map<std::string, std::string> values_;
  std::vector<std::string> unknown_;
  std::vector<std::string> positional_;
};

void printUsage() {
  std::cout <<
      "RM green ball tracker —— 绿球识别与运动矢量(箭头)标注\n"
      "\n"
      "用法: rm_ball_tracker --video <文件> [选项]\n"
      "      rm_ball_tracker --camera <索引> [选项]\n"
      "\n"
      "输入/输出:\n"
      "  -v, --video <path>       输入视频文件\n"
      "  -c, --camera <index>     摄像头索引 (例如 0), 默认 -1 表示不用\n"
      "  -o, --output <path>      输出标注视频 (不指定则不保存)\n"
      "      --csv <path>         导出逐帧数据 CSV (便于画曲线/写报告)\n"
      "      --config <path>      YAML 参数文件 (可选)\n"
      "      --fourcc <code>      输出编码: mp4v(默认)/MJPG/XVID; 失败自动回退 MJPG+avi\n"
      "\n"
      "显示与标注开关 (1=开, 0=关):\n"
      "      --show <0|1>         实时预览窗口 (需要 GUI 环境), 默认 0\n"
      "      --play-fps <f>       预览播放帧率, 0=不限速 (默认: 用 --show 看视频时跟随源帧率)\n"
      "      --chart <0|1>        速度-时间曲线, 默认 1\n"
      "      --trail <0|1>        运动拖尾, 默认 1\n"
      "      --ellipse <0|1>      卡尔曼不确定度椭圆, 默认 1\n"
      "      --roi-box <0|1>      ROI 搜索窗, 默认 1\n"
      "      --hud <0|1>          信息面板, 默认 1\n"
      "      --legend <0|1>       速度色标图例, 默认 1\n"
      "\n"
      "算法参数覆盖:\n"
      "      --hsv hmin,hmax,smin,vmin[,smax,vmax]   覆盖 HSV 阈值\n"
      "      --speed-scale <px/s> 速度满量程, 决定箭头长度与配色 (默认 900)\n"
      "      --arrow-frac <f>     满量程速度对应的箭头长度占对角线比例 (默认 0.15)\n"
      "      --ball-diameter <m>  球的真实直径, 用于把 px/s 换算成 m/s (默认 0.065)\n"
      "      --trail-len <n>      拖尾长度(帧), 默认 60\n"
      "\n"
      "其它:\n"
      "      --selftest           运行卡尔曼滤波自检(不需要视频, 验证滤波实现是否正确)\n"
      "      --scale <f>          处理分辨率缩放, 例如 0.5 提速 (默认 1.0)\n"
      "      --start-frame <n>    从第 n 帧开始处理 (默认 0)\n"
      "      --max-frames <n>     最多处理 n 帧, 0=全部 (默认 0)\n"
      "  -q, --quiet <0|1>        只输出最终汇总, 默认 0\n"
      "  -h, --help               打印本帮助\n"
      "\n"
      "示例:\n"
      "  rm_ball_tracker --video ball.mp4 --output out.avi --csv track.csv\n"
      "  rm_ball_tracker --video ball.mp4 --show 1 --hsv 35,90,60,40\n"
      "  rm_ball_tracker --video ball.mp4 --output out.mp4 --hsv=35,90,60,40\n";
}

/// 替换文件扩展名（输出编码回退时用）
std::string replaceExtension(const std::string& path, const std::string& ext) {
  const std::size_t dot = path.find_last_of('.');
  const std::size_t slash = path.find_last_of("/\\");
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path + ext;
  return path.substr(0, dot) + ext;
}

}  // namespace

int main(int argc, char** argv) {
  const ArgParser args(argc, argv);

  // 自检模式: 只跑卡尔曼滤波的离线测试, 不打开任何视频
  if (args.has("selftest")) return runSelfTest();

  for (std::size_t i = 0; i < args.unknown().size(); ++i) {
    std::cerr << "[WARN] 忽略无法识别的参数: " << args.unknown()[i] << std::endl;
  }
  for (std::size_t i = 0; i < args.positional().size(); ++i) {
    std::cerr << "[WARN] 忽略多余的位置参数: " << args.positional()[i]
              << "  (输入源请用 --video 或 --camera 指定)" << std::endl;
  }

  if (args.has("help") || (argc == 1)) {
    printUsage();
    return (argc == 1) ? 1 : 0;
  }

  try {
    // -------------------------------------------------------------------------
    // 1. 读取参数：默认值 -> YAML -> 命令行覆盖（优先级递增）
    // -------------------------------------------------------------------------
    rmv::AppConfig cfg;
    rmv::VizConfig viz_cfg;

    const std::string config_path = args.getString("config");
    if (!config_path.empty()) {
      if (rmv::loadConfig(config_path, cfg)) {
        std::cout << "[INFO] 已加载参数文件: " << config_path << std::endl;
      } else {
        std::cerr << "[WARN] 参数文件打开失败, 使用默认参数: " << config_path << std::endl;
      }
    }

    const std::string hsv_arg = args.getString("hsv");
    if (!hsv_arg.empty()) {
      int hmin = 0, hmax = 0, smin = 0, vmin = 0, smax = 0, vmax = 0;
      const int cnt =
          std::sscanf(hsv_arg.c_str(), "%d,%d,%d,%d,%d,%d", &hmin, &hmax, &smin, &vmin, &smax, &vmax);
      if (cnt >= 4) {
        cfg.detector.h_min = hmin;
        cfg.detector.h_max = hmax;
        cfg.detector.s_min = smin;
        cfg.detector.v_min = vmin;
        if (cnt >= 6) {
          cfg.detector.s_max = smax;
          cfg.detector.v_max = vmax;
        }
        std::cout << "[INFO] HSV 阈值覆盖为 H[" << cfg.detector.h_min << "," << cfg.detector.h_max
                  << "] S[" << cfg.detector.s_min << "," << cfg.detector.s_max << "] V["
                  << cfg.detector.v_min << "," << cfg.detector.v_max << "]" << std::endl;
      } else {
        std::cerr << "[WARN] --hsv 需要至少 4 个数: hmin,hmax,smin,vmin" << std::endl;
      }
    }

    if (args.getDouble("speed-scale", 0.0) > 0.0) {
      viz_cfg.speed_full_scale = args.getDouble("speed-scale", 900.0);
    }
    if (args.getDouble("arrow-frac", -1.0) > 0.0) {
      viz_cfg.arrow_frac_max = args.getDouble("arrow-frac", 0.15);
    }
    if (args.getDouble("ball-diameter", 0.0) > 0.0) {
      cfg.tracker.ball_diameter_m = args.getDouble("ball-diameter", 0.065);
    }
    if (args.getInt("trail-len", 0) > 0) {
      viz_cfg.trail_len = args.getInt("trail-len", 60);
    }

    viz_cfg.draw_speed_chart = (args.getInt("chart", 1) != 0);
    viz_cfg.draw_trail = (args.getInt("trail", 1) != 0);
    viz_cfg.draw_uncertainty = (args.getInt("ellipse", 1) != 0);
    viz_cfg.draw_roi = (args.getInt("roi-box", 1) != 0);
    viz_cfg.draw_hud = (args.getInt("hud", 1) != 0);
    viz_cfg.draw_legend = (args.getInt("legend", 1) != 0);

    const double scale = args.getDouble("scale", 1.0);
    const int max_frames = args.getInt("max-frames", 0);
    const int start_frame = std::max(0, args.getInt("start-frame", 0));
    const bool show = (args.getInt("show", 0) != 0);
    const bool quiet = (args.getInt("quiet", 0) != 0);
    // 播放限速(帧/秒): 视频文件读帧是不受帧率约束的，不限速时预览会"快进"。
    // 0 = 不限速；用 --show 看视频时默认跟随视频源帧率。
    double play_fps = args.getDouble("play-fps", 0.0);
    const std::string video_path = args.getString("video");
    const int camera_index = args.getInt("camera", -1);
    const std::string csv_path = args.getString("csv");
    std::string out_path = args.getString("output");
    std::string fourcc_str = args.getString("fourcc", "mp4v");

    if (scale <= 0.0) {
      std::cerr << "[ERROR] --scale 必须大于 0" << std::endl;
      return 1;
    }
    if (fourcc_str.size() != 4) {
      std::cerr << "[WARN] --fourcc 必须是 4 个字符, 已回退为 mp4v" << std::endl;
      fourcc_str = "mp4v";
    }

    // -------------------------------------------------------------------------
    // 2. 打开数据源
    // -------------------------------------------------------------------------
    cv::VideoCapture cap;
    bool from_video = false;
    if (!video_path.empty()) {
      cap.open(video_path);
      from_video = true;
    } else if (camera_index >= 0) {
      cap.open(camera_index);
    } else {
      std::cerr << "[ERROR] 请用 --video <路径> 或 --camera <索引> 指定输入源" << std::endl;
      printUsage();
      return 1;
    }

    if (!cap.isOpened()) {
      std::cerr << "[ERROR] 无法打开输入源。视频解码依赖 FFmpeg/GStreamer 后端，"
                   "请确认 OpenCV 编译时启用了相应支持。"
                << std::endl;
      return 1;
    }

    double src_fps = from_video ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    if (!(src_fps > 1.0 && src_fps < 1000.0)) src_fps = 30.0;

    // dt 的来源必须与数据源匹配：
    //   视频文件 -> 用视频自身帧间隔(1/fps)，否则“处理速度”会污染速度估计
    //   摄像头   -> 用真实墙钟间隔(含采集抖动)
    const bool use_wall_dt = !from_video;

    // 预览限速只对视频文件有意义(摄像头采集本身就限速了)；
    // 未显式指定 --play-fps 时，用 --show 看视频就跟随源帧率播放。
    std::string play_desc;
    if (play_fps > 0.0 && from_video) {
      play_desc = cv::format("%.1f fps (手动指定)", play_fps);
    } else if (play_fps <= 0.0 && show && from_video) {
      play_fps = src_fps;
      play_desc = cv::format("%.1f fps (跟随源帧率)", play_fps);
    } else {
      play_fps = 0.0;
      play_desc = "不限速 (按处理速度尽快跑完)";
    }

    std::cout << "======================= RM green ball tracker =======================\n"
              << " 数据源      : "
              << (from_video ? video_path : ("camera " + std::to_string(camera_index))) << "\n"
              << " 源帧率      : " << src_fps << " fps\n"
              << " dt 来源     : " << (use_wall_dt ? "墙钟(实时)" : "视频帧率(离线)") << "\n"
              << " 处理缩放    : " << scale << "\n"
              << " 预览限速    : " << play_desc << "\n"
              << " HSV 阈值    : H[" << cfg.detector.h_min << "," << cfg.detector.h_max << "] S["
              << cfg.detector.s_min << "," << cfg.detector.s_max << "] V[" << cfg.detector.v_min
              << "," << cfg.detector.v_max << "]\n"
              << " 过程噪声 σa : " << cfg.tracker.sigma_a << " px/s^2\n"
              << " 量测噪声 σz : " << cfg.tracker.sigma_z << " px\n"
              << " 速度满量程  : " << viz_cfg.speed_full_scale << " px/s (箭头/配色)\n"
              << " 球直径      : " << cfg.tracker.ball_diameter_m << " m (用于换算 m/s)\n"
              << " 输出视频    : " << (out_path.empty() ? std::string("(不保存)") : out_path)
              << "\n====================================================================="
              << std::endl;

    // -------------------------------------------------------------------------
    // 3. 处理器与可视化器
    // -------------------------------------------------------------------------
    rmv::BallTrackerPipeline pipeline(cfg);
    rmv::Visualizer viz(viz_cfg);

    cv::VideoWriter writer;
    bool writer_ready = false;
    std::string actual_out = out_path;

    std::ofstream csv;
    if (!csv_path.empty()) {
      csv.open(csv_path.c_str());
      if (csv.is_open()) {
        csv << "frame,t,dt,state,used,x,y,vx,vy,speed,speed_display,speed_mps,accel,radius,"
               "mahalanobis,maneuver,roi_w,roi_h,roi_limited,det_found\n";
        std::cout << "[INFO] 逐帧数据将写入: " << csv_path << std::endl;
      } else {
        std::cerr << "[WARN] 无法写入 CSV: " << csv_path << std::endl;
      }
    }

#ifdef RMV_HAVE_HIGHGUI
    const std::string kWindow = "rm_green_ball_tracker";
#else
    if (show) {
      std::cerr << "[WARN] 编译时未启用 highgui, --show 无效; 请改用 --output 保存视频。"
                << std::endl;
    }
#endif

    // -------------------------------------------------------------------------
    // 4. 主循环
    // -------------------------------------------------------------------------
    long frame_index = -1;
    long processed = 0;
    long detected = 0;
    long lost_events = 0;
    double speed_sum = 0.0;
    double speed_max = 0.0;
    double roi_ratio_sum = 0.0;
    long roi_ratio_cnt = 0;
    double fps_measured = 0.0;
    double sim_t = 0.0;
    rmv::TrackState last_state = rmv::TrackState::kInit;

    auto t_prev = std::chrono::steady_clock::now();
    auto t_loop_start = t_prev;  // 播放限速的计时起点（处理第一帧时重置）
    bool loop_timer_started = false;

    while (true) {
      cv::Mat frame;
      if (!cap.read(frame) || frame.empty()) break;
      ++frame_index;
      if (frame_index < start_frame) {
        t_prev = std::chrono::steady_clock::now();  // 跳帧期间不累计 dt
        continue;
      }
      if (max_frames > 0 && processed >= max_frames) break;

      if (!loop_timer_started) {
        t_loop_start = std::chrono::steady_clock::now();
        loop_timer_started = true;
      }

      cv::Mat work;
      if (std::fabs(scale - 1.0) > 1e-6) {
        cv::resize(frame, work, cv::Size(), scale, scale, cv::INTER_LINEAR);
      } else {
        work = frame;
      }

      // --- dt ---
      double dt = 0.0;
      if (use_wall_dt) {
        const auto now = std::chrono::steady_clock::now();
        dt = std::chrono::duration<double>(now - t_prev).count();
        t_prev = now;
      } else {
        dt = 1.0 / src_fps;
      }
      sim_t += dt;

      const auto t_start = std::chrono::steady_clock::now();

      // --- 追踪 ---
      const rmv::TrackResult tr = pipeline.process(work, dt);

      // --- 标注 ---
      cv::Mat canvas = work.clone();
      viz.setFps(fps_measured);
      viz.setFrameIndex(frame_index);
      viz.draw(canvas, tr);

      // --- 统计 ---
      ++processed;
      if (tr.detection.found) ++detected;
      if (tr.state == rmv::TrackState::kLost && last_state != rmv::TrackState::kLost) {
        ++lost_events;
      }
      last_state = tr.state;
      if (tr.state == rmv::TrackState::kTracking || tr.state == rmv::TrackState::kCoasting) {
        speed_sum += tr.speed_display;
        speed_max = std::max(speed_max, tr.speed_display);
      }
      if (tr.roi.limited) {
        const double ratio = std::min(1.0, (2.0 * tr.roi.half) * (2.0 * tr.roi.half) /
                                              (static_cast<double>(canvas.cols) * canvas.rows));
        roi_ratio_sum += ratio;
        ++roi_ratio_cnt;
      }

      // --- 写出 ---
      if (!out_path.empty()) {
        if (!writer_ready) {
          const int code = cv::VideoWriter::fourcc(fourcc_str[0], fourcc_str[1], fourcc_str[2],
                                                   fourcc_str[3]);
          writer.open(out_path, code, src_fps, canvas.size(), true);
          if (!writer.isOpened()) {
            actual_out = replaceExtension(out_path, ".avi");
            std::cerr << "[WARN] 无法以 " << fourcc_str << " 编码写出, 回退 MJPG -> " << actual_out
                      << std::endl;
            writer.open(actual_out, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), src_fps,
                        canvas.size(), true);
          }
          if (!writer.isOpened()) {
            std::cerr << "[ERROR] 输出视频打开失败, 后续不再写出。"
                         "可尝试 --fourcc MJPG 或安装 FFmpeg。"
                      << std::endl;
            out_path.clear();
          } else {
            writer_ready = true;
          }
        }
        if (writer.isOpened()) writer.write(canvas);
      }

      if (csv.is_open()) {
        csv << frame_index << ',' << std::fixed << std::setprecision(4) << sim_t << ',' << tr.dt
            << ',' << rmv::toString(tr.state) << ',' << (tr.measurement_used ? 1 : 0) << ','
            << tr.position.x << ',' << tr.position.y << ',' << tr.velocity.x << ','
            << tr.velocity.y << ',' << tr.speed << ',' << tr.speed_display << ',' << tr.speed_mps
            << ',' << tr.accel << ',' << tr.radius << ',' << tr.mahalanobis << ','
            << (tr.maneuver ? 1 : 0) << ',' << tr.roi.rect.width << ',' << tr.roi.rect.height << ','
            << (tr.roi.limited ? 1 : 0) << ',' << (tr.detection.found ? 1 : 0) << '\n';
      }

      // --- 帧率统计 ---
      const auto t_end = std::chrono::steady_clock::now();
      const double cost = std::max(1e-6, std::chrono::duration<double>(t_end - t_start).count());
      fps_measured = (fps_measured <= 0.0) ? (1.0 / cost) : (0.9 * fps_measured + 0.1 / cost);

      if (!quiet && (processed % 30 == 0)) {
        std::cout << "\r[progress] frame " << frame_index << "  " << rmv::toString(tr.state)
                  << "  v=" << cv::format("%7.1f", tr.speed_display) << " px/s"
                  << "  proc=" << cv::format("%5.1f", fps_measured) << " fps   " << std::flush;
      }

#ifdef RMV_HAVE_HIGHGUI
      if (show) {
        cv::imshow(kWindow, canvas);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
      }
#endif

      // 播放限速：按累计目标时间对齐，避免逐帧累加造成漂移。
      // 处理本身就慢于目标帧率时，wait_s<=0，自然变为"能跑多快跑多快"。
      if (play_fps > 0.0 && from_video) {
        const double target = static_cast<double>(processed) / play_fps;
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop_start).count();
        const double wait_s = target - elapsed;
        if (wait_s > 0.0) {
          std::this_thread::sleep_for(std::chrono::duration<double>(wait_s));
        }
      }
    }

    if (!quiet) std::cout << std::endl;

    if (csv.is_open()) csv.close();
    if (writer.isOpened()) writer.release();
#ifdef RMV_HAVE_HIGHGUI
    if (show) cv::destroyAllWindows();
#endif
    cap.release();

    // -------------------------------------------------------------------------
    // 5. 汇总
    // -------------------------------------------------------------------------
    const double det_rate = (processed > 0) ? (100.0 * detected / processed) : 0.0;
    const double mean_speed = (processed > 0) ? (speed_sum / processed) : 0.0;
    const double mean_roi = (roi_ratio_cnt > 0) ? (100.0 * roi_ratio_sum / roi_ratio_cnt) : 0.0;

    std::cout << "========================= 处理汇总 =========================\n"
              << " 处理帧数        : " << processed << "\n"
              << " 检出帧数        : " << detected << "  (" << cv::format("%.1f", det_rate) << "%)\n"
              << " 丢失重捕次数    : " << lost_events << "\n"
              << " 平均处理帧率    : " << cv::format("%.1f", fps_measured) << " fps\n"
              << " 搜索窗口平均占比: " << cv::format("%.2f", mean_roi)
              << "%  (ROI 局部搜索带来的计算量下降)\n"
              << " 平均速度        : " << cv::format("%.1f", mean_speed) << " px/s\n"
              << " 最大速度        : " << cv::format("%.1f", speed_max) << " px/s\n";
    if (!actual_out.empty() && writer_ready) {
      std::cout << " 输出视频        : " << actual_out << "\n";
    }
    if (!csv_path.empty()) {
      std::cout << " 输出 CSV        : " << csv_path << "\n";
    }
    std::cout << "===========================================================" << std::endl;

    return 0;
  } catch (const cv::Exception& e) {
    std::cerr << "[FATAL] OpenCV 异常: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] 异常: " << e.what() << std::endl;
    return 1;
  }
}


