// =============================================================================
//  main_kalman.cpp —— 绿球识别 + 卡尔曼滤波预测速度，用箭头标出速度方向与大小
//
//  作业要求（first-homework/作业内容）：
//    识别视频中的绿球，用箭头标注其运动方向和速度变化；
//    箭头方向 = 运动方向，箭头长度 = 速度大小（越快越长）。
//
//  本程序的做法（单遍在线处理，绝不回读视频）：
//    1. 检测：HSV 阈值找绿色 -> 开闭运算去噪 -> 找轮廓 -> 用面积/圆度/长宽比
//       三个几何条件滤掉"绿色圆柱"这类干扰 -> 取最大的一块，得到球心与半径。
//    2. 跟踪：4 状态卡尔曼滤波，状态是 [x, y, vx, vy]（单位：像素、像素/帧）。
//       每帧先按物理规律"预测"球到哪、速度会是多少，再拿检测结果"修正"。
//       速度不是靠两帧相减得到的噪声值，而是滤波器估计出来的平滑速度。
//    3. 画箭头：从球心出发，沿速度方向画一条箭头，长度正比于速度（带上下限）。
//       所以箭头方向就是球的运动方向，箭头变长就是变快、变短就是变慢。
//
//  编译（需要 OpenCV 4.x，C++17）：
//    g++ -O2 -std=c++17 main_kalman.cpp -o ball_tracker `pkg-config --cflags --libs opencv4`
//  运行：
//    ./ball_tracker [输入视频] [输出视频] [--no-show]
//    不给参数时默认读 first-homework.mp4、写 first-homework-annotated.mp4。
//
//  ★ 本文件是"逐行注释版"：几乎每一行代码后面都跟了一条中文注释。
//    注释只解释代码，**不改变任何逻辑**，和 main_kalman.cpp 完全等价。
// =============================================================================

#include <opencv2/opencv.hpp>          // OpenCV 总头文件，VideoCapture/Mat/画图全在里面

#include <algorithm>                   // std::min / std::max
#include <cmath>                       // std::sqrt / std::fabs
#include <cstdio>                      // std::printf / std::snprintf / std::fprintf
#include <cstdlib>                     // std::system（用来创建输出目录）
#include <cstring>                     // C 风格字符串（本文件里用得不多，留着无害）
#include <string>                      // std::string
#include <vector>                      // std::vector

namespace {                            // 匿名命名空间：里面的东西只在本文件可见（相当于"私有"）

// ============================== 可调参数（集中放这里，改起来一目了然）========

// ---- 绿色检测 ----
// HSV 里色相 H 是 0~179（OpenCV 把 360° 压成 180 格）。"绿"大约在 40~80 之间。
const int    kHueLow = 35;             // 色相下限（低于它偏黄、偏红）
const int    kHueHigh = 85;            // 色相上限（高于它偏青、偏蓝）
const int    kSatLow = 60;             // 饱和度下限：太低的是灰度色，不算绿
const int    kValLow = 60;             // 明度下限：太暗的不算绿

// 形状筛选的阈值都按"画面短边的比例"来算，换分辨率不会失效。
const double kBallAreaRatio = 0.0005;  // 球的最小面积 / 画面面积（1280x720 → 约 460px²）
const double kBallAreaMaxRatio = 0.25; // 球的最大面积 / 画面面积（防止把整屏绿墙当球）
const double kCircleMin = 0.62;        // 圆度 4*pi*A/P^2 的下限，正圆约 1.0，细长条接近 0
const double kAspectTol = 0.35;        // |宽/高 - 1| 的上限，圆的外接矩形应该接近正方形

// ---- 跟踪状态机 ----
const int    kConfirmHits = 3;         // 连续 3 帧都检测到同一个球才开始跟踪（防单帧噪点）
const int    kMaxMisses = 15;          // 连续 15 帧没有可信检测就放弃这一条轨迹

// ---- 卡尔曼参数 ----
const double kMeasNoise = 4.0;         // 测量噪声方差 R（像素^2）：中心点大约抖 2px
const double kAccNoise = 1.0;          // 过程噪声：球每帧加速度的方差（(像素/帧^2)^2）
const double kGateK = 3.0;             // 3σ 门限：超过就认为这次检测不可信
const double kGateInflateMax = 64.0;   // 不可信时把 R 最多放大 64 倍（软衰减，不是丢弃）

// ---- 折返保护（"投票"）----
// 卡尔曼在球突然掉头时要滞后 6 帧才把速度方向翻过来，那几帧箭头是反的。
// 解决办法：发现"连续两帧的位移方向都跟卡尔曼速度反着走"，就直接采用最近两帧的
// 位移方向，幅度沿用卡尔曼此刻的速度大小（卡尔曼自己正在衰减，等它收敛）。
const double kVoteMinStep = 2.0;       // 差分至少要 2px 才算"真的动了"，否则方向不可靠
const double kVoteMaxStep = 60.0;      // 单帧位移超过 60px 就当是检测跳变，采信方向时不用

// ---- 硬门限 ----
// 一帧之内球不可能跳出画面的 8%（对角线的约 4.7%）。超了就说明检测跳到别的东西上了，
// 直接当这一帧没检测到，把它挡在卡尔曼之外。
const double kMaxStepRatio = 0.08;     // 单帧位移上限 = 画面对角线 * 这个比例

// ---- 候选选择：怎么在多块绿色里认出球 ----
// 实测这段素材每帧恰好两块绿色：球（面积 8600~21500，圆度 0.62~0.83）
// 和绿色圆柱（面积 16600~33900，圆度 0.61~0.78）。375 帧里面积零重叠，
// 所以“球一定比圆柱小”是最可靠的判据——光比圆度不行，圆柱俯视也是圆的。
const double kAreaPenalty = 0.35;      // 面积惩罚系数，越大越偏向选小的那块

// ---- 箭头绘制 ----
const double kArrowPxPerSpeed = 0.15;  // 箭头长度 = 速度(px/s) * 这个系数
const double kArrowMinLen = 18.0;      // 箭头最短长度（px），太小看不见
const double kArrowMaxLen = 140.0;     // 箭头最长长度（px），太长会盖住画面
const double kMinSpeedToDraw = 15.0;   // 低于这个速度就只画个点，不画箭头（px/s）
// 输出速度上限：就是"箭头已经画到最长"对应的那个速度（140 / 0.15 ≈ 933 px/s）。
// 超过它箭头也不会更长了，所以数值上的限幅不影响观感，只是防止极端跳变时报出离谱的数。
const double kSpeedCap = kArrowMaxLen / kArrowPxPerSpeed;   // ≈ 933.3 px/s

// ============================== 一维卡尔曼（x / y 两个方向完全对称，共用一份代码）=

// 每个方向的状态是 [位置, 速度]，用 2x2 协方差描述不确定性。
// 因为 x 和 y 的方程一模一样、且互不相关，所以可以拆成两个独立的一维滤波器，
// 这样代码量不到一半，数学上完全等价。
class Kalman1D {                       // 一维卡尔曼滤波器
public:                                // ↓ 下面是对外公开的成员（外面能调用）
    // z0: 初始位置；r: 测量噪声方差；a: 过程噪声（加速度方差）
    Kalman1D(double z0, double r, double a) : r_(r), a_(a) {   // 构造函数，顺手初始化成员
        x_ = z0;  v_ = 0.0;            // 位置=第一次检测到的坐标，速度未知先给 0
        p00_ = 0.25 * r;               // 初始协方差 P(0,0)：位置"大概差半个像素量级"
        p01_ = 0.5 * r;                // 初始协方差 P(0,1)：位置和速度正相关
        p10_ = 0.5 * r;                //                P(1,0) 与 P(0,1) 始终相等
        p11_ = 1.0 * r;                //                P(1,1)：速度最不确定
        miss_ = 0;                     // 连续丢失帧数清零
    }                                  // 构造结束

    // 预测步：假设两帧之间做匀速运动，位置 += 速度。
    void predict() {                   // 每帧开始都先调它
        x_ += v_;                      // 位置按当前速度往前推一帧（x = x + v）
        // 协方差跟着状态转移矩阵 F=[[1,1],[0,1]] 一起变，再叠加"可能有加速度"的噪声。
        const double n00 = p00_ + p01_ + p10_ + p11_;   // F*P*F^T 的第 (0,0) 项
        const double n01 = p01_ + p11_;                 //                  第 (0,1) 项
        const double n10 = p10_ + p11_;                 //                  第 (1,0) 项（与 n01 相同）
        const double n11 = p11_;                        //                  第 (1,1) 项
        p00_ = n00 + 0.25 * a_;        // 叠加白噪声加速度对位置的增量方差
        p01_ = n01 + 0.50 * a_;        // 位置-速度的耦合增量
        p10_ = n10 + 0.50 * a_;        // 对称项，数值上恒等于 p01_
        p11_ = n11 + a_;               // 加速度对速度的增量方差
    }                                  // 预测结束：不确定性一定变大

    // 当前的新息比值：|测量 - 预测| / 预测标准差。用来判断这次检测可不可信。
    double newRatio(double z) const {  // z 是这一帧检测到的坐标
        const double s = p00_ + r_;    // 新息方差 S = 预测不确定度 + 测量噪声
        if (s <= 1e-12) return 0.0;    // 防御：方差为 0 时避免除零，直接当作 0
        return std::fabs(z - x_) / std::sqrt(s);   // 标准化新息：像"差了几个标准差"
    }                                  // newRatio 结束

    // 更新步：用测量值 z 修正状态。inflate>1 表示"不太相信这次测量"，
    // 等价于把测量噪声放大，卡尔曼增益因此变小，只轻微拉一下状态。
    void correct(double z, double inflate) {   // inflate 默认传 1.0 表示完全相信
        const double s = p00_ + r_ * inflate;  // 放大噪声后的新息方差
        const double k0 = p00_ / s;    // 位置的卡尔曼增益（0~1，越大越信测量）
        const double k1 = p10_ / s;    // 速度的卡尔曼增益
        const double innov = z - x_;   // 新息 = 测量 - 预测
        x_ += k0 * innov;              // 按增益往测量方向拉位置
        v_ += k1 * innov;              // 速度也跟着修正（这就是"预测出速度"的关键）
        const double o00 = p00_, o01 = p01_, o10 = p10_, o11 = p11_;   // 先存旧协方差
        p00_ = o00 - k0 * o00;         // 更新后协方差 P = (I - K*H) * P_old
        p01_ = o01 - k0 * o01;         // 对称更新
        p10_ = o10 - k1 * o00;         // 与 p01_ 保持相等
        p11_ = o11 - k1 * o01;         // 速度的不确定度下降
        miss_ = 0;                     // 有可信测量，丢失计数清零
    }                                  // 更新结束：不确定性一定变小

    // 这一帧没有可信检测：只预测不修正，并累计连续丢失帧数。
    void coast() { ++miss_; }          // 外推滑行：状态不动，只记账

    double pos() const { return x_; }        // 读位置
    double vel() const { return v_; }        // 读速度（像素/帧）
    int    misses() const { return miss_; }  // 读连续丢失帧数

private:                               // ↓ 下面是对外隐藏的成员（只有本类内部能访问）
    double r_, a_;                         // 测量噪声 / 过程噪声
    double x_ = 0.0, v_ = 0.0;             // 位置、速度（成员名带下划线，避免和形参重名）
    double p00_ = 0, p01_ = 0, p10_ = 0, p11_ = 0;   // 2x2 协方差矩阵，p01==p10 始终成立
    int    miss_ = 0;                      // 连续多少帧只预测没修正
};                                         // Kalman1D 结束

// x、y 两个方向合起来的"球的运动模型"。
class BallKalman {                     // 把两个 Kalman1D 包成二维的一个目标
public:                                // ↓ 下面是对外公开的成员
    BallKalman(cv::Point2f center, double r, double a)   // 用首次检测到的球心初始化
        : kx_(center.x, r, a), ky_(center.y, r, a) {}    // x 方向、y 方向各建一个滤波器

    void predict() { kx_.predict(); ky_.predict(); }     // 两方向同时预测

    // 返回两方向新息比值的较大者，作为整帧的"可信度指标"。
    double newRatio(cv::Point2f z) const {               // z 是本帧检测到的球心
        return std::max(kx_.newRatio(z.x), ky_.newRatio(z.y));   // 哪个方向更离谱就按哪个算
    }                                                    // newRatio 结束

    void correct(cv::Point2f z, double inflate) { kx_.correct(z.x, inflate); ky_.correct(z.y, inflate); }   // 两方向一起按同一个可信度修正
    void coast() { kx_.coast(); ky_.coast(); }           // 两方向一起记一次丢失

    cv::Point2f center() const { return cv::Point2f((float)kx_.pos(), (float)ky_.pos()); }   // 合成球心坐标
    cv::Point2f velocity() const { return cv::Point2f((float)kx_.vel(), (float)ky_.vel()); } // 合成速度向量，单位 像素/帧
    int misses() const { return std::max(kx_.misses(), ky_.misses()); }   // 取较大的丢失计数

private:                               // ↓ 下面是对外隐藏的成员
    Kalman1D kx_, ky_;                 // x 方向、y 方向各一个一维滤波器
};                                     // BallKalman 结束

// ============================== 检测：在画面里找那个绿球 =====================

// 返回 true 表示找到，center/radius 是球心和半径。
bool detectGreenBall(const cv::Mat& bgr, cv::Point2f& center, float& radius) {   // 引用输出参数
    if (bgr.empty()) return false;     // 空图直接返回失败

    // 1) 转到 HSV，绿色的区分度比 BGR 好得多。
    cv::Mat hsv;                       // 存放 HSV 图像
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);   // BGR 三通道 -> HSV 三通道

    // 2) 取绿色区间，得到一张"哪里是绿"的二值图。
    cv::Mat mask;                      // 二值掩膜：白色=绿，黑色=不绿
    cv::inRange(hsv, cv::Scalar(kHueLow, kSatLow, kValLow),    // 区间下限 H/S/V
                     cv::Scalar(kHueHigh, 255, 255), mask);    // 区间上限 H/S/V
    // 上一行效果：在区间内的像素置 255，其余置 0

    // 3) 形态学去噪：先开运算（吃掉小噪点），再闭运算（补上球上的小洞）。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));   // 5x5 椭圆核
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);    // 开运算 = 先腐蚀后膨胀，去小噪点
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);   // 闭运算 = 先膨胀后腐蚀，补小洞

    // 4) 找外轮廓。findContours 会修改输入，所以传 mask 自己没关系（后面不用了）。
    std::vector<std::vector<cv::Point>> contours;            // 每个轮廓是一串点
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);   // 只取最外层

    // 5) 面积阈值按画面大小缩放，换分辨率不用改代码。
    const double H_local = (double)bgr.rows;                        // 画面高度（只在本函数用）
    const double frameArea = (double)bgr.cols * H_local;            // 画面总面积（像素）
    const double minArea = kBallAreaRatio * frameArea;              // 允许的最小面积
    const double maxArea = kBallAreaMaxRatio * frameArea;           // 允许的最大面积

    bool hasBall = false;              // "已记录过一个球" 用独立的标志，不靠圆心坐标判断
    double bestArea = 0.0;             // 选中候选的面积（目前只做记录，不参与选择）
    double bestScore = -1e9;           // 选中候选的形状得分，越大越像球
    cv::Point2f bestCenter(0.0f, 0.0f);   // 最大候选的球心，只在 hasBall 为真时有意义
    float bestRadius = 0.0f;              // 最大候选的半径，只在 hasBall 为真时有意义
    for (const auto& c : contours) {              // 逐个轮廓检查
        const double area = cv::contourArea(c);   // 轮廓围成的面积
        if (area < minArea || area > maxArea) continue;   // 太大/太小都不是球，跳过
        const double peri = cv::arcLength(c, true);       // 周长（true = 闭合曲线）
        if (peri <= 1e-6) continue;                       // 周长为 0 没法算圆度，跳过
        // 圆度 = 4*pi*面积 / 周长^2，圆是 1.0，细长条会很小。
        const double circ = 4.0 * CV_PI * area / (peri * peri);   // 先算出来（后面打分要用）
        if (circ < kCircleMin) continue;                  // 不够圆，跳过
        const cv::Rect box = cv::boundingRect(c);         // 外接矩形
        if (box.width <= 0 || box.height <= 0) continue;  // 退化成一条线，跳过
        // 圆的外接矩形应该接近正方形。
        if (std::fabs((double)box.width / (double)box.height - 1.0) > kAspectTol) continue;   // 长宽比不合格
        // 上一行效果：|宽/高 - 1| 超过容差就说明是长条，不是球

        // 同时有多个候选时：选"最像球"的那一块，而不是"最大"的那一块。
        // 绿色圆柱俯视图也是圆的（圆度 0.64~0.69），但它比球更不圆，
        // 而且圆柱几乎从不贴近画面上下边缘。用形状质量打分，圆柱自然排到后面。
        cv::Point2f ctr;                                   // 这一块的外接圆心
        float rad = 0.0f;                                  // 这一块的外接圆半径
        cv::minEnclosingCircle(c, ctr, rad);               // 球心 + 外接圆半径
        const double aspect = (double)box.width / (double)box.height;   // 宽高比
        double score = circ - 0.5 * std::fabs(aspect - 1.0);            // 越圆、越方，分越高
        if (ctr.y > 0.8 * H_local || ctr.y < 0.2 * H_local) score -= 0.10;   // 贴上下边缘的扣分
        // 面积惩罚：球一定比绿色圆柱小（实测 375 帧零重叠）。
        // 取面积相对 minArea 的倍数的对数，免得大物体把分数拉爆。
        const double sizeRatio = area / std::max(minArea, 1.0);        // 面积是最小面积的几倍
        score -= kAreaPenalty * std::log(std::max(sizeRatio, 1.0));    // 越大扣得越多
        if (score > bestScore) {                           // 目前得分最高的
            bestScore = score;                             // 记下得分
            bestArea = area;                               // 记下面积（预留）
            bestCenter = ctr;                              // 记下球心
            bestRadius = rad;                              // 记下半径
            hasBall = true;                                // 标记"已经找到至少一个"
        }                                    // 候选更新结束
        (void)bestArea;                      // 预留：以后若要"得分相近时取大的"就在这里用
    }                                        // 遍历轮廓结束
    if (!hasBall) return false;              // 一个合格候选都没有
    center = bestCenter;                     // 输出球心
    radius = bestRadius;                     // 输出半径
    return true;                             // 找到了
}                                            // detectGreenBall 结束

// ============================== 主程序 =======================================

struct Options {                       // 命令行选项
    std::string input;                 // 输入视频路径
    std::string output;                // 输出视频路径
    bool show = true;                  // 是否弹窗实时播放（没有显示器时用 --no-show）
};                                     // Options 结束

void printUsage(const char* exe) {     // 打印用法，参数是程序名
    std::printf("用法: %s [输入视频] [输出视频] [--no-show]\n", exe);                    // 一行用法
    std::printf("  默认: 输入 first-homework.mp4，输出 first-homework-annotated.mp4\n"); // 说明默认值
    std::printf("  --no-show  不弹窗，只写文件（服务器/无界面环境用）\n");                 // 说明开关
}                                      // printUsage 结束

bool parseArgs(int argc, char** argv, Options& opt) {   // 解析命令行，解析失败返回 false
    for (int i = 1; i < argc; ++i) {   // 从第 1 个参数开始（第 0 个是程序自身）
        const std::string a = argv[i];                          // 当前参数
        if (a == "-h" || a == "--help") { printUsage(argv[0]); return false; }   // 请求帮助
        else if (a == "--no-show") { opt.show = false; }        // 关掉弹窗
        else if (opt.input.empty()) { opt.input = a; }          // 第一个位置参数当输入
        else if (opt.output.empty()) { opt.output = a; }        // 第二个位置参数当输出
        else { std::fprintf(stderr, "多余的参数: %s\n", a.c_str()); printUsage(argv[0]); return false; }   // 参数过多，报错
        // 上一行：既不是 -h/--no-show，输入输出又都填满了，说明参数多余
    }                                                           // 参数循环结束
    if (opt.input.empty())  opt.input = "first-homework.mp4";           // 输入没给就用默认名
    if (opt.output.empty()) opt.output = "first-homework-annotated.mp4"; // 输出没给就用默认名
    return true;                                                // 解析成功
}                                                               // parseArgs 结束

// 画一个带底色的文字标签，保证在任何背景上都看得清。
void putLabel(cv::Mat& img, const std::string& text, cv::Point org,   // img 原地修改
              double scale, cv::Scalar color) {                       // scale 字号，color 颜色
    int base = 0;                                                     // 接收基线偏移
    const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &base);   // 量文字尺寸
    const cv::Rect box(org.x - 2, org.y - sz.height - 2, sz.width + 4, sz.height + base + 4); // 底板矩形
    const cv::Rect clamped = box & cv::Rect(0, 0, img.cols, img.rows);   // 与画面求交，防止越界
    if (clamped.width > 0 && clamped.height > 0) {                    // 交区域非空才画
        cv::Mat roi = img(clamped);       // ROI：只取底板那一块（浅拷贝，改它就是改原图）
        cv::Mat dark = roi * 0.35;        // 亮度乘 0.35，得到一个暗版本
        dark.copyTo(roi);                 // 把暗版本写回去，形成半透明黑底
    }                                     // 底板绘制结束
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);   // 写字
}                                         // putLabel 结束

// 速度越小箭头越偏黄（慢），越大越偏红（快），一眼能看出"变快/变慢"。
cv::Scalar speedColor(double speedPxSec) {                    // 输入速度，输出 BGR 颜色
    const double t = std::min(speedPxSec / 600.0, 1.0);       // 归一化到 0~1（600px/s 封顶）
    return cv::Scalar(0.0, 255.0 * (1.0 - t), 255.0);         // BGR: 红固定 255，绿随速度减小 -> 黄→红
}                                                             // speedColor 结束

// 找不到输出目录时（比如第一次跑、输出在 out/xxx.mp4），自动建出来。
// 只处理"纯字母数字下划线、用 / 或 \ 分隔"的简单路径，省得依赖 C++17 的 <filesystem>。
bool ensureParentDir(const std::string& file) {               // 传入要写的文件路径
    const std::size_t slash = file.find_last_of("/\\");       // 找最后一个路径分隔符
    if (slash == std::string::npos || slash == 0) return true;  // 没有目录部分（当前目录）就不用建
    const std::string dir = file.substr(0, slash);            // 截出目录部分
    for (std::size_t i = 0; i < dir.size(); ++i) {            // 逐字符检查目录名
        const char c = dir[i];                                // 当前字符
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||   // 小写或大写字母
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' ||   // 数字与常见符号
                        c == '/' || c == '\\' || c == ':' || (unsigned char)c >= 0x80;  // 分隔符、冒号、中文等多字节字符
        if (!ok) return true;                                 // 有陌生字符，交给用户自己建，别乱猜
    }                                                         // 字符检查结束
#if defined(_WIN32)                                           // Windows：mkdir 不带 -p，错误丢到 nul
    const std::string cmd = "mkdir \"" + dir + "\" 2>nul";    // 目录已存在时报错也无所谓
#else                                                         // Linux / macOS
    const std::string cmd = "mkdir -p '" + dir + "' 2>/dev/null";   // -p 表示已存在不报错
#endif                                                        // 预处理条件编译到这里结束
    const int rc = std::system(cmd.c_str());                  // 真正执行命令
    (void)rc;                                                 // 明说我不看返回值，防止编译器警告
    return true;                                              // 尽力而为，失败也不阻断
}                                                             // ensureParentDir 结束

}  // namespace                       // 匿名命名空间到此结束

int main(int argc, char** argv) {                          // 程序入口
    cv::setUseOptimized(true);                             // 允许 OpenCV 用 SIMD 指令加速
    Options opt;                                           // 声明选项对象
    if (!parseArgs(argc, argv, opt)) return 1;             // 参数有误：退出码 1
    ensureParentDir(opt.output);                           // 输出目录不存在就先建出来

    cv::VideoCapture cap(opt.input);                       // 打开输入视频
    if (!cap.isOpened()) {                                 // 打不开（路径错/编码不支持）
        std::fprintf(stderr, "打不开输入视频: %s\n", opt.input.c_str());   // 报错信息
        return 2;                                          // 退出码 2
    }                                                      // 打开失败分支结束

    double fps = cap.get(cv::CAP_PROP_FPS);                // 读视频帧率
    if (!(fps > 1.0 && fps < 240.0)) fps = 30.0;           // 有些视频不给 fps，兜底 30

    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);  // 画面宽
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT); // 画面高
    // 不同编码器对 mp4v 的支持不一样，失败了就退到 MJPG。
    cv::VideoWriter writer(opt.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),   // 先试 mp4v
                           fps, cv::Size(W, H));          // 帧率与分辨率必须和输入一致
    if (!writer.isOpened()) {                             // mp4v 不可用
        writer.open(opt.output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(W, H));   // 退回 MJPG
    }                                                     // 回退结束
    if (!writer.isOpened()) {                             // 两种编码器都不行
        std::fprintf(stderr, "打不开输出视频: %s（检查目录是否存在、是否有写权限）\n",   // 报错并给出可能原因
                     opt.output.c_str());                 // 把路径填进 %s（上一行是格式串）
        return 3;                                         // 退出码 3
    }                                                     // 输出失败分支结束

    std::printf("输入: %s  (%dx%d @ %.2f fps)\n", opt.input.c_str(), W, H, fps);   // 打印输入信息
    std::printf("输出: %s\n", opt.output.c_str());                                 // 打印输出信息

    BallKalman* kf = nullptr;                              // 跟踪器指针，锁定目标后才有值
    int confirmHits = 0;                                   // 连续检测到球的帧数（用来做 3 帧确认）
    cv::Point2f prevDet(-1.0f, -1.0f);                     // 上一帧的检测球心（x<0 表示无效）
    int sameDirStreak = 0;                                 // 连续多少帧"检测点跟卡尔曼速度反着走"
    cv::Point2f anchor(-1.0f, -1.0f);                      // 掉头前最后那一帧的球心

    int frameIndex = 0;                                    // 已处理的帧数
    int trackedFrames = 0;                                 // 跟踪到球的帧数（最后打印用）
    // 单帧位移上限：画面宽高的 8%（1280x720 时约 102px）。
    const double maxStep = kMaxStepRatio * std::sqrt((double)W * W + (double)H * H);   // 对角线 × 0.08

    cv::Mat frame;                                         // 一帧图像
    while (cap.read(frame)) {           // 一次读一帧，边读边处理，不回读
        ++frameIndex;                                      // 帧号 +1
        cv::Point2f det;                                   // 本帧检测到的球心
        float detRadius = 0.0f;                            // 本帧检测到的半径
        bool hasDet = detectGreenBall(frame, det, detRadius);   // 执行检测
        // 只在画面里找球：合成视频可能把球"画"到画面外，真实视频里不会有这种点。
        if (hasDet && (det.x < 0.0f || det.x >= W || det.y < 0.0f || det.y >= H)) hasDet = false;   // 越界就作废

        cv::Point2f drawCenter, velocity;   // 最终要画出去的球心与速度（px/帧）
        bool hasTrack = false;              // 本帧是否跟踪到了球（决定画不画箭头）

        if (kf == nullptr) {                                           // 还没锁定目标
            // ---------- 还没锁定：等连续 kConfirmHits 帧都有检测才认为"真的有个球" ----------
            confirmHits = hasDet ? confirmHits + 1 : 0;                // 检测到就累加，断了就清零
            if (confirmHits >= kConfirmHits) {                         // 攒够 3 帧
                kf = new BallKalman(det, kMeasNoise, kAccNoise);       // 用当前球心初始化卡尔曼
                prevDet = det;                                         // 记住这一帧的检测点
                sameDirStreak = 0;                                     // 折返计数清零
                anchor = cv::Point2f(-1.0f, -1.0f);                    // 锚点作废
                confirmHits = 0;                                       // 计数清零
            }                                                          // 锁定分支结束
        } else {                                                       // 已经在跟踪
            kf->predict();                                             // 每帧先预测（位置 += 速度）

            // ---------- 硬门限：一帧之内球跳不了这么远 ----------
            bool stepOk = true;                                        // 先假设位移合理
            if (hasDet && prevDet.x >= 0.0f) {                         // 有检测且上帧也有检测
                stepOk = cv::norm(det - prevDet) <= (float)maxStep;    // 两帧位移是否超上限
            }                                                          // 硬门限判断结束

            bool trusted = false;                                      // 本帧检测是否被完全采信
            if (hasDet && stepOk) {                                    // 检测通过硬门限
                const double ratio = kf->newRatio(det);                // 新息比值（差了几个标准差）
                if (ratio <= kGateK) {                                 // 在 3σ 之内
                    kf->correct(det, 1.0);          // 新息在 3σ 内：正常采信
                    trusted = true;                                    // 标记为完全可信
                } else {                                               // 离得远，不可信
                    // 离得远：不直接扔，而是把测量噪声放大后再修正。
                    // 这样"真的掉头/加速"能被慢慢拉回来，纯噪声也不会带跑滤波器。
                    const double inflate = std::min((ratio / kGateK) * (ratio / kGateK),   // 放大倍数的平方
                                                    kGateInflateMax);                      // 最多 64 倍
                    kf->correct(det, inflate);                         // 用放大后的噪声做"轻推"
                }                                                      // 软门限分支结束
            } else {                                                   // 没检测到 或 位移离谱
                kf->coast();                        // 没检测到：只靠预测撑着
            }                                                          // 采信/滑行判断结束

            if (kf->misses() > kMaxMisses) {                           // 丢失太久
                // 太久没有可信检测，放弃这条轨迹，回到"等锁定"状态。
                delete kf;                                             // 释放旧跟踪器
                kf = nullptr;                                          // 回到未锁定状态
                prevDet = cv::Point2f(-1.0f, -1.0f);                   // 检测点作废
                sameDirStreak = 0;                                     // 折返计数清零
                anchor = cv::Point2f(-1.0f, -1.0f);                    // 锚点作废
            } else {                                                   // 还在跟踪
                velocity = kf->velocity();          // 像素/帧
                drawCenter = trusted ? det : kf->center();             // 可信就画检测点，否则画预测点

                // ---------- 折返保护 ----------
                if (hasDet && prevDet.x >= 0.0f) {                     // 本帧和上帧都有检测
                    const cv::Point2f step = det - prevDet;      // 最近两帧的真实位移
                    const float kfLen = cv::norm(velocity);      // 卡尔曼速度的大小
                    // 这一步是"顺着卡尔曼速度走"还是"反着走"？用点积符号判断。
                    // 别用夹角阈值——掉头过程中卡尔曼速度本身快 0 了。
                    const bool sameDir = kfLen < 1e-9f ||        // 速度几乎为 0 时算"同向"
                        (step.x * velocity.x + step.y * velocity.y > 0.0f);   // 点积 >0 即同向
                    // 注意：这里**不能**用"新息大"来否决反向帧。实测球 300px/s 掉头时，
                    // 头 5 帧新息比值是 6.1/8.8/9.6/9.2/8.3，全都大于门限——那正是掉头本身。
                    if (sameDir) {                                     // 顺着走
                        sameDirStreak = 0;                             // 反向计数清零
                        anchor = cv::Point2f(-1.0f, -1.0f);            // 锚点作废
                    } else {                                           // 反着走
                        ++sameDirStreak;                               // 反向计数 +1
                        if (sameDirStreak == 1) anchor = prevDet;      // 记下掉头那一下的位置
                    }                                                  // 反向分支结束
                    if (sameDirStreak >= 2 && anchor.x >= 0.0f) {      // 连续两帧反向才算数
                        const float stepLen = cv::norm(step);          // 位移长度
                        if (stepLen >= (float)kVoteMinStep && stepLen <= (float)kVoteMaxStep &&   // 位移在合理区间
                            kfLen > 1e-9f) {                           // 且有速度大小可用
                            // 方向取最近两帧的位移方向；幅度**沿用卡尔曼的**。
                            // 若直接拿 anchor 差分当幅度，20~50px 会被算成 600~1500px/s，
                            // 箭头一下顶到最长，反而更错。
                            velocity = step * (kfLen / stepLen);       // 方向换成位移方向，长度保持 kfLen
                            drawCenter = prevDet + step;               // 球心也跟到最新检测点
                        }                                              // 幅度/长度合格分支结束
                    }                                                  // 折返保护结束
                }                                                      // 折返判断结束
                if (hasDet) prevDet = det;                             // 有检测就更新上一帧检测点
                else { prevDet = cv::Point2f(-1.0f, -1.0f); sameDirStreak = 0; }   // 没检测就作废并清零

                hasTrack = true;                                       // 本帧跟踪成功
                ++trackedFrames;                                       // 跟踪帧数 +1
            }                                                          // 继续跟踪分支结束
        }                                                              // 锁定判断结束

        // ---------------- 绘制 ----------------
        if (hasTrack) {                                                // 跟踪到才画
            // 卡尔曼速度是"像素/帧"，乘 fps 换算成"像素/秒"，给人看更直观。
            double shownSpeed = cv::norm(velocity) * fps;              // 速度大小（px/s）
            // 限幅到"箭头最长"对应的速度，防止极端跳变时报出离谱数值。
            shownSpeed = std::min(shownSpeed, kSpeedCap);              // 不超过 kSpeedCap

            // 球：黄色圈 + 红色圆心
            cv::circle(frame, drawCenter, 26, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);   // 黄圈，半径 26
            cv::circle(frame, drawCenter, 3, cv::Scalar(0, 0, 255), -1);                  // 红心，实心

            // 箭头：方向 = 速度方向，长度 = 速度大小
            if (shownSpeed >= kMinSpeedToDraw) {                       // 速度够大才画箭头
                const double dirLen = cv::norm(velocity);              // 速度大小（px/帧）
                cv::Point2f unit(dirLen > 1e-9 ? velocity.x / (float)dirLen : 1.0f,   // 单位向量 x（退化为水平）
                                 dirLen > 1e-9 ? velocity.y / (float)dirLen : 0.0f);  // 单位向量 y
                const double len = std::min(std::max(shownSpeed * kArrowPxPerSpeed, kArrowMinLen),   // 先保证不短于下限
                                            kArrowMaxLen);             // 再保证不长于上限
                const cv::Point2f tip = drawCenter + unit * (float)len;   // 箭头终点 = 球心 + 单位向量 × 长度
                cv::arrowedLine(frame, drawCenter, tip, speedColor(shownSpeed),   // 起点、终点、颜色
                                3, cv::LINE_AA, 0, 0.28);          // 线宽 3、抗锯齿、箭头占比 0.28
            }                                                          // 画箭头结束

            // 屏幕上打一行速度，方便对照箭头长度
            char buf[128];                                             // 文本缓冲
            std::snprintf(buf, sizeof(buf), "speed: %6.1f px/s", shownSpeed);   // 格式化速度
            putLabel(frame, buf, cv::Point(20, 40), 0.8, cv::Scalar(0, 255, 255));   // 左上角写字
        } else {                                                       // 没跟踪到
            putLabel(frame, "searching for green ball...", cv::Point(20, 40), 0.8,   // 提示文字
                     cv::Scalar(180, 180, 180));                       // 灰色
        }                                                              // 文字绘制结束

        {                                                              // 单独代码块，限制 buf 作用域
            char buf[64];                                              // 复用一个缓冲（名字和外层不冲突）
            std::snprintf(buf, sizeof(buf), "frame %d", frameIndex);   // 格式化帧号
            putLabel(frame, buf, cv::Point(20, 76), 0.6, cv::Scalar(220, 220, 220));   // 第二行写字
        }                                                              // 帧号绘制结束

        writer.write(frame);                                           // 写出一帧
        if (opt.show) {                                                // 需要弹窗时
            cv::imshow("green ball tracker", frame);                   // 显示窗口
            const int key = cv::waitKey(1) & 0xFF;                     // 等 1ms 并取按键
            if (key == 27 || key == 'q') break;     // ESC / q 提前结束
        }                                                              // 弹窗部分结束
    }                                                                  // 主循环结束

    delete kf;                                                         // 释放跟踪器
    cap.release();                                                     // 关闭输入
    writer.release();                                                  // 关闭输出（收尾写文件头）
    if (opt.show) cv::destroyAllWindows();                             // 关掉所有窗口

    std::printf("完成：处理 %d 帧，其中跟踪到球 %d 帧。结果已写入 %s\n",   // 三个 % 依次是帧数、跟踪数、输出路径
                frameIndex, trackedFrames, opt.output.c_str());        // 这行是上面格式串的参数
    if (frameIndex == 0) {                                             // 一帧都没读到
        std::fprintf(stderr, "没有读到任何帧，检查输入视频是否有效。\n");   // 报错
        return 4;                                                      // 退出码 4
    }                                                                  // 空输入分支结束
    return 0;                                                          // 一切正常
}                                                                      // main 结束
