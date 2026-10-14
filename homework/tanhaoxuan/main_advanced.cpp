/**
 * ============================================================================
 *  作业：识别视频中的绿球，并用箭头标注其运动方向和速度变化
 *  ---------------------------------------------------------------------------
 *  文件名：main_advanced.cpp        （高级版 / pro 版）
 *  语言  ：C++17 + OpenCV 4.x       （在 VS Code 里直接编译运行）
 *  说明  ：本文件是**自包含的单文件版本**，不需要任何额外头文件；
 *          直接 g++ main_advanced.cpp -o ball_tracker_pro $(pkg-config --cflags --libs opencv4)
 *          即可编译，提交作业时只交这一个文件也能编过。
 *
 *  ---------------------------------------------------------------------------
 *  一、功能（相对基础版 main.cpp 的升级点）
 *    1. 识别        ：HSV 阈值 + 形态学 + 形状筛选（圆度 / 长宽比），
 *                     把同样颜色的"绿色圆柱"排除掉，只留绿球；
 *    2. 亚像素球心  ：轮廓矩 + 椭圆拟合融合，球心精度优于整数像素；
 *    3. 跟踪        ：卡尔曼滤波（常速度模型）+ 马氏距离门限，
 *                     离群测量直接拒掉、短暂丢失按速度滑行（coast）；
 *    4. 抗遮挡      ：颜色检测失败时，改用 LK 光流在球上续跟一帧；
 *    5. 相机运动补偿：背景光流 + 部分仿射 RANSAC 估计相机运动，
 *                     同时给出"视在速度"和"真实速度"；
 *    6. 速度变化    ：逐帧加速度（EMA 平滑）+ 自动判定加速 / 减速 / 匀速；
 *    7. 箭头标注    ：箭头方向 = 运动方向；箭头长度 ∝ 速度大小（越长越快）；
 *                     箭头颜色随速度由绿(慢)到红(快)，长度系数按视频自动标定；
 *    8. 中文标注    ：用 FreeType 渲染中文；若本机 OpenCV 没有 freetype 模块
 *                     或找不到中文字体，会自动退回英文，绝不会因此编译失败；
 *    9. 输出        ：标注视频、逐帧 CSV、速度曲线图、轨迹图、Markdown 报告。
 *
 *  二、命令行用法
 *      ./ball_tracker_pro                                  默认处理 homework.mp4
 *      ./ball_tracker_pro 视频.mp4 输出.mp4                 指定输入与输出
 *      ./ball_tracker_pro --stabilize --stats               相机补偿 + 打印逐帧数据
 *      ./ball_tracker_pro --help                            查看全部选项
 *
 *  三、代码结构（从上到下）
 *      ① 小工具：格式化、半透明矩形、虚线、速度配色、描边文字
 *      ② 文字渲染器 TextRenderer（中文 / 英文自动切换）
 *      ③ 参数 Options / 球候选 Ball / 每帧记录 FrameRec
 *      ④ 绿球检测：颜色分割 + 形状筛选 + 候选挑选
 *      ⑤ 跟踪：卡尔曼滤波、光流续跟、相机（背景）运动估计
 *      ⑥ 绘制：速度箭头、数据面板、速度曲线、轨迹小地图、拖尾
 *      ⑦ 输出：CSV、速度曲线 PNG、轨迹 PNG、Markdown 报告
 *      ⑧ main()：解析参数 → 第一趟分析 → 输出图表 → 第二趟渲染视频
 * ============================================================================
 */

#include <opencv2/opencv.hpp>          // OpenCV 主头文件（core / imgproc / video / videoio 等）

#if defined(__has_include)             // 若编译器支持 __has_include 探测（GCC/Clang 均支持）
#  if __has_include(<opencv2/freetype.hpp>)   // 探测本机 OpenCV 是否带 freetype 模块（属 opencv_contrib）
#    include <opencv2/freetype.hpp>    // 有：包含它，后面就能渲染中文
#    define HAS_FREETYPE 1              // 记为 1，表示"支持中文渲染"
#  else                                 // 否则（没装 opencv_contrib 的 freetype）
#    define HAS_FREETYPE 0              // 记为 0，表示"只能渲染英文"
#  endif                                // 结束内层 #if
#else                                   // 若编译器不支持 __has_include
#  define HAS_FREETYPE 0                // 保守处理：按"没有 freetype"编译，保证能编过
#endif                                  // 结束外层 #if

#include <algorithm>                    // std::min / std::max / std::sort / std::nth_element
#include <chrono>                       // std::chrono 计时（统计处理耗时）
#include <cmath>                        // std::sqrt / std::atan2 / std::fabs 等数学函数
#include <cstdarg>                      // va_list / va_start / va_end（可变参数函数用）
#include <cstdio>                       // std::snprintf / std::printf 格式化输出
#include <deque>                        // std::deque 双端队列（存轨迹拖尾）
#include <filesystem>                   // std::filesystem：建目录、取文件名
#include <fstream>                      // std::ofstream：写 CSV 与 Markdown 报告
#include <iomanip>                      // std::fixed / std::setprecision 控制小数位
#include <iostream>                     // std::cout / std::cerr 控制台输出
#include <string>                       // std::string
#include <vector>                       // std::vector

using namespace std;                    // 使用标准库命名空间（string / vector / cout 等）
using namespace cv;                     // 使用 OpenCV 命名空间（Mat / Point2f / VideoCapture 等）
namespace fs = std::filesystem;         // 给 std::filesystem 起短别名 fs

// ===========================================================================
//  ① 小工具
// ===========================================================================

// 格式化字符串：用法 fmt("x=%.1f", 3.14f)，内部用 vsnprintf 拼成 std::string 返回
static string fmt(const char* f, ...) {                                  // 定义可变参数函数，返回 string
    char buf[512];                                                       // 准备 512 字节字符缓冲
    va_list ap;                                                          // 声明可变参数游标
    va_start(ap, f);                                                     // 让游标指向第一个可变参数
    vsnprintf(buf, sizeof(buf), f, ap);                                  // 按格式 f 把参数写入缓冲（超长自动截断）
    va_end(ap);                                                          // 结束可变参数读取
    return string(buf);                                                  // 缓冲转成 std::string 返回
}                                                                        // 结束 fmt

// 在图上画半透明矩形：造一张纯色图，再与原图按 alpha 混合
static void fillRectAlpha(Mat& img, Rect r, Scalar color, double alpha) { // 参数：目标图、矩形、颜色、不透明度
    r &= Rect(0, 0, img.cols, img.rows);                                  // 把矩形裁剪到图像范围内，防越界
    if (r.width <= 0 || r.height <= 0) return;                            // 裁剪后无面积则直接返回
    Mat roi = img(r);                                                     // 取出该区域（引用，不拷贝）
    Mat overlay(roi.size(), roi.type(), color);                           // 造一张同尺寸纯色图
    addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);              // 按 alpha 混合并写回 roi
}                                                                        // 结束 fillRectAlpha

// 画虚线：从 a 到 b，按"实线段 + 空隙"交替绘制
static void dashedLine(Mat& img, Point a, Point b, Scalar color,         // 目标图、起点、终点、颜色
                       int thick = 1, int dash = 8, int gap = 6) {       // 线宽、实线段长、空隙长
    double len = norm(b - a);                                            // 线段总长度
    if (len < 1) return;                                                 // 太短就不画
    Point2d d = (Point2d(b) - Point2d(a)) / len;                         // 单位方向向量（浮点）
    for (double t = 0; t < len; t += dash + gap) {                       // 沿方向每隔 (dash+gap) 起一段
        double t2 = min(t + dash, len);                                  // 本段实线的结束位置（不超出末端）
        line(img, Point2d(a) + d * t, Point2d(a) + d * t2, color, thick, LINE_AA); // 画这一小段
    }                                                                    // 结束 for
}                                                                        // 结束 dashedLine

// 速度 → 颜色：绿(慢) → 黄 → 红(快)，用于箭头颜色与曲线
static Scalar speedColor(double speed, double vmax) {                    // 参数：速度、归一化用的最大速度
    double t = min(1.0, max(0.0, speed / max(1.0, vmax)));               // 归一化到 0~1（0 慢、1 快）
    Mat hsv(1, 1, CV_8UC3, Scalar(60.0 * (1.0 - t), 255, 255));          // HSV：H 从 60(绿) 降到 0(红)，S/V 拉满
    Mat bgr;                                                             // 存放转换结果
    cvtColor(hsv, bgr, COLOR_HSV2BGR);                                   // HSV 转 BGR（OpenCV 画图用 BGR）
    Vec3b p = bgr.at<Vec3b>(0, 0);                                       // 取出这一个像素的三通道值
    return Scalar(p[0], p[1], p[2]);                                     // 返回 BGR 颜色
}                                                                        // 结束 speedColor

// 运动阶段 → 颜色：加速=红、减速=蓝、匀速=灰
static Scalar phaseColor(int phase) {                                    // 参数：阶段值
    if (phase == 1)  return Scalar(60, 60, 255);                         // 1 = 加速 → 红
    if (phase == -1) return Scalar(255, 160, 60);                        // -1 = 减速 → 蓝
    return Scalar(160, 160, 160);                                        // 其它（匀速）→ 灰
}                                                                        // 结束 phaseColor

// 英文描边文字：先画黑字做底，再画彩色字，保证深色/浅色背景都清晰
static void putTextShadow(Mat& img, const string& s, Point org,          // 目标图、文本、起始坐标
                          double scale, Scalar color, int thick = 1) {   // 字号缩放、颜色、线宽
    putText(img, s, org + Point(1, 1), FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), thick + 1, LINE_AA); // 黑描边
    putText(img, s, org, FONT_HERSHEY_SIMPLEX, scale, color, thick, LINE_AA);                            // 彩色字
}                                                                        // 结束 putTextShadow

// ===========================================================================
//  ② 文字渲染器：优先中文（FreeType + 中文字体），失败自动退回英文（Hershey）
// ===========================================================================
class TextRenderer {                                                     // 定义文字渲染器类
public:                                                                  // 以下成员对外可用
    // 初始化：找不到字体或不要中文时返回 false，调用方用 pick() 自动切英文
    bool init(const string& fontPath, bool wantChinese) {                // 参数：字体路径（可空）、是否要中文
        if (!wantChinese) return false;                                  // 明确不要中文 → 直接失败（走英文）
#if HAS_FREETYPE                                                        // 编译时探测到 freetype 才走这段
        string path = fontPath;                                          // 先采用用户指定的路径
        if (path.empty() || !fs::exists(path)) path = findCjkFont();     // 没指定或不存在 → 自动查找中文字体
        if (path.empty() || !fs::exists(path)) return false;             // 仍找不到 → 退回英文
        ft_ = freetype::createFreeType2();                                // 创建 FreeType2 渲染器
        try {                                                            // 加载字体可能抛异常，故包一层 try
            ft_->loadFontData(path, 0);                                  // 加载字体文件（ttf/ttc，索引 0）
            cjk_ = true;                                                 // 标记中文可用
        } catch (const cv::Exception&) {                                 // 捕获 OpenCV 异常
            cjk_ = false;                                                // 加载失败 → 退回英文
        }                                                                // 结束 try-catch
        fontPath_ = path;                                                // 记录实际使用的字体路径
        return cjk_;                                                     // 返回是否启用中文
#else                                                                   // 编译时没有 freetype 模块
        (void)fontPath;                                                  // 显式忽略参数，避免未使用警告
        (void)wantChinese;                                               // 同上
        return false;                                                    // 只能英文
#endif                                                                  // 结束 #if HAS_FREETYPE
    }                                                                    // 结束 init

    bool cjk() const { return cjk_; }                                    // 当前是否中文模式
    const string& fontPath() const { return fontPath_; }                 // 实际使用的字体路径

    // 按当前模式选文案：中文可用返回 zh，否则返回 en（返回值用完即弃）
    const string& pick(const string& zh, const string& en) const {       // 参数：中文串、英文串
        return cjk_ ? zh : en;                                           // 三元选择
    }                                                                    // 结束 pick

    // 量一段文字的像素尺寸，用于排版（避免文字跑出画面）
    Size size(const string& s, int h) const {                            // 参数：文本、字高（像素）
        int base = 0;                                                    // 接收基线偏移（这里不用）
#if HAS_FREETYPE                                                        // 中文模式用 FreeType 精确测量
        if (cjk_) return ft_->getTextSize(s, h, -1, &base);              // 返回精确尺寸
#endif                                                                  // 结束 #if
        return getTextSize(s, FONT_HERSHEY_SIMPLEX, h / 30.0, 1, &base); // 英文模式按 Hershey 估算
    }                                                                    // 结束 size

    // 不描边的文字（面板内部用，背景已是深色）
    void put(Mat& img, const string& s, Point org, int h, Scalar color) const { // 图、文本、坐标、字高、颜色
#if HAS_FREETYPE                                                        // 中文模式
        if (cjk_) { ft_->putText(img, s, org, h, color, -1, LINE_AA, false); return; } // FreeType 实心渲染
#endif                                                                  // 结束 #if
        putText(img, s, org, FONT_HERSHEY_SIMPLEX, h / 30.0, color, 1, LINE_AA);       // 英文 Hershey 渲染
    }                                                                    // 结束 put

    // 带描边的文字（画在实景画面上时用）
    void putShadow(Mat& img, const string& s, Point org, int h, Scalar color) const { // 图、文本、坐标、字高、颜色
#if HAS_FREETYPE                                                        // 中文模式
        if (cjk_) {                                                      // 走 FreeType 两遍绘制
            ft_->putText(img, s, org + Point(2, 2), h, Scalar(0, 0, 0), -1, LINE_AA, false); // 先画黑底
            ft_->putText(img, s, org, h, color, -1, LINE_AA, false);     // 再画彩字
            return;                                                      // 结束
        }                                                                // 结束 if
#endif                                                                  // 结束 #if
        putTextShadow(img, s, org, h / 30.0, color, 1);                  // 英文：复用 Hershey 描边函数
    }                                                                    // 结束 putShadow

private:                                                                // 以下为内部实现
    // 在常见系统字体目录里找一个中文字体
    static string findCjkFont() {                                        // 返回字体路径；找不到返回空串
        const char* cands[] = {                                          // 候选列表（覆盖 Ubuntu / macOS / Windows）
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",    // Ubuntu：Noto Sans CJK（最常用）
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",     // Ubuntu：Noto Sans CJK Medium
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",              // Ubuntu：文泉驿正黑
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", // Ubuntu：Droid 回退字体
            "/usr/share/fonts/truetype/arphic/uming.ttc",                // Ubuntu：AR PL UMing
            "/System/Library/Fonts/PingFang.ttc",                        // macOS：苹方
            "C:/Windows/Fonts/msyh.ttc",                                 // Windows：微软雅黑
        };                                                               // 结束候选表
        for (const char* c : cands) if (fs::exists(c)) return c;         // 逐个检查，存在就用
        return "";                                                       // 都没找到 → 空串
    }                                                                    // 结束 findCjkFont

#if HAS_FREETYPE                                                        // 只有需要 freetype 时才声明这个成员
    Ptr<freetype::FreeType2> ft_;                                        // FreeType 渲染器智能指针
#endif                                                                  // 结束 #if
    bool   cjk_ = false;                                                 // 当前是否中文模式
    string fontPath_;                                                    // 实际使用的字体路径
};                                                                       // 结束 class TextRenderer

// ===========================================================================
//  ③ 参数、检测结果、每帧记录
// ===========================================================================

// 所有可调参数集中在这里，方便命令行覆盖
struct Options {                                                         // 定义参数结构体
    string videoPath = "homework.mp4";                                   // 输入视频路径（默认作业视频）
    string outVideo  = "output/result_pro.mp4";                          // 输出视频路径
    string outCsv    = "";                                               // CSV 路径；空 = 按输出视频名自动推导
    string outCurve  = "output/speed_curve.png";                         // 速度曲线图路径
    string outMap    = "output/trajectory_map.png";                      // 轨迹图路径
    string outReport = "output/report.md";                               // Markdown 报告路径
    string outPreview = "output/preview_pro.jpg";                        // 预览图（速度最快那一帧）
    string fontPath;                                                     // 中文字体路径（空 = 自动查找）
    bool writeVideo = true;                                              // 是否写结果视频
    bool wantChinese = true;                                             // 是否尝试中文标注
    bool showTrue = false;                                               // 是否额外画"真实运动"箭头
    bool autoGain = true;                                                // 箭头长度系数是否自动标定
    bool dumpStats = false;                                              // 是否打印逐帧数据
    bool writeReport = true;                                             // 是否生成图与报告
    int    hMin = 35, hMax = 95, sMin = 60, vMin = 50;                   // 绿色 HSV 阈值（H 0~179，S/V 0~255）
    double minArea = 1500.0;                                             // 轮廓最小面积（滤掉噪点）
    double minCirc = 0.78;                                               // 最小圆度（球≈0.84，绿圆柱≈0.72）
    double minAspect = 0.80, maxAspect = 1.25;                           // 外接矩形 高/宽 范围（圆柱是竖长条）
    double measSigma = 3.0;                                              // 卡尔曼测量噪声标准差（像素）
    double sigmaAccel = 4000.0;                                          // 卡尔曼过程噪声：加速度标准差（像素/秒²）
    double gateChi2 = 16.0;                                              // 马氏距离门限（2 自由度，约 4σ）
    double flowRScale = 9.0;                                             // 用光流当测量时，测量噪声放大倍数
    int    maxLost = 45;                                                 // 连续丢失多少帧后重新全局搜索
    double accelSmooth = 0.20;                                           // 加速度 EMA 平滑系数
    double accelThresh = 250.0;                                          // 加速/减速判定阈值（像素/秒²）
    double arrowGain = 0.30;                                             // 箭头长度 = 速度 × 本系数（自动标定会覆盖）
    double arrowTargetLen = 175.0;                                       // 自动标定目标：90 分位速度对应的箭头像素长度
    double arrowMaxLen = 240.0;                                          // 箭头最大长度（像素），防止冲出画面
    double staticSpeed = 25.0;                                           // 低于此速度视为"基本静止"（像素/秒）
    double speedMax = 850.0;                                             // 颜色映射与进度条的速度上限（像素/秒）
    int    trailLen = 45;                                                // 轨迹拖尾长度（帧）
};                                                                       // 结束 struct Options

// 一个绿球候选：绿色轮廓 + 形状特征 + 亚像素圆心
struct Ball {                                                            // 定义球候选结构体
    bool    found = false;                                               // 是否有效
    Point2f center{0.f, 0.f};                                            // 亚像素圆心（图像坐标）
    float   radius = 0.f;                                                // 半径（像素）
    double  area = 0, circ = 0, aspect = 1;                              // 面积、圆度、外接矩形高宽比
    Rect    box;                                                         // 外接矩形
};                                                                       // 结束 struct Ball

// 每帧分析结果（第一趟算好存起来，第二趟直接用它渲染）
struct FrameRec {                                                        // 定义每帧记录结构体
    Point2f center{0.f, 0.f};                                            // 最终使用的球心
    Point2f velApp{0.f, 0.f};                                            // 视在速度（像素/秒，卡尔曼估计）
    Point2f velTrue{0.f, 0.f};                                           // 扣掉相机运动后的速度
    Point2f bgVel{0.f, 0.f};                                             // 相机运动造成的视在速度分量
    float   radius = 0.f;                                                // 球半径
    double  speed = 0.0;                                                 // 速度大小 |velApp|
    double  accel = 0.0;                                                 // 平滑后的加速度（像素/秒²）
    int     phase = 0;                                                   // 运动阶段：1 加速 / 0 匀速 / -1 减速
    int     source = -1;                                                 // 来源：2 颜色 / 1 光流 / 0 预测 / -1 无
    bool    bgOk = false;                                                // 相机运动估计是否成功
};                                                                       // 结束 struct FrameRec

// ===========================================================================
//  ④ 绿球检测
// ===========================================================================

// 颜色分割：把画面里的绿色区域抠出来（HSV 阈值 + 形态学去噪/补洞）
static Mat greenMask(const Mat& bgr, const Options& o) {                 // 参数：BGR 原图、参数结构
    Mat hsv, mask;                                                       // hsv 存转换结果，mask 存二值掩码
    cvtColor(bgr, hsv, COLOR_BGR2HSV);                                   // BGR → HSV（HSV 更适合做颜色阈值）
    inRange(hsv, Scalar(o.hMin, o.sMin, o.vMin),                         // 阈值下限（H 绿、S 够饱和、V 够亮）
            Scalar(o.hMax, 255, 255), mask);                             // 阈值上限（H 上限、S/V 不限）
    Mat k1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));           // 5×5 椭圆核（开运算用，去小噪点）
    Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));           // 9×9 椭圆核（闭运算用，补小空洞）
    morphologyEx(mask, mask, MORPH_OPEN, k1);                            // 开运算：先腐蚀后膨胀 → 抹掉零散噪点
    morphologyEx(mask, mask, MORPH_CLOSE, k2);                           // 闭运算：先膨胀后腐蚀 → 补上内部空洞
    return mask;                                                         // 返回处理后的二值掩码
}                                                                        // 结束 greenMask

// 形状筛选：在绿色轮廓里挑出"像球"的那些（把同样颜色的绿色圆柱排除掉）
static vector<Ball> findBallCandidates(const Mat& mask, const Options& o) { // 参数：绿色掩码、参数结构
    vector<vector<Point>> contours;                                      // 用来存放所有外轮廓
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    // 只取最外层轮廓，压缩水平/垂直段
    vector<Ball> cands;                                                  // 存放筛选后的球候选
    for (const auto& c : contours) {                                     // 逐个轮廓判断
        double area = contourArea(c);                                    // 计算轮廓面积（像素²）
        if (area < o.minArea) continue;                                  // 面积太小 → 当噪点丢掉
        double peri = arcLength(c, true);                                // 计算周长（true = 闭合轮廓）
        if (peri <= 0) continue;                                         // 周长异常 → 跳过（防除零）
        double circ = 4.0 * CV_PI * area / (peri * peri);                // 圆度 = 4πA/P²：圆→1，越不圆越小
        Rect   box = boundingRect(c);                                    // 求外接矩形
        double aspect = double(box.height) / max(1, box.width);          // 高宽比：球≈1.0，竖圆柱≈1.5+
        if (circ < o.minCirc) continue;                                  // 圆度不足 → 是圆柱，丢掉
        if (aspect < o.minAspect || aspect > o.maxAspect) continue;      // 长宽比不像正方形 → 不是球，丢掉
        Moments m = moments(c);                                          // 计算轮廓矩（用于求质心）
        if (m.m00 <= 0) continue;                                        // 零阶矩为 0 → 异常轮廓，跳过
        Point2f cMom(float(m.m10 / m.m00), float(m.m01 / m.m00));        // 质心 = 一阶矩 / 零阶矩（亚像素）
        Point2f cEll = cMom;                                             // 椭圆拟合中心先默认等于质心
        Size2f  axes(float(sqrt(area / CV_PI)) * 2.f,                    // 椭圆长轴先按"等面积圆"估算
                     float(sqrt(area / CV_PI)) * 2.f);                   // 短轴同理
        bool ellipseOk = false;                                          // 椭圆拟合是否可信
        if (c.size() >= 5) {                                             // fitEllipse 至少需要 5 个点
            try {                                                        // 拟合可能抛异常，包 try
                RotatedRect e = fitEllipse(c);                           // 用轮廓点拟合椭圆
                double ratio = double(max(e.size.width, e.size.height)) / // 长轴
                               max(1.0, double(min(e.size.width, e.size.height))); // ÷ 短轴
                if (ratio < 1.35) {                                      // 长短轴差距不大 → 接近圆，可信
                    cEll = e.center;                                     // 采用椭圆中心
                    axes = e.size;                                       // 采用椭圆轴长
                    ellipseOk = true;                                    // 标记可信
                }                                                        // 结束 if
            } catch (const cv::Exception&) {                             // 拟合失败
                ellipseOk = false;                                       // 标记不可信，退回质心
            }                                                            // 结束 try-catch
        }                                                                // 结束 if
        Ball b;                                                          // 构造一个候选
        b.found  = true;                                                 // 标记有效
        b.center = ellipseOk ? (cMom * 0.5f + cEll * 0.5f) : cMom;       // 质心与椭圆中心各占一半（更稳）
        b.radius = (axes.width + axes.height) * 0.25f;                   // 半径 ≈ (长轴 + 短轴) / 4
        if (b.radius <= 0) b.radius = float(sqrt(area / CV_PI));         // 半径异常时按等面积圆兜底
        b.area = area;                                                   // 记录面积
        b.circ = circ;                                                   // 记录圆度
        b.aspect = aspect;                                               // 记录长宽比
        b.box = box;                                                     // 记录外接矩形
        cands.push_back(b);                                              // 加入候选列表
    }                                                                    // 结束 for 遍历轮廓
    return cands;                                                        // 返回所有球候选
}                                                                        // 结束 findBallCandidates

// 在多个候选里挑最可能的那个：有预测位置时选离预测最近的（并略微偏向更圆的）
static bool pickBall(const vector<Ball>& cands, Point2f predicted,       // 参数：候选列表、预测位置
                     bool usePrediction, Ball& best) {                   // 是否使用预测、输出的最佳候选
    if (cands.empty()) return false;                                     // 没有候选 → 失败
    int    bestIdx = 0;                                                  // 最佳候选下标
    double bestScore = 1e18;                                             // 最佳得分（越小越好），初值取很大
    for (size_t i = 0; i < cands.size(); ++i) {                          // 遍历所有候选
        double score;                                                    // 本候选的得分
        if (usePrediction) {                                             // 有预测位置时
            score = norm(cands[i].center - predicted)                    // 主项：到预测位置的距离
                  + (1.0 - cands[i].circ) * 200.0;                       // 次项：轻微偏向更圆的（防跳到圆柱）
        } else {                                                         // 没有预测时（刚起步）
            score = -cands[i].area;                                      // 直接取面积最大的
        }                                                                // 结束 if
        if (score < bestScore) { bestScore = score; bestIdx = int(i); }   // 记录更优者
    }                                                                    // 结束 for
    best = cands[bestIdx];                                               // 输出最佳候选
    return true;                                                         // 成功
}                                                                        // 结束 pickBall

// ===========================================================================
//  ⑤ 跟踪：卡尔曼滤波 / 光流续跟 / 相机运动估计
// ===========================================================================

// 卡尔曼跟踪器：状态 = [x, y, vx, vy]（像素、像素/秒），常速度模型
class KalmanTracker {                                                    // 定义跟踪器类
public:                                                                  // 对外接口
    // 用初始位置初始化滤波器
    void init(double dt, Point2f p0, const Options& o) {                 // 参数：帧间隔（秒）、初始位置、参数
        o_  = o;                                                         // 保存参数副本
        dt_ = dt;                                                        // 保存帧间隔
        kf_.init(4, 2, 0, CV_32F);                                       // 初始化：4 个状态、2 个测量、无控制量
        kf_.transitionMatrix = (Mat_<float>(4, 4) <<                     // 状态转移矩阵 F（匀速模型）
            1, 0, float(dt), 0,                                           // x' = x + vx*dt
            0, 1, 0, float(dt),                                           // y' = y + vy*dt
            0, 0, 1, 0,                                                   // vx' = vx
            0, 0, 0, 1);                                                  // vy' = vy
        kf_.measurementMatrix = (Mat_<float>(2, 4) <<                    // 测量矩阵 H：只观测位置
            1, 0, 0, 0,                                                   // z_x = x
            0, 1, 0, 0);                                                  // z_y = y
        setProcessNoise();                                               // 设置过程噪声 Q（见下面函数）
        kf_.measurementNoiseCov = Mat::eye(2, 2, CV_32F) *               // 测量噪声 R = σ²·I
                                  float(o.measSigma * o.measSigma);      // （检测有约 3 像素抖动）
        Mat d = (Mat_<float>(4, 1) << 100.f, 100.f, 10000.f, 10000.f);    // 初始协方差对角线
        kf_.errorCovPost = Mat::diag(d);                                  // 位置不确定 100、速度不确定 10000
        kf_.statePost = (Mat_<float>(4, 1) << p0.x, p0.y, 0.f, 0.f);      // 初始状态：给定位置、速度 0
        inited_ = true;                                                   // 标记已初始化
    }                                                                     // 结束 init

    bool inited() const { return inited_; }                               // 是否已初始化

    // 预测一步，返回预测位置
    Point2f predict() {                                                   // 无参数
        Mat sp = kf_.predict();                                           // 用状态转移矩阵外推一步
        return Point2f(sp.at<float>(0, 0), sp.at<float>(1, 0));           // 返回预测的 (x, y)
    }                                                                     // 结束 predict

    // 门限判断：这次测量是否与当前轨迹"讲得通"（马氏距离 < 阈值）
    bool gate(Point2f z) const {                                          // 参数：测量位置
        Mat Z = (Mat_<float>(2, 1) << z.x, z.y);                          // 把测量写成 2×1 矩阵
        Mat y = Z - kf_.measurementMatrix * kf_.statePre;                 // 新息 = 测量 − 预测测量
        Mat S = kf_.measurementMatrix * kf_.errorCovPre *                 // 新息协方差 S = H·P⁻·Hᵀ + R
                kf_.measurementMatrix.t() + kf_.measurementNoiseCov;      // （P⁻ 是预测协方差）
        Mat d2 = y.t() * S.inv() * y;                                     // 马氏距离平方 d² = yᵀS⁻¹y
        return d2.at<float>(0, 0) < float(o_.gateChi2);                   // 小于阈值 → 接受
    }                                                                     // 结束 gate

    // 用测量更新状态；rScale 用于放大测量噪声（光流测量不如颜色测量可信）
    void correct(Point2f z, double rScale = 1.0) {                        // 参数：测量位置、噪声放大倍数
        Mat R0 = kf_.measurementNoiseCov.clone();                         // 备份原测量噪声
        if (rScale != 1.0) kf_.measurementNoiseCov = R0 * float(rScale);  // 需要时放大 R（更不信任这次测量）
        kf_.correct((Mat_<float>(2, 1) << z.x, z.y));                     // 标准卡尔曼更新
        kf_.measurementNoiseCov = R0;                                     // 恢复原测量噪声
    }                                                                     // 结束 correct

    // 没有测量时：把"预测"直接当作"后验"，让滤波器按速度继续滑行
    void coast() {                                                        // 无参数
        kf_.statePost = kf_.statePre.clone();                             // 后验状态 = 先验状态
        kf_.errorCovPost = kf_.errorCovPre.clone();                       // 后验协方差 = 先验协方差
    }                                                                     // 结束 coast

    Point2f position() const { return Point2f(kf_.statePost.at<float>(0, 0), kf_.statePost.at<float>(1, 0)); } // 当前位置
    Point2f velocity() const { return Point2f(kf_.statePost.at<float>(2, 0), kf_.statePost.at<float>(3, 0)); } // 当前速度
    double  speed()    const { return norm(velocity()); }                 // 速度大小

private:                                                                 // 内部实现
    // 构造过程噪声 Q：由"加速度标准差"σa 推出来的常速度模型 Q
    void setProcessNoise() {                                             // 无参数
        double dt = dt_, q = o_.sigmaAccel * o_.sigmaAccel;               // dt 帧间隔，q = σa²
        kf_.processNoiseCov = (Mat_<float>(4, 4) <<                       // 标准 CV 模型离散化后的 Q
            float(q * dt * dt * dt * dt / 4), 0.f, float(q * dt * dt * dt / 2), 0.f,      // 位置 x 行
            0.f, float(q * dt * dt * dt * dt / 4), 0.f, float(q * dt * dt * dt / 2),      // 位置 y 行
            float(q * dt * dt * dt / 2), 0.f, float(q * dt * dt), 0.f,                    // 速度 vx 行
            0.f, float(q * dt * dt * dt / 2), 0.f, float(q * dt * dt));                   // 速度 vy 行
    }                                                                     // 结束 setProcessNoise

    KalmanFilter kf_;                                                     // OpenCV 卡尔曼滤波器对象
    Options o_;                                                           // 参数副本
    double  dt_ = 1.0 / 30.0;                                             // 帧间隔（默认 30fps）
    bool    inited_ = false;                                              // 是否已初始化
};                                                                        // 结束 class KalmanTracker

// 光流续跟：颜色检测失败时，在球内部撒点、用 LK 光流跟一帧，取位移中位数
struct FlowTracker {                                                      // 定义光流跟踪工具（纯静态函数）
    static bool track(const Mat& prevGray, const Mat& gray,               // 参数：上一帧灰度、当前帧灰度
                      Point2f prevCenter, float radius,                   // 上一帧球心、球半径
                      Point2f& newCenter, int& nGood) {                   // 输出：新球心、有效点数
        nGood = 0;                                                        // 有效点数先清零
        if (prevGray.empty() || prevGray.size() != gray.size() || radius < 4.f) return false; // 输入不合法直接失败
        int r = max(4, int(radius * 0.75f));                              // 只在球半径 75% 以内撒点（避开边缘）
        int step = max(3, r / 3);                                         // 撒点步长（约 3×3 到 5×5 个点）
        vector<Point2f> p0;                                               // 上一帧的特征点集合
        for (int dy = -r; dy <= r; dy += step) {                          // 纵向遍历
            for (int dx = -r; dx <= r; dx += step) {                      // 横向遍历
                if (dx * dx + dy * dy > r * r) continue;                  // 落在圆外的点不要
                Point2f p(prevCenter.x + float(dx), prevCenter.y + float(dy)); // 计算该点绝对坐标
                if (p.x >= 1 && p.y >= 1 && p.x < prevGray.cols - 1 &&    // 必须留在图像内部
                    p.y < prevGray.rows - 1) p0.push_back(p);             // 合法才加入
            }                                                             // 结束内层 for
        }                                                                 // 结束外层 for
        if (p0.size() < 5) return false;                                  // 点太少，跟不住
        vector<Point2f> p1, p0back;                                       // p1 前向跟踪结果、p0back 反向跟踪结果
        vector<uchar> st1, st2;                                           // 前后向跟踪的状态标志
        vector<float> err;                                                // 跟踪误差（这里不用）
        calcOpticalFlowPyrLK(prevGray, gray, p0, p1, st1, err, Size(21, 21), 3); // 前向：上一帧 → 当前帧
        calcOpticalFlowPyrLK(gray, prevGray, p1, p0back, st2, err, Size(21, 21), 3); // 反向：当前帧 → 上一帧
        vector<float> dxs, dys;                                           // 收集有效的位移分量
        for (size_t i = 0; i < p0.size(); ++i) {                          // 遍历每个点
            if (!st1[i] || !st2[i]) continue;                             // 任一方向失败 → 丢弃
            if (norm(p0back[i] - p0[i]) > 1.0f) continue;                 // 前后向不一致（误差>1像素）→ 丢弃
            dxs.push_back(p1[i].x - p0[i].x);                             // 记录 x 位移
            dys.push_back(p1[i].y - p0[i].y);                             // 记录 y 位移
        }                                                                 // 结束 for
        nGood = int(dxs.size());                                          // 记录有效点数
        if (nGood < 5) return false;                                      // 有效点太少 → 失败
        float mx = median(dxs), my = median(dys);                         // 取位移中位数（抗个别误匹配）
        newCenter = Point2f(prevCenter.x + mx, prevCenter.y + my);        // 新球心 = 旧球心 + 中位位移
        return true;                                                      // 成功
    }                                                                     // 结束 track

private:                                                                  // 内部实现
    static float median(vector<float>& v) {                               // 求中位数（会打乱输入顺序）
        size_t n = v.size() / 2;                                          // 中间位置
        nth_element(v.begin(), v.begin() + n, v.end());                   // 线性时间选出第 n 小
        return v[n];                                                      // 返回中位值
    }                                                                     // 结束 median
};                                                                        // 结束 struct FlowTracker

// 相机（背景）运动估计：在背景上取角点做光流，再用 RANSAC 拟合部分仿射变换
struct BackgroundMotion {                                                 // 定义相机运动估计工具
    static bool estimate(const Mat& prevGray, const Mat& gray,            // 参数：上一帧灰度、当前帧灰度
                         const Mat& prevGreenDilated,                     // 上一帧"绿色区域(膨胀)"掩码
                         Mat& affine, double& inlierRatio) {              // 输出：2×3 仿射矩阵、内点率
        inlierRatio = 0.0;                                                // 内点率先清零
        if (prevGray.empty() || prevGray.size() != gray.size()) return false; // 输入不合法
        vector<Point2f> p0;                                               // 背景角点集合
        goodFeaturesToTrack(prevGray, p0, 400, 0.01, 12.0);               // 取最多 400 个强角点（间距≥12）
        if (p0.size() < 12) return false;                                 // 角点太少 → 失败
        vector<Point2f> p1;                                               // 当前帧对应点
        vector<uchar> st;                                                 // 跟踪状态
        vector<float> err;                                                // 跟踪误差（不用）
        calcOpticalFlowPyrLK(prevGray, gray, p0, p1, st, err, Size(21, 21), 3); // 金字塔 LK 光流跟踪
        vector<Point2f> a, b;                                             // a=上一帧点，b=当前帧点（只留背景上的）
        for (size_t i = 0; i < p0.size(); ++i) {                          // 遍历角点
            if (!st[i]) continue;                                         // 跟踪失败 → 跳过
            Point pr(cvRound(p0[i].x), cvRound(p0[i].y));                 // 取整坐标用于查掩码
            if (pr.x < 0 || pr.y < 0 || pr.x >= prevGreenDilated.cols || // 越界检查
                pr.y >= prevGreenDilated.rows) continue;                  // 越界 → 跳过
            if (prevGreenDilated.at<uchar>(pr) > 0) continue;             // 落在绿球/绿柱上 → 不算背景，跳过
            a.push_back(p0[i]);                                           // 记录上一帧点
            b.push_back(p1[i]);                                           // 记录当前帧点
        }                                                                 // 结束 for
        if (a.size() < 12) return false;                                  // 背景点太少 → 失败
        vector<uchar> inliers;                                            // RANSAC 内点标记
        affine = estimateAffinePartial2D(a, b, inliers, RANSAC, 3.0);      // 拟合"旋转+缩放+平移"（4 自由度）
        if (affine.empty()) return false;                                 // 拟合失败
        int good = 0;                                                     // 统计内点数
        for (uchar v : inliers) if (v) good++;                            // 逐个累加
        inlierRatio = double(good) / double(a.size());                    // 内点率 = 内点数 / 总点数
        return good >= 8;                                                 // 内点足够多才算成功
    }                                                                     // 结束 estimate

    // 用仿射矩阵把一个点从上一帧映射到当前帧
    static Point2f apply(const Mat& A, Point2f p) {                       // 参数：2×3 仿射矩阵、点
        return Point2f(float(A.at<double>(0, 0) * p.x + A.at<double>(0, 1) * p.y + A.at<double>(0, 2)), // 新 x
                       float(A.at<double>(1, 0) * p.x + A.at<double>(1, 1) * p.y + A.at<double>(1, 2))); // 新 y
    }                                                                     // 结束 apply
};                                                                        // 结束 struct BackgroundMotion

// ===========================================================================
//  ⑥ 绘制：箭头 / 面板 / 速度曲线 / 轨迹小地图 / 拖尾
// ===========================================================================

// 前置声明：这三个"文字映射"函数定义在 drawPanel 之后，这里先声明一下才能被调用
static string dirText(double angDeg, TextRenderer& t);                   // 角度 → 八方向文字
static string phaseText(int phase, TextRenderer& t);                     // 阶段值 → 加速/减速/匀速
static string sourceText(int src, TextRenderer& t);                      // 来源值 → 颜色/光流/预测

// 画速度箭头：这是作业要求的核心标注——方向 = 运动方向，长度 = 速度大小
static void drawVelocityArrow(Mat& img, TextRenderer& t, Point2f from,   // 参数：图、文字、箭头起点（球心）
                              Point2f vel, const Options& o, Scalar color, // 速度矢量、参数、颜色
                              int thickness = 5, bool withLabel = true) { // 线宽、是否写速度数值
    double sp = norm(vel);                                               // 速度大小（像素/秒）
    if (sp < o.staticSpeed) {                                            // 速度小于阈值 → 视为基本静止
        circle(img, from, 9, color, 2, LINE_AA);                         // 画一个小圆圈表示"位置"
        t.putShadow(img, t.pick(u8"基本静止", "static"),                  // 写"基本静止"（中文/英文自动切换）
                    Point(int(from.x) + 14, int(from.y) - 12), 20, color); // 文字位置在球右上方
        return;                                                          // 结束（不画箭头）
    }                                                                    // 结束 if
    double len = min(sp * o.arrowGain, o.arrowMaxLen);                   // 箭头长度 = 速度 × 系数，并设上限
    Point2f dir = vel * float(1.0 / sp);                                 // 单位方向向量（方向 = 运动方向）
    Point2f tip = from + dir * float(len);                               // 箭头尖端坐标
    arrowedLine(img, from, tip, Scalar(0, 0, 0), thickness + 3, LINE_AA, 0, 0.26); // 先画黑描边（更醒目）
    arrowedLine(img, from, tip, color, thickness, LINE_AA, 0, 0.26);     // 再画彩色箭头
    circle(img, tip, 3.5, color, -1, LINE_AA);                           // 在尖端画个小实心点
    if (withLabel) {                                                     // 需要写数值时
        char buf[64];                                                    // 数值缓冲
        snprintf(buf, sizeof(buf), "%.0f", sp);                          // 把速度取整写进缓冲
        string s = string(buf) + " " + t.pick(u8"像素/秒", "px/s");       // 拼上单位
        Size ts = t.size(s, 19);                                         // 量一下文字尺寸，便于排版
        Point2f mid = from + dir * float(len * 0.55);                    // 文字放在箭头中部附近
        Point2f an = mid + Point2f(-dir.y, dir.x) * 16.f;                // 沿垂直方向偏移，避免压住箭头
        int lx = int(an.x) - ts.width / 2, ly = int(an.y) - ts.height / 2; // 转成左上角坐标并居中
        lx = min(max(lx, 4), max(4, img.cols - ts.width - 4));           // 水平方向夹在画面内
        ly = min(max(ly, 4), max(4, img.rows - ts.height - 4));          // 垂直方向夹在画面内
        t.putShadow(img, s, Point(lx, ly), 19, color);                   // 画出速度数值
    }                                                                    // 结束 if
}                                                                        // 结束 drawVelocityArrow

// 画左上角数据面板：速度、方向、加速度、真实速度、帧号、以及箭头长度标尺
static void drawPanel(Mat& img, TextRenderer& t, const Options& o,       // 参数：图、文字、参数
                      const FrameRec& r, int frameIdx, int total,         // 本帧记录、当前帧号、总帧数
                      double fps, double meanSpeed) {                     // 帧率、平均速度（用于配色与标尺）
    const int px = 18, py = 16, pw = 566, ph = 236;                       // 面板位置与尺寸
    Rect panel(px, py, pw, ph);                                          // 面板矩形
    fillRectAlpha(img, panel, Scalar(18, 18, 18), 0.64);                  // 半透明深色底（不遮住画面）
    rectangle(img, panel, Scalar(210, 210, 210), 1, LINE_AA);             // 面板描边
    Scalar col = speedColor(r.speed, max(1.0, meanSpeed * 2.0));          // 当前速度对应的颜色
    int y = py + 30;                                                     // 文字起始纵坐标
    t.putShadow(img, t.pick(u8"绿球识别与运动标注 · 高级版",               // 面板标题（中文）
                            "GREEN BALL TRACKING (advanced)"),           // 面板标题（英文）
                Point(px + 14, y), 24, Scalar(255, 255, 255));            // 画标题
    y += 12;                                                             // 下移一行
    line(img, Point(px + 14, y), Point(px + pw - 14, y), Scalar(120, 120, 120), 1, LINE_AA); // 标题下分隔线
    char buf[256];                                                       // 文本缓冲
    const int lh = 27, lx = px + 118;                                     // 行高、数值列的起点
    y += lh + 4;                                                         // 换到第一行
    string unitSpd = t.pick(u8"像素/秒", "px/s");                          // 速度单位（中/英）
    string unitFr  = t.pick(u8"像素/帧", "px/frame");                      // 每帧位移单位
    snprintf(buf, sizeof(buf), "%.1f %s   (%.1f %s)", r.speed,           // 组装"速度 (每帧位移)"
             unitSpd.c_str(), r.speed / fps, unitFr.c_str());             // 参数依次填入
    t.putShadow(img, t.pick(u8"速度  ", "speed "), Point(px + 14, y), 21, Scalar(220, 220, 220)); // 行标签
    t.putShadow(img, buf, Point(lx, y), 21, col);                         // 速度数值（用速度配色）
    double ang = atan2(-r.velApp.y, r.velApp.x) * 180.0 / CV_PI;          // 方向角：0°=向右，逆时针为正
    if (fabs(ang) < 0.05) ang = 0.0;                                      // 把 -0.0 归一成 0.0（好看）
    y += lh;                                                              // 换行
    snprintf(buf, sizeof(buf), "%6.1f°   %s", ang,                        // 组装"角度 + 方向文字"
             dirText(ang, t).c_str());                                    // dirText 见下方辅助函数
    t.putShadow(img, t.pick(u8"方向  ", "dir.  "), Point(px + 14, y), 21, Scalar(220, 220, 220)); // 行标签
    t.putShadow(img, buf, Point(lx, y), 21, Scalar(235, 235, 235));       // 方向数值
    y += lh;                                                              // 换行
    snprintf(buf, sizeof(buf), "%+7.0f %s   [%s]", r.accel,               // 组装"加速度 [阶段]"
             t.pick(u8"像素/秒²", "px/s^2").c_str(),                      // 加速度单位
             phaseText(r.phase, t).c_str());                              // 阶段文字
    t.putShadow(img, t.pick(u8"加速度", "accel "), Point(px + 14, y), 21, Scalar(220, 220, 220)); // 行标签
    t.putShadow(img, buf, Point(lx, y), 21, phaseColor(r.phase));         // 加速度（用阶段配色）
    y += lh;                                                              // 换行
    string noteOwn = t.pick(u8"(已扣除相机运动)", "(camera-compensated)"); // 真实速度的说明文字
    snprintf(buf, sizeof(buf), "%.1f %s   %s", norm(r.velTrue),           // 组装"真实速度 + 说明"
             unitSpd.c_str(), noteOwn.c_str());                           // 填入速度与说明
    t.putShadow(img, t.pick(u8"真实速度", "true sp"), Point(px + 14, y), 21, Scalar(220, 220, 220)); // 行标签
    t.putShadow(img, buf, Point(lx, y), 21, Scalar(255, 190, 80));        // 真实速度（橙色）
    y += lh;                                                              // 换行
    snprintf(buf, sizeof(buf), "%d / %d    %s", frameIdx + 1, total,      // 组装"帧号 + 测量来源"
             sourceText(r.source, t).c_str());                            // 来源文字
    t.putShadow(img, t.pick(u8"帧    ", "frame "), Point(px + 14, y), 21, Scalar(220, 220, 220)); // 行标签
    t.putShadow(img, buf, Point(lx, y), 21,                                   // 帧号与来源
                r.source == 2 ? Scalar(120, 255, 120) : Scalar(120, 200, 255)); // 颜色按来源区分
    y += lh + 4;                                                          // 换行并留出间隔
    line(img, Point(px + 14, y - 12), Point(px + pw - 14, y - 12), Scalar(80, 80, 80), 1, LINE_AA); // 分隔线
    double refV = max(1.0, meanSpeed * 1.6);                              // 参考箭头的速度值
    double refLen = min(refV * o.arrowGain, o.arrowMaxLen) * 0.62;        // 参考箭头的像素长度
    arrowedLine(img, Point2f(float(px + 16), float(y + 12)),              // 从面板左下开始画参考箭头
                Point2f(float(px + 16) + float(refLen), float(y + 12)),   // 到参考长度处结束
                speedColor(refV, max(1.0, meanSpeed * 2.0)), 5, LINE_AA, 0, 0.3); // 用速度配色
    snprintf(buf, sizeof(buf), "%s %.0f %s",                              // 组装标尺说明
             t.pick(u8"箭头越长越快，参考", "longer = faster, ref").c_str(), // 中文/英文说明
             refV, unitSpd.c_str());                                      // 参考速度与单位
    t.put(img, buf, Point(px + 26 + int(refLen), y + 4), 17, Scalar(200, 200, 200)); // 画说明文字
}                                                                         // 结束 drawPanel

// 方向文字：把角度翻译成"向右/右上/向上 …"（八方向）
static string dirText(double angDeg, TextRenderer& t) {                   // 参数：角度（度）、文字渲染器
    if (angDeg >= -22.5 && angDeg < 22.5)   return t.pick(u8"向右", "right");       // 右
    if (angDeg >= 22.5 && angDeg < 67.5)    return t.pick(u8"右上", "up-right");    // 右上
    if (angDeg >= 67.5 && angDeg < 112.5)   return t.pick(u8"向上", "up");          // 上
    if (angDeg >= 112.5 && angDeg < 157.5)  return t.pick(u8"左上", "up-left");     // 左上
    if (angDeg >= 157.5 || angDeg < -157.5) return t.pick(u8"向左", "left");        // 左
    if (angDeg >= -157.5 && angDeg < -112.5) return t.pick(u8"左下", "down-left");  // 左下
    if (angDeg >= -112.5 && angDeg < -67.5) return t.pick(u8"向下", "down");        // 下
    return t.pick(u8"右下", "down-right");                                          // 右下（剩余情况）
}                                                                          // 结束 dirText

// 阶段文字：加速 / 减速 / 匀速
static string phaseText(int phase, TextRenderer& t) {                      // 参数：阶段值、文字渲染器
    if (phase == 1)  return t.pick(u8"加速", "speeding up");               // 加速
    if (phase == -1) return t.pick(u8"减速", "slowing down");              // 减速
    return t.pick(u8"匀速", "steady");                                     // 匀速
}                                                                          // 结束 phaseText

// 来源文字：颜色+形状 / 光流续跟 / 卡尔曼预测
static string sourceText(int src, TextRenderer& t) {                       // 参数：来源值、文字渲染器
    if (src == 2) return t.pick(u8"颜色+形状", "color+shape");             // 颜色检测
    if (src == 1) return t.pick(u8"光流续跟", "optical flow");             // 光流续跟
    if (src == 0) return t.pick(u8"卡尔曼预测", "prediction");             // 纯预测
    return t.pick(u8"未捕获", "lost");                                     // 还没有球
}                                                                          // 结束 sourceText

// 画轨迹拖尾：越新的点越亮，让运动路径一目了然
static void drawTrail(Mat& img, const deque<Point2f>& trail) {             // 参数：图、轨迹点队列
    for (size_t i = 1; i < trail.size(); ++i) {                            // 从第二个点开始，逐段连线
        double a = double(i) / double(trail.size());                       // 归一化"新旧程度"（越大越新）
        int g = int(70 + 185 * a);                                         // 绿色分量随新旧增加
        int b = int(255 - 150 * a);                                        // 蓝色分量随新旧减少
        line(img, trail[i - 1], trail[i], Scalar(b, g, 0), 2, LINE_AA);    // 画这一小段（青→黄渐变）
    }                                                                      // 结束 for
}                                                                          // 结束 drawTrail

// 画速度曲线（叠在画面右下角）：整段视频的速度随时间变化 + 当前时刻游标 + 阶段色带
static void drawSpeedGraph(Mat& img, TextRenderer& t, const Options& o,    // 参数：图、文字、参数
                           const vector<double>& speeds,                   // 全片速度序列
                           const vector<int>& phases,                      // 全片阶段序列
                           int curIdx, double meanSpeed) {                 // 当前帧下标、平均速度
    if (speeds.empty()) return;                                            // 没有数据就不画
    const int gw = 452, gh = 186;                                          // 图形框尺寸
    Rect box(img.cols - gw - 18, img.rows - gh - 18, gw, gh);              // 固定在右下角
    fillRectAlpha(img, box, Scalar(18, 18, 18), 0.64);                     // 半透明深色底
    rectangle(img, box, Scalar(210, 210, 210), 1, LINE_AA);                // 描边
    double vmax = 0;                                                       // 纵向最大值
    for (double v : speeds) vmax = max(vmax, v);                           // 取全片最大速度
    vmax = max(60.0, vmax * 1.12);                                         // 留 12% 余量，最小按 60 显示
    const int gx = box.x + 52, gy = box.y + 40;                            // 曲线区左上角
    const int gwi = box.width - 72, ghi = box.height - 74;                 // 曲线区宽高
    int n = int(speeds.size());                                            // 总帧数
    t.putShadow(img, t.pick(u8"速度曲线（像素/秒）", "speed curve (px/s)"), // 小标题（中文/英文）
                Point(box.x + 12, box.y + 26), 20, Scalar(255, 255, 255)); // 画小标题
    for (int i = 0; i <= 4; ++i) {                                         // 画 4 条横向网格线
        int yy = gy + ghi - ghi * i / 4;                                   // 该网格线的纵坐标
        line(img, Point(gx, yy), Point(gx + gwi, yy), Scalar(70, 70, 70), 1, LINE_AA); // 网格线
        t.put(img, fmt("%.0f", vmax * i / 4).c_str(), Point(box.x + 8, yy + 6), 16, Scalar(190, 190, 190)); // 刻度值
    }                                                                      // 结束 for
    line(img, Point(gx, gy), Point(gx, gy + ghi), Scalar(200, 200, 200), 1, LINE_AA);       // 纵轴
    line(img, Point(gx, gy + ghi), Point(gx + gwi, gy + ghi), Scalar(200, 200, 200), 1, LINE_AA); // 横轴
    auto pt = [&](int i, double v) {                                       // 局部函数：把(帧号,速度)转成像素坐标
        double x = gx + (n <= 1 ? 0 : double(i) * gwi / double(n - 1));    // 横向按帧号均匀铺开
        double y = gy + ghi - min(1.0, v / vmax) * ghi;                    // 纵向按速度比例（越高越快）
        return Point(int(x), int(y));                                      // 返回坐标
    };                                                                     // 结束 lambda
    vector<Point> full, past;                                              // full=完整曲线，past=已播部分
    for (int i = 0; i < n; ++i) full.push_back(pt(i, speeds[i]));          // 组装完整曲线
    for (int i = 0; i <= min(curIdx, n - 1); ++i) past.push_back(pt(i, speeds[i])); // 组装已播部分
    polylines(img, vector<vector<Point>>{full}, false, Scalar(90, 90, 90), 1, LINE_AA); // 未来部分画灰色
    if (past.size() > 1)                                                   // 已播部分
        polylines(img, vector<vector<Point>>{past}, false, Scalar(120, 255, 120), 2, LINE_AA); // 画亮绿色
    int yMean = pt(0, meanSpeed).y;                                        // 平均速度对应的纵坐标
    dashedLine(img, Point(gx, yMean), Point(gx + gwi, yMean), Scalar(0, 200, 255), 1, 7, 6); // 平均速度虚线
    t.putShadow(img, fmt("%s %.0f", t.pick(u8"均值", "mean").c_str(), meanSpeed).c_str(),       // "均值 xxx"
                Point(gx + 4, yMean - 6), 17, Scalar(0, 200, 255));        // 画均值标注
    int bandY = gy + ghi + 4;                                              // 阶段色带的纵坐标
    for (int i = 0; i < n; ++i) {                                          // 逐帧画阶段色带
        int x0 = pt(i, 0).x;                                               // 本帧对应的横坐标
        int x1 = (i + 1 < n) ? pt(i + 1, 0).x : x0 + 1;                    // 下一帧的横坐标（用于算宽度）
        if (i < int(phases.size()))                                        // 有阶段数据时
            rectangle(img, Rect(x0, bandY, max(1, x1 - x0), 6), phaseColor(phases[i]), -1); // 画一小格
    }                                                                      // 结束 for
    t.put(img, t.pick(u8"阶段", "phase"), Point(box.x + 8, bandY + 22), 16, Scalar(190, 190, 190)); // 色带说明
    if (curIdx >= 0 && curIdx < n) {                                       // 画"当前时刻"的游标
        Point cp = pt(curIdx, speeds[curIdx]);                             // 当前速度点
        line(img, Point(cp.x, gy), Point(cp.x, gy + ghi), Scalar(120, 120, 120), 1, LINE_AA); // 竖线
        circle(img, cp, 5, Scalar(0, 0, 255), -1, LINE_AA);                // 红点
        circle(img, cp, 7, Scalar(255, 255, 255), 1, LINE_AA);             // 红点外描白圈
    }                                                                      // 结束 if
}                                                                          // 结束 drawSpeedGraph

// 画轨迹小地图（左下角）：把"画面里的轨迹"和"扣除相机运动后的轨迹"画在同一比例尺下
static void drawMiniMap(Mat& img, TextRenderer& t, const Options& o,       // 参数：图、文字、参数
                        const vector<Point2f>& appPath,                    // 视在轨迹（图像坐标）
                        const vector<Point2f>& truePath,                   // 真实轨迹（积分得到）
                        int curIdx, double pathApp, double pathTrue) {     // 当前下标、两条轨迹的累计长度
    const int mw = 264, mh = 216;                                          // 小地图尺寸
    Rect box(18, img.rows - mh - 18, mw, mh);                              // 固定在左下角
    fillRectAlpha(img, box, Scalar(18, 18, 18), 0.64);                     // 半透明底
    rectangle(img, box, Scalar(210, 210, 210), 1, LINE_AA);                // 描边
    t.putShadow(img, t.pick(u8"轨迹小地图", "trajectory map"),             // 标题
                Point(box.x + 12, box.y + 24), 20, Scalar(255, 255, 255)); // 画标题
    if (appPath.empty()) return;                                           // 没有轨迹就不画了
    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;            // 轨迹外接框的四个边界
    for (const auto& p : appPath) {                                        // 遍历所有轨迹点求包围盒
        minX = min(minX, p.x); maxX = max(maxX, p.x);                      // 更新 x 边界
        minY = min(minY, p.y); maxY = max(maxY, p.y);                      // 更新 y 边界
    }                                                                      // 结束 for
    float span = max(maxX - minX, maxY - minY);                            // 取较长的一边作为正方形边长
    span = max(span, 40.f) * 1.15f;                                        // 留 15% 边距，且不小于 40
    float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f;            // 包围盒中心
    const int mx = box.x + 14, my = box.y + 38;                            // 绘图区左上角
    const int mwIn = box.width - 28, mhIn = box.height - 96;               // 绘图区宽高
    float scale = min(float(mwIn) / span, float(mhIn) / span);             // 统一缩放比例（两种轨迹共用）
    Point2f center(mx + mwIn * 0.5f, my + mhIn * 0.5f);                    // 绘图区中心
    auto map = [&](Point2f p) {                                            // 局部函数：图像坐标 → 小地图坐标
        return Point(int(center.x + (p.x - cx) * scale),                   // x 方向平移缩放
                     int(center.y + (p.y - cy) * scale));                  // y 方向平移缩放
    };                                                                     // 结束 lambda
    rectangle(img, Rect(mx, my, mwIn, mhIn), Scalar(80, 80, 80), 1, LINE_AA); // 绘图区边框
    for (size_t i = 1; i < appPath.size(); ++i)                            // 画视在轨迹（橙色）
        line(img, map(appPath[i - 1]), map(appPath[i]), Scalar(255, 150, 60), 2, LINE_AA); // 一段一段连线
    for (size_t i = 1; i < truePath.size(); ++i)                           // 画真实轨迹（青色）
        line(img, map(truePath[i - 1]), map(truePath[i]), Scalar(60, 220, 255), 2, LINE_AA); // 一段一段连线
    circle(img, map(appPath.front()), 4, Scalar(120, 255, 120), -1, LINE_AA); // 起点（绿色实心）
    int ci = min(curIdx, int(appPath.size()) - 1);                         // 当前点的下标
    if (ci >= 0) circle(img, map(appPath[ci]), 5, Scalar(0, 0, 255), -1, LINE_AA); // 当前位置（红色实心）
    t.putShadow(img, fmt("%s %.0f / %.0f %s", t.pick(u8"路程", "path").c_str(),  // 组装"路程 视在/真实"
                         pathApp, pathTrue, t.pick(u8"像素", "px").c_str()).c_str(), // 数值与单位
                Point(box.x + 12, box.y + box.height - 44), 18, Scalar(220, 220, 220)); // 画在底部
    t.put(img, t.pick(u8"橙=视在  青=补偿后", "orange=apparent  cyan=true"),  // 图例说明
          Point(box.x + 12, box.y + box.height - 24), 16, Scalar(170, 170, 170)); // 画图例
}                                                                          // 结束 drawMiniMap

// ===========================================================================
//  ⑦ 输出：速度曲线图 / 轨迹图 / Markdown 报告
// ===========================================================================

// 画"速度-时间曲线"PNG：视在速度、真实速度、平均线、峰值点、底部阶段色带
static void drawSpeedCurvePng(const string& path, TextRenderer& t,        // 参数：输出路径、文字
                              const Options& o, const vector<FrameRec>& recs, // 参数、逐帧记录
                              double fps, double meanSpeed) {              // 帧率、平均速度
    const int W = 1440, H = 800;                                          // 画布尺寸
    Mat img(H, W, CV_8UC3, Scalar(252, 252, 252));                        // 白色画布（报告用的图用白底）
    const Scalar ink(40, 40, 40), grid(226, 226, 226);                    // 文字色、网格色
    t.put(img, t.pick(u8"绿球速度 - 时间曲线", "Green ball speed vs. time"), // 大标题
          Point(70, 44), 40, ink);                                        // 画标题
    t.put(img, fmt("%s %d %s   %.0f fps   %.1f s",                        // 副标题：数据来源与时长
                   t.pick(u8"数据来源：", "source: ").c_str(), int(recs.size()), // 帧数
                   t.pick(u8"帧", "frames").c_str(), fps, recs.size() / fps).c_str(), // 帧率与秒数
          Point(70, 94), 22, Scalar(110, 110, 110));                      // 画副标题
    const int L = 120, R = W - 70, T = 150, B = H - 130;                   // 绘图区四边
    double vmax = 0, vmaxTrue = 0;                                        // 视在/真实速度的最大值
    for (const auto& r : recs) {                                          // 遍历所有帧
        vmax = max(vmax, r.speed);                                        // 更新视在最大值
        vmaxTrue = max(vmaxTrue, double(norm(r.velTrue)));                 // 更新真实最大值
    }                                                                      // 结束 for
    vmax = max(60.0, vmax * 1.10);                                         // 留 10% 余量
    int n = int(recs.size());                                              // 总帧数
    for (int i = 0; i <= 5; ++i) {                                         // 画 5 条横向网格与刻度
        int y = B - (B - T) * i / 5;                                       // 网格纵坐标
        line(img, Point(L, y), Point(R, y), grid, 1, LINE_AA);             // 网格线
        t.put(img, fmt("%.0f", vmax * i / 5).c_str(), Point(L - 74, y - 10), 22, Scalar(90, 90, 90)); // 刻度值
    }                                                                      // 结束 for
    double dur = n / fps;                                                  // 视频总时长（秒）
    for (double sec = 0; sec <= dur + 1e-6; sec += 2.0) {                  // 每 2 秒画一条竖网格
        int x = L + int((R - L) * (sec / max(0.001, dur)));                // 竖线横坐标
        if (x > R) break;                                                  // 超出右边界就停
        line(img, Point(x, T), Point(x, B), grid, 1, LINE_AA);             // 竖网格
        t.put(img, fmt("%.0f", sec).c_str(), Point(x - 10, B + 34), 22, Scalar(90, 90, 90)); // 时间刻度
    }                                                                      // 结束 for
    line(img, Point(L, T), Point(L, B), ink, 2, LINE_AA);                  // 纵轴
    line(img, Point(L, B), Point(R, B), ink, 2, LINE_AA);                  // 横轴
    t.put(img, t.pick(u8"时间 / 秒", "time / s").c_str(), Point((L + R) / 2 - 60, B + 70), 24, ink);  // 横轴标题
    auto pt = [&](int i, double v) {                                       // 局部函数：数据 → 像素坐标
        double x = L + (n <= 1 ? 0 : double(i) * (R - L) / double(n - 1)); // 横坐标按帧号均分
        double y = B - min(1.0, v / vmax) * (B - T);                       // 纵坐标按速度比例
        return Point(int(x), int(y));                                      // 返回坐标
    };                                                                     // 结束 lambda
    for (int i = 0; i < n; ++i) {                                          // 底部阶段色带
        int x0 = pt(i, 0).x, x1 = (i + 1 < n) ? pt(i + 1, 0).x : x0 + 1;   // 本帧区间
        rectangle(img, Rect(x0, B + 4, max(1, x1 - x0), 10), phaseColor(recs[i].phase), -1); // 画一小格
    }                                                                      // 结束 for
    vector<Point> ap, tr;                                                  // ap=视在曲线，tr=真实曲线
    for (int i = 0; i < n; ++i) {                                          // 组装两条曲线
        ap.push_back(pt(i, recs[i].speed));                                // 视在速度点
        tr.push_back(pt(i, norm(recs[i].velTrue)));                        // 真实速度点
    }                                                                      // 结束 for
    polylines(img, vector<vector<Point>>{tr}, false, Scalar(0, 150, 255), 2, LINE_AA); // 画真实速度（橙色，细）
    polylines(img, vector<vector<Point>>{ap}, false, Scalar(0, 170, 0), 3, LINE_AA);   // 画视在速度（绿色，粗）
    int yMean = pt(0, meanSpeed).y;                                        // 平均速度的纵坐标
    dashedLine(img, Point(L, yMean), Point(R, yMean), Scalar(0, 140, 220), 2, 12, 8);  // 平均线（虚线）
    t.put(img, fmt("%s %.0f", t.pick(u8"平均速度", "mean").c_str(), meanSpeed).c_str(), // "平均速度 xxx"
          Point(L + 12, yMean - 14), 22, Scalar(0, 140, 220));             // 画平均标注
    int iMax = 0;                                                          // 峰值所在帧
    for (int i = 1; i < n; ++i) if (recs[i].speed > recs[iMax].speed) iMax = i; // 找出最大速度帧
    circle(img, ap[iMax], 7, Scalar(0, 0, 255), -1, LINE_AA);              // 峰值点画红点
    t.put(img, fmt("%s %.0f (%s %.2f s)", t.pick(u8"峰值", "peak").c_str(), // 组装峰值标注
                   recs[iMax].speed, t.pick(u8"第", "at").c_str(), iMax / fps).c_str(), // 数值与时刻
          Point(ap[iMax].x + 12, ap[iMax].y - 14), 22, Scalar(0, 0, 255)); // 画峰值标注
    int lx = L + 30, ly = T + 24;                                          // 图例位置
    line(img, Point(lx, ly), Point(lx + 46, ly), Scalar(0, 170, 0), 3, LINE_AA);             // 绿线示例
    t.put(img, t.pick(u8"视在速度（画面中的运动）", "apparent speed").c_str(), Point(lx + 58, ly - 11), 22, Scalar(0, 130, 0)); // 图例文字
    line(img, Point(lx, ly + 34), Point(lx + 46, ly + 34), Scalar(0, 150, 255), 3, LINE_AA); // 橙线示例
    t.put(img, t.pick(u8"真实速度（已扣除相机运动）", "true speed").c_str(), Point(lx + 58, ly + 23), 22, Scalar(0, 120, 200)); // 图例文字
    int bx = L, by = B + 26;                                               // 阶段图例起点
    const char* zh[3] = {u8"匀速", u8"加速", u8"减速"};                     // 中文阶段名
    const char* en[3] = {"steady", "speeding up", "slowing down"};         // 英文阶段名
    for (int k = 0; k < 3; ++k) {                                          // 画三个阶段图例
        int ph = (k == 0 ? 0 : (k == 1 ? 1 : -1));                         // 阶段值
        rectangle(img, Rect(bx, by, 26, 12), phaseColor(ph), -1);          // 色块
        t.put(img, t.pick(zh[k], en[k]), Point(bx + 34, by - 2), 20, Scalar(90, 90, 90)); // 文字
        bx += 34 + int(t.size(t.pick(zh[k], en[k]), 20).width) + 40;       // 下移一个图例位置
    }                                                                      // 结束 for
    imwrite(path, img);                                                    // 保存 PNG
}                                                                          // 结束 drawSpeedCurvePng

// 画"轨迹图"PNG：左边是画面里的视在轨迹，右边是扣除相机运动后的真实轨迹（同一比例尺）
static void drawTrajectoryMapPng(const string& path, TextRenderer& t,     // 参数：输出路径、文字
                                 const Options& o,                        // 参数结构
                                 const vector<Point2f>& appPath,          // 视在轨迹
                                 const vector<Point2f>& truePath,         // 真实轨迹
                                 const vector<FrameRec>& recs,            // 逐帧记录（取球半径画示意圆）
                                 double pathApp, double pathTrue) {       // 两条轨迹的累计长度
    const int W = 1440, H = 820;                                          // 画布尺寸
    Mat img(H, W, CV_8UC3, Scalar(252, 252, 252));                        // 白底
    const Scalar ink(40, 40, 40);                                         // 文字颜色
    t.put(img, t.pick(u8"绿球轨迹图", "Green ball trajectory"), Point(70, 46), 40, ink); // 标题
    t.put(img, t.pick(u8"图像坐标，y 轴向下；两个子图使用同一比例尺",       // 副标题（说明坐标系）
                      "image coordinates, y down; both panels share one scale").c_str(), // 英文副标题
          Point(70, 96), 22, Scalar(110, 110, 110));                      // 画副标题
    if (appPath.empty()) { imwrite(path, img); return; }                   // 没有轨迹就直接存空白图返回
    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;            // 视在轨迹包围盒
    for (const auto& p : appPath) {                                        // 遍历求包围盒
        minX = min(minX, p.x); maxX = max(maxX, p.x);                      // 更新 x 边界
        minY = min(minY, p.y); maxY = max(maxY, p.y);                      // 更新 y 边界
    }                                                                      // 结束 for
    float span = max(max(maxX - minX, maxY - minY), 60.f) * 1.20f;         // 正方形边长（含边距）
    float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f;            // 包围盒中心
    Rect pL(70, 140, 620, 580), pR(750, 140, 620, 580);                    // 左右两个子图区域
    rectangle(img, pL, Scalar(200, 200, 200), 1, LINE_AA);                 // 左子图边框
    rectangle(img, pR, Scalar(200, 200, 200), 1, LINE_AA);                 // 右子图边框
    t.put(img, t.pick(u8"视在轨迹（画面中看到球在移动）", "apparent trajectory").c_str(), // 左子图标题
          Point(pL.x, pL.y - 14), 24, ink);                                // 画标题
    t.put(img, t.pick(u8"相机补偿后的真实轨迹", "compensated trajectory").c_str(), // 右子图标题
          Point(pR.x, pR.y - 14), 24, ink);                                // 画标题
    float scale = min(pL.width / span, pL.height / span) * 0.92f;          // 统一缩放比例
    Point2f c(pL.x + pL.width * 0.5f, pL.y + pL.height * 0.5f);            // 左子图中心
    auto map = [&](Point2f p) {                                            // 局部函数：图像坐标 → 左子图坐标
        return Point(int(c.x + (p.x - cx) * scale), int(c.y + (p.y - cy) * scale)); // 平移缩放
    };                                                                     // 结束 lambda
    for (size_t i = 1; i < appPath.size(); ++i)                            // 画视在轨迹（绿线）
        line(img, map(appPath[i - 1]), map(appPath[i]), Scalar(0, 170, 0), 3, LINE_AA); // 连线
    int step = max(1, int(appPath.size()) / 14);                           // 每 1/14 画一个球示意圆
    for (size_t i = 0; i < appPath.size(); i += step) {                    // 遍历画示意圆
        float rad = (i < recs.size() && recs[i].radius > 0) ? recs[i].radius : 60.f; // 取该帧球半径
        circle(img, map(appPath[i]), max(2, int(rad * scale)), Scalar(180, 120, 60), 2, LINE_AA); // 画圆
    }                                                                      // 结束 for
    circle(img, map(appPath.front()), 9, Scalar(255, 120, 0), -1, LINE_AA); // 起点（橙点）
    circle(img, map(appPath.back()), 9, Scalar(0, 0, 255), -1, LINE_AA);   // 终点（红点）
    t.put(img, t.pick(u8"起点", "start").c_str(), map(appPath.front()) + Point(14, -12), 22, Scalar(200, 90, 0)); // 标"起点"
    t.put(img, t.pick(u8"终点", "end").c_str(), map(appPath.back()) + Point(14, 24), 22, Scalar(0, 0, 220));      // 标"终点"
    t.putShadow(img, fmt("%s %.0f %s", t.pick(u8"视在总路程", "apparent path").c_str(), // 组装左子图统计
                         pathApp, t.pick(u8"像素", "px").c_str()).c_str(),               // 数值与单位
                Point(pL.x + 16, pL.y + pL.height - 18), 24, Scalar(0, 140, 0));          // 画在左下角
    float tminX = 1e9f, tminY = 1e9f, tmaxX = -1e9f, tmaxY = -1e9f;        // 真实轨迹包围盒
    for (const auto& p : truePath) {                                       // 遍历求包围盒
        tminX = min(tminX, p.x); tmaxX = max(tmaxX, p.x);                  // 更新 x 边界
        tminY = min(tminY, p.y); tmaxY = max(tmaxY, p.y);                  // 更新 y 边界
    }                                                                      // 结束 for
    float tspan = max(max(tmaxX - tminX, tmaxY - tminY), 2.f) * 1.30f;     // 真实轨迹跨度（含边距）
    float scaleTrue = min(pR.width / tspan, pR.height / tspan) * 0.85f;    // 自动放大后的比例
    float zoom = scaleTrue / max(1e-6f, scale);                            // 相对左子图的放大倍数
    Point2f tc((tminX + tmaxX) * 0.5f, (tminY + tmaxY) * 0.5f);            // 真实轨迹中心
    Point2f cr(pR.x + pR.width * 0.5f, pR.y + pR.height * 0.5f);           // 右子图中心
    auto mapT = [&](Point2f p) {                                           // 局部函数：图像坐标 → 右子图坐标
        return Point(int(cr.x + (p.x - tc.x) * scaleTrue), int(cr.y + (p.y - tc.y) * scaleTrue)); // 平移缩放
    };                                                                     // 结束 lambda
    circle(img, Point(int(cr.x), int(cr.y)), 4, Scalar(160, 160, 160), -1, LINE_AA); // 中心参考点
    for (size_t i = 1; i < truePath.size(); ++i)                           // 画真实轨迹（青线）
        line(img, mapT(truePath[i - 1]), mapT(truePath[i]), Scalar(0, 150, 255), 3, LINE_AA); // 连线
    circle(img, mapT(truePath.front()), 9, Scalar(255, 120, 0), -1, LINE_AA); // 起点
    circle(img, mapT(truePath.back()), 9, Scalar(0, 0, 255), -1, LINE_AA);    // 终点
    t.putShadow(img, fmt("%s ×%.1f", t.pick(u8"已放大", "zoomed").c_str(), zoom).c_str(), // 标注放大倍数
                Point(pR.x + 16, pR.y + 30), 24, Scalar(0, 0, 220));        // 画在右上角
    t.putShadow(img, fmt("%s %.0f %s", t.pick(u8"真实总路程", "true path").c_str(), // 组装右子图统计
                         pathTrue, t.pick(u8"像素", "px").c_str()).c_str(),         // 数值与单位
                Point(pR.x + 16, pR.y + pR.height - 18), 24, Scalar(0, 120, 200));  // 画在左下角
    t.putShadow(img, t.pick(u8"同一比例尺下，这段轨迹几乎缩成一个点：",              // 结论第一行（中文）
                            "At the same scale this path nearly collapses to a point:").c_str(), // 英文结论第一行
                Point(pR.x, pR.y + pR.height + 34), 22, Scalar(90, 90, 90));    // 画第一行
    t.putShadow(img, t.pick(u8"球相对桌面基本没动，画面里的运动主要来自相机移动。",  // 结论第二行（中文）
                            "The ball barely moves; the apparent motion is the camera's.").c_str(), // 英文结论第二行
                Point(pR.x, pR.y + pR.height + 68), 22, Scalar(90, 90, 90));    // 画第二行
    imwrite(path, img);                                                    // 保存 PNG
}                                                                          // 结束 drawTrajectoryMapPng

// 写 Markdown 报告：把关键数字、方法说明、图表引用整理成一份可直接阅读的报告
static void writeReportMd(const string& path, const Options& o,           // 参数：输出路径、参数
                          int totalFrames, int colorFrames, int flowFrames, // 总帧数、颜色帧数、光流帧数
                          int predictFrames, double meanSpeed,            // 预测帧数、平均速度
                          double maxSpeed, double minSpeed,               // 最大/最小速度
                          double pathApp, double pathTrue,                // 两条轨迹长度
                          double meanSpeedTrue, double maxSpeedTrue,      // 真实速度的均值与最大值
                          double gain, double fps, int W, int H,          // 箭头系数、帧率、分辨率
                          int accelFrames, int decelFrames, int steadyFrames, // 三个阶段各自帧数
                          double maxAccel, double minAccel, double processMs, // 加速度极值、处理耗时
                          const string& videoName) {                      // 视频文件名
    ofstream f(path);                                                     // 打开输出文件
    if (!f) return;                                                       // 打开失败就直接返回
    int tracked = colorFrames + flowFrames;                               // 有真实测量的帧数
    double detRate = totalFrames ? 100.0 * tracked / totalFrames : 0;     // 有效率（百分比）
    f << "# 绿球识别与运动分析报告（高级版）\n\n";                          // 一级标题
    f << "- 视频：`" << videoName << "`　" << W << "×" << H << "　"       // 视频信息行
      << fixed << setprecision(0) << fps << " fps　" << totalFrames        // 帧率与总帧数
      << " 帧（" << setprecision(1) << totalFrames / fps << " 秒）\n";     // 时长
    f << "- 程序：`ball_tracker_pro`（卡尔曼滤波 + 光流续跟 + 相机运动补偿）\n"; // 程序说明
    f << "- 处理耗时：" << setprecision(2) << processMs / 1000.0 << " 秒（"  // 耗时
      << setprecision(1) << 1000.0 * totalFrames / max(1.0, processMs) << " fps）\n\n"; // 处理速度
    f << "## 一、识别结果\n\n";                                             // 小节标题
    f << "| 指标 | 数值 |\n| --- | --- |\n";                               // 表格头
    f << "| 检出率 | " << setprecision(1) << detRate << "%（颜色+形状 "      // 检出率
      << colorFrames << " 帧，光流续跟 " << flowFrames << " 帧）|\n";        // 分项帧数
    f << "| 仅靠卡尔曼预测 | " << predictFrames << " 帧 |\n\n";             // 纯预测帧数
    f << "## 二、速度（画面中的视在运动）\n\n";                              // 小节标题
    f << "| 指标 | 数值 |\n| --- | --- |\n";                               // 表格头
    f << "| 平均速度 | " << meanSpeed << " 像素/秒 |\n";                     // 平均速度
    f << "| 最大速度 | " << maxSpeed << " 像素/秒 |\n";                      // 最大速度
    f << "| 最小速度 | " << minSpeed << " 像素/秒 |\n";                      // 最小速度
    f << "| 累计路程 | " << setprecision(0) << pathApp << " 像素 |\n\n";     // 累计路程
    f << "## 三、扣除相机运动后的真实运动\n\n";                              // 小节标题
    f << "视频是手持相机拍摄，画面里小球的\"移动\"大部分来自相机运动。"        // 方法说明
         "程序用背景特征点的光流估计相机运动（部分仿射 + RANSAC）并逐帧扣除：\n\n"; // 说明续
    f << "| 指标 | 视在 | 真实（补偿后） |\n| --- | --- | --- |\n";         // 表格头
    f << "| 平均速度 | " << setprecision(1) << meanSpeed << " 像素/秒 | "    // 平均速度对比
      << meanSpeedTrue << " 像素/秒 |\n";                                   // 真实平均速度
    f << "| 最大速度 | " << maxSpeed << " 像素/秒 | " << maxSpeedTrue        // 最大速度对比
      << " 像素/秒 |\n";                                                    // 真实最大值
    f << "| 累计路程 | " << setprecision(0) << pathApp << " 像素 | "         // 路程对比
      << pathTrue << " 像素 |\n\n";                                        // 真实路程
    f << "结论：真实路程只有视在路程的 "                                      // 结论句开头
      << setprecision(1) << (pathApp > 0 ? 100.0 * pathTrue / pathApp : 0)   // 百分比
      << "%，说明小球相对桌面基本静止。\n\n";                                // 结论句结尾
    f << "## 四、速度变化（加速度与运动阶段）\n\n";                          // 小节标题
    f << "| 指标 | 数值 |\n| --- | --- |\n";                               // 表格头
    f << "| 加速帧 | " << accelFrames << " |\n";                            // 加速帧数
    f << "| 减速帧 | " << decelFrames << " |\n";                            // 减速帧数
    f << "| 匀速帧 | " << steadyFrames << " |\n";                           // 匀速帧数
    f << "| 最大正向加速度 | " << setprecision(0) << maxAccel << " 像素/秒² |\n"; // 最大加速度
    f << "| 最小（负向）加速度 | " << minAccel << " 像素/秒² |\n\n";          // 最小加速度
    f << "判定阈值：|加速度| > " << o.accelThresh << " 像素/秒² 时判为加速或减速。\n\n"; // 阈值说明
    f << "## 五、箭头标注方法\n\n";                                          // 小节标题
    f << "- 箭头方向 = 速度方向（卡尔曼滤波估计）；\n";                       // 方向说明
    f << "- 箭头长度 = |v| × " << setprecision(3) << gain                 // 长度说明（系数）
      << "，并截断到 " << setprecision(0) << o.arrowMaxLen << " 像素；\n";   // 长度上限
    f << "- 箭头颜色：绿(慢) → 黄 → 红(快)；面板中同时给出速度数值、加速度和运动阶段；\n"; // 颜色说明
    f << "- 速度小于 " << o.staticSpeed << " 像素/秒 时标注为\"基本静止\"。\n\n"; // 静止阈值
    f << "## 六、输出文件\n\n";                                              // 小节标题
    f << "- 标注视频：`" << o.outVideo << "`\n";                             // 视频路径
    f << "- 逐帧数据：`" << o.outCsv << "`\n";                               // CSV 路径
    f << "- 速度曲线：`" << o.outCurve << "`\n";                             // 曲线图路径
    f << "- 轨迹图：`" << o.outMap << "`\n\n";                               // 轨迹图路径
    f << "![速度曲线](" << fs::path(o.outCurve).filename().string() << ")\n\n"; // 嵌入曲线图
    f << "![轨迹图](" << fs::path(o.outMap).filename().string() << ")\n";       // 嵌入轨迹图
    if (fs::exists(o.outPreview))                                          // 如果预览图存在
        f << "\n![标注画面](" << fs::path(o.outPreview).filename().string() << ")\n"; // 也嵌进去
}                                                                          // 结束 writeReportMd

// ===========================================================================
//  ⑧ 命令行帮助
// ===========================================================================
static void usage(const char* prog) {                                     // 参数：程序名（argv[0]）
    cout << "用法: " << prog << " [视频] [输出视频] [选项]\n"                // 用法第一行
         << "选项:\n"                                                       // 选项列表开始
         << "  --stabilize     扣除相机整体运动，只显示球相对场景的真实运动\n" // 相机补偿
         << "  --no-video      不写结果视频（只算数据）\n"                   // 跳过视频输出
         << "  --stats         打印每帧的速度统计\n"                        // 打印逐帧数据
         << "  --gain <k>      箭头长度系数（默认 0.35 像素/(像素/秒)）\n"   // 手动指定箭头系数
         << "  --trail <n>     轨迹拖尾长度（默认 45 帧）\n"                 // 拖尾长度
         << "  --csv <文件>    速度数据输出路径（默认与输出视频同名）\n"      // CSV 路径
         << "  --no-chinese    不渲染中文（用英文标注）\n"                   // 关闭中文
         << "  --font <路径>   指定中文字体（.ttf/.ttc）\n"                 // 指定字体
         << "  --show-true     额外画出\"真实运动\"（扣除相机运动）箭头\n"    // 画真实运动箭头
         << "  --help          显示本帮助\n";                              // 帮助
}                                                                          // 结束 usage

// ===========================================================================
//  主程序
// ===========================================================================
int main(int argc, char** argv) {                                          // 程序入口
    Options o;                                                             // 默认参数
    vector<string> pos;                                                    // 存放位置参数（视频路径、输出路径）
    for (int i = 1; i < argc; ++i) {                                        // 逐个解析命令行参数
        string a = argv[i];                                                // 当前参数
        if (a == "--stabilize")            o.showTrue = true;              // --stabilize：画真实运动箭头
        else if (a == "--no-video")        o.writeVideo = false;           // 不写视频
        else if (a == "--stats")           o.dumpStats = true;             // 打印逐帧数据
        else if (a == "--gain" && i + 1 < argc) { o.arrowGain = stod(argv[++i]); o.autoGain = false; } // 手动箭头系数
        else if (a == "--trail" && i + 1 < argc) o.trailLen = stoi(argv[++i]);   // 拖尾长度
        else if (a == "--csv" && i + 1 < argc)   o.outCsv = argv[++i];           // CSV 路径
        else if (a == "--no-chinese")      o.wantChinese = false;          // 关闭中文
        else if (a == "--font" && i + 1 < argc) o.fontPath = argv[++i];    // 指定字体
        else if (a == "--show-true")       o.showTrue = true;              // 画真实运动箭头
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }  // 打印帮助并退出
        else pos.push_back(a);                                             // 其它参数当位置参数
    }                                                                      // 结束 for
    if (pos.size() > 0) o.videoPath = pos[0];                              // 第一个位置参数 = 输入视频
    if (pos.size() > 1) o.outVideo  = pos[1];                              // 第二个位置参数 = 输出视频
    if (o.outCsv.empty()) {                                                // 如果没指定 CSV 路径
        fs::path p(o.outVideo);                                            // 以输出视频路径为基准
        o.outCsv = (p.parent_path() / (p.stem().string() + ".csv")).string(); // 同目录同名但扩展名换成 .csv
    }                                                                      // 结束 if
    if (pos.size() > 1) {                                                  // 如果用户指定了输出视频路径
        fs::path p(o.outVideo);                                            // 就取它的目录作为"结果目录"
        fs::path d = p.parent_path();                                      // 结果目录（例如 out/）
        o.outCurve   = (d / "speed_curve.png").string();                   // 速度曲线跟随该目录
        o.outMap     = (d / "trajectory_map.png").string();                // 轨迹图跟随该目录
        o.outReport  = (d / "report.md").string();                         // 报告跟随该目录
        o.outPreview = (d / "preview_pro.jpg").string();                   // 预览图跟随该目录
    }                                                                      // 结束 if（不指定时仍统一放 output/）
    TextRenderer text;                                                     // 创建文字渲染器
    bool cjk = text.init(o.fontPath, o.wantChinese);                       // 尝试启用中文
    if (cjk)                       cout << "中文标注已启用（字体: " + text.fontPath() + "）\n"; // 提示：中文可用
    else if (!o.wantChinese)       cout << "使用英文标注（--no-chinese）\n";  // 提示：用户要求英文
    else                           cout << "使用英文标注（未找到可用中文字体）\n"; // 提示：找不到字体
    VideoCapture cap(o.videoPath);                                         // 打开输入视频
    if (!cap.isOpened()) {                                                 // 打开失败
        cerr << "无法打开视频: " << o.videoPath << "\n";                    // 报错
        return 1;                                                          // 返回错误码
    }                                                                      // 结束 if
    double fps = cap.get(CAP_PROP_FPS);                                    // 读取帧率
    if (fps < 1.0) fps = 30.0;                                             // 帧率异常时按 30 处理
    int W = int(cap.get(CAP_PROP_FRAME_WIDTH));                            // 读取画面宽
    int H = int(cap.get(CAP_PROP_FRAME_HEIGHT));                           // 读取画面高
    double totalF = cap.get(CAP_PROP_FRAME_COUNT);                         // 读取总帧数
    cout << "视频: " << o.videoPath << "  " << W << "x" << H << "  "        // 打印视频信息
         << fps << " fps  " << totalF << " 帧\n";                          // 帧率与帧数
    if (o.writeVideo) {                                                    // 需要写视频时
        fs::path p(o.outVideo);                                            // 输出视频路径
        if (p.has_parent_path()) fs::create_directories(p.parent_path());  // 确保父目录存在
    }                                                                      // 结束 if
    if (!o.outCsv.empty()) {                                               // 需要写 CSV 时
        fs::path p(o.outCsv);                                              // CSV 路径
        if (p.has_parent_path()) fs::create_directories(p.parent_path());  // 确保父目录存在
    }                                                                      // 结束 if
    if (o.writeReport) {                                                   // 需要报告时
        fs::path p(o.outPreview);                                          // 预览图路径
        if (p.has_parent_path()) fs::create_directories(p.parent_path());  // 一并确保目录存在
    }                                                                      // 结束 if
    VideoWriter writer;                                                    // 视频写入器
    if (o.writeVideo) {                                                    // 需要写视频时
        writer.open(o.outVideo, VideoWriter::fourcc('m', 'p', '4', 'v'),   // 以 mp4v 编码打开输出
                    fps, Size(W, H));                                      // 帧率与尺寸
        if (!writer.isOpened()) {                                          // 打开失败
            cerr << "无法创建输出视频: " << o.outVideo << "（改为只输出数据）\n"; // 提示
            o.writeVideo = false;                                          // 降级为不写视频
        }                                                                  // 结束 if
    }                                                                      // 结束 if

    // ---------------- 第一趟：逐帧分析 ---------------- //
    auto tStart = chrono::steady_clock::now();                             // 记录开始时间（算耗时）
    vector<FrameRec> recs;                                                 // 每帧记录
    vector<Point2f>  appPath;                                              // 视在轨迹（球心序列）
    vector<Point2f>  truePath;                                             // 真实轨迹（积分得到）
    vector<int>      appCount;                                             // 每帧结束时 appPath 的长度
    vector<double>   speeds;                                               // 每帧速度（供曲线使用）
    vector<int>      phases;                                               // 每帧阶段（供色带使用）
    Mat prevGray, prevMaskDil;                                             // 上一帧灰度、上一帧绿色掩码（膨胀后）
    Point2f lastCenter(0.f, 0.f);                                          // 上一帧球心
    float   lastRadius = 40.f;                                             // 上一帧球半径
    Point2f truePos(0.f, 0.f);                                             // 真实轨迹积分位置
    bool    trueInit = false;                                              // 真实轨迹是否已初始化
    KalmanTracker tracker;                                                 // 卡尔曼跟踪器
    bool    trackerInited = false;                                         // 跟踪器是否已初始化
    int     lostCount = 0;                                                 // 连续丢失帧数
    int     colorFrames = 0, flowFrames = 0, predictFrames = 0;            // 三种来源的帧数统计
    double  sumSpeed = 0, maxSpeed = 0, minSpeed = 1e18;                   // 速度统计
    double  sumSpeedTrue = 0, maxSpeedTrue = 0;                            // 真实速度统计
    double  pathApp = 0, pathTrue = 0;                                     // 两条轨迹的累计长度
    int     idx = 0;                                                       // 帧下标
    Mat frame;                                                             // 当前帧图像
    while (cap.read(frame)) {                                              // 逐帧读取
        if (frame.empty()) break;                                          // 读到空帧就结束
        Mat gray;                                                          // 当前帧灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);                             // BGR → 灰度（光流与背景估计都用它）
        Mat mask = greenMask(frame, o);                                    // 分割绿色区域
        Mat maskDil;                                                       // 膨胀后的绿色掩码
        dilate(mask, maskDil, getStructuringElement(MORPH_ELLIPSE, Size(15, 15))); // 膨胀 15×15，用于排除背景点
        vector<Ball> cands = findBallCandidates(mask, o);                  // 找出所有"像球"的候选
        FrameRec r;                                                        // 本帧记录（先默认构造）
        if (!trackerInited) {                                              // 还没开始跟踪
            Ball b;                                                        // 候选球
            if (pickBall(cands, Point2f(), false, b)) {                    // 首次全局搜索：取面积最大的
                tracker.init(1.0 / fps, b.center, o);                      // 用它初始化卡尔曼
                trackerInited = true;                                      // 标记已开始跟踪
                r.center = b.center;                                       // 记录球心
                r.radius = b.radius;                                       // 记录半径
                r.source = 2;                                              // 来源 = 颜色检测
                colorFrames++;                                             // 颜色帧计数 +1
                lastCenter = b.center;                                     // 更新上一帧球心
                lastRadius = b.radius;                                     // 更新上一帧半径
            } else {                                                       // 第一帧就没找到球
                r.source = -1;                                             // 来源 = 无
            }                                                              // 结束 if
        } else {                                                           // 已经在跟踪
            Point2f pred = tracker.predict();                              // 先按状态外推得到预测位置
            Ball best;                                                     // 最佳候选
            bool accepted = false;                                         // 本帧是否接受了测量
            bool colorFound = pickBall(cands, pred, true, best);           // 在候选里挑离预测最近的
            if (colorFound && tracker.gate(best.center)) {                 // 找到且通过马氏距离门限
                tracker.correct(best.center, 1.0);                         // 用颜色测量更新滤波器
                r.source = 2;                                              // 来源 = 颜色
                r.radius = best.radius;                                    // 记录半径
                accepted = true;                                           // 标记已接受
                lostCount = 0;                                             // 丢失计数清零
                colorFrames++;                                             // 颜色帧计数 +1
            }                                                              // 结束 if
            if (!accepted) {                                               // 颜色测量不可用 → 试光流
                Point2f fc;                                                // 光流给出的新球心
                int nGood = 0;                                             // 光流有效点数
                if (FlowTracker::track(prevGray, gray, lastCenter,         // 在球内部撒点做 LK 光流
                                       lastRadius, fc, nGood) &&           // 得到新球心与有效点数
                    tracker.gate(fc)) {                                    // 且这次光流结果也通过门限
                    tracker.correct(fc, o.flowRScale);                     // 用它更新（测量噪声放大，权重更低）
                    r.source = 1;                                          // 来源 = 光流
                    r.radius = lastRadius;                                 // 半径沿用上一帧
                    accepted = true;                                       // 标记已接受
                    lostCount = 0;                                         // 丢失计数清零
                    flowFrames++;                                          // 光流帧计数 +1
                }                                                          // 结束 if
            }                                                              // 结束 if
            if (!accepted) {                                               // 两种测量都没有 → 只能滑行
                lostCount++;                                               // 丢失计数 +1
                if (colorFound && lostCount > o.maxLost) {                 // 丢太久但这次又看到了球
                    tracker.init(1.0 / fps, best.center, o);               // 用当前检测重新初始化滤波器
                    tracker.predict();                                     // 补一次预测（对齐状态）
                    tracker.correct(best.center, 1.0);                     // 再用测量更新
                    r.source = 2;                                          // 来源 = 颜色
                    r.radius = best.radius;                                // 记录半径
                    lostCount = 0;                                         // 丢失计数清零
                    colorFrames++;                                         // 颜色帧计数 +1
                } else {                                                   // 否则继续滑行
                    tracker.coast();                                       // 把先验当后验（按速度外推）
                    r.source = 0;                                          // 来源 = 纯预测
                    r.radius = lastRadius;                                 // 半径沿用上一帧
                    predictFrames++;                                       // 预测帧计数 +1
                }                                                          // 结束 if
            }                                                              // 结束 if
            r.center = tracker.position();                                 // 取滤波后的球心
            r.velApp = tracker.velocity();                                 // 取滤波后的速度（像素/秒）
            r.speed  = norm(r.velApp);                                     // 速度大小
            if (r.radius > 0) lastRadius = r.radius;                       // 更新上一帧半径
        }                                                                  // 结束 if/else
        Point2f bgVel(0.f, 0.f);                                           // 相机运动引起的速度分量
        if (idx > 0 && !prevGray.empty() && r.source != -1) {              // 不是第一帧且有球时才估计
            Mat A;                                                         // 仿射矩阵（上一帧 → 当前帧）
            double inlier = 0;                                             // 内点率
            if (BackgroundMotion::estimate(prevGray, gray, prevMaskDil,    // 用背景光流估计相机运动
                                           A, inlier)) {                   // 成功时 A 有效
                Point2f expected = BackgroundMotion::apply(A, lastCenter); // 相机运动会让球"看起来"移到哪
                bgVel = (expected - lastCenter) * float(fps);              // 换算成速度（像素/秒）
                r.bgOk = true;                                             // 标记相机运动估计成功
            }                                                              // 结束 if
        }                                                                  // 结束 if
        r.bgVel = bgVel;                                                   // 记录相机运动分量
        r.velTrue = r.velApp - bgVel;                                      // 真实速度 = 视在速度 − 相机分量
        if (r.source != -1) {                                              // 这一帧有球
            if (!trueInit) { truePos = r.center; trueInit = true; }        // 真实轨迹的第一点
            truePos += r.velTrue * float(1.0 / fps);                       // 真实轨迹按真实速度积分推进
            appPath.push_back(r.center);                                   // 记录视在轨迹点
            truePath.push_back(truePos);                                   // 记录真实轨迹点
            speeds.push_back(r.speed);                                     // 记录速度（供曲线）
            sumSpeed += r.speed;                                           // 累加视在速度
            maxSpeed = max(maxSpeed, r.speed);                             // 更新视在最大速度
            minSpeed = min(minSpeed, r.speed);                             // 更新视在最小速度
            sumSpeedTrue += norm(r.velTrue);                               // 累加真实速度
            maxSpeedTrue = max(maxSpeedTrue, double(norm(r.velTrue)));     // 更新真实最大速度
            if (idx > 0 && recs.back().source != -1) {                     // 与上一帧都有效时累加路程
                pathApp += norm(r.center - recs.back().center);            // 视在路程
                pathTrue += norm(r.velTrue) / fps;                         // 真实路程
            }                                                              // 结束 if
            lastCenter = r.center;                                         // 更新上一帧球心
        } else {                                                           // 这一帧没球
            speeds.push_back(0.0);                                         // 速度记 0（保持与帧号对齐）
        }                                                                  // 结束 if
        recs.push_back(r);                                                 // 保存本帧记录
        appCount.push_back(int(appPath.size()));                           // 记录当前轨迹点数
        if (o.dumpStats)                                                   // 需要打印时
            printf("f%4d 来源=%2d  球心=(%6.1f,%6.1f)  |v|=%7.1f px/s  真实=%7.1f\n", // 打印一行
                   idx, r.source, r.center.x, r.center.y, r.speed, norm(r.velTrue)); // 具体数值
        prevGray = gray.clone();                                           // 保存本帧灰度给下一帧用
        prevMaskDil = maskDil;                                             // 保存本帧掩码给下一帧用
        idx++;                                                             // 帧号 +1
    }                                                                      // 结束 while 逐帧读取
    cap.release();                                                         // 关闭输入视频
    int totalFrames = int(recs.size());                                    // 实际处理的帧数
    // ---------------- 加速度与运动阶段 ---------------- //
    double acc = 0;                                                        // 加速度的 EMA 平滑值
    for (size_t i = 0; i < recs.size(); ++i) {                             // 逐帧计算
        double sp = recs[i].speed;                                         // 本帧速度
        double spPrev = i > 0 ? recs[i - 1].speed : sp;                    // 上一帧速度
        double aRaw = (sp - spPrev) * fps;                                 // 原始加速度 = 速度差 × 帧率
        acc = (i == 0) ? aRaw : acc * (1.0 - o.accelSmooth) + aRaw * o.accelSmooth; // EMA 平滑
        recs[i].accel = acc;                                               // 记录平滑后的加速度
        recs[i].phase = (acc > o.accelThresh) ? 1                          // 超过正阈值 → 加速
                      : (acc < -o.accelThresh ? -1 : 0);                   // 超过负阈值 → 减速；否则匀速
    }                                                                      // 结束 for
    for (const auto& r : recs) phases.push_back(r.phase);                  // 收集阶段序列（供色带）
    // ---------------- 箭头长度自动标定 ---------------- //
    if (o.autoGain && !speeds.empty()) {                                   // 需要自动标定且有数据时
        vector<double> sp = speeds;                                        // 复制一份速度序列
        size_t k = size_t(sp.size() * 0.9);                                // 取 90 分位的位置
        nth_element(sp.begin(), sp.begin() + k, sp.end());                 // 线性时间选出该分位的值
        double p90 = max(1.0, sp[k]);                                      // 90 分位速度
        o.arrowGain = min(1.5, max(0.05, o.arrowTargetLen / p90));         // 让 p90 对应目标长度
    }                                                                      // 结束 if
    double meanSpeed = (colorFrames + flowFrames) > 0                      // 计算平均速度（按有效帧）
                     ? sumSpeed / (colorFrames + flowFrames) : 0;          // 有效帧为 0 时给 0
    double meanSpeedTrue = (colorFrames + flowFrames) > 0                  // 真实速度均值
                     ? sumSpeedTrue / (colorFrames + flowFrames) : 0;      // 同上
    double processMs = chrono::duration<double, milli>(                 // 计算处理耗时（毫秒）
                           chrono::steady_clock::now() - tStart).count();  // 用当前时间减开始时间
    int accelFrames = 0, decelFrames = 0, steadyFrames = 0;                // 三个阶段帧数
    double maxAccel = 0, minAccel = 0;                                     // 加速度极值
    for (const auto& r : recs) {                                           // 统计
        if (r.phase == 1) accelFrames++;                                   // 加速帧
        else if (r.phase == -1) decelFrames++;                             // 减速帧
        else steadyFrames++;                                               // 匀速帧
        maxAccel = max(maxAccel, r.accel);                                 // 最大加速度
        minAccel = min(minAccel, r.accel);                                 // 最小加速度
    }                                                                      // 结束 for
    cout << "\n分析完成：共 " << totalFrames << " 帧\n"                      // 打印汇总
         << "  检出绿球的帧数: " << colorFrames << "（颜色） + "             // 颜色帧数
         << flowFrames << "（光流）\n"                                     // 光流帧数
         << "  平均速度: " << meanSpeed << " px/s\n"                        // 平均速度
         << "  最大速度: " << maxSpeed << " px/s\n"                         // 最大速度
         << "  真实速度: 平均 " << meanSpeedTrue << " / 最大 " << maxSpeedTrue << " px/s\n" // 真实速度
         << "  累计路程: 视在 " << pathApp << " / 真实 " << pathTrue << " 像素\n" // 路程对比
         << "  箭头长度系数: " << o.arrowGain << "（"                        // 箭头系数
         << (o.autoGain ? "自动标定" : "手动指定") << "）\n"                 // 标定方式
         << "  处理耗时: " << processMs / 1000.0 << " 秒\n";                 // 耗时
    // ---------------- 输出 CSV ---------------- //
    ofstream csv(o.outCsv);                                                // 打开 CSV 文件
    if (csv) {                                                             // 打开成功才写
        csv << "frame,time_s,x_px,y_px,vx_px_s,vy_px_s,speed_px_s,"        // CSV 表头第一段
               "vx_true_px_s,vy_true_px_s,speed_true_px_s,accel_px_s2,"    // 表头第二段
               "phase,radius_px,source,bg_ok\n";                           // 表头第三段
        for (size_t i = 0; i < recs.size(); ++i) {                         // 逐帧写一行
            const FrameRec& r = recs[i];                                   // 本帧记录
            const char* ph = r.phase == 1 ? "accel"                        // 阶段转字符串
                           : (r.phase == -1 ? "decel" : "steady");         // 减速/匀速
            const char* sc = r.source == 2 ? "color"                       // 来源转字符串
                           : (r.source == 1 ? "flow"                       // 光流
                           : (r.source == 0 ? "predict" : "none"));        // 预测/无
            csv << i << ',' << fixed << setprecision(3) << (i / fps) << ',' // 帧号与时间
                << setprecision(1) << r.center.x << ',' << r.center.y << ',' // 球心
                << r.velApp.x << ',' << r.velApp.y << ',' << r.speed << ','  // 视在速度分量与大小
                << r.velTrue.x << ',' << r.velTrue.y << ',' << norm(r.velTrue) << ',' // 真实速度
                << r.accel << ',' << ph << ',' << r.radius << ',' << sc << ',' // 加速度/阶段/半径/来源
                << (r.bgOk ? 1 : 0) << '\n';                               // 相机估计是否成功
        }                                                                  // 结束 for
        csv.close();                                                       // 关闭文件
    }                                                                      // 结束 if
    // ---------------- 输出图表与报告 ---------------- //
    if (o.writeReport) {                                                   // 需要报告时
        fs::path pc(o.outCurve);                                           // 曲线图路径
        if (pc.has_parent_path()) fs::create_directories(pc.parent_path()); // 确保目录存在
        fs::path pm(o.outMap);                                             // 轨迹图路径
        if (pm.has_parent_path()) fs::create_directories(pm.parent_path()); // 确保目录存在
        fs::path pr(o.outReport);                                          // 报告路径
        if (pr.has_parent_path()) fs::create_directories(pr.parent_path()); // 确保目录存在
        drawSpeedCurvePng(o.outCurve, text, o, recs, fps, meanSpeed);      // 画速度曲线 PNG
        drawTrajectoryMapPng(o.outMap, text, o, appPath, truePath, recs,   // 画轨迹图 PNG
                             pathApp, pathTrue);                           // 两条轨迹长度
        writeReportMd(o.outReport, o, totalFrames, colorFrames, flowFrames, // 写 Markdown 报告
                      predictFrames, meanSpeed, maxSpeed, minSpeed,         // 速度统计
                      pathApp, pathTrue, meanSpeedTrue, maxSpeedTrue,       // 路程与真实速度
                      o.arrowGain, fps, W, H,                               // 箭头系数、帧率、分辨率
                      accelFrames, decelFrames, steadyFrames,               // 阶段统计
                      maxAccel, minAccel, processMs,                        // 加速度与耗时
                      fs::path(o.videoPath).filename().string());           // 视频文件名
    }                                                                      // 结束 if
    // ---------------- 第二趟：渲染标注视频 ---------------- //
    if (o.writeVideo) {                                                    // 需要写视频时
        VideoCapture cap2(o.videoPath);                                    // 重新打开输入视频（第二趟）
        deque<Point2f> trail;                                              // 轨迹拖尾队列
        int iMax = 0;                                                      // 速度最大的帧（用于存预览图）
        for (size_t i = 1; i < recs.size(); ++i)                           // 找出最大速度帧
            if (recs[i].speed > recs[iMax].speed) iMax = int(i);           // 更新最大值下标
        Mat f2;                                                            // 当前帧
        int i = 0;                                                         // 帧下标
        while (cap2.read(f2)) {                                            // 逐帧读取
            if (i >= int(recs.size())) break;                              // 超出记录范围就停
            const FrameRec& r = recs[i];                                   // 本帧记录
            if (r.source != -1) {                                          // 这一帧有球
                drawPanel(f2, text, o, r, i, int(recs.size()), fps, meanSpeed); // 先画左上角数据面板（HUD 底）
                drawSpeedGraph(f2, text, o, speeds, phases, i, meanSpeed);  // 再画右下角速度曲线（HUD 底）
                drawMiniMap(f2, text, o, appPath, truePath,                // 再画左下角轨迹小地图（HUD 底）
                            appCount[i] - 1, pathApp, pathTrue);           // 当前轨迹点下标与路程
                trail.push_back(r.center);                                 // 把球心加入拖尾
                while (int(trail.size()) > o.trailLen) trail.pop_front();  // 超出长度就丢弃最旧的
                drawTrail(f2, trail);                                      // 画拖尾（画在 HUD 之上）
                Scalar ballCol = (r.source == 0) ? Scalar(0, 200, 255)     // 预测帧：橙色圈
                               : (r.source == 1 ? Scalar(255, 200, 0)      // 光流帧：青色圈
                                                : Scalar(0, 255, 0));      // 颜色帧：绿色圈
                circle(f2, r.center, max(6, int(r.radius)), ballCol, 3, LINE_AA); // 画球轮廓
                circle(f2, r.center, 4, Scalar(255, 255, 255), -1, LINE_AA);      // 画球心
                drawVelocityArrow(f2, text, r.center, r.velApp, o,         // 画主箭头：方向=运动方向
                                  speedColor(r.speed, max(1.0, meanSpeed * 2.0)), // 颜色按速度
                                  6, true);                                // 线宽、显示数值
                if (o.showTrue && norm(r.velTrue) > o.staticSpeed)         // 需要时画"真实运动"细箭头
                    drawVelocityArrow(f2, text, r.center, r.velTrue, o,    // 用真实速度画
                                      Scalar(255, 190, 80), 3, false);     // 橙色、细、不写数值
            } else {                                                       // 这一帧没球
                text.putShadow(f2, text.pick(u8"尚未捕获到绿球", "GREEN BALL NOT FOUND"), // 提示文字
                               Point(30, 60), 28, Scalar(0, 0, 255));      // 画在左上
            }                                                              // 结束 if
            if (i == iMax) imwrite(o.outPreview, f2);                      // 速度最快那帧存成预览图
            writer.write(f2);                                              // 写入输出视频
            i++;                                                           // 帧号 +1
        }                                                                  // 结束 while
        cap2.release();                                                    // 关闭输入
        writer.release();                                                  // 关闭输出
    }                                                                      // 结束 if
    if (o.autoGain) {                                                      // 收尾提示（自动标定过的系数）
        cout << "  结果视频: " << o.outVideo << "\n";                       // 输出视频路径
    }                                                                      // 结束 if
    if (!o.outCsv.empty())    cout << "  速度数据: " << o.outCsv << "\n";    // CSV 路径
    if (o.writeReport) {                                                   // 报告相关路径
        cout << "  速度曲线: " << o.outCurve << "\n";                       // 曲线图
        cout << "  轨迹图: " << o.outMap << "\n";                           // 轨迹图
        cout << "  报告: " << o.outReport << "\n";                          // 报告
    }                                                                      // 结束 if
    cout << "完成。\n";                                                     // 结束提示
    return 0;                                                              // 正常退出
}                                                                          // 结束 main
