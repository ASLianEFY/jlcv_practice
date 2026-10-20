// ============================================================================
//  机甲大赛视觉作业：识别视频中的绿球，并用箭头标注运动方向和速度变化
//  Green-ball detection & motion tracking with direction/speed arrows
//
//  编译运行详见同目录 README.md
//
//  【整体流程】
//    1. 逐帧读视频 → BGR 转 HSV → 阈值分割出绿色区域（mask）
//    2. 形态学开/闭运算去掉噪点、补上小洞
//    3. findContours 找轮廓 → 用「面积 + 外接矩形长宽比 + 圆形度」三重筛选出球
//    4. 光流估计相机全局运动，把上一帧球心投影到当前帧，得到「球相对场景」的位移
//    5. 位移 × 帧率 = 瞬时速度，再做指数滑动平均(EMA)平滑
//    6. 从球心沿速度方向画箭头，长度正比于速度 → 输出标注后的视频
//
//  作者：xinyusu        环境：Ubuntu + OpenCV 4.x（也可用于 Windows + VS）
// ============================================================================

#include <opencv2/opencv.hpp>

#include <cmath>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// 可调参数集中放在这里，方便对照调试
// ----------------------------------------------------------------------------
struct Params {
    // ---- 绿色阈值（HSV）。H: 色相 0~179, S: 饱和度 0~255, V: 亮度 0~255 ----
    int    h_lo = 30, h_hi = 95;
    int    s_lo = 40, s_hi = 255;
    int    v_lo = 40, v_hi = 255;

    // ---- 轮廓筛选 ----
    double min_area = 1500.0;   // 面积太小的一律当噪点扔掉
    double min_circ = 0.65;     // 圆形度 4*pi*A/P^2，越接近 1 越像圆
    double ar_lo    = 0.75;     // 外接矩形宽高比下限（圆柱侧面是细长条，会被滤掉）
    double ar_hi    = 1.30;     // 宽高比上限
    double assoc_dist = 250.0;  // 与上一帧球心的最大关联距离（像素）
    double area_ratio_lo = 0.35, area_ratio_hi = 2.8;  // 面积突变保护

    // ---- 速度估计 ----
    double alpha      = 0.30;   // EMA 平滑系数：越大越灵敏，越小越平滑
    double decay      = 0.80;   // 丢失目标时速度衰减系数
    double min_speed_show = 25.0;  // 低于该速度(px/s)不画箭头，防止抖动

    // ---- 箭头绘制 ----
    double k_len     = 0.70;    // 箭头长度 = 速度 × k_len（像素）
    double min_arrow = 30.0;    // 箭头最短长度（像素）
    double max_arrow = 260.0;   // 箭头最长长度（像素），防止画到屏幕外
    double speed_ref = 300.0;   // 配色基准速度：达到该速度即显示为最红

    // ---- 其他 ----
    bool  compensate = true;    // 是否做相机全局运动补偿
    int   traj_len   = 90;      // 尾迹长度（帧）
};

// ----------------------------------------------------------------------------
// 一帧里的球检测结果
// ----------------------------------------------------------------------------
struct Ball {
    bool        found = false;
    cv::Point2f center{0.f, 0.f};
    float       radius = 0.f;
    double      area   = 0.0;
};

// ----------------------------------------------------------------------------
// 工具：按 0~max 归一化取值映射到 绿(慢) → 黄 → 红(快) 的颜色
// ----------------------------------------------------------------------------
static cv::Scalar speedColor(double ratio) {
    ratio = std::max(0.0, std::min(1.0, ratio));
    // OpenCV 的 H 在 0~179，60 是绿色，0 是红色
    int hue = static_cast<int>(60.0 * (1.0 - ratio));
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 230, 235));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b p = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(p[0], p[1], p[2]);  // BGR
}

// ----------------------------------------------------------------------------
// 第 1~3 步：颜色分割 + 形态学 + 轮廓筛选，找出球
//   prev_center / prev_area 用于「时间连续性」关联，避免把绿色圆柱当球
// ----------------------------------------------------------------------------
static Ball detectBall(const cv::Mat& frame, const Params& p,
                       const cv::Point2f& prev_center, double prev_area) {
    Ball ball;

    // (1) BGR → HSV，按阈值分割出绿色像素
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv,
                cv::Scalar(p.h_lo, p.s_lo, p.v_lo),
                cv::Scalar(p.h_hi, p.s_hi, p.v_hi),
                mask);

    // (2) 形态学处理：先开运算(去小白点噪声)，再闭运算(补小洞让球更完整)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // (3) 找外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_score = -1e9;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < p.min_area) continue;                       // 太小 → 噪点

        double peri = cv::arcLength(c, true);
        if (peri <= 0) continue;
        double circ = 4.0 * CV_PI * area / (peri * peri);      // 圆形度
        if (circ < p.min_circ) continue;                       // 不够圆 → 圆柱侧面等

        cv::Rect box = cv::boundingRect(c);
        if (box.height <= 0) continue;
        double ar = static_cast<double>(box.width) / box.height;
        if (ar < p.ar_lo || ar > p.ar_hi) continue;            // 细长 → 圆柱

        // 圆度 + 面积（球在画面里投影是圆的，宽高比≈1）
        cv::Point2f ctr;
        float radius = 0.f;
        cv::minEnclosingCircle(c, ctr, radius);

        double score = circ * 100.0;                           // 基础分：越圆越好

        // 时间连续性：离上一帧球心越近、面积越接近，得分越高
        if (prev_area > 0) {
            double d = cv::norm(ctr - prev_center);
            double ratio = area / prev_area;
            if (ratio > p.area_ratio_lo && ratio < p.area_ratio_hi) score += 60.0;
            if (d < p.assoc_dist) score += (1.0 - d / p.assoc_dist) * 120.0;
            else score -= 100.0;                               // 跳太远 → 大概率是别的东西
        }

        if (score > best_score) {
            best_score = score;
            ball.found  = true;
            ball.center = ctr;
            ball.radius = radius;
            ball.area   = area;
        }
    }
    return ball;
}

// ----------------------------------------------------------------------------
// 第 4 步：用稀疏光流估计「相机全局运动」，返回 2x3 仿射矩阵
//   原理：在上一帧取一批角点(goodFeaturesToTrack)，用 LK 光流跟踪到当前帧，
//         再 estimateAffinePartial2D 拟合出整体的旋转+缩放+平移
// ----------------------------------------------------------------------------
static bool estimateCameraMotion(const cv::Mat& prev_gray, const cv::Mat& gray,
                                 const std::vector<cv::Point2f>& prev_pts,
                                 cv::Mat& M) {
    if (prev_gray.empty() || prev_pts.size() < 12) return false;

    std::vector<cv::Point2f> next_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, next_pts, status, err,
                             cv::Size(21, 21), 3);

    std::vector<cv::Point2f> src, dst;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) { src.push_back(prev_pts[i]); dst.push_back(next_pts[i]); }
    }
    if (src.size() < 12) return false;

    M = cv::estimateAffinePartial2D(src, dst, cv::noArray(),
                                    cv::RANSAC, 3.0, 2000, 0.99, 10);
    return !M.empty();
}

// ----------------------------------------------------------------------------
// 主流程
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ---- 命令行参数：输入视频、输出视频，可选 --no-compensate ----
    std::string in_path  = "homework.mp4";
    std::string out_path = "output_green_ball.mp4";
    Params p;
    int pos = 0;   // 位置参数计数：第 1 个是输入，第 2 个是输出
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-compensate") {
            p.compensate = false;
        } else if (a == "--in" && i + 1 < argc) {
            in_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (pos == 0) {
            in_path = a;  ++pos;   // 第一个位置参数 → 输入视频
        } else if (pos == 1) {
            out_path = a; ++pos;   // 第二个位置参数 → 输出视频
        }
    }

    cv::VideoCapture cap(in_path);
    if (!cap.isOpened()) {
        std::cerr << "[错误] 打不开视频：" << in_path << std::endl;
        return -1;
    }
    double fps    = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 30.0;
    int    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double total  = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "[信息] 视频 " << width << "x" << height << " @" << fps
              << "fps, 共 " << total << " 帧" << std::endl;

    // ---- 输出视频：优先 mp4v(mp4)，失败则退回 MJPG(avi) ----
    cv::VideoWriter writer;
    bool ok = writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps, cv::Size(width, height));
    if (!ok) {
        out_path = "output_green_ball.avi";
        writer.open(out_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps, cv::Size(width, height));
        std::cout << "[警告] mp4v 编码不可用，改用 MJPG/avi" << std::endl;
    }
    std::cout << "[信息] 输出文件：" << out_path << std::endl;

    // ---- 状态变量 ----
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_pts;
    cv::Point2f prev_center(0.f, 0.f);
    double      prev_area = 0.0;
    bool        has_prev  = false;
    double      vx = 0.0, vy = 0.0;          // 平滑后的速度 (px/s)
    double      max_speed_seen = 1.0;        // 用于箭头配色的速度上限
    std::deque<cv::Point2f> traj;            // 球心尾迹（稳定坐标系）
    // 把每帧「球相对场景的位移」累加起来，得到不随镜头晃动的轨迹，
    // 这样尾迹才能真正反映球的运动，而不是镜头的运动
    cv::Point2f stab_center(0.f, 0.f);
    bool        stab_init = false;

    int frame_id = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        double t = frame_id / fps;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ===== 1~3. 找球 =====
        Ball ball = detectBall(frame, p,
                               has_prev ? prev_center : cv::Point2f(0, 0),
                               has_prev ? prev_area : 0.0);

        // ===== 4. 相机全局运动补偿 =====
        cv::Mat M;
        bool has_M = false;
        if (p.compensate && has_prev && ball.found)
            has_M = estimateCameraMotion(prev_gray, gray, prev_pts, M);

        // ===== 5. 速度估计 =====
        bool moved_ok = false;
        if (ball.found && has_prev) {
            cv::Point2f ref = prev_center;          // 上一帧球心
            if (has_M) {                            // 投影到当前帧坐标系
                cv::Mat pv = (cv::Mat_<double>(3, 1) << ref.x, ref.y, 1.0);
                cv::Mat pj = M * pv;
                ref = cv::Point2f(static_cast<float>(pj.at<double>(0)),
                                  static_cast<float>(pj.at<double>(1)));
            }
            double dt = 1.0 / fps;
            double ivx = (ball.center.x - ref.x) / dt;   // 瞬时速度
            double ivy = (ball.center.y - ref.y) / dt;

            if (std::hypot(vx, vy) < 1e-6) { vx = ivx; vy = ivy; }   // 首帧直接赋值
            else {                                                    // EMA 平滑
                vx = p.alpha * ivx + (1.0 - p.alpha) * vx;
                vy = p.alpha * ivy + (1.0 - p.alpha) * vy;
            }

            // 累加「球相对场景」的位移 → 得到不随镜头晃动的轨迹
            if (!stab_init) { stab_center = ball.center; stab_init = true; }
            else            { stab_center += (ball.center - ref); }
            moved_ok = true;
        } else if (!ball.found) {
            vx *= p.decay;   // 目标丢失 → 速度逐渐衰减，避免箭头突然消失
            vy *= p.decay;
        }

        double speed = std::hypot(vx, vy);
        max_speed_seen = std::max(max_speed_seen, speed);
        cv::Point2f center = ball.found ? ball.center : prev_center;

        // ===== 尾迹 =====
        if (ball.found) {
            traj.push_back(stab_init ? stab_center : ball.center);
            while (static_cast<int>(traj.size()) > p.traj_len) traj.pop_front();
        }

        // ===== 6. 画标注 =====
        // 颜色映射：慢 → 绿，快 → 红。配色基准取 speed_ref（防止低速时永远全绿）
        cv::Scalar color = speedColor(speed / std::max(p.speed_ref, max_speed_seen));

        if (ball.found) {
            // 球的轮廓圆
            cv::circle(frame, ball.center, static_cast<int>(ball.radius * 1.05),
                       color, 2, cv::LINE_AA);
            cv::circle(frame, ball.center, 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
        }

        // 尾迹折线：越靠近当前位置越亮。因为尾迹记的是稳定坐标系里的位置，
        // 绘制时整体平移到球心，让尾迹末端始终贴合球 → 既反映真实运动又不脱离球
        if (ball.found && traj.size() > 1) {
            cv::Point2f off = ball.center - traj.back();
            for (size_t i = 1; i < traj.size(); ++i) {
                double r = static_cast<double>(i) / traj.size();
                cv::line(frame, traj[i - 1] + off, traj[i] + off,
                         cv::Scalar(60 + 195 * r, 80 + 175 * r, 0), 2, cv::LINE_AA);
            }
        }

        // 速度箭头：方向 = 速度方向，长度 ∝ 速度大小
        if (speed > p.min_speed_show && has_prev) {
            double len = std::min(p.max_arrow, std::max(p.min_arrow, speed * p.k_len));
            cv::Point2f tip = center + cv::Point2f(static_cast<float>(vx / speed * len),
                                                   static_cast<float>(vy / speed * len));
            cv::arrowedLine(frame, center, tip, color, 3, cv::LINE_AA, 0, 0.25);

            // 箭头长度数值提示
            char buf[64];
            std::snprintf(buf, sizeof(buf), "v=%.0f px/s", speed);
            cv::putText(frame, buf, cv::Point(static_cast<int>(center.x) - 55,
                                              static_cast<int>(center.y) - static_cast<int>(ball.radius) - 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
        }

        // 左上角信息栏
        char info[128];
        std::snprintf(info, sizeof(info), "frame %d  t=%.2fs  speed=%.0f px/s  %s",
                      frame_id, t, speed,
                      ball.found ? "TRACKING" : "LOST");
        cv::putText(frame, info, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(frame, info, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        writer.write(frame);

        // ===== 为下一帧准备 =====
        if (ball.found) { prev_center = ball.center; prev_area = ball.area; has_prev = true; }
        prev_gray = gray;
        prev_pts.clear();
        if (p.compensate) {
            std::vector<cv::Point2f> pts;
            cv::goodFeaturesToTrack(gray, pts, 200, 0.01, 20);
            prev_pts = pts;
        }
        ++frame_id;
    }

    cap.release();
    writer.release();
    std::cout << "[完成] 共处理 " << frame_id << " 帧，输出：" << out_path << std::endl;
    return 0;
}
