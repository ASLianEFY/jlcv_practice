/**
 * ============================================================================
 *  main_advanced.cpp —— 高级版主程序
 *  作业：识别视频中的绿球，并用箭头标注其运动方向和速度变化
 * ============================================================================
 *  与基础版（main.cpp）的区别：
 *    基础版：HSV + 圆度筛选 → 相邻帧差分 + EMA 平滑 → 画箭头
 *    高级版：
 *      1. 卡尔曼滤波（常速度模型 + 马氏距离门限）估计位置和速度；
 *      2. 椭圆拟合 + 轮廓矩融合，得到亚像素球心；
 *      3. 颜色检测失败时用 LK 光流在球上续跟，抗短暂遮挡；
 *      4. 背景光流 + 仿射 RANSAC 估计相机运动，区分"球在动"和"相机在动"；
 *      5. 计算加速度，自动判定加速 / 减速 / 匀速阶段；
 *      6. FreeType 渲染中文标注（无中文字体时自动退回英文）；
 *      7. 画面叠加"整段速度曲线"和"轨迹小地图"；
 *      8. 结束后自动生成速度曲线图、轨迹图和 Markdown 分析报告。
 *
 *  两趟处理：第一趟只做分析（检测/跟踪/统计），第二趟重新解码视频做渲染。
 *  好处是渲染时已经知道整段视频的速度分布，箭头长度可以自动标定，
 *  速度曲线也能一次画全。
 *
 *  编译： make pro
 *  运行： ./ball_tracker_pro homework.mp4 output/result_pro.mp4
 * ============================================================================
 */

#include "advanced/common.hpp"
#include "advanced/segmenter.hpp"
#include "advanced/tracker.hpp"
#include "advanced/visualizer.hpp"
#include "advanced/reporter.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>

using namespace adv;
using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    cout << "用法: " << prog << " [视频] [输出视频] [选项]\n"
         << "高级版选项:\n"
         << "  --no-chinese        不渲染中文（用英文标注）\n"
         << "  --font <路径>       指定中文字体（.ttf/.ttc）\n"
         << "  --show-true         额外画出\"真实运动\"（扣除相机运动）箭头\n"
         << "  --gain <k>          箭头长度系数（默认自动标定）\n"
         << "  --no-auto-gain      关闭箭头长度自动标定，使用 --gain\n"
         << "  --accel-thresh <a>  加速/减速判定阈值（像素/秒²，默认 250）\n"
         << "  --gate <chi2>       卡尔曼马氏距离门限（默认 9，越大越信任测量）\n"
         << "  --sigma-a <v>       过程噪声：加速度标准差（像素/秒²，默认 1500）\n"
         << "  --meas-sigma <v>    测量噪声标准差（像素，默认 3）\n"
         << "  --no-video          不写结果视频（只做分析和报告）\n"
         << "  --no-report         不生成 PNG 图和 Markdown 报告\n"
         << "  --stats             打印每帧数据\n"
         << "  --help              显示本帮助\n";
}

int main(int argc, char** argv) {
    Options o;
    vector<string> pos;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--no-chinese")            o.wantChinese = false;
        else if (a == "--font" && i + 1 < argc)      o.fontPath = argv[++i];
        else if (a == "--show-true")        o.showTrue = true;
        else if (a == "--gain" && i + 1 < argc)      { o.arrowGain = stod(argv[++i]); o.autoGain = false; }
        else if (a == "--no-auto-gain")     o.autoGain = false;
        else if (a == "--accel-thresh" && i + 1 < argc) o.accelThresh = stod(argv[++i]);
        else if (a == "--gate" && i + 1 < argc)       o.gateChi2 = stod(argv[++i]);
        else if (a == "--sigma-a" && i + 1 < argc)    o.sigmaAccel = stod(argv[++i]);
        else if (a == "--meas-sigma" && i + 1 < argc) o.measSigma = stod(argv[++i]);
        else if (a == "--no-video")         o.writeVideo = false;
        else if (a == "--no-report")        o.writeReport = false;
        else if (a == "--stats")            o.dumpStats = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else pos.push_back(a);
    }
    if (pos.size() > 0) o.videoPath = pos[0];
    if (pos.size() > 1) o.outVideo  = pos[1];
    if (o.outCsv.empty()) {
        fs::path p(o.outVideo);
        o.outCsv = (p.parent_path() / (p.stem().string() + ".csv")).string();
    }

    // ---------------- 文字渲染器 ----------------
    TextRenderer text;
    bool cjk = text.init(o.fontPath, o.wantChinese);
    if (cjk)                       cout << "中文标注已启用（字体: " + text.fontPath() + "）\n";
    else if (!o.wantChinese)       cout << "使用英文标注（已用 --no-chinese 关闭中文）\n";
    else                           cout << "使用英文标注（未找到可用中文字体，可用 --font 指定）\n";

    // ---------------- 打开视频 ----------------
    VideoCapture cap(o.videoPath);
    if (!cap.isOpened()) {
        cerr << "无法打开视频: " << o.videoPath << "\n";
        return 1;
    }
    double fps   = cap.get(CAP_PROP_FPS); if (fps < 1.0) fps = 30.0;
    int    W     = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int    H     = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    double totalF = cap.get(CAP_PROP_FRAME_COUNT);
    cout << "视频: " << o.videoPath << "  " << W << "x" << H << "  " << fps << " fps  "
         << totalF << " 帧\n";

    for (const auto& p : {o.outVideo, o.outCsv, o.outCurve, o.outMap, o.outReport}) {
        if (!o.writeReport && (p == o.outCurve || p == o.outMap || p == o.outReport)) continue;
        fs::path pp(p);
        if (pp.has_parent_path() && !pp.parent_path().empty()) fs::create_directories(pp.parent_path());
    }

    // =======================================================================
    //  第一趟：分析（检测 + 跟踪 + 相机运动 + 统计）
    // =======================================================================
    auto tStart = chrono::steady_clock::now();

    vector<FrameRec> recs;
    vector<cv::Point2f> appPath, truePath;
    vector<int> appCount;              // appCount[i] = 处理完第 i 帧后轨迹点数
    Stats st;

    KalmanTracker tracker;
    bool  trackerInited = false;
    int   lostCount = 0;
    cv::Mat prevGray, prevMaskDil;
    cv::Point2f lastCenter(0.f, 0.f);
    float lastRadius = 40.f;

    cv::Mat frame;
    int idx = 0;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat mask = greenMask(frame, o);
        cv::Mat maskDil;
        cv::dilate(mask, maskDil, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
        vector<Ball> cands = findBallCandidates(mask, o);

        FrameRec r;

        // --- 首次捕获 ---
        if (!trackerInited) {
            Ball b;
            if (pickBall(cands, cv::Point2f(), false, b)) {
                tracker.init(1.0 / fps, b.center, o);
                trackerInited = true;
                r.center = b.center;
                r.radius = b.radius;
                r.source = SRC_COLOR;
                st.colorFrames++;
                lastCenter = b.center;
                lastRadius = b.radius;
            } else {
                r.source = SRC_NONE;
                recs.push_back(r);
                appCount.push_back(int(appPath.size()));
                idx++;
                prevGray = gray.clone();
                prevMaskDil = maskDil;
                continue;
            }
        } else {
            // --- 卡尔曼预测 ---
            cv::Point2f pred = tracker.predict();

            // --- 测量 1：颜色 + 形状（经过马氏距离门限） ---
            Ball best;
            bool accepted = false;
            bool colorFound = pickBall(cands, pred, true, best);
            if (colorFound && tracker.gate(best.center)) {
                tracker.correct(best.center, 1.0);
                r.source = SRC_COLOR;
                r.radius = best.radius;
                accepted = true;
                lostCount = 0;
                st.colorFrames++;
            }

            // --- 测量 2：光流续跟（颜色检测失败 / 被门限拒绝时） ---
            if (!accepted) {
                cv::Point2f fc;
                int nGood = 0;
                if (FlowTracker::track(prevGray, gray, lastCenter, lastRadius, fc, nGood) &&
                    tracker.gate(fc)) {
                    tracker.correct(fc, o.flowRScale);
                    r.source = SRC_FLOW;
                    r.radius = lastRadius;
                    accepted = true;
                    lostCount = 0;
                    st.flowFrames++;
                }
            }

            // --- 都没有：让滤波器滑行；丢太久就用颜色结果重新捕获 ---
            if (!accepted) {
                lostCount++;
                if (colorFound && lostCount > o.maxLost) {
                    tracker.init(1.0 / fps, best.center, o);
                    tracker.predict();
                    tracker.correct(best.center, 1.0);
                    r.source = SRC_COLOR;
                    r.radius = best.radius;
                    lostCount = 0;
                    st.colorFrames++;
                } else {
                    tracker.coast();
                    r.source = SRC_PREDICT;
                    r.radius = lastRadius;
                    st.predictFrames++;
                }
            }

            r.center  = tracker.position();
            r.velApp  = tracker.velocity();
            r.speed   = cv::norm(r.velApp);
            if (r.radius > 0) lastRadius = r.radius;
        }

        // --- 相机运动估计（背景光流 + 仿射 RANSAC） ---
        // 注意：这里用的 lastCenter 必须还是"上一帧"的球心，
        // 因为仿射矩阵描述的是"上一帧 -> 当前帧"的背景位移。
        cv::Point2f bgVel(0.f, 0.f);
        if (idx > 0 && !prevGray.empty()) {
            cv::Mat A;
            double inlier = 0;
            if (BackgroundMotion::estimate(prevGray, gray, prevMaskDil, A, inlier)) {
                cv::Point2f expected = BackgroundMotion::apply(A, lastCenter);
                bgVel = (expected - lastCenter) * float(fps);   // 相机运动造成的视在速度
                r.bgOk = true;
            }
        }
        r.bgVel   = bgVel;                           // 相机运动造成的视在速度
        r.velTrue = r.velApp - bgVel;                // 真实速度 = 视在速度 - 相机运动
        lastCenter = r.center;                       // 更新为当前帧球心，供下一帧使用

        if (o.dumpStats) {
            printf("f%4d src=%d x=%6.1f y=%6.1f  v=(%7.1f,%7.1f) |v|=%7.1f  vTrue=%7.1f\n",
                   idx, r.source, r.center.x, r.center.y, r.velApp.x, r.velApp.y,
                   r.speed, cv::norm(r.velTrue));
        }

        recs.push_back(r);
        if (r.source != SRC_NONE) appPath.push_back(r.center);
        appCount.push_back(int(appPath.size()));

        prevGray    = gray.clone();
        prevMaskDil = maskDil;
        idx++;
    }
    cap.release();
    st.totalFrames = int(recs.size());

    // ---------------- 后处理：加速度 / 阶段 / 轨迹 / 统计 ----------------
    double acc = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        double sp     = recs[i].speed;
        double spPrev = i > 0 ? recs[i - 1].speed : sp;
        double aRaw   = (sp - spPrev) * fps;
        acc = (i == 0) ? aRaw : acc * (1.0 - o.accelSmooth) + aRaw * o.accelSmooth;
        recs[i].accel = acc;
        recs[i].phase = (acc > o.accelThresh) ? PHASE_ACCEL
                      : (acc < -o.accelThresh ? PHASE_DECEL : PHASE_STEADY);
    }

    // 真实轨迹：把"真实速度"积分起来（相当于把相机固定在桌面上的视角）
    cv::Point2f pTrue = recs.empty() ? cv::Point2f(0, 0) : recs[0].center;
    for (size_t i = 0; i < recs.size(); ++i) {
        const FrameRec& r = recs[i];
        if (i > 0) {
            st.pathApparent += cv::norm(r.center - recs[i - 1].center);
            double dt = 1.0 / fps;
            pTrue += r.velTrue * float(dt);
            st.pathTrue += cv::norm(r.velTrue) / fps;
        }
        truePath.push_back(pTrue);
    }

    double sumSpeedTrue = 0; int tracked = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        const FrameRec& r = recs[i];
        if (r.source == SRC_NONE) continue;
        tracked++;
        st.sumSpeed += r.speed;
        st.maxSpeed = std::max(st.maxSpeed, r.speed);
        st.minSpeed = std::min(st.minSpeed, r.speed);
        double sT = cv::norm(r.velTrue);
        sumSpeedTrue += sT;
        st.maxSpeedTrue = std::max(st.maxSpeedTrue, sT);
        st.maxAccel = std::max(st.maxAccel, r.accel);
        st.minAccel = std::min(st.minAccel, r.accel);
        if (r.phase == PHASE_ACCEL) st.accelFrames++;
        else if (r.phase == PHASE_DECEL) st.decelFrames++;
        else st.steadyFrames++;
        if (r.speed > o.staticSpeed) {
            st.sumDirX += r.velApp.x / r.speed;
            st.sumDirY += r.velApp.y / r.speed;
        }
    }
    double meanSpeed     = tracked ? st.sumSpeed / tracked : 0.0;
    double meanSpeedTrue = tracked ? sumSpeedTrue / tracked : 0.0;

    // ---------------- 箭头长度自动标定 ----------------
    double arrowGain = o.arrowGain;
    if (o.autoGain && !recs.empty()) {
        vector<double> sp;
        for (const auto& r : recs) if (r.source != SRC_NONE) sp.push_back(r.speed);
        if (!sp.empty()) {
            size_t k = size_t(sp.size() * 0.9);
            nth_element(sp.begin(), sp.begin() + k, sp.end());
            double p90 = max(1.0, sp[k]);
            arrowGain = min(1.5, max(0.05, o.arrowTargetLen / p90));
            o.arrowGain = arrowGain;
        }
    }

    auto tEnd = chrono::steady_clock::now();
    st.processMs = chrono::duration<double, milli>(tEnd - tStart).count();

    cout << "分析完成：" << st.totalFrames << " 帧  检出 " << st.colorFrames
         << " 帧、光流续跟 " << st.flowFrames << " 帧、仅预测 " << st.predictFrames << " 帧\n"
         << "  视在速度 平均 " << meanSpeed << " / 最大 " << st.maxSpeed << " 像素/秒\n"
         << "  真实速度 平均 " << meanSpeedTrue << " / 最大 " << st.maxSpeedTrue << " 像素/秒\n"
         << "  累计路程 视在 " << st.pathApparent << " 像素 / 真实 " << st.pathTrue << " 像素\n"
         << "  箭头长度系数 " << arrowGain << "（"
         << (o.autoGain ? "自动标定" : "手动指定") << "）\n"
         << "  分析耗时 " << st.processMs / 1000.0 << " 秒\n";

    // =======================================================================
    //  输出：CSV / 曲线图 / 轨迹图 / 报告
    // =======================================================================
    writeCsv(o.outCsv, recs, fps);
    if (o.writeReport) {
        drawSpeedCurvePng(o.outCurve, text, o, recs, fps, st);
        drawTrajectoryMapPng(o.outMap, text, o, appPath, truePath, recs, st);
        cout << "  速度曲线: " << o.outCurve << "\n";
        cout << "  轨迹图: " << o.outMap << "\n";
    }

    // =======================================================================
    //  第二趟：渲染（重新解码视频，用分析结果绘制）
    // =======================================================================
    if (o.writeVideo) {
        VideoCapture cap2(o.videoPath);
        VideoWriter writer(o.outVideo, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));
        if (!cap2.isOpened() || !writer.isOpened()) {
            cerr << "渲染阶段打开视频/写出视频失败: " << o.outVideo << "\n";
        } else {
            vector<double> speeds;
            vector<int> phases;
            for (const auto& r : recs) { speeds.push_back(r.speed); phases.push_back(r.phase); }

            int iMax = 0;
            for (size_t i = 1; i < recs.size(); ++i) if (recs[i].speed > recs[iMax].speed) iMax = int(i);

            deque<cv::Point2f> trail;
            cv::Mat f2;
            int i = 0;
            while (cap2.read(f2)) {
                if (i >= int(recs.size())) break;
                const FrameRec& r = recs[i];

                if (r.source != SRC_NONE) {
                    trail.push_back(r.center);
                    while (int(trail.size()) > o.trailLen) trail.pop_front();
                    drawTrail(f2, trail);

                    // 绿球轮廓
                    if (r.source == SRC_PREDICT)
                        cv::circle(f2, r.center, std::max(12, int(r.radius)), cv::Scalar(0, 200, 255), 3, cv::LINE_AA);
                    else if (r.source == SRC_FLOW)
                        cv::circle(f2, r.center, std::max(6, int(r.radius)), cv::Scalar(255, 200, 0), 3, cv::LINE_AA);
                    else
                        cv::circle(f2, r.center, std::max(6, int(r.radius)), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
                    cv::circle(f2, r.center, 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);

                    // 主箭头：视在运动（作业要求标注的就是画面里的运动）
                    cv::Scalar col = speedColor(r.speed, std::max(1.0, meanSpeed * 2.0));
                    drawVelocityArrow(f2, text, r.center, r.velApp, o, col, 6, true);

                    // 可选：真实运动箭头（细一些，白色描边）
                    if (o.showTrue) {
                        double sT = cv::norm(r.velTrue);
                        if (sT > o.staticSpeed)
                            drawVelocityArrow(f2, text, r.center, r.velTrue, o,
                                              cv::Scalar(255, 190, 80), 3, false);
                    }

                    drawPanel(f2, text, o, r, i, int(recs.size()), fps, meanSpeed);
                    drawSpeedGraph(f2, text, o, speeds, phases, i, meanSpeed);
                    drawMiniMap(f2, text, o, appPath, truePath, appCount[i] - 1,
                                st.pathApparent, st.pathTrue);
                } else {
                    text.putShadow(f2, text.pick(u8"尚未捕获到绿球", "GREEN BALL NOT FOUND"),
                                   cv::Point(30, 60), 28, cv::Scalar(0, 0, 255));
                }

                if (i == iMax) cv::imwrite("output/preview_pro.jpg", f2);   // 最快那一刻留一张预览图
                writer.write(f2);
                i++;
            }
            cap2.release();
            writer.release();
            cout << "  结果视频: " << o.outVideo << "\n";
        }
    }

    //  报告放在最后写：这样可以把渲染阶段留下的预览图也一起引用进去
    if (o.writeReport) {
        writeReportMd(o.outReport, o, st, fps, W, H, meanSpeed, meanSpeedTrue, arrowGain,
                      fs::path(o.videoPath).filename().string());
        cout << "  报告: " << o.outReport << "\n";
    }

    cout << "完成。\n";
    return 0;
}
