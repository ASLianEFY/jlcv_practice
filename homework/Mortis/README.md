# Week1 绿球识别与运动追踪

**Mortis**

从视频中检测绿色小球，用圆形框标出位置，用固定红色箭头表示运动方向，箭头长度表示速度大小。画面左上角显示帧号、速度（px/s）及球的像素半径。

## 构建与运行

环境：C++17、CMake >= 3.10、OpenCV 4。

在仓库根目录执行：

```bash
cmake -S homework/Mortis -B homework/Mortis/build
cmake --build homework/Mortis/build -j
./homework/Mortis/build/ball_tracker
```

程序仅针对本次作业，固定读取仓库根目录的 `homework.mp4`，输出到 `homework/Mortis/output.mp4`，需从仓库根目录运行。

## 实现方案

1. **绿色分割**：BGR 转 HSV，提取绿色区域，使用开运算去噪、闭运算填补空洞。
2. **形状筛选**：按轮廓面积过滤噪点，再拟合椭圆，以短轴/长轴区分绿球与细长的绿色圆柱。
3. **跨帧匹配**：综合圆度、典型半径及与卡尔曼预测位置的距离选取目标，并拒绝距离或半径突变过大的候选。
4. **位置与速度估计**：卡尔曼滤波先预测，再用本帧检测位置校正，估计平滑的位置和速度。
5. **绘制与输出**：绘制检测圆、圆心、固定红色速度箭头和数值文字，逐帧写入输出视频。

不绘制历史轨迹、类名/评分标签，也不显示加减速提示。漏检时不绘制预测目标，而是重置跟踪状态，下一次检测到目标时重新初始化；首次检测的速度为零。

## 关键参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `kLowerGreen` / `kUpperGreen` | (35, 45, 40) / (95, 255, 255) | HSV 颜色范围 |
| `kMinArea` | 2000 | 最小轮廓面积 |
| `kMinEllipseRoundness` | 0.80 | 椭圆短轴/长轴下限 |
| `kTypicalBallRadius` / `kRadiusTolerance` | 60 / 40 | 形状评分的半径参考值 |
| `kDistancePenalty` / `kRadiusPenalty` | 0.02 / 0.03 | 匹配时的距离、半径惩罚 |
| `kMaxRadiusChangeRatio` | 0.50 | 相邻检测允许的半径变化比例 |
| `kArrowScale` | 4.5 | 速度到箭头长度的比例 |
| `kArrowMinLen` / `kArrowMaxLen` | 25 / 140 | 箭头长度范围 |
| `kMinArrowSpeed` | 1 px/帧 | 低于此速度时隐藏箭头 |
| `kKalmanAccelerationNoise` | 1.0 | 加速度过程噪声 |
| `kKalmanMeasurementNoiseVariance` | 16.0 | 中心点测量噪声方差 |

## 局限

仅跟踪一个绿球。颜色、面积和尺寸参数按当前视频设置，换视频需要重新调整。部分可见的球仍可能因轮廓变形或面积过小被过滤；本程序不通过外推填补漏检帧。速度和半径均为图像像素单位，不代表现实世界中的速度和尺寸。
