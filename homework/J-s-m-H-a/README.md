# week1 — 绿球检测与运动箭头标注

识别 `homework.mp4` 中的绿色小球，用 Kalman 滤波估计其图像运动速度，并用箭头标注运动方向和速度变化。

## 重要说明

经逐帧核实，**绿球相对桌面是静止的**；画面中看到的位移主要来自手持镜头运动。因此本程序的箭头表示的是**图像视运动**（球心在画面坐标系中的运动），而不是球在世界坐标中的真实运动。详细证据见 [`CONTEXT.md`](CONTEXT.md)。

## 环境

- Python 3.12
- OpenCV（`opencv-python-headless`）、numpy

```bash
cd <仓库根目录>
python3 -m venv .venv
source .venv/bin/activate

pip install -r homework/J-s-m-H-a/requirements.txt
# 如需运行测试：
pip install -r homework/J-s-m-H-a/requirements-dev.txt
```

## 运行

```bash
python homework/J-s-m-H-a/main.py \
  --video homework.mp4 \
  --out-dir homework/J-s-m-H-a/output
```

参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--video` | 仓库根 `homework.mp4` | 输入视频路径（默认基于脚本位置解析） |
| `--out-dir` | `homework/J-s-m-H-a/output` | 输出目录 |
| `--no-frames` | 关闭 | 不导出抽样帧与审核拼图 |
| `--trail` | 关闭 | 绘制最近 15 帧球心轨迹 |
| `--debug` | 关闭 | 在 `output/debug/` 输出掩码、候选、ROI、KF 状态 |

程序无 GUI，运行结束后在终端打印检测率、状态统计、最长连续丢失帧数和速度统计。

## 输出

| 文件 | 说明 |
|---|---|
| `output/annotated.mp4` | 标注视频：圆框、标签、箭头、速度数值与状态 |
| `output/speed.csv` | 逐帧数据：`frame,time_s,cx,cy,r,vx,vy,speed_px_s,status` |
| `output/frames/*.png` | 6 张抽样标注帧（约 0/3/6/9/12/14.5 s） |
| `output/contact_sheet.png` | 16 帧审核拼图（4×4） |
| `output/debug/` | `--debug` 时的中间结果 |

生成物不提交进 git。若 `mp4v` 编码器不可用，程序会回退为 `output/annotated.avi`（XVID）。

## 标注规则

- 使用 Kalman 滤波（状态 `[cx, cy, vx, vy]`，匀速模型）估计球心与速度，不使用 EMA。
- 箭头从球心出发，方向 = 平滑后的图像速度方向；长度 `L = clamp(0.25 × speed, 20, 220)` px。
- 速度低于 60 px/s 时不画箭头；速度收敛前（<3 次有效更新或 `σ_v ≥ 300 px/s`）不画箭头。
- 绘制顺序为圆框 → 标签 → 箭头 → 文字，箭头位于球的上层。
- 正常帧：黄色箭头 + 左上角 `v = 123 px/s`。
- 预测帧：灰色虚线箭头 + `v ≈ 123 px/s (pred)`。
- 丢失帧：不画箭头 + `v = --`。
- 正常跟踪时只在预测点周围 ROI 内检测；丢失时全图重捕获，连续 2 帧确认后才恢复。

## 测试

```bash
pytest -v homework/J-s-m-H-a/tests
```

- 共 **44 项测试**：U1–U6 单元测试、I7 合成短视频、I8 真实视频、I9 12 帧人工真值。
- I8 断言检测率 ≥95%、最长连续丢失 ≤5 帧；I9 断言球心误差 ≤10 px。
- 真实视频缺失时 I8/I9 自动跳过。
- 测试代码提交进 git，测试输出与生成物不提交。

## 代码结构

单文件 `main.py`，用分隔注释划分为：配置、视频 I/O、绿色分割、候选筛选、Kalman 跟踪、速度门控、标注、导出、CLI。每个部分配有 docstring，说明输入/输出与关键参数。

## 验证结果（2026-10-19）

| 指标 | 结果 |
|---|---|
| 全片帧数 | 473 |
| 检测率 | **100.0%**（473 detected，0 predicted，0 lost） |
| 最长连续丢失 | **0 帧** |
| 图像速度 | 中位 **375.4 px/s**，p90 **637.7 px/s** |
| 输出视频 | 1280×720、30 fps、473 帧 |
| 真值校验 | 12 帧人工十字线核对全部通过（回归容差 10 px） |
| 测试 | **44 passed** |

镜头在约 401–406 帧出现反向，跟踪器通过"欧氏硬门控 + Mahalanobis 排序/自适应 Q"保持连续检测，未再出现预测圈漂移。标注视频与审计拼图已人工检查：球框居中、箭头在球上层、方向与画面位移一致、长度随速度变化。
