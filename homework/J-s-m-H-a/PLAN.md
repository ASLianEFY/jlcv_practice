# PLAN — week1 绿球检测与运动箭头标注

> 对应任务见 [`CONTEXT.md`](CONTEXT.md)。本计划遵循"小步实现、每步测试、分阶段审核"的原则。
> 最后更新：2026-10-19

## 1. 目标

对 `homework.mp4`（1280×720、30 fps、473 帧）逐帧：

1. 稳定检测画面中的绿色球体（排除同色圆柱）；
2. 用 Kalman 滤波估计球心的图像速度（px/s）；
3. 在每帧画出方向箭头，长度与速度成正比；
4. 在画面左上角显示当前速度数值与状态；
5. 输出标注视频、逐帧速度 CSV、6 张抽样帧和 1 张审核拼图。

## 2. 范围外（明确不做）

- 相机运动补偿 / 世界坐标速度估计（文档说明球实际静止，但不实现补偿）。
- 速度曲线图（仅 CSV + 数值文本）。
- 深度学习检测、相机标定、球体三维定位。
- GUI 实时窗口（无头运行）。
- 将生成物（视频、CSV、图片、调试文件）提交进 git。

## 3. 总体方案

```text
读取视频
  → 绿色区域分割（HSV + 形态学）
  → 绿球候选筛选（ROI + 轮廓形状 + 下方绿色占比 + 半径一致性）
  → Kalman 跟踪（预测、Mahalanobis/欧氏门限、hits/misses、丢失外推、重捕获）
  → 速度门控（早期抑制 + clamp）
  → 标注（圆框、箭头、数值文本、可选轨迹）
  → 导出（MP4 / CSV / 抽样帧 / 审核拼图 / 调试）
```

## 4. 关键算法与参数

所有可调参数集中在 `main.py` 顶部的 `Config`（`dataclass`）中；修改参数必须单独提交。

### 4.1 绿色分割

- BGR → HSV；绿色范围 `H ∈ [40, 90]`、`S ≥ 80`、`V ≥ 40`（OpenCV 的 H 为 0–179）。
- 形态学：3×3 开运算去噪；**不使用大核闭运算**，避免球贴住圆柱时把两者粘连（必要时 3×3 闭运算）。
- 支持 ROI：正常跟踪时只在预测点周围 `±2.5 × r_smooth` 的区域内分割；丢失/重捕获时全图。

### 4.2 绿球候选筛选

对每个绿色轮廓计算：

| 特征 | 规则 |
|---|---|
| 面积 | 首次检测：`800 ≤ area ≤ 40000` px²；轨迹建立后：`[π(0.7r)², π(1.4r)²]` |
| 圆度 A | `4πA / P² ≥ 0.6` |
| 圆度 B | `area / (π × r_minEnclosing²) ≥ 0.6`（周长法对掩码锯齿敏感，两指标同时满足） |
| 长宽比 | `0.7 ≤ w/h ≤ 1.4` |
| 下方绿色占比 | 圆外下方区域绿色占比 `< 0.6`（排除绿色圆柱的柱身） |
| 半径一致性 | 轨迹建立后 `|r - r_smooth| ≤ 0.3 × r_smooth` |

- **不使用"上方绿色占比"**：镜头压低时球位于绿圆柱前方，球上方可能出现大片绿色（实测占比 0.8–1.0），该条件会误杀真球。
- 掩码粘连时用 `HoughCircles` 兜底：在**半分辨率**绿色掩码上检测，半径范围绑定 `r_smooth`，再按 2× 还原；下方绿色占比与半径一致性同样参与筛选。
- 多候选时先按形状得分过滤，再选择与 Kalman 预测最接近者。

### 4.3 Kalman 跟踪

状态与观测：

- 状态 `x = [cx, cy, vx, vy]^T`，匀速模型。
- 观测 `z = [cx, cy]^T`（检测到的球心）。
- `dt = 1 / fps`；`F` 为匀速转移矩阵，`H = [I₂ 0₂]`。
- `R = diag(9, 9)`（约 3 px 观测噪声）；过程噪声由加速度噪声 `σ_a` 生成，初始 `σ_a = 500 px/s²`，作为 `Config` 参数后续调优。
- 初始化：首帧 `x = [cx, cy, 0, 0]`，`P = diag([9, 9, 1e6, 1e6])`；速度由后续观测学习。

关联门限（硬门控）：

1. 欧氏距离 `‖z - z_pred‖ ≤ max(2.5 × r_smooth, 1.5 × |v| × dt + 10)`（与 ROI 同尺度）；
2. 半径偏差 ≤ 30%；
3. 形状筛选通过。

在通过硬门控的候选中，用 Mahalanobis 距离 `d² = innovation^T S^{-1} innovation` 排序并取最小者；当 `d² > 9.21` 时标记为 recovery 帧，下一次预测把 Q 放大 4 倍。

> 不使用硬 Mahalanobis 门限：镜头反向时匀速先验会指向错误方向，硬门限会误杀真球。2026-10-19 实测约 401–406 帧出现该问题，已修复并加回归测试。

轨迹管理：

- 新轨迹需要**连续 2 帧**有效检测才确认（hits ≥ 2）；未确认期间不画箭头。
- 正常跟踪时只在 ROI 内检测；确认后进入 ROI 模式。
- 丢失 ≤ 5 帧：只 `predict()` 不 `update()`，`status=predicted`；继续画灰色虚线箭头。
- 丢失 > 5 帧：`status=lost`，停止画箭头，回到全图搜索；重捕获同样需要连续 2 帧确认，并用新观测**重置 KF**（速度重新学习）。
- 自适应 Q（P2）：当 `d² > 9.21` 时，下一次预测将 `Q` 放大 4 倍，抑制快速晃镜时的滞后；倍数作为 `Config` 参数。

### 4.4 速度门控

- 速度取 KF 后验 `hypot(vx, vy)`；**不使用 EMA**（Kalman 已提供平滑）。
- 早期抑制：首帧 `v=0` 不画箭头；在累计 ≥3 次有效 update 或速度标准差 `σ_v < 300 px/s` 之前不画箭头，避免第 2 帧出现方向错误的长箭头。
- 速度低于 **60 px/s** 不画箭头。
- 显示与预测 clamp 到 **2000 px/s**；CSV 保留未 clamp 的 KF 原始值。

### 4.5 标注

- 球体：黄色圆框 `(0, 255, 255)` + 黑色描边，标签 `ball`。
- 箭头（方案 a）：**从球心出发**，方向 = 速度单位向量；`L = clamp(0.25 × speed, 20, 220)` px；黄色 + 黑色描边，线宽 3，尖端比例 0.25。
- 绘制顺序：**圆框 → 标签 → 箭头 → 文字**，保证箭头在球的上层；箭头从球心出发，利用放大后的长度伸出球外。
- 预测帧：灰色虚线箭头，文本 `v ≈ 123 px/s (pred)`。
- 丢失帧：不画箭头，文本 `v = --`。
- 数值文本：左上角 `v = 123 px/s`；文字带半透明深色背景以保证可读性。
- 可选轨迹（P2）：`--trail` 开启，绘制最近 15 帧球心连线，默认关闭。

### 4.6 输出

- 视频：`output/annotated.mp4`，编码 `mp4v`；写入前检查 `writer.isOpened()`，失败时明确报错；不支持时回退 `XVID` + `.avi`。
- CSV：`output/speed.csv`，字段：
  `frame,time_s,cx,cy,r,vx,vy,speed_px_s,status`
  - `status` 取值 `detected` / `predicted` / `lost`；
  - `lost` 帧的 `cx,cy,r,vx,vy,speed_px_s` 写 `nan`；
  - `predicted` 帧写 KF 预测值。
- 抽样帧：`output/frames/frame_0000.png`、`0090`、`0180`、`0270`、`0360`、`0435`（约 0/3/6/9/12/14.5 s）。
- 审核拼图：`output/contact_sheet.png`，均匀取 16 帧，4×4 网格。
- 调试（`--debug`）：`output/debug/` 下输出掩码、候选框、ROI、KF 预测/后验，供调参；git 忽略。
- 默认路径基于脚本位置：视频默认 `Path(__file__).resolve().parents[2] / "homework.mp4"`，输出默认 `Path(__file__).resolve().parent / "output"`。

## 5. 文件结构

```text
homework/J-s-m-H-a/
├── CONTEXT.md
├── PLAN.md
├── README.md
├── requirements.txt          # opencv-python-headless, numpy（固定版本）
├── requirements-dev.txt      # -r requirements.txt, pytest
├── main.py                   # 单文件，按分隔注释分 9 个部分
├── tests/
│   ├── fixtures/ball_centers.csv   # 12 帧人工核对真值
│   ├── test_segmentation.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_speed.py
│   ├── test_annotation.py
│   ├── test_export.py
│   ├── test_integration_synthetic.py
│   └── test_integration_real.py
└── output/                   # 生成物，git 忽略
```

> 代码结构：单文件 `main.py`（约 250–320 行），用醒目分隔注释划分为：配置、视频 I/O、绿色分割、候选筛选、Kalman 跟踪、速度门控、标注、导出、CLI；每个部分有 docstring。

## 6. 提交计划（共 11 次提交：10 个计划步骤 + 1 次验证修复）

| # | 提交信息 | 内容 | 对应测试 |
|---|---|---|---|
| C1 | `docs(week1): 添加 PLAN/CONTEXT/README` | 三份文档 | 人工审核 |
| C2 | `chore(week1): 添加 Python 依赖、gitignore 与目录骨架` | `.gitignore`、`requirements*.txt`、`main.py` 骨架、测试目录 | 依赖安装、`--help` |
| C3 | `feat(week1): 实现绿色区域分割` | HSV + 3×3 开运算 + ROI 支持 | U1 |
| C4 | `feat(week1): 实现绿球候选筛选与圆柱区分` | 双圆度指标、下方绿色占比、半径自适应、Hough 半分辨率兜底 | U2 |
| C5 | `feat(week1): 实现 Kalman 跟踪与轨迹管理` | KF、门限、hits/misses、ROI、丢失/重捕获、自适应 Q | U3 |
| C6 | `feat(week1): 实现速度门控与早期抑制` | KF 速度、σ_v 抑制、clamp | U4 |
| C7 | `feat(week1): 实现箭头与数值标注` | 方案 a 箭头、虚线预测箭头、文本、可选轨迹 | U5 |
| C8 | `feat(week1): 实现视频/CSV/抽样帧导出与 CLI` | 写入检查、CSV、抽样帧、拼图、调试、路径默认 | U6、I7 |
| F1 | `fix(week1): 放宽关联门限以处理镜头反向` | 欧氏硬门控 + Mahalanobis 排序；镜头反向回归 | U3 |
| C9 | `test(week1): 添加真实视频集成测试与真值校验` | I8、I9、状态统计 | I8、I9 |
| C10 | `docs(week1): 补充验证结果与运行示例` | README 结果与示例 | 全量 pytest + 人工审核 |

约束：每个提交必须"改了什么 + 由哪个测试证明"且测试通过；参数/行为修改单独提交；提交信息用中文 + `类型(范围): 描述`。

## 7. 里程碑与审核检查点

| 里程碑 | 范围 | 审核材料 |
|---|---|---|
| M1 | C1 文档 | 三份文档全文 |
| M2 | C2–C5 检测 + 跟踪 | 改动摘要、`git diff --stat`、U1–U3 的 `pytest -v` |
| M3 | C6–C10 速度 + 标注 + 导出 + 测试 | 全量 `pytest -v`、检测率/状态统计、标注视频、抽样帧、拼图、CSV 样例 |

每个里程碑结束后 push 到 `origin`，等待确认后再进入下一阶段。**全部完成并经确认后才创建 PR。**

## 8. 测试计划

### 单元测试（合成数据）

| 编号 | 文件 | 内容 |
|---|---|---|
| U1 | `test_segmentation.py` | 合成绿球/绿圆柱/红白物体/木纹背景；断言掩码、3×3 开运算、ROI 限制 |
| U2 | `test_detection.py` | 球 vs 竖长圆柱；球贴住圆柱；**球位于圆柱前方（上方绿色占比高）**；双圆度指标；半径自适应；Hough 兜底 |
| U3 | `test_tracking.py` | 匀速/反向/加速轨迹；连续丢失 3/6 帧；重捕获；hits/misses；欧氏硬门控 + Mahalanobis 排序；**镜头反向回归**；自适应 Q |
| U4 | `test_speed.py` | 首帧 v=0；σ_v 抑制；速度低于 60 不画；2000 px/s clamp |
| U5 | `test_annotation.py` | 箭头长度映射与上下限；球心锚定；绘制顺序（箭头在球上层）；虚线箭头；文本位置/状态 |
| U6 | `test_export.py` | CSV 表头/行数/`status`/`r`/`nan`；抽样帧数量与命名；拼图尺寸；writer 打开失败报错 |

### 集成测试

| 编号 | 文件 | 内容 |
|---|---|---|
| I7 | `test_integration_synthetic.py` | 生成 60 帧 320×240 合成视频（绿球已知轨迹 + 绿圆柱干扰），跑完整 CLI；断言检测率 100%、CSV 速度误差 ±10%、`status` 转换正确、输出可读 |
| I8 | `test_integration_real.py` | 真实 `homework.mp4`：473 帧不崩溃、检测率 ≥95%、输出 1280×720/30 fps、CSV 473 行、最长连续丢失 ≤5 帧 |
| I9 | `test_integration_real.py` | 12 帧人工核对真值 `fixtures/ball_centers.csv`：球心误差 ≤10 px |

- 真实视频缺失时 I8/I9 自动 `skip`。
- 合成数据固定随机种子；测试代码提交，`pytest -v` 输出与生成物不提交。
- 真值帧的选取：均匀覆盖 0–15 s，包含球位于圆柱前方、近距离大球等困难场景；由程序给出初值后人工核对。

## 9. 验收标准

1. `pytest -v` 全部通过。
2. I8 检测率 ≥95%，最长连续丢失 ≤5 帧。
3. I9 真值帧球心误差 ≤10 px。
4. 输出视频 1280×720、约 30 fps、473 帧；CSV 473 行。
5. 人工抽查抽样帧与拼图：圆心在球上、箭头方向与画面位移一致、箭头长度随速度变化且箭头在球的上层。
6. `README.md` / `CONTEXT.md` 明确说明"球在世界坐标中静止，箭头表示镜头视运动"。
7. 生成物未进入 git；提交颗粒度符合第 6 节。
8. 你审核测试代码、测试输出与标注结果后明确确认"通过"。

### 实际结果（2026-10-19）

- 473 帧全部 `detected`（检测率 100%），`predicted=0`、`lost=0`，最长连续丢失 0。
- 速度中位 375.4 px/s，p90 637.7 px/s。
- 输出视频 1280×720 / 30 fps / 473 帧；CSV 474 行。
- 12 帧人工真值全部通过（容差 10 px）。
- 全量测试 **44 passed**。
- 约 401–406 帧的镜头反向已由 F1 修复，未再出现预测圈漂移。

## 10. Git 工作流

- 远端保持 **HTTPS**（不改 SSH）：`origin` = 你的 fork，`upstream` = 主仓库。
- **PR 完成前不合并 upstream**，保持工作区只有自己的代码；PR 合并后再 `git fetch upstream && git merge upstream/main`，之后接受工作区出现其他同学的目录。
- PR 前的冲突检查：`git fetch upstream` 后用 `git merge-tree` 或 GitHub 的 mergeable 状态判断；不把 upstream 合进当前工作区。

```bash
git checkout main                      # main = adb7759
git checkout -b week1-green-ball
# 按第 6 节逐次提交
git push origin week1-green-ball       # HTTPS；失败则重试，不改 SSH
# 全部完成并经确认后才创建 PR
gh pr create --repo ASLianEFY/jlcv_practice --base main \
  --head J-s-m-H-a:week1-green-ball \
  --title "[week1] J-s-m-H-a - 绿球检测与运动箭头标注"
```

## 11. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 绿色圆柱顶面近似圆形 | 误检为球 | 下方绿色占比 + 双圆度 + 半径一致性 + 时间连续性；U2 覆盖 |
| 球位于绿圆柱前方时上方出现绿色 | 误杀真球 | 不使用上方绿色占比；U2 增加该场景 |
| 快速晃镜导致 KF 预测滞后 | 门限拒绝真球 | 欧氏硬门控放宽到 2.5r + 自适应 Q；丢失外推 |
| 镜头反向使匀速预测指向错误方向 | 真球被门限拒绝、预测圈漂移 | 欧氏硬门控 + Mahalanobis 仅排序 + 自适应 Q；U3 反向回归（2026-10-19 已修复） |
| ROI 过小导致漏检 | 检测率下降 | ROI 半径 2.5r；丢失时全图搜索 |
| Kalman Q/R 不合适 | 速度抖动或滞后 | 参数集中配置；合成轨迹测试；调参单独提交 |
| hits 确认延迟 1 帧 | 首帧/重捕获后箭头晚出现 | 可接受；文档说明 |
| 箭头从球心出发覆盖球面 | 球被部分遮挡 | 箭头最后画（上层）；系数 0.25 使箭头伸出球外；保留圆框 |
| 近距离球半径超过预期 | 面积筛选漏检 | 全局上限 40000 px² + 半径自适应 |
| `mp4v` 编码器不可用 | 输出失败 | `isOpened()` 检查；回退 `XVID` + `.avi` |
| 真值标注主观误差 | I9 误判 | 12 帧人工核对；容差 10 px |
| HTTPS push 偶发 TLS 错误 | 推送失败 | 重试；不改 SSH；必要时请你确认后再处理 |

## 12. 已确认事项

- 代码结构：单文件 `main.py`，分隔注释 + docstring。
- 检测率：全片 473 帧统计 ≥95%，硬性验收线。
- 跟踪：Kalman 完全取代 EMA（K1 a）。
- 箭头：方案 a——球心锚定，`L = clamp(0.25×v, 20, 220)`，箭头最后画、位于球上层。
- CSV：`status` 列 + `r` 列。
- 优化范围：P0 全部、P1 全部；P2 仅自适应 Q 与可选 `--trail`。
- 关联门限：欧氏硬门控（2.5r）+ Mahalanobis 排序/recovery；镜头反向回归已加入 U3。
- 验证结果：473/473 detected、0 lost、44 项测试通过、12 帧真值通过。
- 远端保持 HTTPS，不改 SSH；PR 完成后再合并 upstream。
