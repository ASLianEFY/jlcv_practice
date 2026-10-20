# 作业名称：绿色小球识别与运动追踪
# 功能说明：HSV 颜色分割 + 圆度/面积筛选定位绿色小球，卡尔曼滤波估计其运动状态，
#           在原视频画面上叠加运动方向箭头和速度标注（箭头越长表示速度越快）。
# 使用的 AI 工具：Claude Code

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def find_default_video() -> Path:
    """优先用脚本同目录的视频，否则向上查找仓库根目录的 homework.mp4。"""
    for folder in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidate = folder / "homework.mp4"
        if candidate.is_file():
            return candidate
    return SCRIPT_DIR / "homework.mp4"


DEFAULT_VIDEO = find_default_video()

# ----------------------------- 检测参数 -----------------------------
GREEN_LOWER = np.array((35, 45, 45), dtype=np.uint8)
GREEN_UPPER = np.array((90, 255, 255), dtype=np.uint8)

MIN_AREA_RATIO = 0.0025  # 轮廓面积占整幅画面的比例下限
MAX_AREA_RATIO = 0.0500
MIN_CIRCULARITY = 0.62  # 4*pi*area / perimeter^2，越接近 1 越圆
MIN_FILL_RATIO = 0.65  # 轮廓面积 / 最小外接圆面积
MIN_ASPECT = 0.60  # 外接矩形宽高比，用于排除细长的绿色物体
MAX_ASPECT = 1.67
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ----------------------------- 跟踪参数 -----------------------------
MAX_LOST_FRAMES = 10  # 连续丢失超过该帧数则重新搜索目标
GATE_BASE = 80.0  # 候选点与卡尔曼预测点的基础距离门限（像素）
SHAPE_PENALTY = 40.0  # 形状得分对匹配代价的影响权重

# 卡尔曼滤波：状态量 [x, y, vx, vy]，观测量 [x, y]
Q_ACCEL = 4.0  # 过程噪声（像素/帧^2）
R_MEASURE = 4.0  # 观测噪声（像素）
P0_POS = 100.0
P0_VEL = 400.0

# ----------------------------- 绘制参数 -----------------------------
ARROW_SECONDS = 0.15  # 箭头长度 = 速度 × 该时间，直观表示“这段时间走的距离”
ARROW_MIN_LENGTH = 12.0
ARROW_MAX_RATIO = 0.25

COLOR_BALL = (0, 255, 255)
COLOR_ARROW = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
COLOR_PREDICT = (0, 165, 255)
WINDOW_NAME = "Green Ball Tracking"


@dataclass
class Detection:
    """一帧中的一个绿色候选目标。"""

    x: float
    y: float
    radius: float
    area: float
    circularity: float
    fill_ratio: float
    aspect: float

    @property
    def score(self) -> float:
        symmetry = min(self.aspect, 1.0 / self.aspect)
        return self.circularity * self.fill_ratio * symmetry


def detect_candidates(frame: np.ndarray) -> List[Detection]:
    """返回当前帧中同时满足颜色和圆形几何约束的候选目标。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = frame.shape[:2]
    frame_area = float(height * width)
    candidates: List[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_AREA_RATIO * frame_area <= area <= MAX_AREA_RATIO * frame_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        _, radius = cv2.minEnclosingCircle(contour)
        if radius <= 0.0:
            continue
        fill_ratio = area / (math.pi * radius * radius)
        if fill_ratio < MIN_FILL_RATIO:
            continue

        _, _, box_width, box_height = cv2.boundingRect(contour)
        if box_height == 0:
            continue
        aspect = box_width / float(box_height)
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue

        candidates.append(
            Detection(
                x=moments["m10"] / moments["m00"],
                y=moments["m01"] / moments["m00"],
                radius=radius,
                area=area,
                circularity=circularity,
                fill_ratio=fill_ratio,
                aspect=aspect,
            )
        )

    return candidates


class BallTracker:
    """HSV 候选 + 卡尔曼滤波的绿球跟踪器。"""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = Q_ACCEL * np.array(
            [
                [0.25, 0, 0.5, 0],
                [0, 0.25, 0, 0.5],
                [0.5, 0, 1.0, 0],
                [0, 0.5, 0, 1.0],
            ],
            dtype=np.float32,
        )
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R_MEASURE
        self.kf.errorCovPost = np.diag(
            [P0_POS, P0_POS, P0_VEL, P0_VEL]
        ).astype(np.float32)

        self.initialized = False
        self.lost_frames = 0
        self.radius = 0.0
        self.center = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)

    def _start(self, detection: Detection) -> None:
        self.kf.statePost = np.array(
            [[detection.x], [detection.y], [0.0], [0.0]], dtype=np.float32
        )
        self.kf.errorCovPost = np.diag(
            [P0_POS, P0_POS, P0_VEL, P0_VEL]
        ).astype(np.float32)
        self.initialized = True
        self.lost_frames = 0
        self.radius = detection.radius
        self.center[:] = (detection.x, detection.y)
        self.velocity[:] = 0.0

    def update(
        self, candidates: Sequence[Detection], frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[np.ndarray], bool]:
        """返回（滤波后的中心, 本帧是否有真实观测）。"""
        prediction = self.kf.predict()
        predicted = prediction[:2].ravel().astype(np.float64)

        if not self.initialized:
            if not candidates:
                return None, False
            self._start(max(candidates, key=lambda item: item.score))
            return self.center.copy(), True

        speed = float(np.linalg.norm(self.velocity))
        gate = GATE_BASE + 0.8 * speed + 25.0 * self.lost_frames

        matched: Optional[Detection] = None
        best_cost = float("inf")
        for candidate in candidates:
            distance = float(
                np.linalg.norm(np.array((candidate.x, candidate.y)) - predicted)
            )
            if distance > gate:
                continue
            # 球的半径在运动过程中基本不变，用半径约束排除其他绿色物体
            if self.radius > 0.0:
                ratio = candidate.radius / self.radius
                if not 0.55 <= ratio <= 1.80:
                    continue
            cost = distance + (1.0 - candidate.score) * SHAPE_PENALTY
            if cost < best_cost:
                best_cost = cost
                matched = candidate

        if matched is not None:
            self.kf.correct(
                np.array([[matched.x], [matched.y]], dtype=np.float32)
            )
            state = self.kf.statePost.ravel()
            self.radius = 0.7 * self.radius + 0.3 * matched.radius
            self.lost_frames = 0
            observed = True
        else:
            # 短暂遮挡或漏检：沿用卡尔曼预测，不立即放弃目标
            self.lost_frames += 1
            if self.lost_frames > MAX_LOST_FRAMES:
                self.initialized = False
                self.radius = 0.0
                self.lost_frames = 0
                return None, False
            # 没有观测时也要把预测状态写回后验，下一帧才能继续按速度外推
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
            state = prediction.ravel()
            observed = False

        self.center[:] = state[:2]
        self.velocity[:] = state[2:4]

        height, width = frame_shape[:2]
        inside = 0.0 <= self.center[0] < width and 0.0 <= self.center[1] < height
        return (self.center.copy() if inside else None), observed


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def draw_result(
    frame: np.ndarray,
    center: Optional[np.ndarray],
    radius: float,
    velocity: np.ndarray,
    fps: float,
    observed: bool,
) -> np.ndarray:
    """在画面上画圆框、速度箭头和速度数值。"""
    height, width = frame.shape[:2]
    if center is None:
        return frame

    center_x, center_y = int(round(center[0])), int(round(center[1]))
    ball_radius = max(6, int(round(radius)))
    color = COLOR_BALL if observed else COLOR_PREDICT
    cv2.circle(frame, (center_x, center_y), ball_radius, color, 2, cv2.LINE_AA)

    speed = float(np.linalg.norm(velocity)) * fps
    if speed > 20.0:
        unit = velocity / np.linalg.norm(velocity)
        max_length = ARROW_MAX_RATIO * min(width, height)
        arrow_length = min(
            max(ARROW_MIN_LENGTH, speed * ARROW_SECONDS), max_length
        )
        # 箭头从小球边缘出发，避免低速时箭头整个藏在球内看不见
        start = np.array((center[0], center[1])) + unit * (radius + 3.0)
        end = start + unit * arrow_length
        cv2.arrowedLine(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            COLOR_ARROW,
            3,
            cv2.LINE_AA,
            tipLength=0.28,
        )

    text = f"{speed:.0f} px/s"
    text_x = center_x + ball_radius + 10
    text_y = center_y - ball_radius - 8
    if text_x > width - 120:
        text_x = max(8, center_x - ball_radius - 115)
    text_y = min(max(text_y, 24), height - 12)
    draw_text(frame, text, (text_x, text_y), 0.60, COLOR_TEXT, 2)

    return frame


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绿色小球识别与运动追踪")
    parser.add_argument(
        "video", nargs="?", type=Path, default=DEFAULT_VIDEO,
        help="输入视频路径，默认为脚本同目录下的 homework.mp4",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="可选：保存标注后的视频，例如 result.mp4",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="不打开预览窗口，适合无图形界面的环境",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"视频文件不存在：{video_path}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"无法打开视频：{video_path}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 1.0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            print(f"无法创建输出视频：{output_path}", file=sys.stderr)
            return 1

    tracker = BallTracker()
    delay_ms = max(1, int(round(1000.0 / fps)))
    frame_count = 0
    tracked_count = 0
    observed_count = 0
    speeds: List[float] = []
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            candidates = detect_candidates(frame)
            center, observed = tracker.update(candidates, frame.shape)
            processed = draw_result(
                frame, center, tracker.radius, tracker.velocity, fps, observed
            )

            frame_count += 1
            if center is not None:
                tracked_count += 1
                speeds.append(float(np.linalg.norm(tracker.velocity)) * fps)
                if observed:
                    observed_count += 1

            if writer is not None:
                writer.write(processed)

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, processed)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if frame_count == 0:
        print("视频中没有可读取的帧", file=sys.stderr)
        return 1

    print(f"共处理 {frame_count} 帧，跟踪成功 {tracked_count} 帧 "
          f"({100.0 * tracked_count / frame_count:.1f}%)")
    print(f"其中真实检测到小球 {observed_count} 帧 "
          f"({100.0 * observed_count / frame_count:.1f}%)")
    if speeds:
        print(f"速度范围：{min(speeds):.0f} ~ {max(speeds):.0f} px/s，"
              f"平均 {sum(speeds) / len(speeds):.0f} px/s")
    print(f"处理耗时：{time.perf_counter() - started:.2f} s")
    if args.output is not None:
        print(f"结果已保存：{args.output.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
