"""求解结果的表达。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    """一套干线协调方案的结果。

    每个字段的格式与示例见下方行间注释。
    核心原始结果包括 plan_choices / segment_times /
    multi_bandwidths / multi_band_starts /
    multi_window_bands / window_band_ranges。
    """

    # 公共信号周期（秒）。
    # 示例: 90.0
    cycle: float

    # 路口名 -> 选中的 SignalPlan 名称。
    # 示例: {"I1": "baseline", "I2": "split_priority"}
    plan_choices: dict[str, str] = field(default_factory=dict)

    # 上行带宽摘要：物理路段名 -> 带宽（秒）。
    # 口径由 solver 的 up_global_output 决定。
    # 示例: {"S12": 8.03, "S23": 10.99}
    bandwidth_up: dict[str, float] = field(default_factory=dict)

    # 下行带宽摘要：物理路段名 -> 带宽（秒）。
    # 示例: {"S12": 0.0, "S23": 0.0}
    bandwidth_down: dict[str, float] = field(default_factory=dict)

    # 上行代表带在各路口的到达时刻（秒，mod cycle）。
    # 当前取第一个活跃段号的轨迹。
    # 示例: {"I1": 4.5, "I2": 18.07, "I3": 32.71}
    band_start_up: dict[str, float] = field(default_factory=dict)

    # 下行代表带在各路口的到达时刻（秒，mod cycle）。
    # 示例: {"I1": 68.4, "I2": 54.97, "I3": 40.69}
    band_start_down: dict[str, float] = field(default_factory=dict)

    # 局部窗口带带宽：窗口 key -> 带宽（秒）。
    # key 格式: "{direction}.win{k}@{start_int}-{end_int}"。
    # 示例: {"down.win3@I2-I4": 13.36}
    window_bands: dict[str, float] = field(default_factory=dict)

    # 局部窗口带时间范围：窗口 key -> [实例, ...]。
    # 每个实例包含 direction/segment_no/bandwidth/intersections/
    # time_min/time_max/intersection_ranges。
    # 示例:
    # {
    #   "down.win3@I2-I4": [
    #     {
    #       "direction": "down",
    #       "segment_no": 1,
    #       "bandwidth": 13.36,
    #       "intersections": ["I2", "I3", "I4"],
    #       "time_min": 18.90,
    #       "time_max": 59.40,
    #       "intersection_ranges": {
    #         "I2": {"start": 46.04, "end": 59.40},
    #         "I3": {"start": 31.76, "end": 45.11},
    #         "I4": {"start": 18.90, "end": 32.26},
    #       },
    #     }
    #   ]
    # }
    window_band_ranges: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    # Stage 2 优化后的段端点时刻：路口名 -> term -> 秒。
    # term 形如 "up.1.start" / "down.2.end"。
    # 示例: {"I2": {"up.1.start": 9.0, "up.1.end": 26.1, ...}}
    segment_times: dict[str, dict[str, float]] = field(default_factory=dict)

    # 全局多段带：方向 -> 段号(1-based) -> 全走廊带宽（秒）。
    # 示例: {"up": {1: 14.0, 2: 8.5}, "down": {1: 12.0}}
    multi_bandwidths: dict[str, dict[int, float]] = field(default_factory=dict)

    # 全局多段带轨迹：方向 -> 段号(1-based) -> 路口名 -> 到达时刻（秒，mod cycle）。
    # 示例: {"up": {1: {"I1": 4.5, "I2": 18.07, "I3": 32.71}}}
    multi_band_starts: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)

    # 局部窗口带按段号拆分：方向 -> 段号(1-based) -> 窗口 key -> 带宽（秒）。
    # 示例: {"down": {1: {"down.win3@I2-I4": 13.36}}}
    multi_window_bands: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)

    # 绿波带层收益（ObjectiveConfig 的 SumGroup + BalanceGroup）。
    # 示例: 32.354
    band_objective: float = 0.0

    # 绿波带层软损失（Stage 2 软边距 + kind="band" 的 SegmentLossSpec）。
    # 示例: 8.843
    band_loss: float = 0.0

    # 带层最终得分: band_objective - band_loss_weight * band_loss。
    # 示例: 25.184
    band_score: float = 0.0

    # 交叉口层损失（信号损失 + 软 LinearSpec slack 违反量）。
    # 示例: 1.350
    intersection_loss: float = 0.0

    # solver 原始目标值（不同 solver / 不同运行模式下语义可能不同）。
    # 示例: 35.751
    objective: float = 0.0

    # 求解状态。
    # 示例: "optimal" / "infeasible" / "optimal|stage1_fallback"
    status: str = "unknown"

    # 底层求解器附加信息。
    # 示例: "HiGHS via scipy: success=True"
    solver_msg: str = ""

    @property
    def total_bandwidth(self) -> float:
        """双向带宽总和（快速评价指标）。"""
        return sum(self.bandwidth_up.values()) + sum(self.bandwidth_down.values())

    def to_dict(self) -> dict:
        """序列化为普通字典，便于存储/展示。"""
        return {
            "cycle": self.cycle,
            "plan_choices": self.plan_choices,
            "bandwidth_up": self.bandwidth_up,
            "bandwidth_down": self.bandwidth_down,
            "band_start_up": self.band_start_up,
            "band_start_down": self.band_start_down,
            "window_bands": self.window_bands,
            "window_band_ranges": self.window_band_ranges,
            "segment_times": self.segment_times,
            "multi_bandwidths": self.multi_bandwidths,
            "multi_band_starts": self.multi_band_starts,
            "multi_window_bands": self.multi_window_bands,
            "band_objective": self.band_objective,
            "band_loss": self.band_loss,
            "band_score": self.band_score,
            "intersection_loss": self.intersection_loss,
            "objective": self.objective,
            "status": self.status,
            "solver_msg": self.solver_msg,
        }
