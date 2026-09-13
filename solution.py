"""求解结果的表达。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    """一套干线协调方案。

    attributes:
        cycle: 公共周期（秒），通常与 Arterial.cycle 一致；
        offsets: 各路口相位差（秒，相对于干线起点路口）；
        plan_choices: 各路口选中的信控方案名；
        bandwidth_up / bandwidth_down: 各路段的上/下行带宽（秒）。
            全局一条带（MAXBAND 风格）时各路段取相同值；
        objective: 目标函数值（语义由求解器定义）；
        status: 求解状态（optimal / feasible / infeasible / ...）；
        solver_msg: 求解器返回的附加信息。
    """

    cycle: float
    offsets: dict[str, float] = field(default_factory=dict)
    plan_choices: dict[str, str] = field(default_factory=dict)
    bandwidth_up: dict[str, float] = field(default_factory=dict)
    bandwidth_down: dict[str, float] = field(default_factory=dict)
    # 每个方向的带宽形态：global = 全局一条；local = 分段/局部带宽
    band_up_style: str = "global"
    band_down_style: str = "global"
    # 带子在各路口的起始时刻（秒，mod 周期），绘制时空图用
    band_start_up: dict[str, float] = field(default_factory=dict)
    band_start_down: dict[str, float] = field(default_factory=dict)
    # 窗口带宽：key 形如 "up.win3@I1-I3" / "down.win3@I1-I3"，
    # value 为该子走廊独立求解得到的局部最优带宽（秒）。
    window_bands: dict[str, float] = field(default_factory=dict)
    # 窗口带时间范围：key -> [range_entry, ...]。
    # 每个 entry 描述一个局部绿波带实例在各路口上的起止时间，
    # 适合直接用于绘图、导出和下游分析。
    window_band_ranges: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    # 相位时长：路口名 -> 相位名 -> 秒（第二阶段优化后回填）
    phase_times: dict[str, dict[str, float]] = field(default_factory=dict)
    # 段级时刻：路口名 -> "up.1.start"/"down.2.end" -> 秒。
    segment_times: dict[str, dict[str, float]] = field(default_factory=dict)
    # 窗口选择：路口名 -> {"plan": 方案名, "up_window": 下标, "down_window": 下标}
    window_choices: dict[str, dict[str, int | str]] = field(default_factory=dict)
    # 多段绿波带：方向 -> 段号(1-based) -> 全走廊公共带宽（秒）。
    multi_bandwidths: dict[str, dict[int, float]] = field(default_factory=dict)
    # 多段绿波带起点：方向 -> 段号(1-based) -> 路口名 -> 到达时刻（秒）。
    multi_band_starts: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    # 多段窗口带宽：方向 -> 段号(1-based) -> "up.win3@I1-I3" -> 秒。
    # 语义同样是“该段号在对应子走廊上的独立局部最优带宽”。
    multi_window_bands: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    # ---- 绿波带层目标 ----
    # band_objective: ObjectiveConfig 的 SumGroup + BalanceGroup 收益
    # band_loss:      绿波带层损失，例如 AlignmentLoss
    # band_score:     band_objective - band_loss_weight * band_loss
    band_objective: float = 0.0
    band_loss: float = 0.0
    band_score: float = 0.0

    # ---- 交叉口/相位层损失 ----
    # 相位 hinge loss + 软 LinearSpec slack 违反量。
    intersection_loss: float = 0.0

    # 兼容旧字段；新代码请直接使用 intersection_loss。
    # 新语义下 total_phase_loss 等于 intersection_loss，
    # 即相位 hinge loss + 软 LinearSpec slack，不再包含 alignment。
    total_phase_loss: float = 0.0
    objective: float = 0.0
    status: str = "unknown"
    solver_msg: str = ""

    @property
    def total_bandwidth(self) -> float:
        """双向带宽总和（快速评价指标）。"""
        return sum(self.bandwidth_up.values()) + sum(self.bandwidth_down.values())

    def is_feasible(self) -> bool:
        """解是否可行（占位：后续实现约束校验）。"""
        # TODO: 校验带宽是否落在各路口绿灯窗内、相位差传递是否一致等
        raise NotImplementedError

    def to_dict(self) -> dict:
        """序列化为普通字典，便于存储/展示。"""
        return {
            "cycle": self.cycle,
            "offsets": self.offsets,
            "plan_choices": self.plan_choices,
            "bandwidth_up": self.bandwidth_up,
            "bandwidth_down": self.bandwidth_down,
            "band_up_style": self.band_up_style,
            "band_down_style": self.band_down_style,
            "band_start_up": self.band_start_up,
            "band_start_down": self.band_start_down,
            "window_bands": self.window_bands,
            "window_band_ranges": self.window_band_ranges,
            "phase_times": self.phase_times,
            "segment_times": self.segment_times,
            "window_choices": self.window_choices,
            "multi_bandwidths": self.multi_bandwidths,
            "multi_band_starts": self.multi_band_starts,
            "multi_window_bands": self.multi_window_bands,
            "band_objective": self.band_objective,
            "band_loss": self.band_loss,
            "band_score": self.band_score,
            "intersection_loss": self.intersection_loss,
            "total_phase_loss": self.total_phase_loss,
            "objective": self.objective,
            "status": self.status,
            "solver_msg": self.solver_msg,
        }
