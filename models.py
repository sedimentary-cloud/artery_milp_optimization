"""干线绿波问题的数据模型。

约定：
- 所有时间相关的比例量均以"占信号周期的比例"（0~1 的小数）表示；
- 方向约定：up = 上行，down = 下行；
- 距离单位：米；速度单位：米/秒；周期单位：秒。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 信控方案
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GreenWindow:
    """一段绿灯窗口，用占周期比例的起止区间表示，如 (0.10, 0.30)。"""

    start: float  # 窗口起点（占周期比例，含）
    end: float    # 窗口终点（占周期比例，含）

    def __post_init__(self) -> None:
        if not (0.0 <= self.start < self.end <= 1.0):
            raise ValueError(f"非法绿灯窗口: ({self.start}, {self.end})")

    @property
    def width(self) -> float:
        """窗口宽度（占周期比例）。"""
        return self.end - self.start


@dataclass
class Phase:
    """一个可调相位的绿灯时长配置。

    attributes:
        name: 相位名称；
        green: 当前/默认绿灯时长（秒）；
        min_green: 该相位绿灯时长下限（秒）；
        max_green: 该相位绿灯时长上限（秒）；
        serves: 该相位服务的方向集合，取值可包含 "up" / "down"。
            例如 serves=("up", "down") 表示该相位同时服务上下行。
    """

    name: str
    green: float = 0.0
    min_green: float = 0.0
    max_green: float = 999.0
    serves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for direction in self.serves:
            if direction not in ("up", "down"):
                raise ValueError(
                    f"Phase {self.name} 的 serves 只能是 up/down，"
                    f"当前为 {direction!r}"
                )


@dataclass
class SignalPlan:
    """一个路口的一种可选信控方案。

    两种用法：
    1. 直接给 up_windows / down_windows（固定窗口，兼容现有代码）；
    2. 给 phases + up_phase + down_phase，表示该方案由若干相位组成，
       上行/下行绿灯窗由对应相位的累计起止时间决定，可做相位时长优化。
    """

    name: str
    up_windows: list[GreenWindow] = field(default_factory=list)
    down_windows: list[GreenWindow] = field(default_factory=list)

    # ---- 可选：相位级配置（用于第二阶段相位时长优化） ----
    phases: list[Phase] = field(default_factory=list)
    up_phase: str | None = None
    down_phase: str | None = None
    lost_time: float = 0.0  # 周期尾部损失时间（秒）

    # 相位间损失时间：{相位名: 该相位绿灯结束后分配的损失秒数}。
    # 例如 {"P1": 5.0} 表示 P1 绿灯结束后有 5 秒黄灯/全红，
    # 然后才进入下一个相位。
    phase_lost_times: dict[str, float] = field(default_factory=dict)

    def total_lost_time(self) -> float:
        """周期总损失时间 = 相位间损失 + 尾部损失。"""
        return float(self.lost_time + sum(self.phase_lost_times.values()))

    def phase_by_name(self, name: str) -> Phase:
        for ph in self.phases:
            if ph.name == name:
                return ph
        raise KeyError(f"方案 {self.name} 没有相位 {name}")

    def serving_phase_names(self, direction: str) -> list[str]:
        """返回通过 serves 声明服务某方向的相位名列表。"""
        return [ph.name for ph in self.phases if direction in ph.serves]

    def up_green_ratio(self) -> float:
        """上行总绿信比。"""
        return sum(w.width for w in self.up_windows)

    def down_green_ratio(self) -> float:
        """下行总绿信比。"""
        return sum(w.width for w in self.down_windows)


# ---------------------------------------------------------------------------
# 路口
# ---------------------------------------------------------------------------

@dataclass
class Intersection:
    """干线上的一个信号交叉口，持有若干可选信控方案。"""

    name: str
    plans: list[SignalPlan] = field(default_factory=list)

    def plan_by_name(self, name: str) -> SignalPlan:
        for p in self.plans:
            if p.name == name:
                return p
        raise KeyError(f"路口 {self.name} 没有方案 {name}")


# ---------------------------------------------------------------------------
# 路段
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """相邻两路口之间的路段（连接顺序由 Arterial 维护）。

    两个方向的长度和车速分别给定；车速预设为固定值。
    """

    name: str
    length_up: float    # 上行方向长度（米）
    length_down: float  # 下行方向长度（米）
    speed_up: float     # 上行车速（米/秒）
    speed_down: float   # 下行车速（米/秒）

    # ---- 行驶时间（秒） ----

    @property
    def travel_time_up(self) -> float:
        return self.length_up / self.speed_up

    @property
    def travel_time_down(self) -> float:
        return self.length_down / self.speed_down

    # ---- 约化量（用于时空图：把时间轴折算成周期数，把空间轴折算成"等效距离"） ----

    def reduced_length_up(self, cycle: float) -> float:
        """上行约化路程：行驶时间占周期数 × 一个周期在图上的单位长度。

        这里定义图上 1 个周期对应 1 个单位长度，因此约化路程 = 行驶时间 / 周期。
        """
        return self.travel_time_up / cycle

    def reduced_length_down(self, cycle: float) -> float:
        """下行约化路程，含义同上行。"""
        return self.travel_time_down / cycle

    def reduced_speed_up(self, cycle: float) -> float:
        """上行约化速度：时空图（x=约化路程, y=周期数）上轨迹斜率的倒数。"""
        return self.speed_up * cycle / cycle  # 占位：约化坐标下恒为 1，保留接口

    def reduced_speed_down(self, cycle: float) -> float:
        """下行约化速度，含义同上行。"""
        return self.speed_down * cycle / cycle


# ---------------------------------------------------------------------------
# 干线
# ---------------------------------------------------------------------------

@dataclass
class Arterial:
    """一条干线：公共周期 + 路口/路段的字典 + 连接顺序。

    order 为路口与路段交替出现的名称列表，例如：
        ["A", "seg_AB", "B", "seg_BC", "C"]
    即路口数 = n，路段数 = n - 1。
    """

    cycle: float  # 公共信号周期（秒）
    intersections: dict[str, Intersection] = field(default_factory=dict)
    segments: dict[str, Segment] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.cycle <= 0:
            raise ValueError("公共周期必须为正")
        if len(self.order) < 1 or len(self.order) % 2 == 0:
            raise ValueError("order 必须是奇数长度：路口、路段交替，首尾为路口")
        for i, key in enumerate(self.order):
            pool = self.intersections if i % 2 == 0 else self.segments
            if key not in pool:
                raise KeyError(f"order 中第 {i} 个元素 {key!r} 未在对应字典中注册")

    # ---- 按顺序访问 ----

    @property
    def intersection_order(self) -> list[Intersection]:
        """沿干线上行方向的路口序列。"""
        return [self.intersections[k] for k in self.order[0::2]]

    @property
    def segment_order(self) -> list[Segment]:
        """沿干线上行方向的路段序列（第 i 段连接第 i 与 i+1 个路口）。"""
        return [self.segments[k] for k in self.order[1::2]]

    @property
    def n_intersections(self) -> int:
        return len(self.intersection_order)

    @property
    def n_segments(self) -> int:
        return len(self.segment_order)
