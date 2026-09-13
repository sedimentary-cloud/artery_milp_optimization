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


@dataclass(frozen=True)
class SignalConstraint:
    """路口级业务约束声明。

    本轮重构后，约束不再直接依赖“相位名”，而是依赖分方向多段绿区间的
    起止时刻。terms 的 key 使用如下记法：

    - ``up.1.start`` / ``up.1.end``
    - ``down.2.start`` / ``down.2.end``

    其中段号从 1 开始。
    """

    terms: dict[str, float]
    sense: str = "<="
    rhs: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        """函数名：__post_init__；参数：无；返回值：无；异常：ValueError。"""
        if self.sense not in ("<=", ">=", "="):
            raise ValueError(f"非法约束方向: {self.sense}")


@dataclass(frozen=True)
class SignalLoss:
    """路口内软损失声明。

    与 ``SignalConstraint`` 一样，terms 只描述单个路口内部的段端点表达式：

    - ``up.1.start`` / ``up.1.end``
    - ``down.2.start`` / ``down.2.end``

    阈值与斜率语义：
    - 下侧：``lower_slope * max(0, lower_threshold - expr)``
    - 上侧：``upper_slope * max(0, expr - upper_threshold)``
    """

    terms: dict[str, float]
    lower_threshold: float | None = None
    lower_slope: float = 1.0
    upper_threshold: float | None = None
    upper_slope: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """函数名：__post_init__；参数：无；返回值：无；异常：ValueError。"""
        if self.lower_threshold is None and self.upper_threshold is None:
            raise ValueError("SignalLoss 至少需要一个阈值")
        if self.lower_slope < 0:
            raise ValueError("SignalLoss.lower_slope 不能为负")
        if self.upper_slope is not None and self.upper_slope < 0:
            raise ValueError("SignalLoss.upper_slope 不能为负")


class SignalPlan:
    """一个路口的一种可选信控方案。

    新主入口：
    - ``up_segments`` / ``down_segments``：按时间顺序给出每个方向的绿区间；
    - ``signal_constraints``：路口内部的业务约束声明，引用段起止时刻。

    兼容旧入口：
    - ``up_windows`` / ``down_windows`` 仍可使用，内部会映射到新字段；
    - ``phases`` 等旧字段暂时保留，供旧求解器继续工作。
    """

    def __init__(
        self,
        name: str,
        up_segments: list[GreenWindow] | None = None,
        down_segments: list[GreenWindow] | None = None,
        signal_constraints: list[SignalConstraint] | None = None,
        signal_losses: list[SignalLoss] | None = None,
        *,
        up_windows: list[GreenWindow] | None = None,
        down_windows: list[GreenWindow] | None = None,
        phases: list[Phase] | None = None,
        up_phase: str | None = None,
        down_phase: str | None = None,
        lost_time: float = 0.0,
        phase_lost_times: dict[str, float] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """函数名：__init__；参数：方案名、多段绿区间、业务约束等；返回值：无；异常：ValueError。"""
        self.name = name
        self.up_segments = list(up_segments if up_segments is not None else (up_windows or []))
        self.down_segments = list(down_segments if down_segments is not None else (down_windows or []))
        self.signal_constraints = list(signal_constraints or [])
        self.signal_losses = list(signal_losses or [])
        self.phases = list(phases or [])
        self.up_phase = up_phase
        self.down_phase = down_phase
        self.lost_time = float(lost_time)
        self.phase_lost_times = dict(phase_lost_times or {})
        self.metadata = dict(metadata or {})
        self._validate_segments("up", self.up_segments)
        self._validate_segments("down", self.down_segments)
        self._validate_signal_constraints()
        self._validate_signal_losses()

    @staticmethod
    def _validate_segments(direction: str, segments: list[GreenWindow]) -> None:
        """函数名：_validate_segments；参数：方向、区间列表；返回值：无；异常：ValueError。"""
        last_end = -1.0
        for idx, segment in enumerate(segments, start=1):
            if segment.start < last_end - 1e-12:
                raise ValueError(
                    f"{direction} 第 {idx} 段与前一段重叠或未按时间排序: "
                    f"{segment.start} < {last_end}"
                )
            last_end = segment.end

    def _value_of_term(self, term: str) -> float:
        """函数名：_value_of_term；参数：term；返回值：常数值；异常：KeyError/ValueError。"""
        try:
            direction, seg_idx, endpoint = term.split(".")
        except ValueError as exc:
            raise ValueError(f"非法约束项标识: {term}") from exc

        if direction not in ("up", "down"):
            raise ValueError(f"未知方向: {direction}")
        if endpoint not in ("start", "end"):
            raise ValueError(f"未知区间端点: {endpoint}")

        index = int(seg_idx) - 1
        segments = self.up_segments if direction == "up" else self.down_segments
        if not (0 <= index < len(segments)):
            raise KeyError(
                f"方案 {self.name} 不存在 {direction}.{seg_idx}.{endpoint}"
            )
        window = segments[index]
        return window.start if endpoint == "start" else window.end

    def term_value(self, term: str) -> float:
        """函数名：term_value；参数：term；返回值：term 当前比例值；异常：KeyError/ValueError。"""
        return self._value_of_term(term)

    def all_segment_terms(self) -> list[str]:
        """函数名：all_segment_terms；参数：无；返回值：全部段级端点名；异常：无。"""
        terms: list[str] = []
        for direction, segments in (("up", self.up_segments), ("down", self.down_segments)):
            for idx, _segment in enumerate(segments, start=1):
                terms.append(f"{direction}.{idx}.start")
                terms.append(f"{direction}.{idx}.end")
        return terms

    def term_bounds(self, term: str) -> tuple[float, float]:
        """函数名：term_bounds；参数：term；返回值：term 的比例上下界；异常：KeyError/ValueError。"""
        default = self._value_of_term(term)
        raw = self.metadata.get("term_bounds", {})
        if not isinstance(raw, dict) or term not in raw:
            return (default, default)

        bounds = raw[term]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"方案 {self.name} 的 term_bounds[{term!r}] 不是二元上下界")
        lower = float(bounds[0])
        upper = float(bounds[1])
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(
                f"方案 {self.name} 的 term_bounds[{term!r}] 越界: {(lower, upper)}"
            )
        return (lower, upper)

    def _validate_signal_constraints(self) -> None:
        """函数名：_validate_signal_constraints；参数：无；返回值：无；异常：ValueError。"""
        for constraint in self.signal_constraints:
            lhs = sum(
                coef * self._value_of_term(term)
                for term, coef in constraint.terms.items()
            )
            if constraint.sense == "<=" and lhs > constraint.rhs + 1e-12:
                raise ValueError(
                    f"方案 {self.name} 的业务约束 {constraint.name or constraint.terms} 不满足: "
                    f"{lhs} <= {constraint.rhs} 失败"
                )
            if constraint.sense == ">=" and lhs < constraint.rhs - 1e-12:
                raise ValueError(
                    f"方案 {self.name} 的业务约束 {constraint.name or constraint.terms} 不满足: "
                    f"{lhs} >= {constraint.rhs} 失败"
                )
            if constraint.sense == "=" and abs(lhs - constraint.rhs) > 1e-12:
                raise ValueError(
                    f"方案 {self.name} 的业务约束 {constraint.name or constraint.terms} 不满足: "
                    f"{lhs} = {constraint.rhs} 失败"
                )

    def _validate_signal_losses(self) -> None:
        """函数名：_validate_signal_losses；参数：无；返回值：无；异常：ValueError。"""
        for loss in self.signal_losses:
            for term in loss.terms:
                self._value_of_term(term)

    @property
    def up_windows(self) -> list[GreenWindow]:
        """函数名：up_windows；参数：无；返回值：上行区间列表；异常：无。"""
        return self.up_segments

    @property
    def down_windows(self) -> list[GreenWindow]:
        """函数名：down_windows；参数：无；返回值：下行区间列表；异常：无。"""
        return self.down_segments

    def total_lost_time(self) -> float:
        """函数名：total_lost_time；参数：无；返回值：总损失时间；异常：无。"""
        return float(self.lost_time + sum(self.phase_lost_times.values()))

    def phase_by_name(self, name: str) -> Phase:
        """函数名：phase_by_name；参数：name；返回值：Phase；异常：KeyError。"""
        for ph in self.phases:
            if ph.name == name:
                return ph
        raise KeyError(f"方案 {self.name} 没有相位 {name}")

    def serving_phase_names(self, direction: str) -> list[str]:
        """函数名：serving_phase_names；参数：direction；返回值：相位名列表；异常：无。"""
        return [ph.name for ph in self.phases if direction in ph.serves]

    def direction_phase_names(self, direction: str) -> list[str]:
        """函数名：direction_phase_names；参数：direction；返回值：相位名列表；异常：ValueError。"""
        if direction not in ("up", "down"):
            raise ValueError(f"未知方向: {direction}")

        explicit = self.up_phase if direction == "up" else self.down_phase
        if explicit is not None:
            names = {ph.name for ph in self.phases}
            if explicit not in names:
                raise ValueError(
                    f"方案 {self.name} 的 {direction}_phase={explicit!r} "
                    f"不在 phases 中"
                )
            return [explicit]

        served = self.serving_phase_names(direction)
        if not served:
            raise ValueError(
                f"方案 {self.name} 无法解析 {direction} 方向相位："
                f"请设置 {direction}_phase 或给 Phase.serves 添加 {direction}"
            )
        return served

    def up_green_ratio(self) -> float:
        """函数名：up_green_ratio；参数：无；返回值：上行总绿信比；异常：无。"""
        return sum(w.width for w in self.up_segments)

    def down_green_ratio(self) -> float:
        """函数名：down_green_ratio；参数：无；返回值：下行总绿信比；异常：无。"""
        return sum(w.width for w in self.down_segments)


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
