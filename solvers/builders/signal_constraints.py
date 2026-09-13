"""段级窗口表达式与约束辅助。

本模块已迁移到“段级输入”体系：
- `SignalPlan.up_segments / down_segments` 是主输入；
- 每个段端点使用 `up.1.start` / `down.2.end` 这类 term 标识；
- 第二阶段求解器直接围绕这些段端点变量建模。

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ...models import SignalPlan


@dataclass
class WindowExpr:
    """一个方案的段级绿灯窗表达式。

    只保存 Stage 2 真正需要的元信息：

    - ``up_labels`` / ``down_labels``：``up.1`` / ``down.2`` 这类段标签；
    - ``term_names``：所有段端点 term；
    - ``term_bounds``：term -> 秒域上下界。
    """

    up_labels: list[str] = field(default_factory=list)
    down_labels: list[str] = field(default_factory=list)
    term_names: list[str] = field(default_factory=list)
    term_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


def window_exprs(plan: SignalPlan, cycle: float) -> WindowExpr:
    """函数名：window_exprs；参数：plan、cycle；返回值：段级窗口表达式；异常：ValueError。"""
    term_names = plan.all_segment_terms()
    term_bounds = {
        term: (
            plan.term_bounds(term)[0] * cycle,
            plan.term_bounds(term)[1] * cycle,
        )
        for term in term_names
    }

    up_labels = [f"up.{idx}" for idx, _window in enumerate(plan.up_segments, start=1)]
    down_labels = [f"down.{idx}" for idx, _window in enumerate(plan.down_segments, start=1)]

    if not up_labels or not down_labels:
        raise ValueError(f"方案 {plan.name} 缺少可用的段级绿区间")

    return WindowExpr(
        up_labels=up_labels,
        down_labels=down_labels,
        term_names=term_names,
        term_bounds=term_bounds,
    )


@dataclass
class SegmentLossSpec:
    """段级端点/段宽表达式的双侧软损失配置。

    定位是**跨路口 / 全局 / 临时策略的高级扩展工具**；单个路口内部
    的期望区间请优先使用 ``SignalPlan.signal_losses`` 里的
    ``SignalLoss``。

    terms 支持：
    - `I2.up.1.start` / `I3.down.2.end`
    - `tU_I2` / `tD_I4`
    - `b_up` / `b_down` / `bD_seg2` / `B_bal`

    损失形式：
    - 下侧：`lower_slope * max(0, lower_threshold - expr)`
    - 上侧：`upper_slope * max(0, expr - upper_threshold)`

    外部声明约定：
    - `lower_threshold / upper_threshold` 均使用“占周期比例”；
    - 求解器内部会在建模前统一乘以 `cycle` 转成秒。
    """

    terms: dict[str, float]
    lower_threshold: float | None = None
    lower_slope: float = 1.0
    upper_threshold: float | None = None
    upper_slope: float | None = None
    name: str = ""
    kind: str = "intersection"
    plan_tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """函数名：__post_init__；参数：无；返回值：无；异常：ValueError。"""
        if self.kind not in ("intersection", "band"):
            raise ValueError(f"非法损失类别: {self.kind}")
        if self.lower_threshold is None and self.upper_threshold is None:
            raise ValueError("SegmentLossSpec 至少需要一个阈值")


class SegmentLossBuilder:
    """把段级双侧软损失转换成 LinearSpec。"""

    def __init__(self, specs: list[SegmentLossSpec] | None = None) -> None:
        """函数名：__init__；参数：specs；返回值：无；异常：无。"""
        self.specs = list(specs or [])

    def add(self, spec: SegmentLossSpec) -> None:
        """函数名：add；参数：spec；返回值：无；异常：无。"""
        self.specs.append(spec)

    def to_linear_specs(self) -> list[tuple[LinearSpec, str]]:
        """函数名：to_linear_specs；参数：无；返回值：[(LinearSpec, kind)]；异常：无。"""
        out: list[tuple[LinearSpec, str]] = []
        for spec in self.specs:
            if spec.lower_threshold is not None:
                out.append((
                    LinearSpec(
                        terms=dict(spec.terms),
                        sense=">=",
                        rhs=float(spec.lower_threshold),
                        soft=True,
                        penalty=float(spec.lower_slope),
                        plan_tags=dict(spec.plan_tags),
                        name=(f"{spec.name}.lower" if spec.name else ""),
                    ),
                    spec.kind,
                ))
            if spec.upper_threshold is not None:
                out.append((
                    LinearSpec(
                        terms=dict(spec.terms),
                        sense="<=",
                        rhs=float(spec.upper_threshold),
                        soft=True,
                        penalty=float(spec.upper_slope if spec.upper_slope is not None else spec.lower_slope),
                        plan_tags=dict(spec.plan_tags),
                        name=(f"{spec.name}.upper" if spec.name else ""),
                    ),
                    spec.kind,
                ))
        return out




@dataclass
class LinearSpec:
    """线性约束声明（跨路口 / 全局 / 临时策略的高级扩展工具）。

    terms 支持：
    - `I2.up.1.start`
    - `I3.down.2.end`
    - `tU_I2` / `tD_I2`
    - `b_up` / `b_down` / `B_bal`
    - `bD_seg1`

    外部声明约定：
    - `rhs` 使用“占周期比例”；
    - 求解器内部会在建模前统一乘以 `cycle` 转成秒；
    - 所有 term 会在组装 MILP 前经过
      :class:`TermValidationContext` 统一校验；非法/不可用的 term
      会抛 ``TermValidationError``，不会被静默丢弃或截短。
    """

    terms: dict[str, float]
    sense: str = ">="
    rhs: float = 0.0
    soft: bool = False
    penalty: float = 1.0
    plan_tags: dict[str, str] = field(default_factory=dict)
    name: str = ""


class ConstraintBuilder:
    """线性硬约束/软约束声明集合。

    定位是**跨路口 / 全局 / 临时策略的高级扩展入口**。路口内部规则
    请优先写进 ``SignalPlan.signal_constraints``，不要用本类外置。
    """

    def __init__(self, specs: list[LinearSpec] | None = None) -> None:
        """函数名：__init__；参数：specs；返回值：无；异常：无。"""
        self.specs = list(specs or [])

    def add(self, spec: LinearSpec) -> None:
        """函数名：add；参数：spec；返回值：无；异常：无。"""
        self.specs.append(spec)


class AlignmentLossBuilder:
    """绿波带层对齐损失配置（只处理上下行全局带）。

    当前实现只支持把“全局带中心”拉到目标时刻：

        up:   tU_i + 0.5 * b_up   ~ targets_up[i]
        down: tD_i + 0.5 * b_down ~ targets_down[i]

    没有写进 ``targets_up`` / ``targets_down`` 的路口不会生成对齐约束。
    局部路段带（``bD_*`` / ``bU_*``）和窗口带（``winK@...``）不在这里处理；
    如果确实需要逐路段对齐，请使用 ``SegmentLossSpec`` 显式构造。

    外部声明约定：
    - ``targets_up / targets_down / tolerance`` 均使用“占周期比例”；
    - 求解器内部会在建模前统一乘以 ``cycle`` 转成秒。
    """

    def __init__(
        self,
        targets_up: dict[str, float] | None = None,
        targets_down: dict[str, float] | None = None,
        tolerance: float = 0.0,
        weight_up: float = 1.0,
        weight_down: float = 1.0,
    ) -> None:
        """函数名：__init__；参数：targets_up、targets_down、tolerance、weight_up、weight_down；返回值：无；异常：无。"""
        self.targets_up = dict(targets_up or {})
        self.targets_down = dict(targets_down or {})
        self.tolerance = tolerance
        self.weight_up = weight_up
        self.weight_down = weight_down

    def to_linear_specs(self, int_names: list[str]) -> list[LinearSpec]:
        """为上下行全局带生成软对齐约束。

        Args:
            int_names: 干线路口名列表，同时决定上下行对齐的遍历顺序。

        Returns:
            上下各一条（``<=`` / ``>=``）软约束；未配置目标的路口自动跳过。
        """
        specs: list[LinearSpec] = []
        tol = self.tolerance

        # 上行：全局带中心 tU_i + 0.5 * b_up
        for name in int_names:
            target = self.targets_up.get(name)
            if target is None:
                continue
            center = {f"tU_{name}": 1.0, "b_up": 0.5}
            specs.append(LinearSpec(center, sense="<=", rhs=target + tol, soft=True,
                                    penalty=self.weight_up, name=f"align.up.{name}.upper"))
            specs.append(LinearSpec(center, sense=">=", rhs=target - tol, soft=True,
                                    penalty=self.weight_up, name=f"align.up.{name}.lower"))

        # 下行：全局带中心 tD_i + 0.5 * b_down
        for name in int_names:
            target = self.targets_down.get(name)
            if target is None:
                continue
            center = {f"tD_{name}": 1.0, "b_down": 0.5}
            specs.append(LinearSpec(center, sense="<=", rhs=target + tol, soft=True,
                                    penalty=self.weight_down, name=f"align.down.{name}.upper"))
            specs.append(LinearSpec(center, sense=">=", rhs=target - tol, soft=True,
                                    penalty=self.weight_down, name=f"align.down.{name}.lower"))

        return specs


def scale_linear_spec(spec: LinearSpec, cycle: float) -> LinearSpec:
    """函数名：scale_linear_spec；参数：spec、cycle；返回值：秒域 LinearSpec；异常：无。"""
    return replace(spec, rhs=float(spec.rhs) * float(cycle))
