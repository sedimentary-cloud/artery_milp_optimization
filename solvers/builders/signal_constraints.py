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

    terms 当前建议只使用：
    - `I2.up.1.start` / `I3.down.2.end`

    带宽类特殊变量 `b_up` / `b_down` / `bD_*` / `bU_*` / `B_bal` 保留，
    但它们表示所有全局 band 的聚合；时间类特殊变量 `tU_*` / `tD_*`
    已停用，不建议再使用。

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

    terms 当前建议只使用：
    - `I2.up.1.start`
    - `I3.down.2.end`

    带宽类特殊变量 `b_up` / `b_down` / `bD_*` / `bU_*` / `B_bal` 保留，
    但它们表示所有全局 band 的聚合；时间类特殊变量 `tU_*` / `tD_*`
    已停用，不建议再使用。

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


def scale_linear_spec(spec: LinearSpec, cycle: float) -> LinearSpec:
    """函数名：scale_linear_spec；参数：spec、cycle；返回值：秒域 LinearSpec；异常：无。"""
    return replace(spec, rhs=float(spec.rhs) * float(cycle))
