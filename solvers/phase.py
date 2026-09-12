"""相位窗口表达式与约束组装辅助。

第二阶段 PhaseTuneSolver 会把方案锁定后，将每个相位时长作为连续变量。
本模块负责把“信控方案”转换成“相位变量的线性表达式”。

表达式约定：
    value = const + Σ coefs[j] * g[j]
其中 j 是该方案 phases 列表里的相位下标；g[j] 是对应相位时长（秒）。

本模块提供三类配置，都是“声明式”的：
1. window_exprs(plan, C)
   - 把 SignalPlan 转成上/下行绿灯窗的相位变量线性表达式；
   - 如果 SignalPlan 没有 phases，则退回固定窗口（常数表达式）。
   - 供 PhaseTuneSolver 内部使用，用户一般不需要直接调用。

2. PhaseLossSpec / PhaseLossBuilder
   - 配置单相位 hinge 损失：
       loss = slope * max(0, threshold - g)
              + upper_slope * max(0, g - upper_threshold)
   - 可指定 intersection，只惩罚某个路口的相位；
   - 所有损失最终汇入 PhaseTuneSolver 的 total_phase_loss。

3. LinearSpec / ConstraintBuilder
   - 声明相位变量的线性硬约束 / 软约束；
   - 软约束自动创建 slack 变量，并把 penalty * slack 汇入损失；
   - 由 PhaseTuneSolver.solve(..., constraint_builder=...) 注入。

典型用法：
    loss_builder = PhaseLossBuilder([
        PhaseLossSpec("P1", threshold=30, slope=2, intersection="I2"),
    ])

    constraints = ConstraintBuilder([
        LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0},
                   sense=">=", rhs=45.0),                    # 硬约束
        LinearSpec({"I4.P1": 1.0, "I4.P2": 1.0},
                   sense=">=", rhs=50.0,
                   soft=True, penalty=3.0),                  # 软约束
    ])

    sol = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
        arterial,
        prior=s1,
        loss_builder=loss_builder,
        constraint_builder=constraints,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import SignalPlan


@dataclass
class LinearExpr:
    """const + Σ coefs[j] * phase_duration[j]。"""
    const: float = 0.0
    coefs: dict[int, float] = field(default_factory=dict)


@dataclass
class WindowExpr:
    """一个方案的上/下行绿灯窗边界，用相位变量线性表示。

    up_start / up_end / down_start / down_end 都是 LinearExpr：
        const + Σ coefs[j] * g[j]
    其中 j 是 plan.phases 的下标，g[j] 是第 j 个相位时长。
    """
    up_start: LinearExpr
    up_end: LinearExpr
    down_start: LinearExpr
    down_end: LinearExpr
    phases: list  # 该方案的 phases 列表


def _widest(windows):
    if not windows:
        raise ValueError("信控方案缺少绿灯窗口")
    return max(windows, key=lambda w: w.width)


def window_exprs(plan: SignalPlan, C: float) -> WindowExpr:
    """把方案转为相位变量线性表达式。

    - 若 plan.phases 非空：按 phases 顺序累计出 up_phase / down_phase 的
      绿灯起止时刻；
    - 否则：退回固定窗口（widest window），表达式退化为常数。
    """
    if plan.phases:
        idx = {ph.name: i for i, ph in enumerate(plan.phases)}
        if plan.up_phase not in idx:
            raise ValueError(f"方案 {plan.name} 缺少 up_phase")
        if plan.down_phase not in idx:
            raise ValueError(f"方案 {plan.name} 缺少 down_phase")

        def start_expr(phase_name: str) -> LinearExpr:
            i = idx[phase_name]
            return LinearExpr(const=0.0, coefs={j: 1.0 for j in range(i)})

        def end_expr(phase_name: str) -> LinearExpr:
            i = idx[phase_name]
            return LinearExpr(const=0.0, coefs={j: 1.0 for j in range(i + 1)})

        return WindowExpr(
            up_start=start_expr(plan.up_phase),
            up_end=end_expr(plan.up_phase),
            down_start=start_expr(plan.down_phase),
            down_end=end_expr(plan.down_phase),
            phases=list(plan.phases),
        )

    win_up = _widest(plan.up_windows)
    win_dn = _widest(plan.down_windows)
    return WindowExpr(
        up_start=LinearExpr(win_up.start * C),
        up_end=LinearExpr(win_up.end * C),
        down_start=LinearExpr(win_dn.start * C),
        down_end=LinearExpr(win_dn.end * C),
        phases=[],
    )


def expr_coefs(expr: LinearExpr, phase_indices: list[int]) -> dict[int, float]:
    """把表达式里的相位下标映射到 MILP 变量下标。"""
    return {phase_indices[j]: c for j, c in expr.coefs.items()}


@dataclass
class PhaseLossSpec:
    """单个相位的 hinge 损失配置。

    损失公式：
        loss = slope * max(0, threshold - g)
               + upper_slope * max(0, g - upper_threshold)

    参数：
        phase: 相位名，例如 "P1"；
        threshold: 过低惩罚阈值（秒）。g < threshold 时开始产生损失；
        slope: 过低惩罚的单位斜率（元/秒）；
        intersection: 可选。为 None 时对所有同名相位生效；
            填路口名时只惩罚该路口；
        upper_threshold: 可选。过高惩罚阈值（秒）。g > upper_threshold
            时开始产生损失；None 表示不惩罚过大；
        upper_slope: 可选。过高惩罚斜率；None 时复用 slope。

    注意：
        - 该损失属于“相位时长可调”阶段，即 PhaseTuneSolver / TwoStageSolver；
        - PhaseTuneSolver 会为每个 spec 创建连续变量 ℓ，并加入：
              ℓ >= threshold - g
              ℓ >= g - upper_threshold（如配置）
          最小化时 ℓ 自动收紧为 hinge 值。
    """
    phase: str
    threshold: float
    slope: float = 1.0
    intersection: str | None = None
    upper_threshold: float | None = None
    upper_slope: float | None = None


class PhaseLossBuilder:
    """把 PhaseLossSpec 列表转换成 MILP 变量/约束/目标表达式。

    用法：
        loss_builder = PhaseLossBuilder([
            PhaseLossSpec("P1", threshold=30, slope=2),
            PhaseLossSpec("P2", threshold=35, slope=1.5, intersection="I4",
                          upper_threshold=50, upper_slope=3.0),
        ])

    然后传给 PhaseTuneSolver.solve(..., loss_builder=loss_builder, ...)。
    如果同时要扫描帕累托前沿，则传给 EpsilonConstraintRunner(loss_builder=...)。
    """

    def __init__(self, specs: list[PhaseLossSpec] | None = None) -> None:
        self.specs = list(specs or [])

    def spec_for(self, phase_name: str,
                 intersection_name: str | None = None) -> PhaseLossSpec | None:
        """按相位名和路口名查找损失配置。

        优先返回 intersection 精确匹配的路口级配置；
        其次返回 intersection=None 的全局配置。
        当前实现按列表顺序返回第一个匹配项。
        """
        for s in self.specs:
            if s.phase != phase_name:
                continue
            if s.intersection is None or s.intersection == intersection_name:
                return s
        return None


@dataclass
class LinearSpec:
    """相位变量的线性组合约束声明。

    表示：
        Σ a_i * g_i  (sense)  rhs

    例如：
        {"I2.P1": 1.0, "I2.P2": 1.0} >= 45.0
    表示路口 I2 的 P1 + P2 之和不得小于 45 秒。

    terms 的 key 支持两种写法：
      - "I3.P1"：路口 I3 的相位 P1（路口级）；
      - "P1"：所有选中方案里的相位 P1（全局）。

    sense:
      - ">="：大于等于
      - "<="：小于等于
      - "="：等于（仅硬约束）

    soft=False:
        硬约束，直接加入 MILP。
    soft=True:
        自动创建 slack 变量 s，并把 penalty*s 汇入损失表达式；
        支持 sense=">="（缺口在下侧）和 sense="<="（缺口在上侧）；
        sense="=" 的软约束暂不支持，可拆成两条单侧软约束。
    """
    terms: dict[str, float]
    sense: str = ">="  # ">=" / "<=" / "="
    rhs: float = 0.0
    soft: bool = False
    penalty: float = 1.0


class ConstraintBuilder:
    """声明式约束集合，统一管理硬约束和软约束。

    用法：
        constraints = ConstraintBuilder([
            LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0},
                       sense=">=", rhs=60.0),                 # 硬约束
            LinearSpec({"I5.P1": 1.0, "I5.P2": 1.0},
                       sense=">=", rhs=90.0,
                       soft=True, penalty=3.0),               # 软约束
        ])

        sol = PhaseTuneSolver(...).solve(
            arterial,
            prior=s1,
            constraint_builder=constraints,
        )

    PhaseTuneSolver 会自动：
      - 把 terms 解析成相位变量下标；
      - 硬约束直接加一行；
      - 软约束创建 slack 变量并改写为带 slack 的行；
      - 把所有 soft penalty 汇入损失表达式，供 ε-约束扫描使用。
    """

    def __init__(self, specs: list[LinearSpec] | None = None) -> None:
        self.specs = list(specs or [])

    def add(self, spec: LinearSpec) -> None:
        """追加一条约束声明。"""
        self.specs.append(spec)


class AlignmentLossBuilder:
    """带边/带中心对齐损失配置。

    对每个路口（或 oneway 模式下的每个路段）定义：
        center = 带中心（现有变量的线性组合）
        a      = 到达高峰中心（常数）
        loss   = max(0, |center - a| - tolerance)

    本 builder 负责把上述 hinge 损失转成两条软 LinearSpec：
        center - slack_upper <= a + tolerance
        center + slack_lower >= a - tolerance
    然后复用 ConstraintBuilder 的软约束机制，自动进入总损失表达式。
    """

    def __init__(self,
                 targets_up: dict[str, float] | None = None,
                 targets_down: dict[str, float] | None = None,
                 tolerance: float = 0.0,
                 weight_up: float = 1.0,
                 weight_down: float = 1.0) -> None:
        self.targets_up = dict(targets_up or {})
        self.targets_down = dict(targets_down or {})
        self.tolerance = tolerance
        self.weight_up = weight_up
        self.weight_down = weight_down

    def to_linear_specs(self, mode: str,
                        int_names: list[str],
                        seg_names: list[str]) -> list[LinearSpec]:
        specs: list[LinearSpec] = []
        tol = self.tolerance

        # 上行：全局一条带，逐路口对齐
        for n in int_names:
            a = self.targets_up.get(n)
            if a is None:
                continue
            center = {f"tU_{n}": 1.0, "b_up": 0.5}
            specs.append(LinearSpec(center, sense="<=", rhs=a + tol,
                                    soft=True, penalty=self.weight_up))
            specs.append(LinearSpec(center, sense=">=", rhs=a - tol,
                                    soft=True, penalty=self.weight_up))

        # 下行
        if mode == "global":
            for n in int_names:
                a = self.targets_down.get(n)
                if a is None:
                    continue
                center = {f"tD_{n}": 1.0, "b_down": 0.5}
                specs.append(LinearSpec(center, sense="<=", rhs=a + tol,
                                        soft=True, penalty=self.weight_down))
                specs.append(LinearSpec(center, sense=">=", rhs=a - tol,
                                        soft=True, penalty=self.weight_down))
        elif mode == "oneway":
            # 下行带宽分段：按段中点近似
            for idx, sname in enumerate(seg_names):
                n0 = int_names[idx]
                n1 = int_names[idx + 1]
                a0 = self.targets_down.get(n0)
                a1 = self.targets_down.get(n1)
                if a0 is None or a1 is None:
                    continue
                a_mid = (a0 + a1) / 2.0
                center = {
                    f"tD_{n0}": 0.5,
                    f"tD_{n1}": 0.5,
                    f"bD_{sname}": 0.5,
                }
                specs.append(LinearSpec(center, sense="<=", rhs=a_mid + tol,
                                        soft=True, penalty=self.weight_down))
                specs.append(LinearSpec(center, sense=">=", rhs=a_mid - tol,
                                        soft=True, penalty=self.weight_down))
        else:
            raise ValueError(f"unknown mode: {mode}")
        return specs
