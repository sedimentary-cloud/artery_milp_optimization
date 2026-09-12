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
   - 相位 hinge / 软约束汇入 intersection_loss；
     AlignmentLossBuilder 的 alignment 进入 band_loss。

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
    """一个方案的上/下行候选绿灯窗集合。

    每个候选窗口是 (start_expr, end_expr)，其中表达式为：
        const + Σ coefs[j] * g[j]

    一个方向可能有多个候选窗口，例如多个相位都服务上行时，
    每个服务相位对应一个候选窗口。
    """
    up_windows: list[tuple[LinearExpr, LinearExpr]] = field(default_factory=list)
    down_windows: list[tuple[LinearExpr, LinearExpr]] = field(default_factory=list)
    phases: list = field(default_factory=list)  # 该方案的 phases 列表
    up_phase_names: list[str | None] = field(default_factory=list)
    down_phase_names: list[str | None] = field(default_factory=list)

    @property
    def up_start(self) -> LinearExpr:
        """兼容单窗口访问。多窗口时请使用 up_windows。"""
        if len(self.up_windows) != 1:
            raise RuntimeError("WindowExpr 有多个上行窗口，请使用 up_windows")
        return self.up_windows[0][0]

    @property
    def up_end(self) -> LinearExpr:
        if len(self.up_windows) != 1:
            raise RuntimeError("WindowExpr 有多个上行窗口，请使用 up_windows")
        return self.up_windows[0][1]

    @property
    def down_start(self) -> LinearExpr:
        if len(self.down_windows) != 1:
            raise RuntimeError("WindowExpr 有多个下行窗口，请使用 down_windows")
        return self.down_windows[0][0]

    @property
    def down_end(self) -> LinearExpr:
        if len(self.down_windows) != 1:
            raise RuntimeError("WindowExpr 有多个下行窗口，请使用 down_windows")
        return self.down_windows[0][1]


def _widest(windows):
    if not windows:
        raise ValueError("信控方案缺少绿灯窗口")
    return max(windows, key=lambda w: w.width)


def direction_phase_name(plan: SignalPlan, direction: str) -> str:
    """兼容接口：当某方向只有一个服务相位时返回该相位名。

    多窗口场景请使用 plan.direction_phase_names(direction)。
    """
    names = plan.direction_phase_names(direction)
    if len(names) != 1:
        raise NotImplementedError(
            f"方案 {plan.name} 的 {direction} 方向有多个服务相位: {names}；"
            f"请使用 direction_phase_names() 或 window_exprs().{direction}_windows。"
        )
    return names[0]


def phase_start_times(plan: SignalPlan, phase_times: dict[str, float] | None = None) -> dict[str, float]:
    """计算每个相位的开始时刻（秒），包含 phase_lost_times 常量。

    Args:
        plan: 信控方案。
        phase_times: 可选的相位时长覆盖，通常来自 Solution.phase_times。
            未提供时使用 Phase.green。
    """
    starts: dict[str, float] = {}
    acc = 0.0
    for ph in plan.phases:
        starts[ph.name] = acc
        g = (float(phase_times[ph.name])
             if phase_times is not None and ph.name in phase_times
             else float(ph.green))
        acc += g + float(plan.phase_lost_times.get(ph.name, 0.0))
    return starts


def window_exprs(plan: SignalPlan, C: float) -> WindowExpr:
    """把方案转为多个候选绿灯窗的相位变量线性表达式。

    - 若 plan.phases 非空：每个服务该方向的相位产生一个候选窗口；
      相位间损失 phase_lost_times 作为常数项加入后续相位的起点。
    - 若 plan.phases 为空：plan.up_windows / down_windows 中每个固定窗口
      都作为一个候选窗口，表达式退化为常数。
    """
    if plan.phases:
        idx = {ph.name: i for i, ph in enumerate(plan.phases)}

        # 校验 phase_lost_times 的 key 是否都是有效相位名。
        for pname in plan.phase_lost_times:
            if pname not in idx:
                raise ValueError(
                    f"方案 {plan.name} 的 phase_lost_times 含未知相位: {pname}"
                )

        def start_expr(phase_name: str) -> LinearExpr:
            i = idx[phase_name]
            const = sum(
                float(plan.phase_lost_times.get(plan.phases[j].name, 0.0))
                for j in range(i)
            )
            return LinearExpr(const=const, coefs={j: 1.0 for j in range(i)})

        def end_expr(phase_name: str) -> LinearExpr:
            i = idx[phase_name]
            start = start_expr(phase_name)
            coefs = dict(start.coefs)
            coefs[i] = coefs.get(i, 0.0) + 1.0
            return LinearExpr(const=start.const, coefs=coefs)

        up_names = plan.direction_phase_names("up")
        down_names = plan.direction_phase_names("down")
        up_windows = [(start_expr(name), end_expr(name)) for name in up_names]
        down_windows = [(start_expr(name), end_expr(name)) for name in down_names]

        return WindowExpr(
            up_windows=up_windows,
            down_windows=down_windows,
            phases=list(plan.phases),
            up_phase_names=list(up_names),
            down_phase_names=list(down_names),
        )

    if not plan.up_windows or not plan.down_windows:
        raise ValueError(f"方案 {plan.name} 缺少可用的固定绿灯窗口")

    up_windows = [
        (LinearExpr(w.start * C), LinearExpr(w.end * C))
        for w in plan.up_windows
    ]
    down_windows = [
        (LinearExpr(w.start * C), LinearExpr(w.end * C))
        for w in plan.down_windows
    ]
    return WindowExpr(
        up_windows=up_windows,
        down_windows=down_windows,
        phases=[],
        up_phase_names=[None] * len(up_windows),
        down_phase_names=[None] * len(down_windows),
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
