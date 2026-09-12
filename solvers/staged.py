"""两阶段求解：先选方案，再锁方案优化相位时长。

Stage 1: 现有 CompositeBandSolver / OneWayPrioritySolver，只负责确定 plan_choices。
Stage 2: PhaseTuneSolver 锁定方案后，把相位时长作为连续变量重新优化带宽。
TwoStageSolver 对外提供统一接口：stage2 失败时 fallback 到 stage1 的解。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..solution import Solution
from .base import Solver
from .phase import (AlignmentLossBuilder, ConstraintBuilder, LinearExpr,
                    LinearSpec, PhaseLossBuilder, PhaseLossSpec, WindowExpr,
                    expr_coefs, window_exprs)


class _LegacyPhaseTuneSolver(Solver):
    """第二阶段：锁定方案，优化相位时长和绿波带宽。

    mode:
        "global": 全局上下行带宽模型（对应 CompositeBandSolver 的第二阶段）；
        "oneway": 上行全局 + 下行分段/窗口带宽模型（对应 OneWayPrioritySolver）。
    """

    name = "phase-tune"

    def __init__(self,
                 mode: str = "global",
                 down_weight: float = 1.0,
                 up_weight: float = 1.0,
                 window_weights: dict[int, float] | None = None,
                 max_loops: int = 3,
                 objective_mode: str = "sum",
                 balance_eps: float = 0.1,
                 balance_terms: tuple[str, ...] = ("up", "down"),
                 tunable_intersections: set[str] | None = None) -> None:
        self.mode = mode
        self.down_weight = down_weight
        self.up_weight = up_weight
        raw = window_weights or {2: 1.0}
        self.window_weights = {k: w for k, w in raw.items() if 2 <= k <= 3}
        if not self.window_weights:
            self.window_weights = {2: 1.0}
        self.max_loops = max_loops
        # global 模式下的带宽目标：
        #   "sum"                  max b_up + down_weight * b_down
        #   "balanced"             max B_bal = min(b_up, b_down)
        #   "balanced_composite"   max b_up + b_down + balance_eps * B_bal
        self.objective_mode = objective_mode
        self.balance_eps = balance_eps
        self.balance_terms = tuple(balance_terms)
        # 只允许这些路口调相位；None 表示所有路口都可调。
        self.tunable_intersections = tunable_intersections

    def solve(self,
              arterial,
              prior: Solution | None = None,
              loss_builder: PhaseLossBuilder | None = None,
              constraint_builder: ConstraintBuilder | None = None,
              alignment_builder: AlignmentLossBuilder | None = None,
              max_loss: float | None = None,
              objective: str = "bandwidth") -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)

        # 锁定方案：从 prior.plan_choices 读取；没有 prior 时退回第一个方案。
        selected = []
        for inter in ints:
            name = prior.plan_choices.get(inter.name) if prior else None
            plan = inter.plan_by_name(name) if name else inter.plans[0]
            selected.append(plan)

        int_names = [v.name for v in ints]
        name_to_i = {n: i for i, n in enumerate(int_names)}
        seg_name_to_idx = {s.name: i for i, s in enumerate(segs)}
        seg_names = [s.name for s in segs]

        tunable = [
            self.tunable_intersections is None or name in self.tunable_intersections
            for name in int_names
        ]

        exprs: list[WindowExpr] = []
        for i, plan in enumerate(selected):
            if tunable[i]:
                exprs.append(window_exprs(plan, C))
            else:
                wu = max(plan.up_windows, key=lambda w: w.width)
                wd = max(plan.down_windows, key=lambda w: w.width)
                exprs.append(WindowExpr(
                    up_start=LinearExpr(wu.start * C),
                    up_end=LinearExpr(wu.end * C),
                    down_start=LinearExpr(wd.start * C),
                    down_end=LinearExpr(wd.end * C),
                    phases=[],
                ))

        # ---------------- 变量布局 ----------------
        if self.mode == "global":
            idx_bu, idx_bd = 0, 1
            idx_tU = 2
            idx_tD = 2 + n
            idx_mU = 2 + 2 * n
            idx_mD = 2 + 2 * n + m
            base = 2 + 2 * n + 2 * m
            idx_bD = None
            idx_win = {}
        elif self.mode == "oneway":
            idx_bu = 0
            idx_bd = None
            idx_tU = 1
            idx_mU = 1 + n
            idx_tD = 1 + n + m
            idx_mD = 1 + 2 * n + m
            idx_bD = 1 + 2 * n + 2 * m
            idx_win: dict[tuple[int, int], int] = {}
            cur = idx_bD + m
            for k in self.window_weights:
                if k != 3:
                    continue
                for j in range(n - k + 1):
                    idx_win[(k, j)] = cur
                    cur += 1
            base = cur
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        # 均衡目标变量 B_bal（仅 global 模式）
        idx_bal = None
        if self.mode == "global" and self.objective_mode in ("balanced", "balanced_composite"):
            idx_bal = base
            base += 1

        phase_idx: list[list[int]] = []
        cur = base
        for i in range(n):
            if tunable[i]:
                cnt = len(selected[i].phases)
                phase_idx.append(list(range(cur, cur + cnt)))
                cur += cnt
            else:
                phase_idx.append([])

        # hinge 损失变量：ℓ_{i,p}
        loss_vars: list[tuple[int, int, int, PhaseLossSpec, str]] = []
        fixed_loss = 0.0
        if loss_builder is not None:
            for i, plan in enumerate(selected):
                for p, ph in enumerate(plan.phases):
                    spec = loss_builder.spec_for(ph.name, int_names[i])
                    if spec is None:
                        continue
                    if tunable[i]:
                        loss_vars.append((i, p, cur, spec, "lower"))
                        cur += 1
                        if spec.upper_threshold is not None:
                            loss_vars.append((i, p, cur, spec, "upper"))
                            cur += 1
                    else:
                        g = ph.green
                        fixed_loss += spec.slope * max(0.0, spec.threshold - g)
                        if spec.upper_threshold is not None:
                            slope_up = spec.upper_slope or spec.slope
                            fixed_loss += slope_up * max(0.0, g - spec.upper_threshold)

        # 声明式线性约束：硬约束直接加行，软约束额外创建 slack 变量。
        # 对齐损失 builder 自动转成软 LinearSpec，与用户显式约束合并。
        combined_specs: list[LinearSpec] = []
        if constraint_builder is not None:
            combined_specs.extend(constraint_builder.specs)
        if alignment_builder is not None:
            combined_specs.extend(
                alignment_builder.to_linear_specs(self.mode, int_names, seg_names)
            )
        combined_constraints = ConstraintBuilder(combined_specs) if combined_specs else None

        resolved_constraints: list[tuple[LinearSpec, list[tuple[int, float]], int | None]] = []
        if combined_constraints is not None:
            for spec in combined_constraints.specs:
                terms: list[tuple[int, float]] = []
                for name, coef in spec.terms.items():
                    # 带宽变量
                    if name in ("b_up", "b_down", "B_bal"):
                        var = None
                        if name == "b_up":
                            var = idx_bu
                        elif name == "b_down":
                            var = idx_bd
                        elif name == "B_bal":
                            var = idx_bal
                        if var is not None:
                            terms.append((var, coef))
                        continue
                    # 分段下行带宽：bD_seg1
                    if name.startswith("bD_"):
                        if idx_bD is not None:
                            sname = name[3:]
                            si = seg_name_to_idx.get(sname)
                            if si is not None:
                                terms.append((idx_bD + si, coef))
                        continue
                    # 带前沿到达时刻：tU_I2 / tD_I2
                    if name.startswith("tU_"):
                        iname = name[3:]
                        i = name_to_i.get(iname)
                        if i is not None:
                            terms.append((idx_tU + i, coef))
                        continue
                    if name.startswith("tD_"):
                        iname = name[3:]
                        i = name_to_i.get(iname)
                        if i is not None:
                            terms.append((idx_tD + i, coef))
                        continue
                    # 相位变量：带 "." 表示路口.相位，不带则对所有同名相位生效
                    if "." in name:
                        iname, pname = name.split(".", 1)
                        i = name_to_i.get(iname)
                        if i is None:
                            continue
                        p = next((idx for idx, ph in enumerate(selected[i].phases)
                                  if ph.name == pname), None)
                        if p is not None:
                            terms.append((phase_idx[i][p], coef))
                    else:
                        pname = name
                        for i, plan in enumerate(selected):
                            p = next((idx for idx, ph in enumerate(plan.phases)
                                      if ph.name == pname), None)
                            if p is not None:
                                terms.append((phase_idx[i][p], coef))
                if not terms:
                    continue
                slack = None
                if spec.soft:
                    slack = cur
                    cur += 1
                resolved_constraints.append((spec, terms, slack))
        nvar = cur

        # ---------------- 目标 ----------------
        c = np.zeros(nvar)
        if objective == "loss":
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                c[var] = slope
            for spec, _, slack in resolved_constraints:
                if spec.soft and slack is not None:
                    c[slack] = c[slack] + spec.penalty
        elif self.mode == "global":
            if self.objective_mode == "balanced":
                c[idx_bal] = -1.0
            elif self.objective_mode == "balanced_composite":
                c[idx_bu] = -1.0
                c[idx_bd] = -self.down_weight
                c[idx_bal] = -self.balance_eps
            else:
                c[idx_bu], c[idx_bd] = -1.0, -self.down_weight
        else:
            c[idx_bu] = -self.up_weight
            w2 = self.window_weights.get(2, 0.0)
            c[idx_bD:idx_bD + m] = -w2
            for (k, j), v in idx_win.items():
                c[v] = -self.window_weights[k]

        # ---------------- 变量界 ----------------
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C
        if self.mode == "global":
            ub[idx_bu] = C
            ub[idx_bd] = C
            if idx_bal is not None:
                ub[idx_bal] = C
        else:
            ub[idx_bu] = C
            ub[idx_bD:idx_bD + m] = C
            for v in idx_win.values():
                ub[v] = C
        lb[idx_mU:idx_mU + m] = -self.max_loops
        ub[idx_mU:idx_mU + m] = self.max_loops
        lb[idx_mD:idx_mD + m] = -self.max_loops
        ub[idx_mD:idx_mD + m] = self.max_loops
        for row in phase_idx:
            ub[row] = C
        for _, _, var, _, _ in loss_vars:
            ub[var] = C
        for _, _, slack in resolved_constraints:
            if slack is not None:
                ub[slack] = C

        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1

        rows, lo_list, hi_list = [], [], []

        def add_row(coefs: dict[int, float], lo: float, hi: float) -> None:
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] = v
            rows.append(row)
            lo_list.append(lo)
            hi_list.append(hi)

        # ---------------- 相位自身约束 ----------------
        for i, plan in enumerate(selected):
            if not tunable[i]:
                continue
            for p, ph in enumerate(plan.phases):
                add_row({phase_idx[i][p]: 1.0}, ph.min_green, ph.max_green)
            if plan.phases:
                total = {phase_idx[i][p]: 1.0 for p in range(len(plan.phases))}
                add_row(total, C - plan.lost_time, C - plan.lost_time)

        # ---------------- 均衡目标约束 ----------------
        if idx_bal is not None:
            if "up" in self.balance_terms:
                add_row({idx_bal: 1.0, idx_bu: -1.0}, -np.inf, 0.0)
            if "down" in self.balance_terms:
                add_row({idx_bal: 1.0, idx_bd: -1.0}, -np.inf, 0.0)

        # ---------------- 相位 hinge 损失 ----------------
        # 过小惩罚：ℓ >= threshold - g   ->   ℓ + g >= threshold
        # 过大惩罚：ℓ >= g - upper_threshold -> ℓ - g >= -upper_threshold
        for i, p, var, spec, side in loss_vars:
            if side == "lower":
                add_row({var: 1.0, phase_idx[i][p]: 1.0}, spec.threshold, np.inf)
            else:
                add_row({var: 1.0, phase_idx[i][p]: -1.0},
                        -spec.upper_threshold, np.inf)

        # ---------------- 声明式线性约束 ----------------
        for spec, terms, slack in resolved_constraints:
            row_coefs = {var: coef for var, coef in terms}
            if not spec.soft:
                if spec.sense == ">=":
                    add_row(row_coefs, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row(row_coefs, -np.inf, spec.rhs)
                elif spec.sense == "=":
                    add_row(row_coefs, spec.rhs, spec.rhs)
                else:
                    raise ValueError(f"unknown sense: {spec.sense}")
            else:
                if spec.sense == ">=":
                    add_row({**row_coefs, slack: 1.0}, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row({**row_coefs, slack: -1.0}, -np.inf, spec.rhs)
                else:
                    raise ValueError("soft '=' constraint is not supported; "
                                     "use two one-sided soft constraints")

        if max_loss is not None:
            loss_coefs = {}
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                loss_coefs[var] = slope
            for spec, _, slack in resolved_constraints:
                if spec.soft and slack is not None:
                    loss_coefs[slack] = loss_coefs.get(slack, 0.0) + spec.penalty
            add_row(loss_coefs, -np.inf, max_loss - fixed_loss)

        # ---------------- 绿波传递约束 ----------------
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # ---------------- 绿灯窗约束 ----------------
        def lower_row(t_var: int, expr: LinearExpr, i: int) -> None:
            """t_var >= expr.const + Σ coefs*g -> t_var - Σ coefs*g >= const。"""
            coefs = {t_var: 1.0}
            coefs.update({phase_idx[i][j]: -v for j, v in expr.coefs.items()})
            add_row(coefs, expr.const, np.inf)

        def upper_row(t_var: int, b_var: int, expr: LinearExpr, i: int) -> None:
            """t_var + b_var <= expr.const + Σ coefs*g。"""
            coefs = {t_var: 1.0, b_var: 1.0}
            coefs.update({phase_idx[i][j]: -v for j, v in expr.coefs.items()})
            add_row(coefs, -np.inf, expr.const)

        if self.mode == "global":
            for i, expr in enumerate(exprs):
                lower_row(idx_tU + i, expr.up_start, i)
                upper_row(idx_tU + i, idx_bu, expr.up_end, i)
                lower_row(idx_tD + i, expr.down_start, i)
                upper_row(idx_tD + i, idx_bd, expr.down_end, i)
        else:
            # 上行全局带
            for i, expr in enumerate(exprs):
                lower_row(idx_tU + i, expr.up_start, i)
                upper_row(idx_tU + i, idx_bu, expr.up_end, i)
            # 下行分段带
            for i, expr in enumerate(exprs):
                lower_row(idx_tD + i, expr.down_start, i)
            for i in range(m):
                upper_row(idx_tD + i, idx_bD + i, exprs[i].down_end, i)
                upper_row(idx_tD + i + 1, idx_bD + i, exprs[i + 1].down_end, i + 1)
            # 窗口带宽：B <= bD_i
            for (k, j), v in idx_win.items():
                for i in range(j, j + k - 1):
                    add_row({v: 1.0, idx_bD + i: -1.0}, -np.inf, 0.0)

        res = milp(c=c,
                   constraints=LinearConstraint(np.array(rows), lo_list, hi_list),
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        seg_names = [s.name for s in segs]
        int_names = [v.name for v in ints]

        # 方案选择和相位时长
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}
        sol.phase_times = {}
        for i, plan in enumerate(selected):
            if plan.phases:
                sol.phase_times[int_names[i]] = {
                    ph.name: float(x[j])
                    for ph, j in zip(plan.phases, phase_idx[i])
                }

        if self.mode == "global":
            sol.band_up_style = "global"
            sol.band_down_style = "global"
            sol.bandwidth_up = {name: float(x[idx_bu]) for name in seg_names}
            sol.bandwidth_down = {name: float(x[idx_bd]) for name in seg_names}
        else:
            sol.band_up_style = "global"
            sol.band_down_style = "local"
            sol.bandwidth_up = {name: float(x[idx_bu]) for name in seg_names}
            sol.bandwidth_down = {seg_names[i]: float(x[idx_bD + i]) for i in range(m)}
            for i in range(m):
                sol.window_bands[f"win2@{int_names[i]}-{int_names[i + 1]}"] = float(x[idx_bD + i])
            for (k, j), v in idx_win.items():
                key = f"win{k}@{int_names[j]}-{int_names[j + k - 1]}"
                sol.window_bands[key] = float(x[v])

        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}
        sol.objective = -float(res.fun)
        total_loss = fixed_loss
        for i, p, _, spec, side in loss_vars:
            g = float(x[phase_idx[i][p]])
            if side == "lower":
                viol = max(0.0, spec.threshold - g)
                slope = spec.slope
            else:
                viol = max(0.0, g - spec.upper_threshold)
                slope = spec.upper_slope or spec.slope
            total_loss += slope * viol
        for spec, terms, slack in resolved_constraints:
            if not spec.soft:
                continue
            expr_val = sum(coef * float(x[var])
                           for var, coef in terms)
            if spec.sense == ">=":
                total_loss += spec.penalty * max(0.0, spec.rhs - expr_val)
            elif spec.sense == "<=":
                total_loss += spec.penalty * max(0.0, expr_val - spec.rhs)
        sol.total_phase_loss = float(total_loss)
        sol.status = "optimal" if res.success else res.message
        return sol


class TwoStageSolver(Solver):
    """编排器：Stage1 选方案，Stage2 锁方案优化相位；失败时 fallback。"""

    name = "two-stage"

    def __init__(self,
                 stage1: Solver,
                 mode: str = "global",
                 loss_builder: PhaseLossBuilder | None = None,
                 constraint_builder: ConstraintBuilder | None = None,
                 **tune_kwargs) -> None:
        self.stage1 = stage1
        self.mode = mode
        self.loss_builder = loss_builder
        self.constraint_builder = constraint_builder
        self.tune_kwargs = tune_kwargs

    def solve(self, arterial) -> Solution:
        s1 = self.stage1.solve(arterial)
        try:
            from .flexible_phase_solver import PhaseTuneSolver as _NewPhaseTuneSolver
            tuner = _NewPhaseTuneSolver(mode=self.mode, **self.tune_kwargs)
            s2 = tuner.solve(arterial, prior=s1,
                             loss_builder=self.loss_builder,
                             constraint_builder=self.constraint_builder)
            if s2.status == "optimal":
                return s2
        except Exception:
            pass
        s1.status = f"{s1.status}|stage1_fallback"
        return s1


class EpsilonConstraintRunner:
    """ε-约束法扫描 PhaseTuneSolver 的带宽-损失帕累托前沿。

    用法：
        runner = EpsilonConstraintRunner(
            mode="global",
            loss_builder=PhaseLossBuilder([...]),
            n_points=10,
        )
        frontier = runner.run(arterial, prior=s1)
        # frontier: list[(eps, bandwidth, loss, Solution)]
    """

    def __init__(self,
                 mode: str = "global",
                 loss_builder: PhaseLossBuilder | None = None,
                 constraint_builder: ConstraintBuilder | None = None,
                 alignment_builder: AlignmentLossBuilder | None = None,
                 n_points: int = 5,
                 down_weight: float = 1.0,
                 up_weight: float = 1.0,
                 window_weights: dict[int, float] | None = None,
                 max_loops: int = 3,
                 metric: str | None = None,
                 objective_mode: str = "sum",
                 balance_eps: float = 0.1,
                 balance_terms: tuple[str, ...] = ("up", "down")) -> None:
        self.mode = mode
        self.loss_builder = loss_builder
        self.constraint_builder = constraint_builder
        self.alignment_builder = alignment_builder
        self.n_points = n_points
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.window_weights = window_weights
        self.max_loops = max_loops
        # metric: 帕累托前沿 x 轴的带宽口径。
        # 默认由 objective_mode 自动推导，保证“优化目标”和“展示口径”同源；
        # 仅当用户确实想报告另一个指标时才手动覆盖。
        if metric is None:
            if objective_mode == "balanced":
                metric = "balanced"
            elif objective_mode == "balanced_composite":
                metric = "objective"
            else:
                metric = "sum"
        self.metric = metric
        self.objective_mode = objective_mode
        self.balance_eps = balance_eps
        self.balance_terms = tuple(balance_terms)
        self.frontier: list[tuple[float, float, float, Solution]] = []

    def _make_tuner(self) -> PhaseTuneSolver:
        from .flexible_phase_solver import PhaseTuneSolver as _NewPhaseTuneSolver
        return _NewPhaseTuneSolver(mode=self.mode,
                               down_weight=self.down_weight,
                               up_weight=self.up_weight,
                               window_weights=self.window_weights,
                               max_loops=self.max_loops,
                               objective_mode=self.objective_mode,
                               balance_eps=self.balance_eps,
                               balance_terms=self.balance_terms)

    def _bandwidth(self, sol: Solution) -> float:
        if self.mode == "global":
            bu = next(iter(sol.bandwidth_up.values()), 0.0)
            bd = next(iter(sol.bandwidth_down.values()), 0.0)
            if self.metric == "balanced":
                return float(min(bu, bd))
            if self.metric == "objective":
                return float(sol.objective)
            return float(bu + bd)
        return float(sol.objective)

    def run(self, arterial, prior: Solution) -> list[tuple[float, float, float, Solution]]:
        tuner = self._make_tuner()

        # P1: 最大带宽
        s_hi = tuner.solve(arterial, prior=prior, loss_builder=self.loss_builder,
                           constraint_builder=self.constraint_builder,
                           alignment_builder=self.alignment_builder,
                           objective="bandwidth")
        if s_hi.status != "optimal":
            return []
        L_hi = s_hi.total_phase_loss

        # P2: 最小损失
        s_lo = tuner.solve(arterial, prior=prior, loss_builder=self.loss_builder,
                           constraint_builder=self.constraint_builder,
                           alignment_builder=self.alignment_builder,
                           objective="loss")
        if s_lo.status != "optimal":
            return []

        L_lo = s_lo.total_phase_loss
        eps_values = list(np.linspace(L_lo, L_hi, max(self.n_points, 2)))

        frontier: list[tuple[float, float, float, Solution]] = []
        for eps in eps_values:
            s = tuner.solve(arterial, prior=prior, loss_builder=self.loss_builder,
                            constraint_builder=self.constraint_builder,
                            alignment_builder=self.alignment_builder,
                            max_loss=eps, objective="bandwidth")
            if s.status == "optimal":
                frontier.append((float(eps), self._bandwidth(s),
                                 s.total_phase_loss, s))

        self.frontier = frontier
        return frontier

    def knee_point(self) -> tuple[float, float, Solution] | None:
        """简单拐点启发：到两端点连线距离最大的前沿点。"""
        if len(self.frontier) < 3:
            return None
        # 端点为 (带宽, 损失)
        p0 = (self._bandwidth(self.frontier[0][3]), self.frontier[0][2])
        p1 = (self._bandwidth(self.frontier[-1][3]), self.frontier[-1][2])
        best = None
        best_d = -1.0
        for _, b, l, sol in self.frontier:
            d = abs((p1[0] - p0[0]) * (p0[1] - l)
                    - (p0[0] - b) * (p1[1] - p0[1]))
            if d > best_d:
                best_d = d
                best = (b, l, sol)
        if best is None:
            return None
        return best
