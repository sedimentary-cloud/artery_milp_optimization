"""FullFlexiblePhaseTuneSolver：相位变量 + BandModel + ObjectiveConfig。

这是新架构的完整第二阶段求解器（第一版）：
- 从 prior 锁定方案；
- 为每个相位创建连续变量 g；
- 相位约束：min/max、Σg + lost_time = C；
- 窗口边界由相位变量线性表达；
- 基础段带宽 b[d,i] 落在两端窗口内；
- 窗口带格 B[d,k,j] <= b[d,i]；
- 目标由 ObjectiveConfig 决定。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..models import Arterial
from ..solution import Solution
from .band_model import BandModel
from .base import Solver
from .objective_config import ObjectiveConfig
from .phase import window_exprs


class FullFlexiblePhaseTuneSolver(Solver):
    """相位变量 + 带格 + 目标配置。"""

    name = "full-flexible-phase"

    def __init__(self,
                 config: ObjectiveConfig,
                 max_loops: int = 3,
                 up_global_output: bool = True,
                 down_global_output: bool = False) -> None:
        self.config = config
        self.max_loops = max_loops
        self.up_global_output = up_global_output
        self.down_global_output = down_global_output

    def solve(self, arterial: Arterial, prior: Solution | None = None) -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)
        self.config.validate(n)

        # 锁定方案
        selected = []
        for inter in ints:
            name = prior.plan_choices.get(inter.name) if prior and prior.plan_choices else None
            plan = inter.plan_by_name(name) if name else inter.plans[0]
            selected.append(plan)

        exprs = [window_exprs(p, C) for p in selected]
        n_phases = [len(p.phases) for p in selected]

        # ---------------- 变量布局 ----------------
        cur = 0
        idx_g: list[list[int]] = []
        for i in range(n):
            idx_g.append(list(range(cur, cur + n_phases[i])))
            cur += n_phases[i]

        idx_tU = cur; cur += n
        idx_mU = cur; cur += m
        idx_tD = cur; cur += n
        idx_mD = cur; cur += m
        idx_bU = cur; cur += m
        idx_bD = cur; cur += m

        band_model = BandModel(n, m)
        band_offset = cur
        cur += band_model.nvar

        balance_vars: dict[int, int] = {}
        for gidx in range(len(self.config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1
        nvar = cur

        # ---------------- 目标 ----------------
        c = np.zeros(nvar)

        def band_var_by_key(key: str) -> int:
            from .objective_config import parse_band_key
            band = parse_band_key(key, n)
            return band_offset + band_model.var_of(band)

        for group in self.config.sum_groups:
            for key, weight in group.terms.items():
                c[band_var_by_key(key)] += -weight

        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            c[gvar] += -group.weight
            if group.eps > 0:
                for member in group.members:
                    c[band_var_by_key(member)] += -group.weight * group.eps

        # ---------------- 变量界 ----------------
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        for i, plan in enumerate(selected):
            for p, ph in enumerate(plan.phases):
                lb[idx_g[i][p]] = ph.min_green
                ub[idx_g[i][p]] = ph.max_green
        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C
        lb[idx_mU:idx_mU + m] = -self.max_loops
        ub[idx_mU:idx_mU + m] = self.max_loops
        lb[idx_mD:idx_mD + m] = -self.max_loops
        ub[idx_mD:idx_mD + m] = self.max_loops
        ub[idx_bU:idx_bU + m] = C
        ub[idx_bD:idx_bD + m] = C
        for gvar in balance_vars.values():
            ub[gvar] = C

        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1

        rows, lo_list, hi_list = [], [], []

        def add_row(coefs, lo, hi):
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] = v
            rows.append(row)
            lo_list.append(lo)
            hi_list.append(hi)

        def expr_coefs(expr, phase_indices):
            return {phase_indices[j]: v for j, v in expr.coefs.items()}

        # ---------------- 相位约束 ----------------
        for i, plan in enumerate(selected):
            if not plan.phases:
                continue
            total = {idx_g[i][p]: 1.0 for p in range(len(plan.phases))}
            add_row(total, C - plan.lost_time, C - plan.lost_time)

        # ---------------- 带前沿传递 ----------------
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # ---------------- 窗口约束 ----------------
        # t_i >= start_expr  ->  t_i - Σ c*g >= const
        # t_i + b_i <= end_expr -> t_i + b_i - Σ c*g <= const
        for i, expr in enumerate(exprs):
            gcoefs = expr_coefs(expr.up_start, idx_g[i])
            add_row({idx_tU + i: 1.0, **{k: -v for k, v in gcoefs.items()}},
                    expr.up_start.const, np.inf)
            gcoefs = expr_coefs(expr.up_end, idx_g[i])
            add_row({idx_tU + i: 1.0, idx_bU + i: 1.0,
                     **{k: -v for k, v in gcoefs.items()}},
                    -np.inf, expr.up_end.const)
            gcoefs = expr_coefs(expr.down_start, idx_g[i])
            add_row({idx_tD + i: 1.0, **{k: -v for k, v in gcoefs.items()}},
                    expr.down_start.const, np.inf)
            gcoefs = expr_coefs(expr.down_end, idx_g[i])
            add_row({idx_tD + i: 1.0, idx_bD + i: 1.0,
                     **{k: -v for k, v in gcoefs.items()}},
                    -np.inf, expr.down_end.const)

        # ---------------- 带格约束 ----------------
        for d in ("up", "down"):
            b_idx = idx_bU if d == "up" else idx_bD
            for k in range(2, n + 1):
                for j in range(n - k + 1):
                    B_var = band_offset + band_model.B_idx[d][k][j]
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, b_idx + i: -1.0}, -np.inf, 0.0)

        # ---------------- 均衡组约束 ----------------
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                add_row({gvar: 1.0, band_var_by_key(member): -1.0}, -np.inf, 0.0)

        res = milp(c=c,
                   constraints=LinearConstraint(np.array(rows), lo_list, hi_list),
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        int_names = [v.name for v in ints]
        seg_names = [s.name for s in segs]
        sol.objective = -float(res.fun)
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}
        sol.phase_times = {}
        for i, plan in enumerate(selected):
            if plan.phases:
                sol.phase_times[int_names[i]] = {
                    ph.name: float(x[idx_g[i][p]])
                    for p, ph in enumerate(plan.phases)
                }

        # 带宽输出：默认全局带用 B[n,0]，分段带用 b_i
        if self.up_global_output:
            b_up_global = float(x[band_offset + band_model.B_idx["up"][n][0]])
            sol.bandwidth_up = {name: b_up_global for name in seg_names}
        else:
            sol.bandwidth_up = {seg_names[i]: float(x[idx_bU + i]) for i in range(m)}

        if self.down_global_output:
            b_down_global = float(x[band_offset + band_model.B_idx["down"][n][0]])
            sol.bandwidth_down = {name: b_down_global for name in seg_names}
        else:
            sol.bandwidth_down = {seg_names[i]: float(x[idx_bD + i]) for i in range(m)}

        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}
        sol.status = "optimal" if res.success else res.message
        return sol
