"""FullFlexiblePhaseTuneSolver：相位变量 + BandModel + ObjectiveConfig。

新架构完整第二阶段：
- 从 prior 锁定方案；
- 为每个相位创建变量 g；不可调路口的 g 用上下界固定为初始值；
- 相位约束：min/max、Σg + lost_time = C；
- 窗口边界由 g 线性表达；
- 基础段带宽 b[d,i] 与窗口带格 B[d,k,j] 都由 BandModel 提供；
- 目标由 ObjectiveConfig 决定；
- 支持 loss_builder / constraint_builder / alignment_builder / max_loss。

模型一句话：
    g -> 绿灯窗
    绿灯窗 + t + b -> 基础段带宽
    b -> B (B[d,k,j] <= b[d,i] for i in j ... j+k-1)
    ObjectiveConfig -> 目标
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..models import Arterial
from ..solution import Solution
from .band_model import BandModel
from .base import Solver
from .objective_config import ObjectiveConfig, parse_band_key
from .phase import (ConstraintBuilder, LinearExpr, LinearSpec,
                    PhaseLossBuilder, PhaseLossSpec, window_exprs)


class FullFlexiblePhaseTuneSolver(Solver):
    """相位变量 + BandModel + ObjectiveConfig + 损失/约束。"""

    name = "full-flexible-phase"

    def __init__(self,
                 config: ObjectiveConfig,
                 max_loops: int = 3,
                 mode: str = "global",
                 up_global_output: bool = True,
                 down_global_output: bool = False,
                 up_style: str | None = None,
                 down_style: str | None = None) -> None:
        self.config = config
        self.max_loops = max_loops
        self.mode = mode
        self.up_global_output = up_global_output
        self.down_global_output = down_global_output
        self.up_style = up_style or ("global" if up_global_output else "local")
        self.down_style = down_style or ("global" if down_global_output else "local")

    # ------------------------------------------------------------------
    # 辅助：下界/上界约束
    # ------------------------------------------------------------------
    @staticmethod
    def _lower_phase_row(var: int, expr: LinearExpr,
                         phase_indices: list[int]) -> dict[int, float]:
        """生成 var >= expr 的行系数：var - Σ coef_j*g_j >= const。"""
        row = {var: 1.0}
        for j, coef in expr.coefs.items():
            row[phase_indices[j]] = row.get(phase_indices[j], 0.0) - coef
        return row

    @staticmethod
    def _upper_phase_row(t_var: int, b_var: int, expr: LinearExpr,
                         phase_indices: list[int]) -> dict[int, float]:
        """生成 t_var + b_var <= expr 的行系数。"""
        row = {t_var: 1.0, b_var: 1.0}
        for j, coef in expr.coefs.items():
            row[phase_indices[j]] = row.get(phase_indices[j], 0.0) - coef
        return row

    def solve(self, arterial: Arterial,
              prior: Solution | None = None,
              loss_builder: PhaseLossBuilder | None = None,
              constraint_builder: ConstraintBuilder | None = None,
              alignment_builder=None,
              max_loss: float | None = None,
              objective: str = "bandwidth",
              tunable_intersections: set[str] | None = None) -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)
        if n < 2:
            raise ValueError("FullFlexiblePhaseTuneSolver 至少需要两个路口")
        self.config.validate(n)

        int_names = [v.name for v in ints]
        name_to_i = {name: i for i, name in enumerate(int_names)}
        seg_names = [s.name for s in segs]
        seg_name_to_idx = {name: i for i, name in enumerate(seg_names)}

        # ------------------------------------------------------------------
        # 0) 锁定方案，并把每个方案的绿灯窗写成相位变量的线性表达式
        # ------------------------------------------------------------------
        selected = []
        for inter in ints:
            if not inter.plans:
                raise ValueError(f"路口 {inter.name} 没有可选方案")
            name = (prior.plan_choices.get(inter.name)
                    if prior and prior.plan_choices else None)
            plan = inter.plan_by_name(name) if name else inter.plans[0]
            selected.append(plan)

        exprs = [window_exprs(plan, C) for plan in selected]
        n_phases = [len(plan.phases) for plan in selected]
        tunable = [
            (tunable_intersections is None or name in tunable_intersections)
            for name in int_names
        ]

        # ------------------------------------------------------------------
        # 1) 变量布局
        #    g_{i,p} -> tU -> mU -> tD -> mD -> BandModel(b, B) -> balance -> loss -> slack
        # ------------------------------------------------------------------
        cur = 0
        idx_g: list[list[int]] = []
        for i in range(n):
            idx_g.append(list(range(cur, cur + n_phases[i])))
            cur += n_phases[i]

        idx_tU = cur; cur += n
        idx_mU = cur; cur += m
        idx_tD = cur; cur += n
        idx_mD = cur; cur += m

        # BandModel 自己同时提供基础段 b[d,i] 和窗口带 B[d,k,j] 的局部索引。
        band_model = BandModel(n, m)
        band_offset = cur
        idx_bU = band_offset + band_model.b_idx["up"][0]
        idx_bD = band_offset + band_model.b_idx["down"][0]
        cur += band_model.nvar

        balance_vars: dict[int, int] = {}
        for gidx in range(len(self.config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1

        # hinge 损失变量 ℓ_{i,p,side}
        loss_vars: list[tuple[int, int, int, PhaseLossSpec, str]] = []
        if loss_builder is not None:
            for i, plan in enumerate(selected):
                for p, ph in enumerate(plan.phases):
                    spec = loss_builder.spec_for(ph.name, int_names[i])
                    if spec is None:
                        continue
                    loss_vars.append((i, p, cur, spec, "lower"))
                    cur += 1
                    if spec.upper_threshold is not None:
                        loss_vars.append((i, p, cur, spec, "upper"))
                        cur += 1

        # ------------------------------------------------------------------
        # 2) 解析声明式约束（LinearSpec / AlignmentLossBuilder 转换来的软约束）
        # ------------------------------------------------------------------
        def band_var(key: str) -> int:
            band = parse_band_key(key, n)
            return band_offset + band_model.var_of(band)

        def resolve_special(name: str) -> int | None:
            """把 b_up / tU_I2 / bD_seg1 / B_bal 等特殊名字映射到变量。"""
            if name == "b_up":
                return band_var("up.global")
            if name == "b_down":
                return band_var("down.global")
            if name == "B_bal":
                return balance_vars.get(0)

            if name.startswith("tU_") or name.startswith("tD_"):
                iname = name[3:]
                i = name_to_i.get(iname)
                if i is None and iname.startswith("I"):
                    try:
                        i = int(iname[1:]) - 1
                    except ValueError:
                        i = None
                if i is None or not (0 <= i < n):
                    return None
                return (idx_tU + i) if name.startswith("tU_") else (idx_tD + i)

            if name.startswith("bD_"):
                sname = name[3:]
                si = seg_name_to_idx.get(sname)
                if si is None and sname.startswith("seg"):
                    try:
                        si = int(sname[3:]) - 1
                    except ValueError:
                        si = None
                if si is None or not (0 <= si < m):
                    return None
                return idx_bD + si

            return None

        def resolve_phase(name: str) -> list[int]:
            """把 I2.P1 或 P1 映射到所有匹配的相位变量。"""
            out: list[int] = []
            if "." in name:
                iname, pname = name.split(".", 1)
                i = name_to_i.get(iname)
                if i is None:
                    return out
                p = next((j for j, ph in enumerate(selected[i].phases)
                          if ph.name == pname), None)
                if p is not None and p < len(idx_g[i]):
                    out.append(idx_g[i][p])
                return out

            pname = name
            for i, plan in enumerate(selected):
                p = next((j for j, ph in enumerate(plan.phases)
                          if ph.name == pname), None)
                if p is not None and p < len(idx_g[i]):
                    out.append(idx_g[i][p])
            return out

        combined_specs: list[LinearSpec] = []
        if constraint_builder is not None:
            combined_specs.extend(constraint_builder.specs)
        if alignment_builder is not None:
            combined_specs.extend(
                alignment_builder.to_linear_specs(self.mode, int_names, seg_names)
            )

        resolved_constraints: list[tuple[LinearSpec, list[tuple[int, float]], int | None]] = []
        for spec in combined_specs:
            terms: list[tuple[int, float]] = []
            for name, coef in spec.terms.items():
                var = resolve_special(name)
                if var is not None:
                    terms.append((var, coef))
                    continue
                for pvar in resolve_phase(name):
                    terms.append((pvar, coef))
            if not terms:
                continue
            slack = None
            if spec.soft:
                slack = cur
                cur += 1
            resolved_constraints.append((spec, terms, slack))

        nvar = cur

        # ------------------------------------------------------------------
        # 3) 目标函数：把 ObjectiveConfig 翻译成 c 向量
        # ------------------------------------------------------------------
        c = np.zeros(nvar)
        if objective == "loss":
            # min 总损失
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                c[var] += slope
            for spec, _, slack in resolved_constraints:
                if spec.soft and slack is not None:
                    c[slack] += spec.penalty
        else:
            # max 带宽目标 -> milp 求 min，所以系数取负
            for group in self.config.sum_groups:
                for key, weight in group.terms.items():
                    c[band_var(key)] += -weight
            for gidx, group in enumerate(self.config.balance_groups):
                gvar = balance_vars[gidx]
                c[gvar] += -group.weight
                if group.eps > 0:
                    for member in group.members:
                        c[band_var(member)] += -group.weight * group.eps

            # ε-约束扫描时，给损失一个极小的二次权重。
            # 这不改变“带宽优先”的主目标，但能让同一带宽下的输出
            # 尽量落在最小损失点，避免帕累托前沿出现被支配点。
            if max_loss is not None:
                tiny = 1e-7
                for _, _, var, spec, side in loss_vars:
                    slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                    c[var] += tiny * slope
                for spec, _, slack in resolved_constraints:
                    if spec.soft and slack is not None:
                        c[slack] += tiny * spec.penalty

        # ------------------------------------------------------------------
        # 4) 变量界
        # ------------------------------------------------------------------
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)

        for i, plan in enumerate(selected):
            for p, ph in enumerate(plan.phases):
                if tunable[i]:
                    lb[idx_g[i][p]] = ph.min_green
                    ub[idx_g[i][p]] = ph.max_green
                else:
                    lb[idx_g[i][p]] = ph.green
                    ub[idx_g[i][p]] = ph.green

        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C

        lb[idx_mU:idx_mU + m] = -self.max_loops
        ub[idx_mU:idx_mU + m] = self.max_loops
        lb[idx_mD:idx_mD + m] = -self.max_loops
        ub[idx_mD:idx_mD + m] = self.max_loops

        # b 和 B 全部非负，并以 C 为上界
        lb[band_offset:band_offset + band_model.nvar] = 0.0
        ub[band_offset:band_offset + band_model.nvar] = C

        for gvar in balance_vars.values():
            ub[gvar] = C
        for _, _, var, _, _ in loss_vars:
            ub[var] = C
        for _, _, slack in resolved_constraints:
            if slack is not None:
                ub[slack] = C

        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1

        rows: list[np.ndarray] = []
        lo_list: list[float] = []
        hi_list: list[float] = []

        def add_row(coefs: dict[int, float], lo: float, hi: float) -> None:
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] += v
            rows.append(row)
            lo_list.append(lo)
            hi_list.append(hi)

        # ------------------------- 相位自身约束 -------------------------
        # 不可调路口的 g 已由上下界固定，不再额外强加 Σg = C-lost_time；
        # 这样 tunable_intersections 的行为与“固定绿灯窗”一致，也避免
        # 初始数据相位和 lost_time 不一致时把整个模型判为不可行。
        for i, plan in enumerate(selected):
            if not plan.phases or not tunable[i]:
                continue
            total = {idx_g[i][p]: 1.0 for p in range(len(plan.phases))}
            add_row(total, C - plan.lost_time, C - plan.lost_time)

        # ------------------------- hinge 损失约束 -------------------------
        # 过小惩罚：ℓ >= threshold - g   ->   ℓ + g >= threshold
        # 过大惩罚：ℓ >= g - upper_threshold -> ℓ - g >= -upper_threshold
        for i, p, var, spec, side in loss_vars:
            if side == "lower":
                add_row({var: 1.0, idx_g[i][p]: 1.0}, spec.threshold, np.inf)
            else:
                add_row({var: 1.0, idx_g[i][p]: -1.0},
                        -spec.upper_threshold, np.inf)

        # ------------------------- 声明式约束 -------------------------
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
                    raise ValueError("soft '=' constraint is not supported")

        # ------------------------- max_loss 约束 -------------------------
        if max_loss is not None:
            loss_coefs: dict[int, float] = {}
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                loss_coefs[var] = loss_coefs.get(var, 0.0) + slope
            for spec, _, slack in resolved_constraints:
                if spec.soft and slack is not None:
                    loss_coefs[slack] = loss_coefs.get(slack, 0.0) + spec.penalty
            add_row(loss_coefs, -np.inf, max_loss)

        # ------------------------- 带前沿传递 -------------------------
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # ------------------------- 绿灯窗约束 -------------------------
        # 每个路口：
        #   tU_i >= up_start_i
        #   tD_i >= down_start_i
        # 每个路段 i：
        #   tU_i     + b_up_i <= up_end_i
        #   tU_{i+1} + b_up_i <= up_end_{i+1}
        #   tD_i     + b_down_i <= down_end_i
        #   tD_{i+1} + b_down_i <= down_end_{i+1}
        for i, expr in enumerate(exprs):
            add_row(self._lower_phase_row(idx_tU + i, expr.up_start, idx_g[i]),
                    expr.up_start.const, np.inf)
            add_row(self._lower_phase_row(idx_tD + i, expr.down_start, idx_g[i]),
                    expr.down_start.const, np.inf)

        for i in range(m):
            expr0, expr1 = exprs[i], exprs[i + 1]
            add_row(self._upper_phase_row(idx_tU + i, idx_bU + i,
                                          expr0.up_end, idx_g[i]),
                    -np.inf, expr0.up_end.const)
            add_row(self._upper_phase_row(idx_tU + i + 1, idx_bU + i,
                                          expr1.up_end, idx_g[i + 1]),
                    -np.inf, expr1.up_end.const)
            add_row(self._upper_phase_row(idx_tD + i, idx_bD + i,
                                          expr0.down_end, idx_g[i]),
                    -np.inf, expr0.down_end.const)
            add_row(self._upper_phase_row(idx_tD + i + 1, idx_bD + i,
                                          expr1.down_end, idx_g[i + 1]),
                    -np.inf, expr1.down_end.const)

        # ------------------------- 窗口带格 -------------------------
        for d in ("up", "down"):
            b_base = idx_bU if d == "up" else idx_bD
            for k in range(2, n + 1):
                for j in range(n - k + 1):
                    B_var = band_offset + band_model.B_idx[d][k][j]
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, b_base + i: -1.0}, -np.inf, 0.0)

        # ------------------------- 均衡组约束 -------------------------
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                add_row({gvar: 1.0, band_var(member): -1.0}, -np.inf, 0.0)

        constraints = (LinearConstraint(np.array(rows), np.array(lo_list), np.array(hi_list))
                       if rows else ())
        res = milp(c=c,
                   constraints=constraints,
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        sol.objective = -float(res.fun)
        sol.band_up_style = self.up_style
        sol.band_down_style = self.down_style
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}

        sol.phase_times = {}
        for i, plan in enumerate(selected):
            if plan.phases:
                sol.phase_times[int_names[i]] = {
                    ph.name: float(x[idx_g[i][p]])
                    for p, ph in enumerate(plan.phases)
                }

        # ------------------------- 结果提取 -------------------------
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

        # one-way / local 模式下，把下行窗口带格回填给绘图。
        if self.down_style == "local":
            for band in self.config.all_band_keys(n):
                if band.direction != "down" or band.k >= n:
                    continue
                j0 = band.start
                j1 = band.start + band.k - 1
                wkey = f"win{band.k}@{int_names[j0]}-{int_names[j1]}"
                sol.window_bands[wkey] = float(
                    x[band_offset + band_model.var_of(band)]
                )

        # ------------------------- 总损失回填 -------------------------
        total_loss = 0.0
        for i, p, _, spec, side in loss_vars:
            g = float(x[idx_g[i][p]])
            if side == "lower":
                total_loss += spec.slope * max(0.0, spec.threshold - g)
            else:
                total_loss += ((spec.upper_slope or spec.slope)
                               * max(0.0, g - spec.upper_threshold))
        for spec, terms, _ in resolved_constraints:
            if not spec.soft:
                continue
            expr_val = sum(coef * float(x[var]) for var, coef in terms)
            if spec.sense == ">=":
                total_loss += spec.penalty * max(0.0, spec.rhs - expr_val)
            elif spec.sense == "<=":
                total_loss += spec.penalty * max(0.0, expr_val - spec.rhs)
        sol.total_phase_loss = float(total_loss)
        sol.status = "optimal" if res.success else res.message
        return sol
