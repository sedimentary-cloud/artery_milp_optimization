"""FlexibleBandSolver：读取 BandModel + ObjectiveConfig 组装 MILP。

当前版本假设：
- 每个路口使用第一个信控方案；
- 每个方向取第一个绿灯窗口（后续再接方案/窗口选择层）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..models import Arterial
from ..solution import Solution
from .band_model import BandModel
from .base import Solver
from .objective_config import BalanceGroup, ObjectiveConfig, SumGroup


class FlexibleBandSolver(Solver):
    """根据 ObjectiveConfig 求解双向绿波带宽组合目标。"""

    name = "flexible-band"

    def __init__(self,
                 config: ObjectiveConfig,
                 max_loops: int = 3) -> None:
        self.config = config
        self.max_loops = max_loops

    def solve(self, arterial: Arterial) -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)
        self.config.validate(n)

        # 取每个路口的第一个方案、第一个窗口
        wu = []
        wd = []
        for inter in ints:
            plan = inter.plans[0]
            if not plan.up_windows or not plan.down_windows:
                raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少绿灯窗口")
            wu.append(plan.up_windows[0])
            wd.append(plan.down_windows[0])

        # 变量布局
        idx_tU = 0
        idx_mU = n
        idx_tD = n + m
        idx_mD = 2 * n + m
        idx_bU = 2 * n + 2 * m
        idx_bD = 2 * n + 3 * m
        cur = 2 * n + 4 * m

        band_model = BandModel(n, m)
        band_offset = cur
        cur += band_model.nvar

        # BalanceGroup 的组 min 变量
        balance_vars: dict[int, int] = {}
        for gidx, group in enumerate(self.config.balance_groups):
            balance_vars[gidx] = cur
            cur += 1
        nvar = cur

        # 目标：scipy 默认最小化，这里 c 取负
        c = np.zeros(nvar)
        for group in self.config.sum_groups:
            for key, weight in group.terms.items():
                band = _parse_band_key(key, n)
                var = _band_var(band_model, band, band_offset)
                c[var] += -weight
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            c[gvar] += -group.weight
            if group.eps > 0:
                for member in group.members:
                    band = _parse_band_key(member, n)
                    var = _band_var(band_model, band, band_offset)
                    c[var] += -group.weight * group.eps

        # 变量界
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
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

        # 带前沿传递
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # 上行：每个路段两端窗口约束
        for i in range(n):
            add_row({idx_tU + i: 1}, wu[i].start * C, np.inf)
        for i in range(m):
            add_row({idx_tU + i: 1, idx_bU + i: 1}, -np.inf, wu[i].end * C)
            add_row({idx_tU + i + 1: 1, idx_bU + i: 1}, -np.inf, wu[i + 1].end * C)

        # 下行：每个路段两端窗口约束
        for i in range(n):
            add_row({idx_tD + i: 1}, wd[i].start * C, np.inf)
        for i in range(m):
            add_row({idx_tD + i: 1, idx_bD + i: 1}, -np.inf, wd[i].end * C)
            add_row({idx_tD + i + 1: 1, idx_bD + i: 1}, -np.inf, wd[i + 1].end * C)

        # 带格约束：B <= b
        for d in ("up", "down"):
            b_idx = idx_bU if d == "up" else idx_bD
            for k in range(2, n + 1):
                for j in range(n - k + 1):
                    B_var = band_offset + band_model.B_idx[d][k][j]
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, b_idx + i: -1.0},
                                -np.inf, 0.0)

        # BalanceGroup: B_g <= member
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                band = _parse_band_key(member, n)
                var = _band_var(band_model, band, band_offset)
                add_row({gvar: 1.0, var: -1.0}, -np.inf, 0.0)

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
        sol.objective = -float(res.fun)
        sol.bandwidth_up = {seg_names[i]: float(x[idx_bU + i]) for i in range(m)}
        sol.bandwidth_down = {seg_names[i]: float(x[idx_bD + i]) for i in range(m)}
        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}
        sol.plan_choices = {name: ints[i].plans[0].name for i, name in enumerate(int_names)}
        sol.window_choices = {
            name: {"plan": ints[i].plans[0].name, "up_window": 0, "down_window": 0}
            for i, name in enumerate(int_names)
        }
        sol.status = "optimal" if res.success else res.message
        return sol


def _parse_band_key(text: str, n: int):
    from .objective_config import parse_band_key
    return parse_band_key(text, n)


def _band_var(band_model: BandModel, band, offset: int) -> int:
    return offset + band_model.var_of(band)


def composite_config(up_weight: float = 1.0,
                     down_weight: float = 1.0,
                     objective_mode: str = "sum",
                     balance_eps: float = 0.1,
                     balance_terms: tuple[str, ...] = ("up", "down")) -> ObjectiveConfig:
    """CompositeBandSolver 的预设目标配置。"""
    if objective_mode == "sum":
        return ObjectiveConfig(sum_groups=[SumGroup({
            "up.global": up_weight,
            "down.global": down_weight,
        })])
    if objective_mode == "balanced":
        members = [f"{d}.global" for d in balance_terms]
        return ObjectiveConfig(balance_groups=[BalanceGroup(members, weight=1.0)])
    if objective_mode == "balanced_composite":
        members = [f"{d}.global" for d in balance_terms]
        return ObjectiveConfig(
            sum_groups=[SumGroup({
                "up.global": up_weight,
                "down.global": down_weight,
            })],
            balance_groups=[BalanceGroup(members, weight=balance_eps)],
        )
    raise ValueError(f"unknown objective_mode: {objective_mode}")


def oneway_config(up_weight: float = 1.0,
                  window_weights: dict[int, float] | None = None,
                  segment_down_weights: dict[str, float] | None = None,
                  n_intersections: int = 0) -> ObjectiveConfig:
    """OneWayPrioritySolver 的预设目标配置（下行分段 + 窗口带）。"""
    ww = window_weights or {2: 1.0}
    seg_weights = dict(segment_down_weights or {})
    terms: dict[str, float] = {"up.global": up_weight}
    for i in range(n_intersections - 1):
        seg_name = f"seg{i+1}"
        terms[f"down.{seg_name}"] = seg_weights.get(seg_name, ww.get(2, 1.0))
    w3 = ww.get(3, 0.0)
    if w3 > 0:
        for j in range(1, n_intersections - 1):
            terms[f"down.win3@I{j}-I{j+2}"] = w3
    return ObjectiveConfig(sum_groups=[SumGroup(terms)])
