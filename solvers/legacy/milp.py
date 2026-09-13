"""Legacy MILP 求解器实现。

注意：
    当前推荐入口是 solvers/stage1/flexible_band.py：
    - CompositeBandSolver  = FlexibleBandSolver(composite_config(...))
    - OneWayPrioritySolver = FlexibleBandSolver(oneway_config(...))

本文件仍保留旧的 CompositeBandSolver / OneWayPrioritySolver / MaxBandMILPSolver
实现，用于兼容和参照。

旧模型要点：
    - 全局带：b_up、b_down 只有方向级变量；
    - 分段带：下行可为每段 bD_i 独立带宽；
    - 窗口带：B <= b 的线性不等式；
    - 方案选择：δ 0-1 变量 + 仿射窗口边界。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...models import Arterial, GreenWindow
from ...solution import Solution
from ..core.base import Solver


def _widest_window(windows: list[GreenWindow]) -> GreenWindow:
    """取最宽的一段绿灯窗（简化处理）。"""
    if not windows:
        raise ValueError("信控方案缺少绿灯窗口")
    return max(windows, key=lambda w: w.width)


class CompositeBandSolver(Solver):
    """双向全局带宽最大化（MAXBAND 风格），基于 scipy.optimize.milp（HiGHS）。

    变量（时间单位均为秒，周期 C = arterial.cycle）：
        b_up, b_down               上/下行带宽
        tU_0..tU_{n-1}             上行带前沿到达各路口的时刻（mod C）
        tD_0..tD_{n-1}             下行带前沿到达各路口的时刻（mod C）
        mU_0..mU_{n-2}             上行各段整数圈数
        mD_0..mD_{n-2}             下行各段整数圈数

    约束：
        tU_{i+1} - tU_i - C * mU_i = tau_up_i      （带前沿传递 + 圈数吸收）
        s_i * C <= t_i <= e_i * C - b              （带子落在绿灯窗内）
        下行同理。
    """

    name = "global-maxband"

    def __init__(self, down_weight: float = 1.0, up_weight: float = 1.0,
                 max_loops: int = 3,
                 max_window: int = 3, objective_mode: str = "sum",
                 balance_eps: float = 0.1,
                 balance_terms: tuple[str, ...] = ("up", "down")) -> None:
        """
        args:
            up_weight: 目标中上行带宽的权重；
            down_weight: 目标中下行带宽的权重（常取上下行流量比）；
            max_loops: 整数圈数变量 m 的取值范围 [-max_loops, max_loops]；
            max_window: 返回 window_bands 的最大窗口长度。这里是在已求出的
                全局下行轨迹上计算各窗口可容纳的公共带宽，不重新优化窗口；
            objective_mode:
                "sum" 最大化 b_up + k*b_down；
                "balanced" 最大化 B_bal = min(b_up, b_down)；
                "balanced_composite" 最大化 b_up + b_down + eps*B_bal。
            balance_eps: balanced_composite 目标中 B_bal 的权重；
            balance_terms: B_bal 的作用域，默认 ("up","down")，表示
                B_bal <= b_up 且 B_bal <= b_down。
        """
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.max_loops = max_loops
        self.max_window = max_window
        self.objective_mode = objective_mode
        self.balance_eps = balance_eps
        self.balance_terms = tuple(balance_terms)

    def solve(self, arterial: Arterial) -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)

        # 把“方案 × 上行窗口 × 下行窗口”拍平成联合选项。
        # options[i] 中每一项 = (方案下标, 上窗下标, 下窗下标, 上窗, 下窗)。
        options: list[list[tuple[int, int, int, GreenWindow, GreenWindow]]] = []
        for inter in ints:
            opts: list[tuple[int, int, int, GreenWindow, GreenWindow]] = []
            for p_idx, plan in enumerate(inter.plans):
                if not plan.up_windows or not plan.down_windows:
                    raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少绿灯窗口")
                for q, wu in enumerate(plan.up_windows):
                    for r, wd in enumerate(plan.down_windows):
                        opts.append((p_idx, q, r, wu, wd))
            options.append(opts)

        # 变量布局: [b_up, b_dn, tU(n), tD(n), mU(m), mD(m), δ(选项...)]
        idx_bu, idx_bd = 0, 1
        idx_tU = 2
        idx_tD = 2 + n
        idx_mU = 2 + 2 * n
        idx_mD = 2 + 2 * n + m
        cur = 2 + 2 * n + 2 * m
        idx_opt: list[list[int]] = []
        for i in range(n):
            idx_opt.append(list(range(cur, cur + len(options[i]))))
            cur += len(options[i])
        idx_bal = None
        if self.objective_mode in ("balanced", "balanced_composite"):
            idx_bal = cur
            cur += 1
        nvar = cur

        # 目标
        c = np.zeros(nvar)
        if self.objective_mode == "balanced":
            c[idx_bal] = -1.0
        elif self.objective_mode == "balanced_composite":
            c[idx_bu] = -self.up_weight
            c[idx_bd] = -self.down_weight
            c[idx_bal] = -self.balance_eps
        else:
            c[idx_bu] = -self.up_weight
            c[idx_bd] = -self.down_weight

        # 变量界
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        ub[idx_bu] = min(
            max(wu.width for _, _, _, wu, _ in options[i]) for i in range(n)
        ) * C
        ub[idx_bd] = min(
            max(wd.width for _, _, _, _, wd in options[i]) for i in range(n)
        ) * C
        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C
        lb[idx_mU:idx_mD + m] = -self.max_loops
        ub[idx_mU:idx_mD + m] = self.max_loops
        for row in idx_opt:
            ub[row] = 1.0
        if idx_bal is not None:
            ub[idx_bal] = C

        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mD + m] = 1  # 圈数为整数
        for row in idx_opt:
            integrality[row] = 1  # 方案+窗口联合选择变量为 0-1

        rows, lhs_lb, lhs_ub = [], [], []

        def add_row(coefs: dict[int, float], lo: float, hi: float) -> None:
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] = v
            rows.append(row)
            lhs_lb.append(lo)
            lhs_ub.append(hi)

        # 0) 方案+窗口选择：每个路口只能选一个联合选项
        for i in range(n):
            add_row({idx_opt[i][o]: 1.0 for o in range(len(options[i]))}, 1.0, 1.0)

        # 0.5) 均衡目标约束：按配置的作用域添加 B_bal <= b_*
        if idx_bal is not None:
            if "up" in self.balance_terms:
                add_row({idx_bal: 1.0, idx_bu: -1.0}, -np.inf, 0.0)
            if "down" in self.balance_terms:
                add_row({idx_bal: 1.0, idx_bd: -1.0}, -np.inf, 0.0)

        # 1) 带前沿传递（等式）：tU_{i+1} - tU_i - C*mU_i = tau_i
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # 2) 干涉约束：窗口边界替换为选中联合选项的线性组合（无 big-M）
        for i in range(n):
            up_starts = {idx_opt[i][o]: wu.start * C
                         for o, (_, _, _, wu, _) in enumerate(options[i])}
            up_ends = {idx_opt[i][o]: -wu.end * C
                       for o, (_, _, _, wu, _) in enumerate(options[i])}
            dn_starts = {idx_opt[i][o]: wd.start * C
                         for o, (_, _, _, _, wd) in enumerate(options[i])}
            dn_ends = {idx_opt[i][o]: -wd.end * C
                       for o, (_, _, _, _, wd) in enumerate(options[i])}

            # tU_i >= Σ δ * start*C
            add_row({idx_tU + i: -1.0, **up_starts}, -np.inf, 0.0)
            # tU_i + b_up <= Σ δ * end*C
            add_row({idx_tU + i: 1.0, idx_bu: 1.0, **up_ends}, -np.inf, 0.0)
            # tD_i >= Σ δ * start*C
            add_row({idx_tD + i: -1.0, **dn_starts}, -np.inf, 0.0)
            # tD_i + b_down <= Σ δ * end*C
            add_row({idx_tD + i: 1.0, idx_bd: 1.0, **dn_ends}, -np.inf, 0.0)

        res = milp(c=c,
                   constraints=LinearConstraint(np.array(rows), lhs_lb, lhs_ub),
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, status=res.message,
                       solver_msg=f"HiGHS via scipy: success={res.success}")
        sol.band_up_style = "global"
        sol.band_down_style = "global"
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        sol.objective = -float(res.fun)
        seg_names = [s.name for s in segs]
        int_names = [v.name for v in ints]

        # 每个路口实际选中的联合选项（δ 中值最大的那个）
        chosen_opt: list[int] = []
        for i in range(n):
            vals = [float(x[j]) for j in idx_opt[i]]
            chosen_opt.append(max(range(len(vals)), key=lambda o: vals[o]))
        win_up_sel = [options[i][chosen_opt[i]][3] for i in range(n)]
        win_dn_sel = [options[i][chosen_opt[i]][4] for i in range(n)]

        sol.bandwidth_up = {name: float(x[idx_bu]) for name in seg_names}
        sol.bandwidth_down = {name: float(x[idx_bd]) for name in seg_names}
        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}

        # 在固定全局轨迹上计算下行 window_bands。
        # 对窗口 [j, j+k-1]，公共带宽 = min_i(e_i*C - tD_i)，i 在窗口内。
        if self.max_window >= 2:
            for k in range(2, min(self.max_window, n) + 1):
                for j in range(n - k + 1):
                    upper = min(
                        win_dn_sel[i].end * C - float(x[idx_tD + i])
                        for i in range(j, j + k)
                    )
                    bw = max(0.0, float(upper))
                    key = f"win{k}@{int_names[j]}-{int_names[j + k - 1]}"
                    sol.window_bands[key] = bw

        # 相位差取选中方案的上行绿灯窗起点（物理相位参考）
        sol.offsets = {name: win_up_sel[i].start * C for i, name in enumerate(int_names)}
        sol.plan_choices = {}
        sol.window_choices = {}
        for i, name in enumerate(int_names):
            p_idx, q, r, _, _ = options[i][chosen_opt[i]]
            plan = ints[i].plans[p_idx]
            sol.plan_choices[name] = plan.name
            sol.window_choices[name] = {
                "plan": plan.name,
                "up_window": q,
                "down_window": r,
            }
        sol.status = "optimal" if res.success else res.message
        return sol


class OneWayPrioritySolver(Solver):
    """单向全局绿波 + 另一方向分路段/分窗口带宽加权和最大化。

    - 上行：全局一条带 bU，必须穿过所有路口的上行绿灯窗（经典 MAXBAND）；
    - 下行：共享一条带轨迹 tD_i（含整数圈数），但每个路段有独立带宽 bD_i，
      只需塞进该段两端路口的下行绿灯窗（MULTIBAND 风格）；
    - 窗口带宽：相邻 k 个路口（k-1 个路段）的窗口带宽 B = 窗口内各段 bD_i
      的最小值，用 B <= bD_i 不等式组线性表达；
    - 目标：max up_weight * bU + Σ_k window_weights[k] * Σ（k 窗口带宽）。

    注：window_weights = {2: w2} 即"相邻两个路口带宽之和"；{2: w2, 3: w3}
    表示同时奖励相邻 2 个和相邻 3 个路口的贯通带宽。
    当前 Solution.window_bands 只构造到“三个一组”，即 k <= 3；
    传入 k > 3 的权重会被忽略。
    """

    name = "one-way-priority"

    def __init__(self,
                 up_weight: float = 1.0,
                 window_weights: dict[int, float] | None = None,
                 segment_down_weights: dict[str, float] | None = None,
                 max_loops: int = 3) -> None:
        self.up_weight = up_weight
        self.segment_down_weights = dict(segment_down_weights or {})
        raw_weights = window_weights or {2: 1.0}
        # 窗口带宽只需要构造到三个一组；k > 3 的权重忽略。
        self.window_weights = {k: w for k, w in raw_weights.items() if k <= 3}
        if not self.window_weights:
            self.window_weights = {2: 1.0}
        self.max_loops = max_loops

    def solve(self, arterial: Arterial) -> Solution:
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)

        # 把“方案 × 上行窗口 × 下行窗口”拍平成联合选项。
        options: list[list[tuple[int, int, int, GreenWindow, GreenWindow]]] = []
        for inter in ints:
            opts = []
            for p_idx, plan in enumerate(inter.plans):
                if not plan.up_windows or not plan.down_windows:
                    raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少绿灯窗口")
                for q, wu in enumerate(plan.up_windows):
                    for r, wd in enumerate(plan.down_windows):
                        opts.append((p_idx, q, r, wu, wd))
            options.append(opts)

        # 变量布局: [bU, tU(n), mU(m), tD(n), mD(m), bD(m), 窗口变量..., δ...]
        idx_bU = 0
        idx_tU, idx_mU = 1, 1 + n
        idx_tD, idx_mD = 1 + n + m, 1 + 2 * n + m
        idx_bD = 1 + 2 * n + 2 * m
        idx_win: dict[tuple[int, int], int] = {}  # (窗口大小k, 起点路口j) -> 变量下标
        cur = idx_bD + m
        for k in self.window_weights:
            if k != 3:
                continue  # k=2 的窗口带宽就是 bD_i 本身；k>3 不再构造
            for j in range(n - k + 1):
                idx_win[(k, j)] = cur
                cur += 1
        idx_opt: list[list[int]] = []
        for i in range(n):
            idx_opt.append(list(range(cur, cur + len(options[i]))))
            cur += len(options[i])
        nvar = cur

        # 目标（minimize 取负）
        c = np.zeros(nvar)
        c[idx_bU] = -self.up_weight
        w2 = self.window_weights.get(2, 0.0)
        for i, seg in enumerate(segs):
            w = self.segment_down_weights.get(seg.name, w2)
            c[idx_bD + i] = -w
        for (k, j), v in idx_win.items():
            c[v] = -self.window_weights[k]

        # 变量界
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        ub[idx_bU] = min(
            max(wu.width for _, _, _, wu, _ in options[i]) for i in range(n)
        ) * C
        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C
        for i in range(m):
            ub[idx_bD + i] = min(
                max(wd.width for _, _, _, _, wd in options[i]),
                max(wd.width for _, _, _, _, wd in options[i + 1]),
            ) * C
        lb[idx_mU:idx_mU + m] = -self.max_loops
        ub[idx_mU:idx_mU + m] = self.max_loops
        lb[idx_mD:idx_mD + m] = -self.max_loops
        ub[idx_mD:idx_mD + m] = self.max_loops
        for v in idx_win.values():
            ub[v] = C
        for row in idx_opt:
            ub[row] = 1.0

        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1
        for row in idx_opt:
            integrality[row] = 1  # 方案+窗口联合选择变量为 0-1

        rows, lo_list, hi_list = [], [], []

        def add_row(coefs, lo, hi):
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] = v
            rows.append(row)
            lo_list.append(lo)
            hi_list.append(hi)

        # 0) 方案+窗口选择：每个路口只能选一个联合选项
        for i in range(n):
            add_row({idx_opt[i][o]: 1.0 for o in range(len(options[i]))}, 1.0, 1.0)

        # 1) 带前沿传递（两个方向）
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # 2) 上行：全局带 bU 落进每个路口选中联合选项的上行绿灯窗
        for i in range(n):
            up_starts = {idx_opt[i][o]: wu.start * C
                         for o, (_, _, _, wu, _) in enumerate(options[i])}
            up_ends = {idx_opt[i][o]: -wu.end * C
                       for o, (_, _, _, wu, _) in enumerate(options[i])}
            # tU_i >= Σ δ * start*C
            add_row({idx_tU + i: -1.0, **up_starts}, -np.inf, 0.0)
            # tU_i + bU <= Σ δ * end*C
            add_row({idx_tU + i: 1.0, idx_bU: 1.0, **up_ends}, -np.inf, 0.0)

        # 3) 下行：每段带宽 bD_i 落进该段两端路口选中联合选项的下行绿灯窗
        for i in range(n):
            dn_starts = {idx_opt[i][o]: wd.start * C
                         for o, (_, _, _, _, wd) in enumerate(options[i])}
            # tD_i >= Σ δ * start*C
            add_row({idx_tD + i: -1.0, **dn_starts}, -np.inf, 0.0)
        for i in range(m):
            i0_ends = {idx_opt[i][o]: -wd.end * C
                       for o, (_, _, _, _, wd) in enumerate(options[i])}
            i1_ends = {idx_opt[i + 1][o]: -wd.end * C
                       for o, (_, _, _, _, wd) in enumerate(options[i + 1])}
            # tD_i + bD_i <= Σ δ_i * end_i*C
            add_row({idx_tD + i: 1.0, idx_bD + i: 1.0, **i0_ends}, -np.inf, 0.0)
            # tD_{i+1} + bD_i <= Σ δ_{i+1} * end_{i+1}*C
            add_row({idx_tD + i + 1: 1.0, idx_bD + i: 1.0, **i1_ends}, -np.inf, 0.0)

        # 4) 窗口带宽：B(k,j) <= 窗口内每个路段的 bD_i
        for (k, j), v in idx_win.items():
            for i in range(j, j + k - 1):
                add_row({v: 1, idx_bD + i: -1}, -np.inf, 0.0)

        res = milp(c=c,
                   constraints=LinearConstraint(np.array(rows), lo_list, hi_list),
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        sol.band_up_style = "global"
        sol.band_down_style = "local"
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        seg_names = [s.name for s in segs]
        int_names = [v.name for v in ints]
        sol.objective = -float(res.fun)
        sol.bandwidth_up = {name: float(x[idx_bU]) for name in seg_names}
        sol.bandwidth_down = {seg_names[i]: float(x[idx_bD + i]) for i in range(m)}
        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}

        # 读取每个路口选中的联合选项
        chosen_opt: list[int] = []
        for i in range(n):
            vals = [float(x[j]) for j in idx_opt[i]]
            chosen_opt.append(max(range(len(vals)), key=lambda o: vals[o]))
        sol.plan_choices = {}
        sol.window_choices = {}
        for i, name in enumerate(int_names):
            p_idx, q, r, _, _ = options[i][chosen_opt[i]]
            plan = ints[i].plans[p_idx]
            sol.plan_choices[name] = plan.name
            sol.window_choices[name] = {
                "plan": plan.name,
                "up_window": q,
                "down_window": r,
            }

        # 两两路口之间的带宽（k=2）也属于 window_bands
        for i in range(m):
            sol.window_bands[f"win2@{int_names[i]}-{int_names[i + 1]}"] = float(x[idx_bD + i])
        # 三个一组的窗口带宽（k=3）
        for (k, j), v in idx_win.items():
            wname = f"win{k}@{int_names[j]}-{int_names[j + k - 1]}"
            sol.window_bands[wname] = float(x[v])
        sol.status = "optimal" if res.success else res.message
        return sol


class MaxBandMILPSolver(Solver):
    """分段带宽最大化（MULTIBAND 风格）的完整版求解器，占位。"""

    name = "maxband-milp"

    def __init__(self, backend: str = "highs") -> None:
        self.backend = backend
        self._model = None

    def build(self, arterial: Arterial) -> None:
        raise NotImplementedError

    def solve(self, arterial: Arterial) -> Solution:
        raise NotImplementedError
