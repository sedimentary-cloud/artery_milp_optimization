"""FlexibleBandSolver：读取 BandModel + ObjectiveConfig 组装 MILP。

数学模型
--------
变量：
    b_up_i / b_down_i      第 i 段基础带宽
    tU_i / tD_i            带前沿到达各路口时刻（mod C）
    mU_i / mD_i            整数圈数修正量
    B[d,k,j]               窗口带宽度
    δ                      方案/窗口联合选择 0-1 变量

约束：
    1) 带前沿传递
         tU_{i+1} = tU_i + tau_up_i + C * mU_i
         tD_i = tD_{i+1} + tau_down_i + C * mD_i

    2) 每个路段两端窗口约束
         t_i >= Σ δ * start * C
         t_i + b_i <= Σ δ * end * C

    3) 窗口带格
         B[d,k,j] <= b[d,i]   for i = j ... j+k-1

    4) 方案/窗口选择
         Σ_options δ = 1

目标：
    由 ObjectiveConfig 给出；
    SumGroup 直接加权求和；
    BalanceGroup 创建组 min 变量 B_g，并加 B_g <= member。

可选绿波带层损失：
    alignment_builder 会转成软约束 slack，band_loss_weight 控制它
    以加权和形式进入目标：
        max band_objective - band_loss_weight * band_loss
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
                 max_loops: int = 3,
                 name: str = "flexible-band",
                 up_style: str = "global",
                 down_style: str = "global",
                 up_global_output: bool = False,
                 down_global_output: bool = False) -> None:
        self.config = config
        self.max_loops = max_loops
        self.name = name
        self.up_style = up_style
        self.down_style = down_style
        # 输出兼容：若为 True，则 Solution.bandwidth_* 填全局带值，
        # 而不是逐路段基础带宽。
        self.up_global_output = up_global_output
        self.down_global_output = down_global_output

    def solve(self,
              arterial: Arterial,
              alignment_builder=None,
              band_loss_weight: float = 0.0) -> Solution:
        """求解第一阶段带宽组合 MILP。

        Args:
            arterial: 干线数据，绿灯窗已经固定。
            alignment_builder: 可选，绿波带层对齐损失配置。
                它会被转成线性软约束 slack，并以加权和形式进入目标。
            band_loss_weight: band_loss 的权重 λ。
                目标为：
                    max band_objective - λ * band_loss
        """
        C = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)
        self.config.validate(n)

        # ============================================================
        # 先把数学符号翻译成“初中生也能懂”的话：
        #
        # 想象每个路口是一个公交站，绿波带是一辆“绿波车”。
        # 我们想让这辆车尽量宽、尽量顺地通过所有站。
        #
        # C: 一个信号周期的秒数。
        # n: 路口数量；m = n - 1 是路段数量。
        #
        # b_up_i / b_down_i:
        #   第 i 段路上，绿波带的“宽度”（单位秒）。
        #   带宽越宽，说明能一次通过的车队越长。
        #
        # tU_i / tD_i:
        #   绿波带车头到达第 i 个路口的时刻。
        #   因为信号周期会重复，所以只记录在 [0, C) 内的时刻。
        #
        # mU_i / mD_i:
        #   整数圈数。车头到达下一站可能跨了 1 个周期、2 个周期，
        #   用整数 m 记录“少算/多算”了几个周期。
        #
        # B[d, k, j]:
        #   对 d 方向而言从第 j 个路口开始、连续 k 个路口的窗口带宽度。
        #   k=2 就是一段路；k=n 就是整条干线。
        #
        # δ:
        #   0/1 开关。每个路口有多个“方案×上行窗口×下行窗口”选项，
        #   δ=1 表示选中某个选项，δ=0 表示不选。
        # ============================================================

        # 方案 × 上行窗口 × 下行窗口 联合选项
        options: list[list[tuple[int, int, int, object, object]]] = []
        for inter in ints:
            opts = []
            for p_idx, plan in enumerate(inter.plans):
                if not plan.up_windows or not plan.down_windows:
                    raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少绿灯窗口")
                for q, wu in enumerate(plan.up_windows):
                    for r, wd in enumerate(plan.down_windows):
                        opts.append((p_idx, q, r, wu, wd))
            if not opts:
                raise ValueError(f"路口 {inter.name} 没有可用方案/窗口")
            options.append(opts)

        # ============================================================
        # 变量在一条长向量 x 里的排列顺序。
        # 求解器 HiGHS 只认识一维向量，所以我们要记住每个变量在第几位。
        # ============================================================
        # 所有变量排成一条长向量 x，下面这些 idx_* 是各变量块在 x 中的起始下标。
        # n = 路口数量，m = 路段数量。
        idx_tU = 0                  # tU_0..tU_{n-1} 从 0 开始，共 n 个
        idx_mU = n                  # mU_0..mU_{m-1} 紧接在 tU 后面，从 n 开始，共 m 个
        idx_tD = n + m              # tD_0..tD_{n-1} 紧接在 mU 后面，从 n+m 开始，共 n 个
        idx_mD = 2 * n + m          # mD_0..mD_{m-1} 紧接在 tD 后面，从 2n+m 开始，共 m 个
        idx_bU = 2 * n + 2 * m      # b_up_0..b_up_{m-1} 紧接在 mD 后面，从 2n+2m 开始，共 m 个
        idx_bD = 2 * n + 3 * m      # b_down_0..b_down_{m-1} 紧接在 bU 后面，从 2n+3m 开始，共 m 个
        cur = 2 * n + 4 * m         # 以上基础变量一共占 2n+4m 个；cur 是下一个空闲下标

        # 创建窗口带格 B[d,k,j] 的变量索引表，并把它们排在基础变量之后。
        band_model = BandModel(n, m)
        band_offset = cur              # 带格变量的起始下标
        cur += band_model.nvar         # 为所有 B 变量预留位置

        # 每个 BalanceGroup 需要一个“组内最小值”变量：
        #     B_g <= 每个成员
        # 最大化 B_g 时，它自动变成成员里的最小值。
        balance_vars: dict[int, int] = {}
        for gidx, group in enumerate(self.config.balance_groups):
            balance_vars[gidx] = cur
            cur += 1

        # 每个路口一组 0-1 变量 δ，用来选择：
        #     (方案, 上行窗口, 下行窗口)
        # idx_opt[i] 存第 i 个路口所有选项的变量下标。
        idx_opt: list[list[int]] = []
        for i in range(n):
            idx_opt.append(list(range(cur, cur + len(options[i]))))
            cur += len(options[i])

        # ============================================================
        # 绿波带层对齐损失：AlignmentLossBuilder
        #
        # 第一阶段没有相位变量 g，因此 AlignmentLossBuilder 里引用的
        # tU_* / tD_* / b_up / b_down / bD_* 都要在这里映射到
        # 第一阶段的变量。
        # ============================================================
        seg_names = [s.name for s in segs]
        int_names = [v.name for v in ints]
        name_to_i = {name: i for i, name in enumerate(int_names)}
        seg_name_to_idx = {name: i for i, name in enumerate(seg_names)}

        def resolve_band_loss_name(name: str) -> int | None:
            """把 alignment 约束里的变量名映射到第一阶段 MILP 变量。"""
            if name == "b_up":
                return band_offset + band_model.B_idx["up"][n][0]
            if name == "b_down":
                return band_offset + band_model.B_idx["down"][n][0]
            if name == "B_bal" and balance_vars:
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

            # 第一阶段没有相位变量，相位名直接忽略。
            return None

        # 每个元素：(原始 LinearSpec, [(变量下标, 系数), ...], slack 变量下标)
        band_loss_resolved: list[tuple[object, list[tuple[int, float]], int]] = []
        if alignment_builder is not None:
            # 无论权重是否为 0，都解析并回填真实 band_loss；
            # 权重为 0 时只是不把 loss 放进目标函数。
            align_mode = "oneway" if self.down_style == "local" else "global"
            align_specs = alignment_builder.to_linear_specs(
                align_mode, int_names, seg_names
            )
            for spec in align_specs:
                terms: list[tuple[int, float]] = []
                for name, coef in spec.terms.items():
                    var = resolve_band_loss_name(name)
                    if var is not None:
                        terms.append((var, coef))
                if not terms:
                    continue
                slack = cur
                cur += 1
                band_loss_resolved.append((spec, terms, slack))

        # 到这里所有变量都排完了，nvar 是变量总数。
        nvar = cur

        # scipy.optimize.milp 默认求最小值：min c^T x。
        # 但我们的目标是“最大化”各项，所以把系数取负：
        #     max w*x  ==  min (-w)*x
        c = np.zeros(nvar)

        # SumGroup：每个带标识直接按权重加进目标。
        for group in self.config.sum_groups:                       # 遍历每个加权和组
            for key, weight in group.terms.items():                # 取出本组里的“带标识 -> 权重”
                band = _parse_band_key(key, n)                     # 把字符串解析成 BandKey(direction, k, start)
                var = _band_var(band_model, band, band_offset)     # 找到这个带对应的 MILP 变量下标
                c[var] += -weight                                  # 目标原本是 +weight，但 milp 求最小值，所以取负

        # BalanceGroup：组 min 变量按组权重加负号；
        # 再给每个成员加一个很小的 ε 托底项，避免均衡达标后其他带摆烂。
        for gidx, group in enumerate(self.config.balance_groups):  # 遍历每个均衡组
            gvar = balance_vars[gidx]                              # 取出该均衡组的组 min 变量下标
            c[gvar] += -group.weight                               # 最大化 B_g，所以目标系数取 -weight
            if group.eps > 0:                                      # 如果开启了 ε 托底
                for member in group.members:                       # 遍历该均衡组的所有成员带
                    band = _parse_band_key(member, n)              # 解析成员带标识
                    var = _band_var(band_model, band, band_offset) # 找到成员带对应的变量下标
                    c[var] += -group.weight * group.eps            # 给成员带加一个很小的正权重，防止“只均衡、不榨总量”

        # 绿波带层损失：
        #   目标 = max band_objective - band_loss_weight * band_loss
        # milp 求 min，所以 slack 系数写成 + band_loss_weight * penalty。
        for spec, _, slack in band_loss_resolved:
            c[slack] += band_loss_weight * spec.penalty

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
        for row in idx_opt:
            ub[row] = 1.0
        for _, _, slack in band_loss_resolved:
            ub[slack] = C

        # integrality[j] 告诉求解器第 j 个变量是什么类型：
        #   0 -> 连续变量，可以是小数；
        #   1 -> 整数变量，只能取整数。
        # 这里先把所有变量都看成连续的。
        integrality = np.zeros(nvar)

        # mU_0..mU_{m-1} 和 mD_0..mD_{m-1} 是“跨了几个周期”的圈数，
        # 必须是整数。例如 m=-1 表示往回减一个周期，m=0 表示不跨周期。
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1

        # δ 是“方案/窗口选择开关”，只能取 0 或 1。
        # 在 MILP 里 0-1 变量也算整数变量，所以 integrality 设为 1，
        # 再配合 bounds 里的 0 <= δ <= 1，就变成二进制变量。
        for row in idx_opt:
            integrality[row] = 1

        rows, lo_list, hi_list = [], [], []

        def add_row(coefs, lo, hi):
            """添加一行线性约束：lo <= Σ coefs[j] * x[j] <= hi。

            参数：
                coefs: {变量下标: 系数}
                       例如 {idx_tU+1: 1, idx_tU: -1, idx_mU: -C}
                       表示 1*tU_{i+1} + (-1)*tU_i + (-C)*mU_i
                lo:    这一行线性表达式的下界；
                hi:    这一行线性表达式的上界。

            用法：
                lo == hi  -> 等式约束；
                lo = -inf -> 只有上界的不等式；
                hi = +inf -> 只有下界的不等式。
            """
            row = np.zeros(nvar)
            for j, v in coefs.items():
                row[j] = v
            rows.append(row)
            lo_list.append(lo)
            hi_list.append(hi)

        # 方案/窗口选择：每个路口一个联合选项
        for i in range(n):
            add_row({idx_opt[i][o]: 1.0 for o in range(len(options[i]))}, 1.0, 1.0)

        # ============================================================
        # 约束 1：带前沿传递
        #
        # 上行：车头从路口 i 出发，开过第 i 段路需要 tau_up_i 秒。
        #       所以 tU_{i+1} = tU_i + tau_up_i + C * mU_i。
        #       移项后就是下面代码里的等式。
        #
        # 下行：车是从右往左开，所以用 tD_i 和 tD_{i+1} 的关系。
        # ============================================================
        for i, seg in enumerate(segs):
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            add_row({idx_tD + i: 1, idx_tD + i + 1: -1, idx_mD + i: -C},
                    seg.travel_time_down, seg.travel_time_down)

        # ============================================================
        # 约束 2：绿波带必须落在绿灯窗口里（上行）
        #
        # 设路口 i 选中的上行绿灯窗是 [start_i*C, end_i*C]。
        # 带子从 tU_i 开始，宽度是 b_up_i，所以带子占据：
        #     [tU_i, tU_i + b_up_i]
        #
        # 要整条带子都在绿灯里，就要：
        #     tU_i >= start_i*C
        #     tU_i + b_up_i <= end_i*C
        #
        # 因为窗口可能还没定，所以边界写成“选中的 δ 乘以对应 start/end”的和。
        # ============================================================
        for i in range(n):
            up_starts = {idx_opt[i][o]: wu.start * C
                         for o, (_, _, _, wu, _) in enumerate(options[i])}
            add_row({idx_tU + i: -1.0, **up_starts}, -np.inf, 0.0)
        for i in range(m):
            i0_ends = {idx_opt[i][o]: -wu.end * C
                       for o, (_, _, _, wu, _) in enumerate(options[i])}
            i1_ends = {idx_opt[i + 1][o]: -wu.end * C
                       for o, (_, _, _, wu, _) in enumerate(options[i + 1])}
            add_row({idx_tU + i: 1.0, idx_bU + i: 1.0, **i0_ends}, -np.inf, 0.0)
            add_row({idx_tU + i + 1: 1.0, idx_bU + i: 1.0, **i1_ends}, -np.inf, 0.0)

        # ============================================================
        # 约束 3：绿波带必须落在绿灯窗口里（下行）
        #
        # 和上行完全一样的道理，只是方向相反：
        #     tD_i >= start_i*C
        #     tD_i + b_down_i <= end_i*C
        # ============================================================
        for i in range(n):
            dn_starts = {idx_opt[i][o]: wd.start * C
                         for o, (_, _, _, _, wd) in enumerate(options[i])}
            add_row({idx_tD + i: -1.0, **dn_starts}, -np.inf, 0.0)
        for i in range(m):
            i0_ends = {idx_opt[i][o]: -wd.end * C
                       for o, (_, _, _, _, wd) in enumerate(options[i])}
            i1_ends = {idx_opt[i + 1][o]: -wd.end * C
                       for o, (_, _, _, _, wd) in enumerate(options[i + 1])}
            add_row({idx_tD + i: 1.0, idx_bD + i: 1.0, **i0_ends}, -np.inf, 0.0)
            add_row({idx_tD + i + 1: 1.0, idx_bD + i: 1.0, **i1_ends}, -np.inf, 0.0)

        # ============================================================
        # 约束 4：窗口带格
        #
        # B[d,k,j] 表示“连续 k 个路口”的公共带宽。
        # 公共带宽不能超过窗口内任何一段路的基础带宽：
        #     B[d,k,j] <= b[d,i]    i = j ... j+k-1
        #
        # 例如 k=3 时：
        #     B <= b_1, B <= b_2, B <= b_3
        # 所以 B 自动变成三段里最小的那个，也就是“公共瓶颈”。
        # ============================================================
        for d in ("up", "down"):
            b_idx = idx_bU if d == "up" else idx_bD
            for k in range(2, n + 1):
                for j in range(n - k + 1):
                    B_var = band_offset + band_model.B_idx[d][k][j]
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, b_idx + i: -1.0}, -np.inf, 0.0)

        # ============================================================
        # 约束 5：均衡组取 min
        #
        # 如果用户想让“上下行一样宽”，可以配置 BalanceGroup：
        #     B_g <= b_up, B_g <= b_down
        # 最大化 B_g 时，它自动变成两者的较小值。
        # ============================================================
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                band = _parse_band_key(member, n)
                var = _band_var(band_model, band, band_offset)
                add_row({gvar: 1.0, var: -1.0}, -np.inf, 0.0)

        # 绿波带层对齐损失约束：
        #   sense <= : expr - slack <= rhs
        #   sense >= : expr + slack >= rhs
        for spec, terms, slack in band_loss_resolved:
            row_coefs = {var: coef for var, coef in terms}
            if spec.sense == "<=":
                add_row({**row_coefs, slack: -1.0}, -np.inf, spec.rhs)
            elif spec.sense == ">=":
                add_row({**row_coefs, slack: 1.0}, spec.rhs, np.inf)
            else:
                raise ValueError("band alignment soft '=' constraint is not supported")

        res = milp(c=c,
                   constraints=LinearConstraint(np.array(rows), lo_list, hi_list),
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        sol.band_up_style = self.up_style
        sol.band_down_style = self.down_style
        if res.x is None:
            sol.status = "infeasible"
            return sol

        x = res.x
        seg_names = [s.name for s in segs]
        int_names = [v.name for v in ints]
        sol.objective = -float(res.fun)
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

        # 回填绿波带层损失和目标值。
        band_loss = 0.0
        for spec, terms, _ in band_loss_resolved:
            expr_val = sum(coef * float(x[var]) for var, coef in terms)
            if spec.sense == ">=":
                violation = max(0.0, spec.rhs - expr_val)
            elif spec.sense == "<=":
                violation = max(0.0, expr_val - spec.rhs)
            else:
                violation = 0.0
            band_loss += spec.penalty * violation

        band_score = -float(res.fun)
        sol.band_loss = float(band_loss)
        sol.band_score = float(band_score)
        sol.band_objective = float(band_score + band_loss_weight * band_loss)
        sol.intersection_loss = 0.0
        sol.total_phase_loss = 0.0

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
        sol.status = "optimal" if res.success else res.message
        return sol


def _parse_band_key(text: str, n: int):
    from .objective_config import parse_band_key
    return parse_band_key(text, n)


def _band_var(band_model: BandModel, band, offset: int) -> int:
    return offset + band_model.var_of(band)


def composite_config(up_weight: float = 1.0,                 # 上行全局带权重
                     down_weight: float = 1.0,               # 下行全局带权重
                     objective_mode: str = "sum",            # 目标模式：sum / balanced / balanced_composite
                     balance_eps: float = 0.1,               # 复合目标里 min 项的权重
                     balance_terms: tuple[str, ...] = ("up", "down")) -> ObjectiveConfig:
    """CompositeBandSolver 的预设目标配置。

    生成目标：
        sum                 -> max up_weight*b_up + down_weight*b_down
        balanced            -> max min(b_up, b_down)
        balanced_composite  -> max up_weight*b_up + down_weight*b_down
                                    + balance_eps * min(b_up, b_down)
    """
    if objective_mode == "sum":                              # 情况 1：纯加权和
        return ObjectiveConfig(sum_groups=[SumGroup({        # 创建一个 SumGroup
            "up.global": up_weight,                          # 上行全局带权重
            "down.global": down_weight,                      # 下行全局带权重
        })])

    if objective_mode == "balanced":                         # 情况 2：纯均衡
        members = [f"{d}.global" for d in balance_terms]     # 例如 ["up.global", "down.global"]
        return ObjectiveConfig(balance_groups=[BalanceGroup(members, weight=1.0)])  # max min(members)

    if objective_mode == "balanced_composite":               # 情况 3：加权和 + 均衡托底
        members = [f"{d}.global" for d in balance_terms]     # 均衡组包含哪些全局带
        return ObjectiveConfig(
            sum_groups=[SumGroup({                           # 1) 先加常规加权和
                "up.global": up_weight,                      # 上行全局带权重
                "down.global": down_weight,                  # 下行全局带权重
            })],
            balance_groups=[BalanceGroup(members, weight=balance_eps)],  # 2) 再加 eps*min
        )

    raise ValueError(f"unknown objective_mode: {objective_mode}")  # 未知模式直接报错


def oneway_config(up_weight: float = 1.0,                        # 上行全局带权重
                  window_weights: dict[int, float] | None = None,  # 任意窗口权重：{k: w_k}
                  segment_down_weights: dict[str, float] | None = None,  # 下行逐段权重（可选）
                  n_intersections: int = 0) -> ObjectiveConfig:    # 路口数量
    """OneWayPrioritySolver 的预设目标配置（下行分段 + 任意窗口带）。

    参数：
        up_weight: 上行全局带 b_up_global 的权重；
        window_weights: 下行窗口权重字典：
            {2: w2}          -> 只奖励每个下行路段；
            {2: w2, 3: w3}   -> 再奖励每个下行三路口窗口；
            {2: w2, 3: w3, 4: w4, ...} -> 支持任意 k <= n；
            如果不写 2，默认 w2 = 1.0；
        segment_down_weights: 下行逐段权重，优先级高于 window_weights[2]；
            例如 {"seg1": 2.0, "seg3": 0.5}；
        n_intersections: 路口数量 n，用于展开所有 k 窗口。

    生成目标：
        max up_weight * b_up_global
          + Σ 下行每段权重 * b_down_seg
          + Σ_k w_k * 所有下行 k 窗口带
    """
    ww = dict(window_weights or {})                              # 复制窗口权重，避免修改原字典
    if not ww:                                                   # 如果没有提供任何窗口权重
        ww = {2: 1.0}                                            # 默认只奖励下行每个路段

    seg_weights = dict(segment_down_weights or {})               # 复制逐段权重
    terms: dict[str, float] = {"up.global": up_weight}           # 先放上行全局带

    # k=2：每个下行路段。即使 ww 里没有 2，也用默认权重 1.0。
    w2 = ww.get(2, 1.0)                                          # k=2 的默认权重
    for i in range(n_intersections - 1):                         # 遍历所有相邻路口段
        seg_name = f"seg{i+1}"                                   # seg1, seg2, ...
        terms[f"down.{seg_name}"] = seg_weights.get(seg_name, w2)  # 该段权重

    # k>=3：任意窗口大小。
    for k, weight in ww.items():                                 # 遍历用户配置的每个 k
        if k == 2:                                               # k=2 已在上面处理
            continue
        if k < 2 or k > n_intersections:                         # 越界窗口直接报错
            raise ValueError(
                f"oneway_config: 非法窗口大小 k={k}, "
                f"要求 2 <= k <= {n_intersections}"
            )
        if weight <= 0:                                          # 非正权重不加入目标
            continue
        for start1 in range(1, n_intersections - k + 2):         # 起点路口编号从 I1 开始
            end1 = start1 + k - 1                                # 终点路口编号
            key = f"down.win{k}@I{start1}-I{end1}"               # 例如 down.win4@I1-I4
            terms[key] = weight                                  # 加入该窗口带

    return ObjectiveConfig(sum_groups=[SumGroup(terms)])         # 所有项放进一个 SumGroup


def make_composite_solver(down_weight: float = 1.0,
                          up_weight: float = 1.0,
                          max_loops: int = 3,
                          max_window: int = 3,
                          objective_mode: str = "sum",
                          balance_eps: float = 0.1,
                          balance_terms: tuple[str, ...] = ("up", "down")) -> FlexibleBandSolver:
    """旧 CompositeBandSolver 的薄工厂。"""
    cfg = composite_config(up_weight=up_weight,
                           down_weight=down_weight,
                           objective_mode=objective_mode,
                           balance_eps=balance_eps,
                           balance_terms=balance_terms)
    return FlexibleBandSolver(cfg, max_loops=max_loops,
                              name="composite-band",
                              up_style="global", down_style="global",
                              up_global_output=True, down_global_output=True)


def make_oneway_solver(up_weight: float = 1.0,
                       window_weights: dict[int, float] | None = None,
                       segment_down_weights: dict[str, float] | None = None,
                       max_loops: int = 3,
                       n_intersections: int | None = None) -> FlexibleBandSolver:
    """旧 OneWayPrioritySolver 的薄工厂。"""
    n = n_intersections or 0
    cfg = oneway_config(up_weight=up_weight,
                        window_weights=window_weights,
                        segment_down_weights=segment_down_weights,
                        n_intersections=n)
    return FlexibleBandSolver(cfg, max_loops=max_loops,
                              name="one-way-priority",
                              up_style="global", down_style="local",
                              up_global_output=True, down_global_output=False)
