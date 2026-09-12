"""FullFlexiblePhaseTuneSolver：相位变量 + BandModel + ObjectiveConfig。

这个模块是“新架构”的核心求解器。它把原来的 PhaseTuneSolver 从
“自己一套相位/带宽模型”改成下面这条统一链路：

    g_{i,p}  ──►  绿灯窗  ──►  基础段带宽 b[d,i]
                                 │
                                 ▼
                        窗口带格 B[d,k,j] <= b[d,i]
                                 │
                                 ▼
                         ObjectiveConfig 决定主目标

其中：
- g_{i,p}：路口 i 第 p 个相位的绿灯时长（秒，连续变量）。
- b[d,i]：方向 d 第 i 段的基础带宽（秒，连续变量）。
- B[d,k,j]：方向 d、从路口 j 开始连续 k 个路口的窗口带宽（秒）。
- tU_i / tD_i：上行/下行带前沿到达路口 i 的时刻（秒）。
- mU_i / mD_i：上行/下行的整周期圈数（整数变量）。

约束链路：
1. 相位约束：
       min_green_{i,p} <= g_{i,p} <= max_green_{i,p}
       Σ_p g_{i,p} + Σ_p phase_lost_times_p + lost_time_i = C
2. 相位生成绿灯窗：
       start(P_k) = Σ_{j<k} (g_j + phase_lost_times_j)
       end(P_k)   = start(P_k) + g_k
       up_start_i   = start(up_phase)
       up_end_i     = end(up_phase)
       下行同理；未显式设置 up_phase/down_phase 时从 Phase.serves 解析。
3. 基础段带宽必须落在路段两端路口的绿灯窗内：
       tU_i     + b_up_i <= up_end_i
       tU_{i+1} + b_up_i <= up_end_{i+1}
       tU_i     >= up_start_i
       tU_{i+1} >= up_start_{i+1}
       下行同理。
4. 带前沿传递：
       tU_{i+1} = tU_i + tau_up_i + C * mU_i
       tD_i     = tD_{i+1} + tau_down_i + C * mD_i
5. 窗口带格：
       B[d,k,j] <= b[d,i]    i = j ... j+k-1
6. 目标：
       ObjectiveConfig 里的 SumGroup / BalanceGroup 统一翻译成 c 向量。
7. 损失系统：
       PhaseLossBuilder 的 hinge loss、LinearSpec 软约束 slack、
       AlignmentLossBuilder 的对齐损失进入 band_loss；
       EpsilonConstraintRunner 再做
           max band_score s.t. intersection_loss <= eps。

代码组织说明：
- 变量在一条一维向量 x 中排列，idx_* 保存各变量块的起始下标。
- BandModel 负责 b 和 B 的局部索引，本模块负责把它们映射到全局 x。
- 本求解器仍然是 MILP：mU/mD 是整数变量，其余都是连续变量。
"""


from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..models import Arterial
from ..solution import Solution
from .band_model import BandModel, fill_solution_window_bands
from .base import Solver
from .objective_config import ObjectiveConfig, parse_band_key
from .phase import (ConstraintBuilder, LinearExpr, LinearSpec,
                    PhaseLossBuilder, PhaseLossSpec, window_exprs)


class FullFlexiblePhaseTuneSolver(Solver):
    """完整新架构求解器：相位变量 + BandModel + ObjectiveConfig + 损失/约束。

    它不负责“选方案”——方案由 prior.plan_choices 锁定；
    如果 prior 为空，则每个路口退化为第一个方案。

    Attributes:
        config: ObjectiveConfig，主目标声明。
        max_loops: mU/mD 整数圈数的绝对值上界，默认 3。
        mode: "global" 或 "oneway"，用于：
            - 选择 AlignmentLossBuilder 的下行表达式；
            - 决定默认输出风格；
            - 决定 window_bands 是否回填。
        up_global_output: 若为 True，Solution.bandwidth_up 填 B_up_global；
            否则逐路段填 b_up_i。
        down_global_output: 若为 True，Solution.bandwidth_down 填
            B_down_global；否则逐路段填 b_down_i。
        up_style / down_style: 绘图使用的 global/local 风格标记。
    """

    name = "full-flexible-phase"

    def __init__(self,
                 config: ObjectiveConfig,
                 max_loops: int = 3,
                 mode: str = "global",
                 up_global_output: bool = True,
                 down_global_output: bool = False,
                 up_style: str | None = None,
                 down_style: str | None = None) -> None:
        """初始化求解器。

        Args:
            config: 主目标配置。
            max_loops: mU/mD 整数圈数上界，防止变量无界。
            mode: "global" 或 "oneway"。
            up_global_output: 上行输出全局带还是逐路段带。
            down_global_output: 下行输出全局带还是逐路段带。
            up_style: 绘图风格，默认由 up_global_output 推导。
            down_style: 绘图风格，默认由 down_global_output 推导。
        """
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
        """生成“变量下界”约束的行系数。

        数学形式：
            var >= expr.const + Σ_j coef_j * g_j

        移项后：
            var - Σ_j coef_j * g_j >= expr.const

        Returns:
            {变量下标: 系数}，可直接交给 add_row。
        """
        # 变量 var 的系数固定为 +1。
        row = {var: 1.0}
        # expr.coefs 里使用的是相位在 plan.phases 中的局部下标 j，
        # 这里要用 phase_indices[j] 映射到 MILP 全局变量下标。
        for j, coef in expr.coefs.items():
            row[phase_indices[j]] = row.get(phase_indices[j], 0.0) - coef
        return row

    @staticmethod
    def _upper_phase_row(t_var: int, b_var: int, expr: LinearExpr,
                         phase_indices: list[int]) -> dict[int, float]:
        """生成“带子末端不超过绿灯窗终点”的行系数。

        数学形式：
            t_var + b_var <= expr.const + Σ_j coef_j * g_j

        移项后：
            t_var + b_var - Σ_j coef_j * g_j <= expr.const

        Returns:
            {变量下标: 系数}，其中 t_var 和 b_var 系数都是 +1。
        """
        # 带前沿 t 和带宽 b 的系数都是 +1。
        row = {t_var: 1.0, b_var: 1.0}
        # 窗口终点表达式中的相位项移到不等式左边后变为负号。
        for j, coef in expr.coefs.items():
            row[phase_indices[j]] = row.get(phase_indices[j], 0.0) - coef
        return row

    def solve(self, arterial: Arterial,
              prior: Solution | None = None,
              loss_builder: PhaseLossBuilder | None = None,
              constraint_builder: ConstraintBuilder | None = None,
              alignment_builder=None,
              max_loss: float | None = None,
              max_intersection_loss: float | None = None,
              band_loss_weight: float = 0.0,
              objective: str = "bandwidth",
              tunable_intersections: set[str] | None = None) -> Solution:
        """求解第二阶段相位/带宽 MILP。

        Args:
            arterial: 干线数据。
            prior: Stage 1 的解，用 plan_choices 锁定每个路口的方案。
                为 None 时取每个路口的第一个方案。
            loss_builder: 相位 hinge 损失声明，进入 intersection_loss。
            constraint_builder: 线性硬/软约束声明；软约束进入
                intersection_loss。
            alignment_builder: 绿波带层对齐损失声明，进入 band_loss；
                通过 band_loss_weight 以加权和形式进入 band_score。
            max_loss: 兼容旧接口。等价于 max_intersection_loss。
            max_intersection_loss: 交叉口损失上界，添加
                intersection_loss <= max_intersection_loss。
            band_loss_weight: band_loss 的权重 λ。
                objective="bandwidth" 时目标为：
                    max band_objective - λ * band_loss
            objective: "bandwidth"/"band_score" 或 "loss"/"intersection_loss"。
            tunable_intersections: 允许调整相位的路口名集合；
                None 表示全部可调。

        Returns:
            Solution。若 MILP 不可行，status="infeasible"。
        """
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
        # selected[i] 表示第 i 个路口最终锁定的 SignalPlan。
        # 这里不做 0-1 方案选择，因为 Stage 1 已经用 prior.plan_choices 选好了。
        selected = []
        for inter in ints:
            if not inter.plans:
                raise ValueError(f"路口 {inter.name} 没有可选方案")
            # prior 可能为空；空则退化为每个路口的第一个方案。
            name = (prior.plan_choices.get(inter.name)
                    if prior and prior.plan_choices else None)
            plan = inter.plan_by_name(name) if name else inter.plans[0]
            selected.append(plan)

        # window_exprs 把“相位结构”转成绿灯窗边界的线性表达式：
        #   up_start_i = Σ_{up_phase 之前的相位} g
        #   up_end_i   = up_start_i + g_{up_phase}
        # 如果方案没有 phases，则表达式中只有常数项（固定窗口）。
        exprs = [window_exprs(plan, C) for plan in selected]

        # 每个路口的相位变量个数。
        n_phases = [len(plan.phases) for plan in selected]

        # tunable[i] 表示第 i 个路口的相位是否允许调整。
        # None 表示全部可调；否则只有集合里的路口可调。
        tunable = [
            (tunable_intersections is None or name in tunable_intersections)
            for name in int_names
        ]

        # ------------------------------------------------------------------
        # 1) 变量布局
        #    g_{i,p} -> tU -> mU -> tD -> mD -> BandModel(b, B) -> balance -> loss -> slack
        # ------------------------------------------------------------------
        # cur 是“下一个空闲变量下标”。每声明一个变量块，就把 cur 推进。
        cur = 0

        # 变量块 1：g_{i,p}。idx_g[i][p] 是路口 i 第 p 个相位的全局变量下标。
        idx_g: list[list[int]] = []
        for i in range(n):
            idx_g.append(list(range(cur, cur + n_phases[i])))
            cur += n_phases[i]

        # 变量块 2：tU_i、mU_i、tD_i、mD_i。
        # 注意 m 是路段数 = n-1，因为 t 有 n 个、m 有 m 个。
        idx_tU = cur; cur += n      # 上行带前沿 tU_0..tU_{n-1}
        idx_mU = cur; cur += m      # 上行圈数 mU_0..mU_{m-1}
        idx_tD = cur; cur += n      # 下行带前沿 tD_0..tD_{n-1}
        idx_mD = cur; cur += m      # 下行圈数 mD_0..mD_{m-1}

        # 变量块 3：BandModel 同时提供基础段 b[d,i] 和窗口带 B[d,k,j]。
        # band_model.nvar = 两个方向的基础段 + 两个方向的窗口带。
        band_model = BandModel(n, m)
        band_offset = cur
        # b_idx["up"]/["down"] 是相对 band_offset 的局部下标。
        # 因为每个方向的基础段连续排列，所以这里取第 0 个再往后加 i 即可。
        idx_bU = band_offset + band_model.b_idx["up"][0]
        idx_bD = band_offset + band_model.b_idx["down"][0]
        cur += band_model.nvar

        # 变量块 4：BalanceGroup 的组最小值变量 B_group。
        # 每个组一个变量，约束 B_group <= 每个成员带。
        balance_vars: dict[int, int] = {}
        for gidx in range(len(self.config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1

        # 变量块 5：hinge 损失变量 ℓ_{i,p,side}。
        # lower 对应 g 过小，upper 对应 g 过大；每个变量都有下界 0。
        loss_vars: list[tuple[int, int, int, PhaseLossSpec, str]] = []
        if loss_builder is not None:
            for i, plan in enumerate(selected):
                for p, ph in enumerate(plan.phases):
                    # 按“相位名 + 路口名”查找损失配置。
                    spec = loss_builder.spec_for(ph.name, int_names[i])
                    if spec is None:
                        continue
                    # 过小惩罚变量：ℓ >= threshold - g。
                    loss_vars.append((i, p, cur, spec, "lower"))
                    cur += 1
                    # 如果配置了过大约束，再创建一个 upper 变量。
                    if spec.upper_threshold is not None:
                        loss_vars.append((i, p, cur, spec, "upper"))
                        cur += 1

        # ------------------------------------------------------------------
        # 2) 解析声明式约束（LinearSpec / AlignmentLossBuilder 转换来的软约束）
        # ------------------------------------------------------------------
        def band_var(key: str) -> int:
            """把 ObjectiveConfig 的带标识解析成 MILP 全局变量下标。

            例如："up.global" -> B_up 全局带变量；
                  "down.seg2" -> b_down_2 基础段变量；
                  "down.win3@I1-I3" -> 对应窗口带变量。
            """
            # parse_band_key 把字符串解析成 BandKey(direction, k, start)。
            band = parse_band_key(key, n)
            # BandModel 给出的是局部下标，加上 band_offset 后才是全局下标。
            return band_offset + band_model.var_of(band)

        def resolve_special(name: str) -> int | None:
            """把约束里的特殊变量名映射到 MILP 全局变量下标。

            支持：
                "b_up"       -> 上行全局带 B_up_global
                "b_down"     -> 下行全局带 B_down_global
                "B_bal"      -> 第一个 BalanceGroup 的组最小值变量
                "tU_I2"      -> 上行带前沿时刻 tU_2
                "tD_seg1"    -> 下行带前沿时刻 tD_1
                "bD_seg1"    -> 下行第 1 段基础带宽 b_down_1

            Returns:
                全局变量下标；无法识别时返回 None。
            """
            # b_up / b_down 是旧接口里最常用的两个全局带别名。
            if name == "b_up":
                return band_var("up.global")
            if name == "b_down":
                return band_var("down.global")
            # B_bal 对应第一个 BalanceGroup 的组最小值变量。
            if name == "B_bal":
                return balance_vars.get(0)

            # tU_I2 / tD_I2 这类名字：按路口名找到 t 变量下标。
            if name.startswith("tU_") or name.startswith("tD_"):
                iname = name[3:]  # 去掉 "tU_" 或 "tD_"
                i = name_to_i.get(iname)
                # 兼容 "tU_2" 这种纯数字写法。
                if i is None and iname.startswith("I"):
                    try:
                        i = int(iname[1:]) - 1
                    except ValueError:
                        i = None
                if i is None or not (0 <= i < n):
                    return None
                return (idx_tU + i) if name.startswith("tU_") else (idx_tD + i)

            # bD_seg1 这类名字：按路段名找到下行基础带宽变量下标。
            if name.startswith("bD_"):
                sname = name[3:]  # 去掉 "bD_"
                si = seg_name_to_idx.get(sname)
                # 兼容 bD_seg1 这种按编号的写法。
                if si is None and sname.startswith("seg"):
                    try:
                        si = int(sname[3:]) - 1
                    except ValueError:
                        si = None
                if si is None or not (0 <= si < m):
                    return None
                return idx_bD + si

            # 不是特殊变量名，交给 resolve_phase 继续尝试。
            return None

        def resolve_phase(name: str) -> list[int]:
            """把相位名映射到 MILP 全局变量下标列表。

            支持两种写法：
                "I2.P1" -> 只匹配路口 I2 的 P1 相位；
                "P1"    -> 匹配所有选中方案里的 P1 相位。

            Returns:
                匹配到的相位变量下标列表；可能为空。
            """
            out: list[int] = []
            # 情况 1：带点的 "I2.P1"，只匹配指定路口。
            if "." in name:
                iname, pname = name.split(".", 1)
                i = name_to_i.get(iname)
                if i is None:
                    return out
                # 在该路口的相位列表里找同名相位。
                p = next((j for j, ph in enumerate(selected[i].phases)
                          if ph.name == pname), None)
                if p is not None and p < len(idx_g[i]):
                    out.append(idx_g[i][p])
                return out

            # 情况 2：不带点的 "P1"，对所有选中方案里的同名相位生效。
            pname = name
            for i, plan in enumerate(selected):
                p = next((j for j, ph in enumerate(plan.phases)
                          if ph.name == pname), None)
                if p is not None and p < len(idx_g[i]):
                    out.append(idx_g[i][p])
            return out

        # ============================================================
        # 两类约束分开解析：
        #   band_specs         -> 绿波带层 loss，进入 band_loss
        #   intersection_specs -> 路口/相位层 loss，进入 intersection_loss
        #
        # AlignmentLossBuilder 本质是绿波带层损失；
        # constraint_builder 里的软约束则视为交叉口/相位层损失。
        # ============================================================
        band_specs: list[LinearSpec] = []
        if alignment_builder is not None:
            band_specs.extend(
                alignment_builder.to_linear_specs(self.mode, int_names, seg_names)
            )

        intersection_specs: list[LinearSpec] = []
        if constraint_builder is not None:
            intersection_specs.extend(constraint_builder.specs)

        # resolved_constraints 的每项是：
        #   (原始 LinearSpec, [(MILP变量下标, 系数), ...], slack变量下标或None, kind)
        # kind ∈ {"band", "intersection"}。
        # 解析失败、一个变量都没匹配到的约束会被跳过。
        resolved_constraints: list[
            tuple[LinearSpec, list[tuple[int, float]], int | None, str]
        ] = []

        for kind, specs in (("band", band_specs),
                            ("intersection", intersection_specs)):
            for spec in specs:
                terms: list[tuple[int, float]] = []
                for name, coef in spec.terms.items():
                    # 先尝试特殊变量名：b_up / tU_* / bD_* / B_bal 等。
                    var = resolve_special(name)
                    if var is not None:
                        terms.append((var, coef))
                        continue
                    # 再尝试相位名：I2.P1 或全局 P1。
                    for pvar in resolve_phase(name):
                        terms.append((pvar, coef))
                if not terms:
                    continue
                # 软约束需要一个非负 slack 变量，目标里会给它 penalty。
                slack = None
                if spec.soft:
                    slack = cur
                    cur += 1
                resolved_constraints.append((spec, terms, slack, kind))

        nvar = cur

        # ------------------------------------------------------------------
        # 3) 目标函数：把 ObjectiveConfig 翻译成 c 向量
        # ------------------------------------------------------------------
        # scipy.optimize.milp 默认求 min c^T x。
        # 我们的带宽目标是 max，所以带宽系数要写成 -weight。
        effective_max_intersection = (
            max_intersection_loss if max_intersection_loss is not None else max_loss
        )

        c = np.zeros(nvar)
        if objective in ("loss", "intersection_loss"):
            # 目标 = min intersection_loss。
            # 只最小化相位 hinge 和交叉口软约束 slack；
            # alignment 属于 band_loss，不进入这个目标。
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                c[var] += slope
            for spec, _, slack, kind in resolved_constraints:
                if kind == "intersection" and spec.soft and slack is not None:
                    c[slack] += spec.penalty
        else:
            # 目标 = max band_objective - band_loss_weight * band_loss。
            # 先加入 ObjectiveConfig 的带宽收益。
            for group in self.config.sum_groups:
                for key, weight in group.terms.items():
                    c[band_var(key)] += -weight
            for gidx, group in enumerate(self.config.balance_groups):
                gvar = balance_vars[gidx]
                c[gvar] += -group.weight
                if group.eps > 0:
                    for member in group.members:
                        c[band_var(member)] += -group.weight * group.eps

            # 再把带层损失以加权和形式放进目标：
            #   max ... - λ * band_loss
            # milp 求 min，所以 band slack 的系数写成 +λ*penalty。
            for spec, _, slack, kind in resolved_constraints:
                if kind == "band" and spec.soft and slack is not None:
                    c[slack] += band_loss_weight * spec.penalty

            # ε-约束扫描时，给交叉口损失一个极小的二次权重。
            # 这不改变“带宽优先”的主目标，但能让同一带宽下的输出
            # 尽量落在 intersection_loss 最小点。
            if effective_max_intersection is not None:
                tiny = 1e-7
                for _, _, var, spec, side in loss_vars:
                    slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                    c[var] += tiny * slope
                for spec, _, slack, kind in resolved_constraints:
                    if kind == "intersection" and spec.soft and slack is not None:
                        c[slack] += tiny * spec.penalty

        # ------------------------------------------------------------------
        # 4) 变量界
        # ------------------------------------------------------------------
        # lb/ub 是每个变量的下界/上界向量。
        # 默认连续变量下界为 0、上界为 +inf，后面再逐块覆盖。
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)

        # 相位 g：可调路口用 [min_green, max_green]；
        # 不可调路口用 [green, green] 固定成初始值。
        for i, plan in enumerate(selected):
            for p, ph in enumerate(plan.phases):
                if tunable[i]:
                    lb[idx_g[i][p]] = ph.min_green
                    ub[idx_g[i][p]] = ph.max_green
                else:
                    lb[idx_g[i][p]] = ph.green
                    ub[idx_g[i][p]] = ph.green

        # 带前沿时刻在 [0, C] 内。
        ub[idx_tU:idx_tU + n] = C
        ub[idx_tD:idx_tD + n] = C

        # 圈数 m 是整数，限制在 [-max_loops, max_loops]。
        lb[idx_mU:idx_mU + m] = -self.max_loops
        ub[idx_mU:idx_mU + m] = self.max_loops
        lb[idx_mD:idx_mD + m] = -self.max_loops
        ub[idx_mD:idx_mD + m] = self.max_loops

        # b 和 B 全部非负，并以周期 C 为上界。
        lb[band_offset:band_offset + band_model.nvar] = 0.0
        ub[band_offset:band_offset + band_model.nvar] = C

        # 均衡组最小值变量、损失变量、软约束 slack 都非负且不超过 C。
        for gvar in balance_vars.values():
            ub[gvar] = C
        for _, _, var, _, _ in loss_vars:
            ub[var] = C
        for _, _, slack, _ in resolved_constraints:
            if slack is not None:
                ub[slack] = C

        # integrality=0 表示连续变量，=1 表示整数变量。
        # 只有 mU/mD 是整数，其余 g、t、b、B、slack 都是连续变量。
        integrality = np.zeros(nvar)
        integrality[idx_mU:idx_mU + m] = 1
        integrality[idx_mD:idx_mD + m] = 1

        # 下面开始逐行组装线性约束：
        #   rows[k] 是第 k 行的系数向量，
        #   lo_list[k] <= rows[k] @ x <= hi_list[k]。
        rows: list[np.ndarray] = []
        lo_list: list[float] = []
        hi_list: list[float] = []

        def add_row(coefs: dict[int, float], lo: float, hi: float) -> None:
            """添加一行线性约束 lo <= Σ coefs[j] * x[j] <= hi。

            Args:
                coefs: {MILP变量下标: 系数}。
                lo: 表达式下界；用 -np.inf 表示只有上界。
                hi: 表达式上界；用 np.inf 表示只有下界。

            常见用法：
                add_row({v: 1.0}, 3.0, 3.0)  -> v = 3
                add_row({v: 1.0}, 3.0, np.inf) -> v >= 3
                add_row({v: 1.0}, -np.inf, 3.0) -> v <= 3
            """
            # 先建一条全 0 的行，再按 coefs 填入系数。
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
            # 等式：Σ_p g_{i,p} = C - 总损失时间。
            # 总损失 = phase_lost_times + 尾部 lost_time。
            # add_row 的 lo=hi 表示等式约束。
            total = {idx_g[i][p]: 1.0 for p in range(len(plan.phases))}
            total_loss = plan.total_lost_time()
            add_row(total, C - total_loss, C - total_loss)

        # ------------------------- hinge 损失约束 -------------------------
        # 过小惩罚：ℓ >= threshold - g   ->   ℓ + g >= threshold
        # 过大惩罚：ℓ >= g - upper_threshold -> ℓ - g >= -upper_threshold
        for i, p, var, spec, side in loss_vars:
            if side == "lower":
                # ℓ + g >= threshold  <=>  ℓ >= threshold - g。
                # 目标是最小化 ℓ，所以最优时 ℓ = max(0, threshold-g)。
                add_row({var: 1.0, idx_g[i][p]: 1.0}, spec.threshold, np.inf)
            else:
                # ℓ - g >= -upper_threshold  <=>  ℓ >= g - upper_threshold。
                # 最优时 ℓ = max(0, g-upper_threshold)。
                add_row({var: 1.0, idx_g[i][p]: -1.0},
                        -spec.upper_threshold, np.inf)

        # ------------------------- 声明式约束 -------------------------
        for spec, terms, slack, kind in resolved_constraints:
            # terms 是 [(变量下标, 系数), ...]，转成 add_row 需要的字典。
            row_coefs = {var: coef for var, coef in terms}
            if not spec.soft:
                # 硬约束：直接按 sense 加行。
                if spec.sense == ">=":
                    add_row(row_coefs, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row(row_coefs, -np.inf, spec.rhs)
                elif spec.sense == "=":
                    add_row(row_coefs, spec.rhs, spec.rhs)
                else:
                    raise ValueError(f"unknown sense: {spec.sense}")
            else:
                # 软约束：
                #   >= 时：expr + slack >= rhs，slack >= max(0, rhs-expr)。
                #   <= 时：expr - slack <= rhs，slack >= max(0, expr-rhs)。
                # slack 在目标中带 penalty，因此会被尽量压到最小。
                if spec.sense == ">=":
                    add_row({**row_coefs, slack: 1.0}, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row({**row_coefs, slack: -1.0}, -np.inf, spec.rhs)
                else:
                    raise ValueError("soft '=' constraint is not supported")

        # ------------------------- max_intersection_loss 约束 -------------------------
        if effective_max_intersection is not None:
            # 构造 intersection_loss = Σ slope*ℓ + Σ penalty*slack，
            # 然后添加约束 intersection_loss <= max_intersection_loss。
            # alignment 属于 band_loss，不进入这个约束。
            loss_coefs: dict[int, float] = {}
            for _, _, var, spec, side in loss_vars:
                slope = spec.slope if side == "lower" else (spec.upper_slope or spec.slope)
                loss_coefs[var] = loss_coefs.get(var, 0.0) + slope
            for spec, _, slack, kind in resolved_constraints:
                if kind == "intersection" and spec.soft and slack is not None:
                    loss_coefs[slack] = loss_coefs.get(slack, 0.0) + spec.penalty
            add_row(loss_coefs, -np.inf, effective_max_intersection)

        # ------------------------- 带前沿传递 -------------------------
        for i, seg in enumerate(segs):
            # 上行：tU_{i+1} - tU_i - C*mU_i = tau_up_i。
            # add_row 的 lo=hi=travel_time_up 表示等式。
            add_row({idx_tU + i + 1: 1, idx_tU + i: -1, idx_mU + i: -C},
                    seg.travel_time_up, seg.travel_time_up)
            # 下行：tD_i - tD_{i+1} - C*mD_i = tau_down_i。
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
        # 每个路口先加“带前沿不能早于窗口起点”的下界约束。
        for i, expr in enumerate(exprs):
            # tU_i >= up_start_i。
            add_row(self._lower_phase_row(idx_tU + i, expr.up_start, idx_g[i]),
                    expr.up_start.const, np.inf)
            # tD_i >= down_start_i。
            add_row(self._lower_phase_row(idx_tD + i, expr.down_start, idx_g[i]),
                    expr.down_start.const, np.inf)

        # 每个路段 i 有 4 条“带子末端不超过绿灯窗终点”的约束：
        # 上行在左端路口、上行在右端路口、下行在左端、下行在右端。
        for i in range(m):
            # expr0 是路段左端路口 i 的窗口表达式，
            # expr1 是路段右端路口 i+1 的窗口表达式。
            expr0, expr1 = exprs[i], exprs[i + 1]

            # tU_i + b_up_i <= up_end_i
            add_row(self._upper_phase_row(idx_tU + i, idx_bU + i,
                                          expr0.up_end, idx_g[i]),
                    -np.inf, expr0.up_end.const)
            # tU_{i+1} + b_up_i <= up_end_{i+1}
            add_row(self._upper_phase_row(idx_tU + i + 1, idx_bU + i,
                                          expr1.up_end, idx_g[i + 1]),
                    -np.inf, expr1.up_end.const)
            # tD_i + b_down_i <= down_end_i
            add_row(self._upper_phase_row(idx_tD + i, idx_bD + i,
                                          expr0.down_end, idx_g[i]),
                    -np.inf, expr0.down_end.const)
            # tD_{i+1} + b_down_i <= down_end_{i+1}
            add_row(self._upper_phase_row(idx_tD + i + 1, idx_bD + i,
                                          expr1.down_end, idx_g[i + 1]),
                    -np.inf, expr1.down_end.const)

        # ------------------------- 窗口带格 -------------------------
        # 窗口带格：B[d,k,j] 不能超过它覆盖的任意一段基础带宽。
        # 对每个方向 d、每个窗口大小 k、每个起点 j：
        #   B[d,k,j] <= b[d,i]   for i = j ... j+k-1
        for d in ("up", "down"):
            # 根据方向选择基础带宽块在 x 中的起始下标。
            b_base = idx_bU if d == "up" else idx_bD
            for k in range(2, n + 1):
                for j in range(n - k + 1):
                    # 这个窗口带变量在全局 x 中的下标。
                    B_var = band_offset + band_model.B_idx[d][k][j]
                    # 窗口带覆盖的路段是 j ... j+k-1 中相邻段的下标 j...j+k-2，
                    # 因此 Python range 写 range(j, j + k - 1)。
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, b_base + i: -1.0}, -np.inf, 0.0)

        # ------------------------- 均衡组约束 -------------------------
        # 均衡组：B_group <= 每个成员带。
        # 最大化 B_group 时，它会自动等于成员里的最小值。
        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                add_row({gvar: 1.0, band_var(member): -1.0}, -np.inf, 0.0)

        # 把所有行打包成 scipy 的 LinearConstraint。
        # rows 为空时传空 tuple，避免构造空矩阵。
        constraints = (LinearConstraint(np.array(rows), np.array(lo_list), np.array(hi_list))
                       if rows else ())
        # 调用 HiGHS（通过 scipy.optimize.milp）求解。
        # c 是最小化系数，bounds 是变量界，integrality 指定整数变量。
        res = milp(c=c,
                   constraints=constraints,
                   bounds=Bounds(lb, ub),
                   integrality=integrality)

        sol = Solution(cycle=C, solver_msg=f"HiGHS via scipy: success={res.success}")
        if res.x is None:
            # 不可行时没有最优解，直接返回。
            sol.status = "infeasible"
            return sol

        x = res.x
        # milp 求的是 min c^T x；带宽目标系数取过负，所以取负还原展示值。
        sol.objective = -float(res.fun)
        # 告诉绘图模块：上/下行是画一条全局带，还是画逐路段/窗口带。
        sol.band_up_style = self.up_style
        sol.band_down_style = self.down_style
        # 锁定方案回填。
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}

        # 相位时长回填：路口名 -> 相位名 -> 秒。
        sol.phase_times = {}
        for i, plan in enumerate(selected):
            if plan.phases:
                sol.phase_times[int_names[i]] = {
                    ph.name: float(x[idx_g[i][p]])
                    for p, ph in enumerate(plan.phases)
                }

        # ------------------------- 结果提取 -------------------------
        # 上行带宽：
        #   up_global_output=True  -> 输出 B_up_global，所有路段展示同一个值；
        #   up_global_output=False -> 输出每个路段自己的 b_up_i。
        if self.up_global_output:
            b_up_global = float(x[band_offset + band_model.B_idx["up"][n][0]])
            sol.bandwidth_up = {name: b_up_global for name in seg_names}
        else:
            sol.bandwidth_up = {seg_names[i]: float(x[idx_bU + i]) for i in range(m)}

        # 下行带宽：逻辑同上，只是一个输出 B_down_global，另一个输出 b_down_i。
        if self.down_global_output:
            b_down_global = float(x[band_offset + band_model.B_idx["down"][n][0]])
            sol.bandwidth_down = {name: b_down_global for name in seg_names}
        else:
            sol.bandwidth_down = {seg_names[i]: float(x[idx_bD + i]) for i in range(m)}

        # 带前沿时刻回填，绘制时空图时会根据路段行驶时间递推绝对轨迹。
        sol.band_start_up = {name: float(x[idx_tU + i]) for i, name in enumerate(int_names)}
        sol.band_start_down = {name: float(x[idx_tD + i]) for i, name in enumerate(int_names)}

        # 后处理：不修改优化变量，用最终带前沿 t 和最终绿灯窗
        # 重新计算两个方向、k=2..5 的可行窗口绿波带，供 plot 绘制。
        fill_solution_window_bands(sol, arterial, max_window=5)

        # ------------------------- 两类损失回填 -------------------------
        # 1) 绿波带层损失：band_loss。
        #    只统计 kind="band" 的软约束违反量，典型来源是 AlignmentLossBuilder。
        band_loss = 0.0
        # 2) 交叉口/相位层损失：intersection_loss。
        #    包含相位 hinge loss 和 kind="intersection" 的软约束违反量。
        intersection_loss = 0.0

        # 相位 hinge 损失。
        for i, p, _, spec, side in loss_vars:
            g = float(x[idx_g[i][p]])
            if side == "lower":
                intersection_loss += spec.slope * max(0.0, spec.threshold - g)
            else:
                intersection_loss += ((spec.upper_slope or spec.slope)
                                      * max(0.0, g - spec.upper_threshold))

        # 软约束违反量 * penalty，按 kind 分别汇总。
        for spec, terms, _, kind in resolved_constraints:
            if not spec.soft:
                continue
            expr_val = sum(coef * float(x[var]) for var, coef in terms)
            if spec.sense == ">=":
                violation = max(0.0, spec.rhs - expr_val)
            elif spec.sense == "<=":
                violation = max(0.0, expr_val - spec.rhs)
            else:
                violation = 0.0
            weighted = spec.penalty * violation
            if kind == "band":
                band_loss += weighted
            else:
                intersection_loss += weighted

        # 原始绿波带收益 band_objective：
        #   SumGroup: Σ weight * band_value
        #   BalanceGroup: weight * B_group + weight*eps * Σ member_value
        band_objective = 0.0
        for group in self.config.sum_groups:
            for key, weight in group.terms.items():
                band_objective += weight * float(x[band_var(key)])
        for gidx, group in enumerate(self.config.balance_groups):
            band_objective += group.weight * float(x[balance_vars[gidx]])
            if group.eps > 0:
                for member in group.members:
                    band_objective += group.weight * group.eps * float(x[band_var(member)])

        sol.band_objective = float(band_objective)
        sol.band_loss = float(band_loss)
        sol.band_score = float(band_objective - band_loss_weight * band_loss)
        sol.intersection_loss = float(intersection_loss)
        sol.total_phase_loss = float(intersection_loss)

        # SciPy 的 success 不一定等价于“最优”状态字符串，这里沿用原策略：
        # success=True 写 optimal，否则把求解器消息写进 status。
        sol.status = "optimal" if res.success else res.message
        return sol
