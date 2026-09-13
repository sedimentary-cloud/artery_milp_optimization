"""多段绿波带第一阶段求解器。

当前版本统一了两类 Stage 1 能力：
- 多段绿波带传播；
- 路口多方案选择；
- ObjectiveConfig 目标 DSL；

约束说明：
- 每个方向建模固定数量 band，band 在每个路口自由选择候选方案中的绿灯窗口；
- 同一方向多条 band 通过顺序约束保持时间上不重叠，避免复制解刷爆目标；
- 全局带、分段带和局部窗口带都使用相同的“band 编号固定、窗口自由选择”语义。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...models import Arterial, SignalPlan
from ...solution import Solution
from ..builders.signal_constraints import scale_linear_spec
from ..builders.term_validation import (TermValidationContext,
                                        common_endpoint_terms)
from ..core.band_lattice import (fill_solution_local_band_records,
                                 fill_solution_missing_window_bands)
from ..core.base import Solver, compute_effective_max_loops
from ..core.margin import BandMarginConfig
from ..core.objective import ObjectiveConfig, parse_band_key


BandInstance = tuple[str, int]
LatticeInstance = tuple[str, int, int, int]


def _evaluate_plan_signal_losses(selected: list[SignalPlan], cycle: float) -> float:
    """按名义端点计算选中方案的 plan 级 SignalLoss。

    Stage 1 不建模段端点微调，因此选中方案的端点就是方案里的名义窗口。
    这里把 ``SignalLoss`` 的阈值统一乘周期转成秒，再按软损失公式累加。
    这部分不计入 MILP 目标，只用于让 Stage 1 结果也带有可比的
    ``intersection_loss``。
    """
    total = 0.0
    for plan in selected:
        for spec in plan.signal_losses:
            expr_value = sum(
                coef * plan.term_value(term) * cycle
                for term, coef in spec.terms.items()
            )
            if spec.lower_threshold is not None:
                violation = max(0.0, spec.lower_threshold * cycle - expr_value)
                total += spec.lower_slope * violation
            if spec.upper_threshold is not None:
                upper_slope = (
                    spec.upper_slope
                    if spec.upper_slope is not None
                    else spec.lower_slope
                )
                violation = max(0.0, expr_value - spec.upper_threshold * cycle)
                total += upper_slope * violation
    return total


class SegmentedBandSolver(Solver):
    """统一 Stage 1 核心：多段传播 + 多方案选择 + ObjectiveConfig。"""

    name = "segmented-band"

    def __init__(
        self,
        config: ObjectiveConfig,
        *,
        max_segments: int = 3,
        max_bands: int | None = None,
        max_loops: int = 3,
        band_gap: float = 0.0,
        name: str = "segmented-band",
        up_global_output: bool = False,
        down_global_output: bool = False,
        margin: BandMarginConfig | None = None,
    ) -> None:
        """函数名：__init__；参数：config、最大 band 数、最大圈数、band 间隔、输出口径、边距；返回值：无；异常：ValueError。"""
        self.config = config
        self.margin = margin or BandMarginConfig()
        # max_segments 保留为旧参数名，语义改为每个方向最多建模多少个 band。
        self.max_bands = int(max_bands if max_bands is not None else max_segments)
        self.max_segments = self.max_bands
        self.max_loops = int(max_loops)
        self.band_gap = float(band_gap)
        self.name = name
        self.up_global_output = bool(up_global_output)
        self.down_global_output = bool(down_global_output)

        if self.max_bands <= 0:
            raise ValueError("max_bands 必须为正")
        if self.max_loops < 0:
            raise ValueError("max_loops 不能为负")
        if self.band_gap < 0:
            raise ValueError("band_gap 不能为负")

    def solve(
        self,
        arterial: Arterial,
        band_loss_weight: float = 0.0,
        constraint_builder=None,
        loss_builder=None,
    ) -> Solution:
        """Stage 1 主流程：用 MILP 找最佳绿波带方案。

        先讲一个“小学生版”的故事：

        1. 有一条路，路上有几个路口，每个路口都有红绿灯。
        2. 每个路口的红绿灯不是一直绿，而是在一个固定周期 C 内，分成一段或几段绿灯窗口。
        3. 我们要安排若干条“绿波带”。每条带就像一队车，沿着路走，经过每个路口时，
           必须在某个绿灯窗口内通过。
        4. 每个路口可能有多个候选信控方案；每个方案又可能有多个绿灯窗口。
           所以要决定：这个路口用哪个方案？这条带在这个路口走哪个窗口？
        5. 带子越宽越好。Stage 1 就是要在这些选择中，找一组让总带宽最大的方案。

        本方法里的主要变量：

        - C: 周期。比如 90 秒。时间像钟表一样，走满 C 秒回到 0。
        - i: 路口编号，0,1,2,...
        - e: 物理路段编号，0,1,2,...；第 e 段连接路口 e 和 e+1。
        - direction: "up" 上行 / "down" 下行。
        - band_no: 第几条绿波带，1,2,...
        - t[direction,band_no,i]: 这条带到达路口 i 的时刻（0~C 之间的钟表时间）。
        - loop[direction,band_no,e]: 从路口 e 到 e+1 之间，带子可能多绕了整数圈；
          因为有周期，实际传播时间是“行驶时间 + loop*C”。
        - width[direction,band_no,e]: 这条带在路段 e 上的宽度（秒）。
        - global_B[direction,band_no]: 整条带在所有路段上都能保持的最小宽度。
        - y[direction,band_no,i,plan,window]: 0/1 选择变量。
          y=1 表示：第 band_no 条带，在路口 i，使用 plan 方案的第 window 个绿灯窗口。
        - delta[i,plan]: 0/1 选择变量。每个路口只能选一个方案。
        - pi[direction,r,s,i]: 0/1，表示路口 i 上第 r 条带在第 s 条带前面还是后面。
        - balance_var: 均衡组的最小值变量，用来做 max-min 公平性目标。

        公式背后的含义：

        - 传播公式：t_{i+1} - t_i = 行驶时间 + loop*C
          意思是：到下一个路口的钟表时间，差一个“行驶时间”；如果跨了周期，就补整数个 C。
        - 窗口起点公式：t_i >= 所选窗口的起点 + margin
          意思是：带子不能早于绿灯窗口开始。
        - 窗口终点公式：t_i + width_e <= 所选窗口的终点 - margin
          意思是：带子通过时，必须在绿灯窗口结束前完成。
        - 全局宽度公式：global_B <= 每个路段的 width
          意思是：整条带能稳定通过的宽度，等于最窄路段的宽度。
        """
        # ============================================================
        # 0. 先认识这条路：周期、路口、路段
        # ============================================================
        # cycle 是信号周期，例如 90 秒。所有时间都在这 90 秒里循环。
        cycle = arterial.cycle
        # ints 是按上行方向排好的路口列表，例如 [I1, I2, I3]。
        ints = arterial.intersection_order
        # segs 是相邻路口之间的物理路段列表，例如 [S12, S23]。
        segs = arterial.segment_order
        int_names = [inter.name for inter in ints]          # ["I1", "I2", ...]
        seg_names = [seg.name for seg in segs]              # ["S12", "S23", ...]
        name_to_i = {name: idx for idx, name in enumerate(int_names)}   # 路口名 -> 下标
        seg_name_to_idx = {name: idx for idx, name in enumerate(seg_names)}  # 路段名 -> 下标
        n, m = len(ints), len(segs)                         # n 个路口，m 个路段

        # 用户给的 self.max_loops 只是“手动下限”。
        # 对长路段，自动按 ceil(travel_time / cycle) + 1 放大，
        # 并让所有路口/路段共用同一个安全上界。
        effective_max_loops = compute_effective_max_loops(arterial, self.max_loops)

        if n < 2:
            raise ValueError("SegmentedBandSolver 至少需要两个路口")

        # ============================================================
        # 1. 把“比例边距”换算成“秒”
        # ============================================================
        # BandMarginConfig 里存的是占周期的比例，例如 0.01 表示 1% 周期。
        # 乘以 cycle 后，得到真正的秒数。
        # 硬边距的意思是：绿波带不能贴住绿灯窗口边缘，要向内缩这么多。
        hard_margin_up = self.margin.hard_margin_up * cycle
        hard_margin_down = self.margin.hard_margin_down * cycle

        def effective_start(plan: SignalPlan, direction: str, window_no: int) -> float:
            """一个绿灯窗口扣掉硬边距后的“有效起点”（秒）。"""
            margin = hard_margin_up if direction == "up" else hard_margin_down
            return self._segment_window(plan, direction, window_no).start * cycle + margin

        def effective_end(plan: SignalPlan, direction: str, window_no: int) -> float:
            """一个绿灯窗口扣掉硬边距后的“有效终点”（秒）。"""
            margin = hard_margin_up if direction == "up" else hard_margin_down
            return self._segment_window(plan, direction, window_no).end * cycle - margin

        # ============================================================
        # 2. 准备“候选方案”和“要建模的 band”
        # ============================================================
        # config 是用户给的目标规则，例如“上行全局带权重 1.0”。
        config = self.config
        config.validate(n)
        # options[i] 是第 i 个路口的所有候选信控方案。
        options = self._build_plan_options(ints)
        # band_instances 是要建模的所有 (方向, band 编号)。
        # 例如 max_bands=2 时：[(up,1),(up,2),(down,1),(down,2)]。
        band_instances = self._band_instances()
        if not band_instances:
            raise ValueError("没有可用的绿波带 band")

        # 对每个路口、每个方向，列出所有可选的 (方案下标 o, 窗口号 q)。
        # 例如 options[i] 有 2 个方案，第 1 个方案有 1 个上行窗口，
        # 第 2 个方案有 2 个上行窗口，那么 plan_window_options[(i,"up")]
        # 就是 [(0,1), (1,1), (1,2)]。
        plan_window_options: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for i, opt_list in enumerate(options):
            for direction in ("up", "down"):
                opts: list[tuple[int, int]] = []
                for o, plan in enumerate(opt_list):
                    windows = plan.up_segments if direction == "up" else plan.down_segments
                    for q in range(1, len(windows) + 1):
                        opts.append((o, q))
                plan_window_options[(i, direction)] = opts

        # ============================================================
        # 3. 给 MILP 里的每一个未知量安排一个“座位号”
        # ============================================================
        # MILP 求解器只认 0,1,2,... 这样的变量下标。
        # 下面所有 *_idx 字典都是：业务含义 -> 变量下标。
        # cur 是“下一个空座位号”。
        cur = 0
        idx_opt: list[list[int]] = []
        for opt_list in options:
            idx_opt.append(list(range(cur, cur + len(opt_list))))
            cur += len(opt_list)

        # 全局 band 的变量：
        # - t_idx: 到达每个路口的时刻 t
        # - loop_idx: 每段路多绕的整数圈数
        # - width_idx: 每段路的带宽
        # - global_idx: 整条 band 的公共最小宽度 B
        t_idx: dict[BandInstance, list[int]] = {}
        loop_idx: dict[BandInstance, list[int]] = {}
        width_idx: dict[BandInstance, list[int]] = {}
        global_idx: dict[BandInstance, int] = {}
        # 局部 band 的变量：
        # - local_t_idx: 局部带在子走廊各路口的时间
        # - local_loop_idx: 局部带在子走廊各段的整数圈数
        # - lattice_idx: 局部带宽度
        lattice_idx: dict[LatticeInstance, int] = {}
        local_t_idx: dict[LatticeInstance, list[int]] = {}
        local_loop_idx: dict[LatticeInstance, list[int]] = {}
        for key in band_instances:
            t_idx[key] = list(range(cur, cur + n))
            cur += n
            loop_idx[key] = list(range(cur, cur + m))
            cur += m
            width_idx[key] = list(range(cur, cur + m))
            cur += m
            global_idx[key] = cur
            cur += 1

        # 局部 band 是“只看连续 k 个路口”的小绿波带。
        # local_specs 里只包含 ObjectiveConfig 明确写到的局部 key，
        # 例如 down.seg2、down.win3@I2-I4。
        local_specs = self._local_objective_specs(config, n)
        # local_bands_by_spec: (方向,k,起点) -> 要建模的 band 编号列表
        # 例如 {(down,3,1): [1,2,3]}。
        local_bands_by_spec: dict[tuple[str, int, int], list[int]] = {}
        for direction, k, start in local_specs:
            band_numbers = self._local_band_numbers(direction, start, k)
            local_bands_by_spec[(direction, k, start)] = band_numbers
            for band_no in band_numbers:
                local_key = (direction, band_no, k, start)
                local_t_idx[local_key] = list(range(cur, cur + k))
                cur += k
                local_loop_idx[local_key] = list(range(cur, cur + k - 1))
                cur += k - 1
                lattice_idx[local_key] = cur
                cur += 1

        # 自由窗口选择变量 y[d, band, i, plan, window]。
        # y=1 表示：第 band 条带，在路口 i，选择第 plan 个方案的第 window 个窗口。
        assign_idx: dict[tuple[str, int, int, int, int], int] = {}
        for direction, band_no in band_instances:
            for i in range(n):
                for o, q in plan_window_options[(i, direction)]:
                    assign_idx[(direction, band_no, i, o, q)] = cur
                    cur += 1

        # 局部带也要自由选择窗口，所以也有自己的 y 变量。
        local_assign_idx: dict[tuple[str, int, int, int, int, int, int], int] = {}
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            for band_no in band_numbers:
                for offset, abs_idx in enumerate(range(start, start + k)):
                    for o, q in plan_window_options[(abs_idx, direction)]:
                        local_assign_idx[(direction, band_no, k, start, offset, o, q)] = cur
                        cur += 1

        # 顺序变量 pi：同方向两条带，谁在前、谁在后。
        # 没有它，两条带可以完全重叠，目标分数会被“复制”刷高。
        order_idx: dict[tuple[str, int, int, int], int] = {}
        for direction in ("up", "down"):
            nums = [band_no for d, band_no in band_instances if d == direction]
            for a in range(len(nums)):
                for b in range(a + 1, len(nums)):
                    r, s = nums[a], nums[b]
                    for i in range(n):
                        order_idx[(direction, r, s, i)] = cur
                        cur += 1

        # 局部带之间的顺序变量。
        local_order_idx: dict[tuple[str, int, int, int, int, int], int] = {}
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            for a in range(len(band_numbers)):
                for b in range(a + 1, len(band_numbers)):
                    r, s = band_numbers[a], band_numbers[b]
                    for offset in range(k):
                        local_order_idx[(direction, r, s, k, start, offset)] = cur
                        cur += 1

        # 均衡组变量：B_bal <= 组内每个成员。
        # 最大化 B_bal 就等于最大化“最差成员”，用于公平性目标。
        balance_vars: dict[int, int] = {}
        for gidx in range(len(config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1

        # 按方向整理一下有哪些 band 编号，方便后面查。
        active_by_direction = {
            "up": [band_no for direction, band_no in band_instances if direction == "up"],
            "down": [band_no for direction, band_no in band_instances if direction == "down"],
        }

        # ObjectiveConfig 里的 key（如 up.global 或 down.win3@I2-I4）
        # 最终要对应到一个或多个实际变量。这里就是做这个翻译。
        def band_member_vars(key_text: str) -> list[int]:
            """把目标 key 翻译成 MILP 变量下标列表。"""
            band = parse_band_key(key_text, n)
            out: list[int] = []
            if key_text.endswith(".global"):
                for band_no in active_by_direction.get(band.direction, []):
                    out.append(global_idx[(band.direction, band_no)])
                return out

            for band_no in local_bands_by_spec.get((band.direction, band.k, band.start), []):
                lattice_key = (band.direction, band_no, band.k, band.start)
                if lattice_key in lattice_idx:
                    out.append(lattice_idx[lattice_key])
            return out

        # 特殊变量名 b_up、b_down、B_bal、tU_I2、bD_S12 等，
        # 也要翻译成实际的 MILP 变量下标。
        def resolve_band_loss_name(name: str) -> list[int]:
            """把特殊损失/约束变量名翻译成变量下标列表。

            带宽相关的特殊变量保留：
            - b_up / b_down：对应方向所有全局 band 的 B 求和；
            - bU_* / bD_*：对应方向所有全局 band 在指定物理路段上的宽度求和；
            - B_bal：第一个 BalanceGroup 的组内最小值。

            时间相关的 tU_* / tD_* 已停用：它们过去只取第一条 band，
            在多 band 模型下容易误导，因此只保留注释并主动报错。
            """
            if name == "b_up":
                return [global_idx[("up", band_no)] for band_no in active_by_direction["up"]]
            if name == "b_down":
                return [global_idx[("down", band_no)] for band_no in active_by_direction["down"]]
            if name == "B_bal":
                gvar = balance_vars.get(0)
                return [] if gvar is None else [gvar]

            # 时间相关特殊变量 tU_* / tD_* 已停用，解析逻辑保留为注释。
            # if name.startswith("tU_"):
            #     i = name_to_i.get(name[3:])
            #     if i is None or not active_by_direction["up"]:
            #         return []
            #     return [t_idx[("up", active_by_direction["up"][0])][i]]
            # if name.startswith("tD_"):
            #     i = name_to_i.get(name[3:])
            #     if i is None or not active_by_direction["down"]:
            #         return []
            #     return [t_idx[("down", active_by_direction["down"][0])][i]]

            if name.startswith(("tU_", "tD_")):
                raise ValueError(
                    f"时间特殊变量 {name!r} 已停用；"
                    "它在多 band 模型下只代表第一条 band，容易产生歧义。"
                    "请改用 ObjectiveConfig 或 SignalConstraint / SignalLoss。"
                )

            if name.startswith("bD_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [width_idx[("down", band_no)][seg_idx] for band_no in active_by_direction["down"]]
            if name.startswith("bU_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [width_idx[("up", band_no)][seg_idx] for band_no in active_by_direction["up"]]
            return []


        # ============================================================
        # 4. 把外部的业务约束/软损失收集起来
        # ============================================================
        # all_specs 里每项是 (LinearSpec, kind)。
        # kind="intersection" 表示交叉口层，kind="band" 表示带层。
        all_specs = []
        if constraint_builder is not None:
            for spec in constraint_builder.specs:
                all_specs.append((spec, "intersection"))
        if loss_builder is not None:
            all_specs.extend(loss_builder.to_linear_specs())

        # 在真正建模前，先检查用户写的 I1.up.2.start 这类 term 是否合法。
        # 这样错误会在“组装模型前”被清楚报出来，而不是求解时才崩。
        validation_ctx = TermValidationContext(
            intersection_names=int_names,
            segment_names=seg_names,
            active_directions={
                direction for direction, nums in active_by_direction.items() if nums
            },
            active_segment_numbers={
                direction: set(nums)
                for direction, nums in active_by_direction.items()
                if nums
            },
            balance_group_count=len(config.balance_groups),
            endpoint_terms_by_intersection={
                inter.name: common_endpoint_terms(inter.plans) for inter in ints
            },
            plan_names_by_intersection={
                inter.name: {plan.name for plan in inter.plans} for inter in ints
            },
        )
        validation_ctx.validate_specs(all_specs)

        # resolved_specs 存放已经翻译成“变量下标 + 系数”的约束/损失。
        # 后面既可以加到约束矩阵，也可以加到目标函数。
        resolved_specs = []

        def resolve_term(name: str) -> dict[int, float]:
            """将标识符解析为变量下标与系数的映射。"""
            parts = name.split(".")
            if len(parts) == 4 and parts[0] in name_to_i:
                i = name_to_i[parts[0]]
                d, idx_1, attr = parts[1], int(parts[2]), parts[3]
                res = {}
                for o, plan in enumerate(options[i]):
                    win = self._segment_window(plan, d, idx_1)
                    val = win.start if attr == "start" else win.end
                    res[idx_opt[i][o]] = val * cycle
                return res

            vars_found = resolve_band_loss_name(name)
            return {v: 1.0 for v in vars_found}

        for spec, kind in all_specs:
            scaled_spec = scale_linear_spec(spec, cycle)
            combined_terms: dict[int, float] = {}
            for t_name, t_coef in scaled_spec.terms.items():
                v_map = resolve_term(t_name)
                for v_idx, v_coef in v_map.items():
                    combined_terms[v_idx] = combined_terms.get(v_idx, 0.0) + t_coef * v_coef

            if not combined_terms:
                continue

            slack = None
            if scaled_spec.soft:
                slack = cur
                cur += 1

            resolved_specs.append((scaled_spec, list(combined_terms.items()), slack, kind))

        # ============================================================
        # 5. 组装目标函数：给“好”的变量一个负系数（因为 milp 默认求最小）
        # ============================================================
        nvar = cur          # 变量总数
        c = np.zeros(nvar)  # 目标函数系数；c[j] 表示变量 x_j 的系数

        # SumGroup：加权和。
        # 例如 SumGroup({"up.global": 2.0}) 就是 2 * B_up_global。
        # 因为 milp 求最小，所以这里写 c[var] -= weight。
        for group in config.sum_groups:
            for key_text, weight in group.terms.items():
                if weight == 0.0:
                    continue
                member_vars = band_member_vars(key_text)
                if not member_vars:
                    raise ValueError(f"目标项 {key_text} 在当前多段配置下没有可用变量")
                for var in member_vars:
                    c[var] += -weight

        # BalanceGroup：最大化组内最小值。
        # 目标里给 B_bal 一个负系数；约束里会让 B_bal <= 每个成员。
        for gidx, group in enumerate(config.balance_groups):
            c[balance_vars[gidx]] += -group.weight
            if group.eps > 0:
                for member in group.members:
                    member_vars = band_member_vars(member)
                    if not member_vars:
                        raise ValueError(f"均衡项 {member} 在当前多段配置下没有可用变量")
                    for var in member_vars:
                        c[var] += -group.weight * group.eps

        # 软约束的 slack 表示“违反了多少”。
        # 目标里给 slack 正系数，让求解器尽量减少违反。
        for spec, _, slack, kind in resolved_specs:
            if slack is not None:
                weight = band_loss_weight if kind == "band" else spec.penalty
                c[slack] += weight

        # ============================================================
        # 6. 给变量写上下界，并标记哪些必须是整数
        # ============================================================
        # lb: 下界；ub: 上界；integrality=1 表示这个变量必须是整数。
        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        integrality = np.zeros(nvar)

        # 方案选择 delta：0/1 变量。
        for row in idx_opt:
            ub[row] = 1.0
            integrality[row] = 1
        # 自由窗口选择 y：0/1 变量。
        for var in assign_idx.values():
            ub[var] = 1.0
            integrality[var] = 1
        for var in local_assign_idx.values():
            ub[var] = 1.0
            integrality[var] = 1
        # band 顺序 pi：0/1 变量。
        for var in order_idx.values():
            ub[var] = 1.0
            integrality[var] = 1
        for var in local_order_idx.values():
            ub[var] = 1.0
            integrality[var] = 1

        # 全局 band 变量：
        # - t 在 [0, cycle]
        # - loop 是整数，范围 [-effective_max_loops, effective_max_loops]
        # - width/B 在 [0, cycle]
        for key in band_instances:
            ub[t_idx[key]] = cycle
            lb[loop_idx[key]] = -effective_max_loops
            ub[loop_idx[key]] = effective_max_loops
            ub[width_idx[key]] = cycle
            ub[global_idx[key]] = cycle
            integrality[loop_idx[key]] = 1
        # 局部 band 变量同理。
        for local_key, band_var in lattice_idx.items():
            ub[local_t_idx[local_key]] = cycle
            lb[local_loop_idx[local_key]] = -effective_max_loops
            ub[local_loop_idx[local_key]] = effective_max_loops
            ub[band_var] = cycle
            integrality[local_loop_idx[local_key]] = 1

        # 均衡组最小值变量和软约束 slack 都不会超过一个周期。
        for gvar in balance_vars.values():
            ub[gvar] = cycle
        for _spec, _terms, slack, _kind in resolved_specs:
            if slack is not None:
                ub[slack] = cycle

        # ============================================================
        # 7. 开始拼约束矩阵
        # ============================================================
        # rows/lo_list/hi_list 组成 A*x in [lo, hi]。
        # add_row 表示：sum(coef_j * x_j) 必须落在 [lower, upper] 之间。
        rows: list[np.ndarray] = []
        lo_list: list[float] = []
        hi_list: list[float] = []

        def add_row(coefs: dict[int, float], lower: float, upper: float) -> None:
            """函数名：add_row；参数：coefs、lower、upper；返回值：无；异常：无。"""
            row = np.zeros(nvar)
            for j, value in coefs.items():
                row[j] += value
            rows.append(row)
            lo_list.append(lower)
            hi_list.append(upper)

        def incident_edge_indices(intersection_idx: int) -> list[int]:
            edges: list[int] = []
            if intersection_idx > 0:
                edges.append(intersection_idx - 1)
            if intersection_idx < m:
                edges.append(intersection_idx)
            return edges

        for i, opt_list in enumerate(options):
            add_row({idx_opt[i][o]: 1.0 for o in range(len(opt_list))}, 1.0, 1.0)

        # ---------------- 全局 band：传播 + 自由窗口 ----------------
        for key in band_instances:
            direction, band_no = key

            # ---- 7.1 传播公式：带子从上一个路口开到下一个路口 ----
            # 上行：t_{e+1} - t_e = 上行行驶时间 + loop*C
            # 下行：t_e - t_{e+1} = 下行行驶时间 + loop*C
            # loop 是整数，表示因为周期循环，实际可能多绕了整数圈。
            for e, segment in enumerate(segs):
                if direction == "up":
                    add_row(
                        {
                            t_idx[key][e + 1]: 1.0,
                            t_idx[key][e]: -1.0,
                            loop_idx[key][e]: -cycle,
                        },
                        segment.travel_time_up,
                        segment.travel_time_up,
                    )
                else:
                    add_row(
                        {
                            t_idx[key][e]: 1.0,
                            t_idx[key][e + 1]: -1.0,
                            loop_idx[key][e]: -cycle,
                        },
                        segment.travel_time_down,
                        segment.travel_time_down,
                    )

            # ---- 7.2 每个 band 在每个路口必须“恰好选一个窗口” ----
            # 所有候选 (方案,窗口) 的 y 加起来必须等于 1。
            for i in range(n):
                options_i = plan_window_options[(i, direction)]
                add_row(
                    {assign_idx[(direction, band_no, i, o, q)]: 1.0 for o, q in options_i},
                    1.0,
                    1.0,
                )
                # 只有路口选中了方案 o，band 才可能用方案 o 的窗口 q。
                # 公式：y_{i,o,q} <= delta_{i,o}
                for o, q in options_i:
                    add_row(
                        {
                            assign_idx[(direction, band_no, i, o, q)]: 1.0,
                            idx_opt[i][o]: -1.0,
                        },
                        -np.inf,
                        0.0,
                    )

            # ---- 7.3 起点公式：带子不能早于所选绿灯窗口的有效起点 ----
            # t_i >= 所选窗口的有效起点
            for i in range(n):
                options_i = plan_window_options[(i, direction)]
                start_terms = {
                    assign_idx[(direction, band_no, i, o, q)]:
                        effective_start(options[i][o], direction, q)
                    for o, q in options_i
                }
                add_row({t_idx[key][i]: -1.0, **start_terms}, -np.inf, 0.0)

            # ---- 7.4 终点公式：带子必须在所选绿灯窗口的有效终点前通过 ----
            # t_i + 路段宽度 <= 所选窗口的有效终点
            # 同一条路段 e 的两个端点路口都要满足。
            for e in range(m):
                for i in (e, e + 1):
                    options_i = plan_window_options[(i, direction)]
                    end_terms = {
                        assign_idx[(direction, band_no, i, o, q)]:
                            -effective_end(options[i][o], direction, q)
                        for o, q in options_i
                    }
                    add_row(
                        {
                            t_idx[key][i]: 1.0,
                            width_idx[key][e]: 1.0,
                            **end_terms,
                        },
                        -np.inf,
                        0.0,
                    )

            # ---- 7.5 全局宽度：整条带不能比任何一段还宽 ----
            # B_global <= width_e，对每个路段 e 都成立。
            for e in range(m):
                add_row({global_idx[key]: 1.0, width_idx[key][e]: -1.0}, -np.inf, 0.0)

        # ---- 7.6 同方向 band 顺序约束：不能完全重叠 ----
        # pi=1 表示 r 在 s 前面；pi=0 表示 s 在 r 前面。
        # 这样两条带可以共用同一窗口，但必须一前一后，不能复制刷分。
        order_m = 2.0 * cycle
        for direction in ("up", "down"):
            nums = active_by_direction[direction]
            for a in range(len(nums)):
                for b in range(a + 1, len(nums)):
                    r, s = nums[a], nums[b]
                    for i in range(n):
                        pi = order_idx[(direction, r, s, i)]
                        for e in incident_edge_indices(i):
                            # 第一条不等式：
                            # t_r + w_r + gap <= t_s + M*(1-pi)
                            # 当 pi=1 时，等价于 t_r + w_r + gap <= t_s。
                            add_row(
                                {
                                    t_idx[(direction, r)][i]: 1.0,
                                    width_idx[(direction, r)][e]: 1.0,
                                    t_idx[(direction, s)][i]: -1.0,
                                    pi: order_m,
                                },
                                -np.inf,
                                order_m - self.band_gap,
                            )
                            # 第二条不等式：
                            # t_s + w_s + gap <= t_r + M*pi
                            # 当 pi=0 时，等价于 t_s + w_s + gap <= t_r。
                            add_row(
                                {
                                    t_idx[(direction, s)][i]: 1.0,
                                    width_idx[(direction, s)][e]: 1.0,
                                    t_idx[(direction, r)][i]: -1.0,
                                    pi: -order_m,
                                },
                                -np.inf,
                                -self.band_gap,
                            )

        # ---------------- 局部窗口/segment band：传播 + 自由窗口 ----------------
        # 局部带只看连续 k 个路口，例如 down.win3@I2-I4。
        # 它的变量更少：每个参与路口一个 t，每段一个整数圈，一个公共宽度。
        for local_key, band_var in lattice_idx.items():
            direction, band_no, k, start = local_key
            t_vars = local_t_idx[local_key]
            loop_vars = local_loop_idx[local_key]
            local_segs = segs[start:start + k - 1]
            for offset, segment in enumerate(local_segs):
                if direction == "up":
                    add_row(
                        {
                            t_vars[offset + 1]: 1.0,
                            t_vars[offset]: -1.0,
                            loop_vars[offset]: -cycle,
                        },
                        segment.travel_time_up,
                        segment.travel_time_up,
                    )
                else:
                    add_row(
                        {
                            t_vars[offset]: 1.0,
                            t_vars[offset + 1]: -1.0,
                            loop_vars[offset]: -cycle,
                        },
                        segment.travel_time_down,
                        segment.travel_time_down,
                    )

            for offset, abs_idx in enumerate(range(start, start + k)):
                options_i = plan_window_options[(abs_idx, direction)]
                # 局部带在这个路口也必须恰好选一个窗口。
                add_row(
                    {
                        local_assign_idx[(direction, band_no, k, start, offset, o, q)]: 1.0
                        for o, q in options_i
                    },
                    1.0,
                    1.0,
                )
                # 只有该路口选中方案 o，局部带才能用方案 o 的窗口 q。
                for o, q in options_i:
                    add_row(
                        {
                            local_assign_idx[(direction, band_no, k, start, offset, o, q)]: 1.0,
                            idx_opt[abs_idx][o]: -1.0,
                        },
                        -np.inf,
                        0.0,
                    )
                # 局部带起点/终点也必须落在所选窗口的有效范围内。
                start_terms = {
                    local_assign_idx[(direction, band_no, k, start, offset, o, q)]:
                        effective_start(options[abs_idx][o], direction, q)
                    for o, q in options_i
                }
                end_terms = {
                    local_assign_idx[(direction, band_no, k, start, offset, o, q)]:
                        -effective_end(options[abs_idx][o], direction, q)
                    for o, q in options_i
                }
                add_row({t_vars[offset]: -1.0, **start_terms}, -np.inf, 0.0)
                add_row(
                    {t_vars[offset]: 1.0, band_var: 1.0, **end_terms},
                    -np.inf,
                    0.0,
                )

        # 局部带之间也加顺序，避免同一个局部 key 用多条相同带刷分。
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            for a in range(len(band_numbers)):
                for b in range(a + 1, len(band_numbers)):
                    r, s = band_numbers[a], band_numbers[b]
                    for offset in range(k):
                        pi = local_order_idx[(direction, r, s, k, start, offset)]
                        t_r = local_t_idx[(direction, r, k, start)][offset]
                        t_s = local_t_idx[(direction, s, k, start)][offset]
                        b_r = lattice_idx[(direction, r, k, start)]
                        b_s = lattice_idx[(direction, s, k, start)]
                        # 局部带之间也用同样的先后关系：
                        # pi=1 -> t_r + b_r + gap <= t_s；
                        # pi=0 -> t_s + b_s + gap <= t_r。
                        add_row(
                            {t_r: 1.0, b_r: 1.0, t_s: -1.0, pi: order_m},
                            -np.inf,
                            order_m - self.band_gap,
                        )
                        add_row(
                            {t_s: 1.0, b_s: 1.0, t_r: -1.0, pi: -order_m},
                            -np.inf,
                            -self.band_gap,
                        )

        # 均衡组约束：B_bal <= 每个成员。
        # 例如 B_bal <= B_up_global 且 B_bal <= B_down_global，
        # 最大化 B_bal 就等于让上下行中较小的那个尽量大。
        for gidx, group in enumerate(config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                row = {gvar: 1.0}
                for var in band_member_vars(member):
                    row[var] = row.get(var, 0.0) - 1.0
                add_row(row, -np.inf, 0.0)

        # ---------------- 业务约束/损失：Big-M + plan_tags ----------------
        # plan_tags 表示“只有当某些路口选了指定方案时，这条约束才生效”。
        # Big-M 是一个很大的数：条件不满足时，约束右边变得很大/左边变得很小，
        # 等价于“放松这条约束”。
        BIG_M = cycle * 10.0
        for spec, terms, slack, _kind in resolved_specs:
            row = {var: coef for var, coef in terms}
            condition_terms: dict[int, float] = {}
            if spec.plan_tags:
                for int_name, plan_name in spec.plan_tags.items():
                    if int_name not in name_to_i:
                        continue
                    i = name_to_i[int_name]
                    found = False
                    for o, plan in enumerate(options[i]):
                        if plan.name == plan_name:
                            condition_terms[idx_opt[i][o]] = 1.0
                            found = True
                            break
                    if not found:
                        continue

            n_tags = len(condition_terms)

            if spec.sense == "<=":
                m_row = dict(row)
                if slack is not None:
                    m_row[slack] = -1.0
                if n_tags > 0:
                    for v_idx, v_coef in condition_terms.items():
                        m_row[v_idx] = m_row.get(v_idx, 0.0) + BIG_M * v_coef
                    add_row(m_row, -np.inf, spec.rhs + BIG_M * n_tags)
                else:
                    add_row(m_row, -np.inf, spec.rhs)
            elif spec.sense == ">=":
                m_row = dict(row)
                if slack is not None:
                    m_row[slack] = 1.0
                if n_tags > 0:
                    for v_idx, v_coef in condition_terms.items():
                        m_row[v_idx] = m_row.get(v_idx, 0.0) - BIG_M * v_coef
                    add_row(m_row, spec.rhs - BIG_M * n_tags, np.inf)
                else:
                    add_row(m_row, spec.rhs, np.inf)

        # ============================================================
        # 8. 把目标、约束、上下界、整数要求交给 MILP 求解器
        # ============================================================
        constraints = (
            LinearConstraint(np.array(rows), np.array(lo_list), np.array(hi_list))
            if rows
            else ()
        )
        result = milp(
            c=c,                         # 目标函数系数
            constraints=constraints,     # 所有约束
            bounds=Bounds(lb, ub),       # 每个变量的上下界
            integrality=integrality,     # 哪些变量必须是整数/0-1
        )

        sol = Solution(cycle=cycle, solver_msg=f"HiGHS via scipy: success={result.success}")
        if result.x is None:
            sol.status = "infeasible"
            return sol

        # x 是求解器返回的最优变量值数组：x[j] 就是第 j 号变量的值。
        x = result.x
        # ============================================================
        # 9. 从变量值还原出人能看懂的结果
        # ============================================================
        # 每个路口的 delta 变量中，接近 1 的那个方案就是被选中的方案。
        chosen_plan_idx = {
            int_names[i]: max(range(len(options[i])), key=lambda o: float(x[idx_opt[i][o]]))
            for i in range(n)
        }
        selected = [options[i][chosen_plan_idx[int_names[i]]] for i in range(n)]
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}

        # 从 y 变量里读出每条 band 在每个路口实际选了哪个方案、哪个窗口。
        sol.band_window_choices = {"up": {}, "down": {}}
        sol.band_order_choices = {"up": {}, "down": {}}
        for direction, band_no in band_instances:
            sol.band_window_choices[direction][band_no] = {}
            for i in range(n):
                options_i = plan_window_options[(i, direction)]
                o_best, q_best = max(
                    options_i,
                    key=lambda item: float(x[assign_idx[(direction, band_no, i, item[0], item[1])]]),
                )
                sol.band_window_choices[direction][band_no][int_names[i]] = {
                    "plan": options[i][o_best].name,
                    "window": q_best,
                }
        for direction in ("up", "down"):
            nums = active_by_direction[direction]
            sol.band_order_choices[direction] = {}
            for a in range(len(nums)):
                for b in range(a + 1, len(nums)):
                    r, s = nums[a], nums[b]
                    sol.band_order_choices[direction][r] = sol.band_order_choices[direction].get(r, {})
                    sol.band_order_choices[direction][r][s] = {}
                    for i in range(n):
                        sol.band_order_choices[direction][r][s][int_names[i]] = int(
                            round(float(x[order_idx[(direction, r, s, i)]]))
                        )

        # 局部带的窗口选择和顺序也一样还原出来。
        sol.local_band_window_choices = {}
        sol.local_band_order_choices = {}
        local_records = []
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            key = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
            sol.local_band_window_choices.setdefault(key, {})
            sol.local_band_order_choices.setdefault(key, {})
            for band_no in band_numbers:
                choices: dict[str, dict[str, object]] = {}
                raw_times: list[float] = []
                for offset, abs_idx in enumerate(range(start, start + k)):
                    options_i = plan_window_options[(abs_idx, direction)]
                    o_best, q_best = max(
                        options_i,
                        key=lambda item: float(
                            x[local_assign_idx[(direction, band_no, k, start, offset, item[0], item[1])]]
                        ),
                    )
                    choices[int_names[abs_idx]] = {
                        "plan": options[abs_idx][o_best].name,
                        "window": q_best,
                    }
                    raw_times.append(float(x[local_t_idx[(direction, band_no, k, start)][offset]]))
                sol.local_band_window_choices[key][band_no] = choices
                local_records.append({
                    "direction": direction,
                    "band_no": band_no,
                    "key": key,
                    "start": start,
                    "k": k,
                    "bandwidth": float(x[lattice_idx[(direction, band_no, k, start)]]),
                    "times": raw_times,
                    "window_choices": choices,
                })
            for a in range(len(band_numbers)):
                for b in range(a + 1, len(band_numbers)):
                    r, s = band_numbers[a], band_numbers[b]
                    sol.local_band_order_choices[key].setdefault(r, {})[s] = {}
                    for offset, abs_idx in enumerate(range(start, start + k)):
                        sol.local_band_order_choices[key][r][s][int_names[abs_idx]] = int(
                            round(float(x[local_order_idx[(direction, r, s, k, start, offset)]]))
                        )

        # ============================================================
        # 10. 汇总最终带宽、轨迹和局部带结果
        # ============================================================
        sol.multi_bandwidths = {"up": {}, "down": {}}
        sol.multi_band_starts = {"up": {}, "down": {}}
        sol.multi_window_bands = {"up": {}, "down": {}}

        aggregated_edge_widths = {
            "up": {seg_name: 0.0 for seg_name in seg_names},
            "down": {seg_name: 0.0 for seg_name in seg_names},
        }
        for direction, band_no in band_instances:
            key = (direction, band_no)
            edge_widths = [float(x[index]) for index in width_idx[key]]
            global_width = float(x[global_idx[key]])
            sol.multi_bandwidths[direction][band_no] = global_width
            sol.multi_band_starts[direction][band_no] = {
                int_names[i]: float(x[t_idx[key][i]])
                for i in range(n)
            }
            for e, seg_name in enumerate(seg_names):
                aggregated_edge_widths[direction][seg_name] += edge_widths[e]

        up_total_global = sum(sol.multi_bandwidths["up"].values())
        down_total_global = sum(sol.multi_bandwidths["down"].values())
        if self.up_global_output:
            sol.bandwidth_up = {name: up_total_global for name in seg_names}
        else:
            sol.bandwidth_up = dict(aggregated_edge_widths["up"])
        if self.down_global_output:
            sol.bandwidth_down = {name: down_total_global for name in seg_names}
        else:
            sol.bandwidth_down = dict(aggregated_edge_widths["down"])

        # 把主 MILP 的局部带结果先写进 Solution；
        # 再把所有方向、所有长度的“缺失局部带”自动补齐，方便画图。
        fill_solution_local_band_records(sol, arterial, local_records, clear=True)
        fill_solution_missing_window_bands(
            sol, arterial, max_window=5, max_loops=effective_max_loops, margin=self.margin
        )

        # ============================================================
        # 11. 统计软损失：band_loss 和 intersection_loss
        # ============================================================
        band_loss = 0.0
        intersection_loss = 0.0
        for spec, terms, _slack, kind in resolved_specs:
            if not spec.soft:
                continue
            expr_value = sum(coef * float(x[var]) for var, coef in terms)
            active = True
            if spec.plan_tags:
                for int_name, plan_name in spec.plan_tags.items():
                    if sol.plan_choices.get(int_name) != plan_name:
                        active = False
                        break
            if not active:
                continue
            if spec.sense == ">=":
                violation = max(0.0, spec.rhs - expr_value)
            else:
                violation = max(0.0, expr_value - spec.rhs)
            weight = band_loss_weight if kind == "band" else spec.penalty
            weighted = weight * violation
            if kind == "band":
                band_loss += weighted
            else:
                intersection_loss += weighted

        # plan 级 SignalLoss 没有作为变量进入 Stage 1 MILP，但选中方案
        # 已有名义端点，因此这里补算，避免 Stage 1 的 intersection_loss
        # 永远为 0，导致 Pareto/汇报图错误。
        intersection_loss += _evaluate_plan_signal_losses(selected, cycle)

        # 计算目标里的带层收益（不含损失）。
        band_objective = 0.0
        for group in config.sum_groups:
            for key_text, weight in group.terms.items():
                if weight == 0.0:
                    continue
                band_objective += weight * sum(float(x[var]) for var in band_member_vars(key_text))
        for gidx, group in enumerate(config.balance_groups):
            band_objective += group.weight * float(x[balance_vars[gidx]])
            if group.eps > 0:
                for member in group.members:
                    band_objective += group.weight * group.eps * sum(
                        float(x[var]) for var in band_member_vars(member)
                    )

        # 把最终统计写进 Solution。
        # band_score 由 band_objective / band_loss / band_loss_weight
        # 通过 Solution.band_score property 推导。
        sol.band_objective = float(band_objective)
        sol.band_loss = float(band_loss)
        sol.band_loss_weight = float(band_loss_weight)
        sol.intersection_loss = float(intersection_loss)
        sol.objective = -float(result.fun)   # milp 求最小，取负就是最大化目标值
        sol.status = "optimal" if result.success else result.message
        return sol

    @staticmethod
    def _build_plan_options(intersections) -> list[list[SignalPlan]]:
        """收集每个路口的候选方案。

        返回值例如：
            [[I1方案A, I1方案B], [I2方案A], ...]
        每个路口至少要有一个方案，而且每个方案都必须同时有上行和下行绿灯窗口。
        """
        options: list[list[SignalPlan]] = []
        for inter in intersections:
            if not inter.plans:
                raise ValueError(f"路口 {inter.name} 没有可选方案")
            for plan in inter.plans:
                if not plan.up_segments or not plan.down_segments:
                    raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少上下行绿灯分段")
            options.append(list(inter.plans))
        return options

    def _band_instances(self) -> list[BandInstance]:
        """生成要建模的所有全局 band。

        例如 max_bands=2 时返回：
            [(up,1), (up,2), (down,1), (down,2)]
        这里的 band 编号只是“第几条带”，不再等于路口窗口号。
        """
        keys: list[BandInstance] = []
        for direction in ("up", "down"):
            for band_no in range(1, self.max_bands + 1):
                keys.append((direction, band_no))
        return keys

    @staticmethod
    def _local_objective_specs(config: ObjectiveConfig,
                               n: int) -> list[tuple[str, int, int]]:
        """从目标配置里找出所有需要优化的小窗口带。

        返回 (direction, k, start)：
        - direction: up/down
        - k: 连续多少个路口，例如 3
        - start: 起点路口下标，0-based

        例如 down.win3@I2-I4 会变成 (down, 3, 1)。
        只有明确出现在 ObjectiveConfig 里的局部 key 才会在这里出现。
        """
        specs: set[tuple[str, int, int]] = set()
        for group in config.sum_groups:
            for key_text, weight in group.terms.items():
                # 权重为 0 的目标项不需要建立局部 band 变量。
                if weight == 0.0 or key_text.endswith(".global"):
                    continue
                band = parse_band_key(key_text, n)
                specs.add((band.direction, band.k, band.start))
        for group in config.balance_groups:
            for key_text in group.members:
                if key_text.endswith(".global"):
                    continue
                band = parse_band_key(key_text, n)
                specs.add((band.direction, band.k, band.start))
        return sorted(specs)

    def _local_band_numbers(self, direction: str, start: int, k: int) -> list[int]:
        """一个局部窗口里要建模几条 band。

        局部窗口最多需要 2 条 band，因此这里取 ``min(max_bands, 2)``。
        这样可以减少对称性和 0/1 变量数量。
        """
        return list(range(1, min(self.max_bands, 2) + 1))

    @staticmethod
    def _segment_window(plan: SignalPlan, direction: str, window_no: int):
        """取出某方案在某方向的第 window_no 个绿灯窗口。

        window_no 从 1 开始：1 表示第一个窗口，2 表示第二个窗口，以此类推。
        """
        windows = plan.up_segments if direction == "up" else plan.down_segments
        return windows[window_no - 1]
