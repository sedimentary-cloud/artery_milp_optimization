"""多段绿波带第一阶段求解器。

当前版本统一了两类 Stage 1 能力：
- 多段绿波带传播；
- 路口多方案选择；
- ObjectiveConfig 目标 DSL；

约束说明：
- 每个方向的第 r 段只与各路口同编号第 r 段协调；
- 为避免“某些方案缺少第 r 段”带来的激活变量复杂度，
  当前只对“所有候选方案都共同拥有”的段号建模；
- 同一个 BandKey 会把所有同编号段实例聚合求和，语义与
  Stage 2 的 FullFlexiblePhaseTuneSolver 保持一致。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...models import Arterial, SignalPlan
from ...solution import Solution
from ..builders.signal_constraints import scale_linear_spec
from ..builders.term_validation import (TermValidationContext,
                                        common_endpoint_terms)
from ..core.band_lattice import (fill_solution_multi_window_bands,
                                 fill_solution_window_band_ranges)
from ..core.base import Solver
from ..core.margin import BandMarginConfig
from ..core.objective import ObjectiveConfig, parse_band_key


BandInstance = tuple[str, int]
LatticeInstance = tuple[str, int, int, int]


class SegmentedBandSolver(Solver):
    """统一 Stage 1 核心：多段传播 + 多方案选择 + ObjectiveConfig。"""

    name = "segmented-band"

    def __init__(
        self,
        config: ObjectiveConfig,
        *,
        max_segments: int = 3,
        max_loops: int = 3,
        name: str = "segmented-band",
        up_global_output: bool = False,
        down_global_output: bool = False,
        margin: BandMarginConfig | None = None,
    ) -> None:
        """函数名：__init__；参数：config、最大段数、最大圈数、输出口径、边距；返回值：无；异常：ValueError。"""
        self.config = config
        self.margin = margin or BandMarginConfig()
        self.max_segments = int(max_segments)
        self.max_loops = int(max_loops)
        self.name = name
        self.up_global_output = bool(up_global_output)
        self.down_global_output = bool(down_global_output)

        if self.max_segments <= 0:
            raise ValueError("max_segments 必须为正")
        if self.max_loops < 0:
            raise ValueError("max_loops 不能为负")

    def solve(
        self,
        arterial: Arterial,
        band_loss_weight: float = 0.0,
        constraint_builder=None,
        loss_builder=None,
    ) -> Solution:
        """函数名：solve；参数：arterial、band_loss_weight、业务约束/损失；返回值：Solution；异常：ValueError。"""
        cycle = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        int_names = [inter.name for inter in ints]
        seg_names = [seg.name for seg in segs]
        name_to_i = {name: idx for idx, name in enumerate(int_names)}
        seg_name_to_idx = {name: idx for idx, name in enumerate(seg_names)}
        n, m = len(ints), len(segs)

        if n < 2:
            raise ValueError("SegmentedBandSolver 至少需要两个路口")

        hard_margin_up = self.margin.hard_margin_up * cycle
        hard_margin_down = self.margin.hard_margin_down * cycle

        def effective_start(plan: SignalPlan, direction: str, segment_no: int) -> float:
            margin = hard_margin_up if direction == "up" else hard_margin_down
            return self._segment_window(plan, direction, segment_no).start * cycle + margin

        def effective_end(plan: SignalPlan, direction: str, segment_no: int) -> float:
            margin = hard_margin_up if direction == "up" else hard_margin_down
            return self._segment_window(plan, direction, segment_no).end * cycle - margin

        config = self.config
        config.validate(n)
        options = self._build_plan_options(ints)
        band_instances = self._common_band_instances(options)
        if not band_instances:
            raise ValueError("没有所有候选方案共同拥有的可用绿波段")

        cur = 0
        idx_opt: list[list[int]] = []
        for opt_list in options:
            idx_opt.append(list(range(cur, cur + len(opt_list))))
            cur += len(opt_list)

        t_idx: dict[BandInstance, list[int]] = {}
        loop_idx: dict[BandInstance, list[int]] = {}
        width_idx: dict[BandInstance, list[int]] = {}
        global_idx: dict[BandInstance, int] = {}
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

        local_specs = self._local_objective_specs(config, n)
        local_segments_by_spec: dict[tuple[str, int, int], list[int]] = {}
        for direction, k, start in local_specs:
            segment_numbers = self._window_common_segment_numbers(options, direction, start, k)
            local_segments_by_spec[(direction, k, start)] = segment_numbers
            for segment_no in segment_numbers:
                local_key = (direction, segment_no, k, start)
                local_t_idx[local_key] = list(range(cur, cur + k))
                cur += k
                local_loop_idx[local_key] = list(range(cur, cur + k - 1))
                cur += k - 1
                lattice_idx[local_key] = cur
                cur += 1

        balance_vars: dict[int, int] = {}
        for gidx in range(len(config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1

        active_by_direction = {
            "up": [segment_no for direction, segment_no in band_instances if direction == "up"],
            "down": [segment_no for direction, segment_no in band_instances if direction == "down"],
        }

        def band_member_vars(key_text: str) -> list[int]:
            """函数名：band_member_vars；参数：key_text；返回值：聚合变量列表；异常：ValueError。"""
            band = parse_band_key(key_text, n)
            out: list[int] = []
            if key_text.endswith(".global"):
                for segment_no in active_by_direction.get(band.direction, []):
                    out.append(global_idx[(band.direction, segment_no)])
                return out

            for segment_no in local_segments_by_spec.get((band.direction, band.k, band.start), []):
                lattice_key = (band.direction, segment_no, band.k, band.start)
                if lattice_key in lattice_idx:
                    out.append(lattice_idx[lattice_key])
            return out

        def resolve_band_loss_name(name: str) -> list[int]:
            """函数名：resolve_band_loss_name；参数：name；返回值：变量下标列表；异常：无。"""
            if name == "b_up":
                return [global_idx[("up", segment_no)] for segment_no in active_by_direction["up"]]
            if name == "b_down":
                return [global_idx[("down", segment_no)] for segment_no in active_by_direction["down"]]
            if name == "B_bal":
                gvar = balance_vars.get(0)
                return [] if gvar is None else [gvar]
            if name.startswith("tU_"):
                i = name_to_i.get(name[3:])
                if i is None or not active_by_direction["up"]:
                    return []
                return [t_idx[("up", active_by_direction["up"][0])][i]]
            if name.startswith("tD_"):
                i = name_to_i.get(name[3:])
                if i is None or not active_by_direction["down"]:
                    return []
                return [t_idx[("down", active_by_direction["down"][0])][i]]
            if name.startswith("bD_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [width_idx[("down", segment_no)][seg_idx] for segment_no in active_by_direction["down"]]
            if name.startswith("bU_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [width_idx[("up", segment_no)][seg_idx] for segment_no in active_by_direction["up"]]
            return []

        # 业务约束与损失
        all_specs: list[tuple[LinearSpec, str]] = []
        if constraint_builder is not None:
            for spec in constraint_builder.specs:
                all_specs.append((spec, "intersection"))
        if loss_builder is not None:
            all_specs.extend(loss_builder.to_linear_specs())

        # 统一 term 校验：在组装 MILP 之前把所有非法/不可用 term 一次性
        # 变成清晰的 TermValidationError，避免静默丢项、静默截短、
        # 或 Stage 1 遍历候选方案时抛出难懂的 IndexError。
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

        resolved_specs: list[tuple[LinearSpec, list[tuple[int, float]], int | None, str]] = []

        def resolve_term(name: str) -> dict[int, float]:
            """将标识符解析为变量下标与系数的映射。"""
            # 1. 检查是否是路口相位端点：{IntName}.{dir}.{idx}.{start|end}
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

            # 2. 检查是否是带宽层变量
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

        nvar = cur
        c = np.zeros(nvar)

        for group in config.sum_groups:
            for key_text, weight in group.terms.items():
                member_vars = band_member_vars(key_text)
                if not member_vars:
                    raise ValueError(f"目标项 {key_text} 在当前多段配置下没有可用变量")
                for var in member_vars:
                    c[var] += -weight

        for gidx, group in enumerate(config.balance_groups):
            c[balance_vars[gidx]] += -group.weight
            if group.eps > 0:
                for member in group.members:
                    member_vars = band_member_vars(member)
                    if not member_vars:
                        raise ValueError(f"均衡项 {member} 在当前多段配置下没有可用变量")
                    for var in member_vars:
                        c[var] += -group.weight * group.eps

        for spec, _, slack, kind in resolved_specs:
            if slack is not None:
                weight = band_loss_weight if kind == "band" else spec.penalty
                c[slack] += weight

        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        integrality = np.zeros(nvar)

        for row in idx_opt:
            ub[row] = 1.0
            integrality[row] = 1

        for key in band_instances:
            ub[t_idx[key]] = cycle
            lb[loop_idx[key]] = -self.max_loops
            ub[loop_idx[key]] = self.max_loops
            ub[width_idx[key]] = cycle
            ub[global_idx[key]] = cycle
            integrality[loop_idx[key]] = 1
        for local_key, band_var in lattice_idx.items():
            ub[local_t_idx[local_key]] = cycle
            lb[local_loop_idx[local_key]] = -self.max_loops
            ub[local_loop_idx[local_key]] = self.max_loops
            ub[band_var] = cycle
            integrality[local_loop_idx[local_key]] = 1

        for gvar in balance_vars.values():
            ub[gvar] = cycle
        for _spec, _terms, slack, _kind in resolved_specs:
            if slack is not None:
                ub[slack] = cycle

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

        for i, opt_list in enumerate(options):
            add_row({idx_opt[i][o]: 1.0 for o in range(len(opt_list))}, 1.0, 1.0)

        for key in band_instances:
            direction, segment_no = key
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

            for i, opt_list in enumerate(options):
                start_terms = {
                    idx_opt[i][o]: effective_start(plan, direction, segment_no)
                    for o, plan in enumerate(opt_list)
                }
                add_row({t_idx[key][i]: -1.0, **start_terms}, -np.inf, 0.0)

            for e in range(m):
                left_end_terms = {
                    idx_opt[e][o]: -effective_end(plan, direction, segment_no)
                    for o, plan in enumerate(options[e])
                }
                right_end_terms = {
                    idx_opt[e + 1][o]: -effective_end(plan, direction, segment_no)
                    for o, plan in enumerate(options[e + 1])
                }
                add_row(
                    {t_idx[key][e]: 1.0, width_idx[key][e]: 1.0, **left_end_terms},
                    -np.inf,
                    0.0,
                )
                add_row(
                    {t_idx[key][e + 1]: 1.0, width_idx[key][e]: 1.0, **right_end_terms},
                    -np.inf,
                    0.0,
                )
                add_row({global_idx[key]: 1.0, width_idx[key][e]: -1.0}, -np.inf, 0.0)

        for local_key, band_var in lattice_idx.items():
            direction, segment_no, k, start = local_key
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
                start_terms = {
                    idx_opt[abs_idx][o]: effective_start(plan, direction, segment_no)
                    for o, plan in enumerate(options[abs_idx])
                }
                end_terms = {
                    idx_opt[abs_idx][o]: -effective_end(plan, direction, segment_no)
                    for o, plan in enumerate(options[abs_idx])
                }
                add_row({t_vars[offset]: -1.0, **start_terms}, -np.inf, 0.0)
                add_row({t_vars[offset]: 1.0, band_var: 1.0, **end_terms}, -np.inf, 0.0)

        for gidx, group in enumerate(config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                row = {gvar: 1.0}
                for var in band_member_vars(member):
                    row[var] = row.get(var, 0.0) - 1.0
                add_row(row, -np.inf, 0.0)

        # 业务约束与损失建模（支持 plan_tags 与 Big-M）
        BIG_M = cycle * 10.0
        for spec, terms, slack, _kind in resolved_specs:
            row = {var: coef for var, coef in terms}
            
            # 处理 plan_tags 条件
            # 如果指定了 plan_tags，约束仅在对应的 δ_i_o = 1 时生效
            # Big-M 逻辑: Expr >= rhs - M * (N_tags - Σ δ_i_o)
            condition_terms: dict[int, float] = {}
            if spec.plan_tags:
                for int_name, plan_name in spec.plan_tags.items():
                    if int_name not in name_to_i:
                        continue
                    i = name_to_i[int_name]
                    # 找到该路口对应的方案下标
                    found = False
                    for o, plan in enumerate(options[i]):
                        if plan.name == plan_name:
                            condition_terms[idx_opt[i][o]] = 1.0
                            found = True
                            break
                    if not found:
                        # 如果该方案根本不存在，则该约束永远不生效，跳过
                        continue
            
            n_tags = len(condition_terms)
            
            if spec.sense == "<=":
                # Expr <= rhs + M * (N_tags - Σ δ_i_o) + slack
                # Expr + M * Σ δ_i_o - slack <= rhs + M * N_tags
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
                # Expr >= rhs - M * (N_tags - Σ δ_i_o) - slack
                # Expr - M * Σ δ_i_o + slack >= rhs - M * N_tags
                m_row = dict(row)
                if slack is not None:
                    m_row[slack] = 1.0
                if n_tags > 0:
                    for v_idx, v_coef in condition_terms.items():
                        m_row[v_idx] = m_row.get(v_idx, 0.0) - BIG_M * v_coef
                    add_row(m_row, spec.rhs - BIG_M * n_tags, np.inf)
                else:
                    add_row(m_row, spec.rhs, np.inf)

        constraints = (
            LinearConstraint(np.array(rows), np.array(lo_list), np.array(hi_list))
            if rows
            else ()
        )
        result = milp(
            c=c,
            constraints=constraints,
            bounds=Bounds(lb, ub),
            integrality=integrality,
        )

        sol = Solution(cycle=cycle, solver_msg=f"HiGHS via scipy: success={result.success}")
        sol.band_up_style = "multi"
        sol.band_down_style = "multi"
        if result.x is None:
            sol.status = "infeasible"
            return sol

        x = result.x
        chosen_plan_idx = {
            int_names[i]: max(range(len(options[i])), key=lambda o: float(x[idx_opt[i][o]]))
            for i in range(n)
        }
        selected = [options[i][chosen_plan_idx[int_names[i]]] for i in range(n)]
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}
        sol.window_choices = {
            int_names[i]: {
                "plan": selected[i].name,
                "up_window": 0 if selected[i].up_segments else -1,
                "down_window": 0 if selected[i].down_segments else -1,
            }
            for i in range(n)
        }

        sol.multi_bandwidths = {"up": {}, "down": {}}
        sol.multi_band_starts = {"up": {}, "down": {}}
        sol.multi_window_bands = {"up": {}, "down": {}}

        aggregated_edge_widths = {
            "up": {seg_name: 0.0 for seg_name in seg_names},
            "down": {seg_name: 0.0 for seg_name in seg_names},
        }
        for direction, segment_no in band_instances:
            key = (direction, segment_no)
            edge_widths = [float(x[index]) for index in width_idx[key]]
            global_width = float(x[global_idx[key]])
            sol.multi_bandwidths[direction][segment_no] = global_width
            sol.multi_band_starts[direction][segment_no] = {
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

        if active_by_direction["up"]:
            first_segment = active_by_direction["up"][0]
            sol.band_start_up = {
                int_names[i]: float(x[t_idx[("up", first_segment)][i]])
                for i in range(n)
            }
        if active_by_direction["down"]:
            first_segment = active_by_direction["down"][0]
            sol.band_start_down = {
                int_names[i]: float(x[t_idx[("down", first_segment)][i]])
                for i in range(n)
            }
        fill_solution_multi_window_bands(sol, arterial, max_loops=self.max_loops, margin=self.margin)
        fill_solution_window_band_ranges(sol, arterial)

        band_loss = 0.0
        intersection_loss = 0.0
        for spec, terms, _slack, kind in resolved_specs:
            if not spec.soft:
                continue
            expr_value = sum(coef * float(x[var]) for var, coef in terms)
            
            # 同样需要考虑 plan_tags 是否满足
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

        band_objective = 0.0
        for group in config.sum_groups:
            for key_text, weight in group.terms.items():
                band_objective += weight * sum(float(x[var]) for var in band_member_vars(key_text))
        for gidx, group in enumerate(config.balance_groups):
            band_objective += group.weight * float(x[balance_vars[gidx]])
            if group.eps > 0:
                for member in group.members:
                    band_objective += group.weight * group.eps * sum(
                        float(x[var]) for var in band_member_vars(member)
                    )

        sol.band_objective = float(band_objective)
        sol.band_loss = float(band_loss)
        sol.band_score = float(band_objective - band_loss_weight * band_loss)
        sol.intersection_loss = float(intersection_loss)
        sol.objective = -float(result.fun)
        sol.status = "optimal" if result.success else result.message
        return sol

    @staticmethod
    def _build_plan_options(intersections) -> list[list[SignalPlan]]:
        """函数名：_build_plan_options；参数：intersections；返回值：候选方案列表；异常：ValueError。"""
        options: list[list[SignalPlan]] = []
        for inter in intersections:
            if not inter.plans:
                raise ValueError(f"路口 {inter.name} 没有可选方案")
            for plan in inter.plans:
                if not plan.up_segments or not plan.down_segments:
                    raise ValueError(f"路口 {inter.name} 方案 {plan.name} 缺少上下行绿灯分段")
            options.append(list(inter.plans))
        return options

    def _common_band_instances(self, options: list[list[SignalPlan]]) -> list[BandInstance]:
        """函数名：_common_band_instances；参数：options；返回值：全候选共同拥有的段实例列表；异常：无。"""
        keys: list[BandInstance] = []
        for direction in ("up", "down"):
            max_count = min(
                len(plan.up_segments if direction == "up" else plan.down_segments)
                for opt_list in options
                for plan in opt_list
            )
            for segment_no in range(1, min(max_count, self.max_segments) + 1):
                keys.append((direction, segment_no))
        return keys

    @staticmethod
    def _local_objective_specs(config: ObjectiveConfig,
                               n: int) -> list[tuple[str, int, int]]:
        """函数名：_local_objective_specs；参数：config、n；返回值：局部窗口规格；异常：无。"""
        specs: set[tuple[str, int, int]] = set()
        for group in config.sum_groups:
            for key_text in group.terms:
                if key_text.endswith(".global"):
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

    def _window_common_segment_numbers(self,
                                       options: list[list[SignalPlan]],
                                       direction: str,
                                       start: int,
                                       k: int) -> list[int]:
        """函数名：_window_common_segment_numbers；参数：options、direction、start、k；返回值：段号列表；异常：无。"""
        max_count = min(
            len(plan.up_segments if direction == "up" else plan.down_segments)
            for abs_idx in range(start, start + k)
            for plan in options[abs_idx]
        )
        return list(range(1, min(max_count, self.max_segments) + 1))

    @staticmethod
    def _segment_window(plan: SignalPlan, direction: str, segment_no: int):
        """函数名：_segment_window；参数：plan、direction、segment_no；返回值：GreenWindow；异常：IndexError。"""
        windows = plan.up_segments if direction == "up" else plan.down_segments
        return windows[segment_no - 1]
