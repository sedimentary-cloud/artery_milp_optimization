"""FullFlexiblePhaseTuneSolver：段级窗口变量 + 多段带传播 + ObjectiveConfig。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...models import Arterial, SignalConstraint, SignalLoss, SignalPlan
from ...solution import Solution
from ..core.base import Solver
from ..core.band_lattice import (fill_solution_multi_window_bands,
                                 fill_solution_window_band_ranges)
from ..core.objective import BandKey, ObjectiveConfig, parse_band_key
from ..builders.signal_constraints import (ConstraintBuilder, LinearSpec,
                                           SegmentLossBuilder,
                                           scale_linear_spec, window_exprs)
from ..builders.term_validation import TermValidationContext


BandInstance = tuple[str, int]
LatticeInstance = tuple[str, int, int, int]


class FullFlexiblePhaseTuneSolver(Solver):
    """完整段级求解器：段端点变量 + 多段带传播 + 业务约束。"""

    name = "full-flexible-phase"

    def __init__(
        self,
        config: ObjectiveConfig,
        max_loops: int = 3,
        up_global_output: bool = True,
        down_global_output: bool = False,
    ) -> None:
        """函数名：__init__；参数：config、max_loops、输出口径；返回值：无；异常：无。"""
        self.config = config
        self.max_loops = max_loops
        self.up_global_output = up_global_output
        self.down_global_output = down_global_output

    @staticmethod
    def _available_bands(plans: list[SignalPlan]) -> list[BandInstance]:
        """函数名：_available_bands；参数：plans；返回值：全走廊共享的带段键；异常：无。"""
        keys: list[BandInstance] = []
        for direction in ("up", "down"):
            max_count = min(
                len(plan.up_segments if direction == "up" else plan.down_segments)
                for plan in plans
            )
            for segment_no in range(1, max_count + 1):
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

    @staticmethod
    def _window_common_segment_numbers(plans: list[SignalPlan],
                                       direction: str,
                                       start: int,
                                       k: int) -> list[int]:
        """函数名：_window_common_segment_numbers；参数：plans、direction、start、k；返回值：段号列表；异常：无。"""
        max_count = min(
            len(plan.up_segments if direction == "up" else plan.down_segments)
            for plan in plans[start:start + k]
        )
        return list(range(1, max_count + 1))

    def solve(
        self,
        arterial: Arterial,
        prior: Solution | None = None,
        loss_builder: SegmentLossBuilder | None = None,
        constraint_builder: ConstraintBuilder | None = None,
        max_loss: float | None = None,
        max_intersection_loss: float | None = None,
        band_loss_weight: float = 0.0,
        objective: str = "bandwidth",
        tunable_intersections: set[str] | None = None,
    ) -> Solution:
        """函数名：solve；参数：arterial 等；返回值：Solution；异常：ValueError。"""
        cycle = arterial.cycle
        ints = arterial.intersection_order
        segs = arterial.segment_order
        n, m = len(ints), len(segs)
        if n < 2:
            raise ValueError("FullFlexiblePhaseTuneSolver 至少需要两个路口")
        self.config.validate(n)

        int_names = [inter.name for inter in ints]
        name_to_i = {name: idx for idx, name in enumerate(int_names)}
        seg_names = [seg.name for seg in segs]
        seg_name_to_idx = {name: idx for idx, name in enumerate(seg_names)}

        selected: list[SignalPlan] = []
        for inter in ints:
            if not inter.plans:
                raise ValueError(f"路口 {inter.name} 没有可选方案")
            pname = prior.plan_choices.get(inter.name) if prior and prior.plan_choices else None
            selected.append(inter.plan_by_name(pname) if pname else inter.plans[0])

        band_instances = self._available_bands(selected)
        if not band_instances:
            raise ValueError("当前方案下没有可用的全走廊多段绿波带")

        exprs = [window_exprs(plan, cycle) for plan in selected]
        tunable = [
            tunable_intersections is None or name in tunable_intersections
            for name in int_names
        ]
        active_by_direction = {
            "up": [segment_no for direction, segment_no in band_instances if direction == "up"],
            "down": [segment_no for direction, segment_no in band_instances if direction == "down"],
        }

        cur = 0
        idx_terms: list[list[int]] = []
        for expr in exprs:
            idx_terms.append(list(range(cur, cur + len(expr.term_names))))
            cur += len(expr.term_names)

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

        local_specs = self._local_objective_specs(self.config, n)
        local_segments_by_spec: dict[tuple[str, int, int], list[int]] = {}
        for direction, k, start in local_specs:
            segment_numbers = self._window_common_segment_numbers(selected, direction, start, k)
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
        for gidx in range(len(self.config.balance_groups)):
            balance_vars[gidx] = cur
            cur += 1

        def build_prefixed_specs(
            plan_constraints: list[SignalConstraint],
            intersection_name: str,
        ) -> list[LinearSpec]:
            """函数名：build_prefixed_specs；参数：plan_constraints、intersection_name；返回值：LinearSpec 列表；异常：无。"""
            specs: list[LinearSpec] = []
            for item in plan_constraints:
                specs.append(
                    LinearSpec(
                        terms={f"{intersection_name}.{term}": coef for term, coef in item.terms.items()},
                        sense=item.sense,
                        rhs=item.rhs,
                        soft=False,
                        penalty=1.0,
                        name=item.name,
                    )
                )
            return specs

        def build_prefixed_loss_specs(
            plan_losses: list[SignalLoss],
            intersection_name: str,
        ) -> list[LinearSpec]:
            """函数名：build_prefixed_loss_specs；参数：plan_losses、intersection_name；返回值：LinearSpec 列表；异常：无。"""
            specs: list[LinearSpec] = []
            for item in plan_losses:
                terms = {f"{intersection_name}.{term}": coef for term, coef in item.terms.items()}
                if item.lower_threshold is not None:
                    specs.append(
                        LinearSpec(
                            terms=terms,
                            sense=">=",
                            rhs=item.lower_threshold,
                            soft=True,
                            penalty=item.lower_slope,
                            name=(f"{item.name}.lower" if item.name else ""),
                        )
                    )
                if item.upper_threshold is not None:
                    specs.append(
                        LinearSpec(
                            terms=terms,
                            sense="<=",
                            rhs=item.upper_threshold,
                            soft=True,
                            penalty=(
                                item.upper_slope
                                if item.upper_slope is not None
                                else item.lower_slope
                            ),
                            name=(f"{item.name}.upper" if item.name else ""),
                        )
                    )
            return specs

        band_specs: list[LinearSpec] = []
        intersection_specs: list[LinearSpec] = []
        for i, plan in enumerate(selected):
            intersection_specs.extend(build_prefixed_specs(plan.signal_constraints, int_names[i]))
            intersection_specs.extend(build_prefixed_loss_specs(plan.signal_losses, int_names[i]))
        if constraint_builder is not None:
            intersection_specs.extend(constraint_builder.specs)
        if loss_builder is not None:
            for spec, kind in loss_builder.to_linear_specs():
                if kind == "band":
                    band_specs.append(spec)
                else:
                    intersection_specs.append(spec)

        # 统一 term 校验：Stage 2 方案已锁定，要求选中方案定义相应端点。
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
            balance_group_count=len(self.config.balance_groups),
            endpoint_terms_by_intersection={
                name: set(exprs[i].term_names) for i, name in enumerate(int_names)
            },
            plan_names_by_intersection={
                inter.name: {plan.name for plan in inter.plans} for inter in ints
            },
        )
        validation_ctx.validate_specs(
            [(spec, "band") for spec in band_specs]
            + [(spec, "intersection") for spec in intersection_specs]
        )

        def band_member_vars_text(key_text: str) -> list[int]:
            """函数名：band_member_vars_text；参数：key_text；返回值：聚合变量列表；异常：无。"""
            band = parse_band_key(key_text, n)
            vars_out: list[int] = []
            if key_text.endswith(".global"):
                for segment_no in active_by_direction.get(band.direction, []):
                    vars_out.append(global_idx[(band.direction, segment_no)])
                return vars_out

            for segment_no in local_segments_by_spec.get((band.direction, band.k, band.start), []):
                key = (band.direction, segment_no, band.k, band.start)
                if key in lattice_idx:
                    vars_out.append(lattice_idx[key])
            return vars_out

        def resolve_special_terms(name: str, coef: float) -> list[tuple[int, float]]:
            """函数名：resolve_special_terms；参数：name、coef；返回值：[(变量下标, 系数)]；异常：无。"""
            if name == "b_up":
                return [
                    (global_idx[("up", segment_no)], coef)
                    for segment_no in active_by_direction["up"]
                ]
            if name == "b_down":
                return [
                    (global_idx[("down", segment_no)], coef)
                    for segment_no in active_by_direction["down"]
                ]
            if name == "B_bal":
                gvar = balance_vars.get(0)
                return [] if gvar is None else [(gvar, coef)]
            if name.startswith("tU_"):
                i = name_to_i.get(name[3:])
                if i is None or not active_by_direction["up"]:
                    return []
                first_segment = active_by_direction["up"][0]
                return [(t_idx[("up", first_segment)][i], coef)]
            if name.startswith("tD_"):
                i = name_to_i.get(name[3:])
                if i is None or not active_by_direction["down"]:
                    return []
                first_segment = active_by_direction["down"][0]
                return [(t_idx[("down", first_segment)][i], coef)]
            if name.startswith("bD_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [
                    (width_idx[("down", segment_no)][seg_idx], coef)
                    for segment_no in active_by_direction["down"]
                ]
            if name.startswith("bU_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [
                    (width_idx[("up", segment_no)][seg_idx], coef)
                    for segment_no in active_by_direction["up"]
                ]
            return []

        def resolve_term_terms(name: str, coef: float) -> list[tuple[int, float]]:
            """函数名：resolve_term_terms；参数：name、coef；返回值：[(变量下标, 系数)]；异常：无。"""
            parts = name.split(".")
            if len(parts) != 4:
                return []
            inter_name = parts[0]
            local_term = ".".join(parts[1:])
            i = name_to_i.get(inter_name)
            if i is None:
                return []
            try:
                local_index = exprs[i].term_names.index(local_term)
            except ValueError:
                return []
            return [(idx_terms[i][local_index], coef)]

        resolved_constraints: list[tuple[LinearSpec, list[tuple[int, float]], int | None, str]] = []
        for kind, specs in (("band", band_specs), ("intersection", intersection_specs)):
            for spec in specs:
                # 过滤 plan_tags
                if spec.plan_tags:
                    skip = False
                    for int_name, plan_name in spec.plan_tags.items():
                        if int_name not in name_to_i:
                            continue
                        i = name_to_i[int_name]
                        if selected[i].name != plan_name:
                            skip = True
                            break
                    if skip:
                        continue

                scaled_spec = scale_linear_spec(spec, cycle)
                terms: list[tuple[int, float]] = []
                for name, coef in scaled_spec.terms.items():
                    mapped = resolve_special_terms(name, coef)
                    if not mapped:
                        mapped = resolve_term_terms(name, coef)
                    terms.extend(mapped)
                if not terms:
                    continue
                slack = None
                if scaled_spec.soft:
                    slack = cur
                    cur += 1
                resolved_constraints.append((scaled_spec, terms, slack, kind))

        nvar = cur
        effective_max_intersection = (
            max_intersection_loss if max_intersection_loss is not None else max_loss
        )

        c = np.zeros(nvar)
        if objective in ("loss", "intersection_loss"):
            for spec, _, slack, kind in resolved_constraints:
                if kind == "intersection" and spec.soft and slack is not None:
                    c[slack] += spec.penalty
        else:
            for group in self.config.sum_groups:
                for key, weight in group.terms.items():
                    member_vars = band_member_vars_text(key)
                    if not member_vars:
                        raise ValueError(f"目标项 {key} 在当前多段方案下没有可用带宽变量")
                    for var in member_vars:
                        c[var] += -weight
            for gidx, group in enumerate(self.config.balance_groups):
                c[balance_vars[gidx]] += -group.weight
                for member in group.members:
                    member_vars = band_member_vars_text(member)
                    if not member_vars:
                        raise ValueError(f"均衡项 {member} 在当前多段方案下没有可用带宽变量")
                    if group.eps > 0:
                        for var in member_vars:
                            c[var] += -group.weight * group.eps
            for spec, _, slack, kind in resolved_constraints:
                if kind == "band" and spec.soft and slack is not None:
                    c[slack] += band_loss_weight * spec.penalty
            if effective_max_intersection is not None:
                tiny = 1e-7
                for spec, _, slack, kind in resolved_constraints:
                    if kind == "intersection" and spec.soft and slack is not None:
                        c[slack] += tiny * spec.penalty

        lb = np.zeros(nvar)
        ub = np.full(nvar, np.inf)
        integrality = np.zeros(nvar)

        for i, expr in enumerate(exprs):
            for j, term in enumerate(expr.term_names):
                lower, upper = expr.term_bounds[term]
                if not tunable[i]:
                    value = selected[i].term_value(term) * cycle
                    lower = value
                    upper = value
                lb[idx_terms[i][j]] = lower
                ub[idx_terms[i][j]] = upper

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
        for spec, _, slack, _kind in resolved_constraints:
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

        for i, expr in enumerate(exprs):
            for direction in ("up", "down"):
                labels = expr.up_labels if direction == "up" else expr.down_labels
                count = len(labels)
                for seg_no in range(1, count + 1):
                    start_name = f"{direction}.{seg_no}.start"
                    end_name = f"{direction}.{seg_no}.end"
                    start_var = idx_terms[i][expr.term_names.index(start_name)]
                    end_var = idx_terms[i][expr.term_names.index(end_name)]
                    add_row({end_var: 1.0, start_var: -1.0}, 0.0, np.inf)
                    if seg_no < count:
                        next_start_name = f"{direction}.{seg_no + 1}.start"
                        next_start_var = idx_terms[i][expr.term_names.index(next_start_name)]
                        add_row({next_start_var: 1.0, end_var: -1.0}, 0.0, np.inf)

        for spec, terms, slack, _kind in resolved_constraints:
            row: dict[int, float] = {}
            for var, coef in terms:
                row[var] = row.get(var, 0.0) + coef
            if not spec.soft:
                if spec.sense == ">=":
                    add_row(row, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row(row, -np.inf, spec.rhs)
                elif spec.sense == "=":
                    add_row(row, spec.rhs, spec.rhs)
                else:
                    raise ValueError(f"unknown sense: {spec.sense}")
            else:
                if spec.sense == ">=":
                    add_row({**row, slack: 1.0}, spec.rhs, np.inf)
                elif spec.sense == "<=":
                    add_row({**row, slack: -1.0}, -np.inf, spec.rhs)
                else:
                    raise ValueError("soft '=' constraint is not supported")

        if effective_max_intersection is not None:
            loss_row: dict[int, float] = {}
            for spec, _, slack, kind in resolved_constraints:
                if kind == "intersection" and spec.soft and slack is not None:
                    loss_row[slack] = loss_row.get(slack, 0.0) + spec.penalty
            add_row(loss_row, -np.inf, effective_max_intersection)

        for direction, segment_no in band_instances:
            key = (direction, segment_no)
            if direction == "up":
                start_name = f"up.{segment_no}.start"
                end_name = f"up.{segment_no}.end"
            else:
                start_name = f"down.{segment_no}.start"
                end_name = f"down.{segment_no}.end"

            for i, seg in enumerate(segs):
                if direction == "up":
                    add_row(
                        {t_idx[key][i + 1]: 1.0, t_idx[key][i]: -1.0, loop_idx[key][i]: -cycle},
                        seg.travel_time_up,
                        seg.travel_time_up,
                    )
                else:
                    add_row(
                        {t_idx[key][i]: 1.0, t_idx[key][i + 1]: -1.0, loop_idx[key][i]: -cycle},
                        seg.travel_time_down,
                        seg.travel_time_down,
                    )

            for i, expr in enumerate(exprs):
                start_var = idx_terms[i][expr.term_names.index(start_name)]
                add_row({t_idx[key][i]: 1.0, start_var: -1.0}, 0.0, np.inf)

            for e in range(m):
                left_end_var = idx_terms[e][exprs[e].term_names.index(end_name)]
                right_end_var = idx_terms[e + 1][exprs[e + 1].term_names.index(end_name)]
                add_row(
                    {t_idx[key][e]: 1.0, width_idx[key][e]: 1.0, left_end_var: -1.0},
                    -np.inf,
                    0.0,
                )
                add_row(
                    {t_idx[key][e + 1]: 1.0, width_idx[key][e]: 1.0, right_end_var: -1.0},
                    -np.inf,
                    0.0,
                )
                add_row(
                    {global_idx[key]: 1.0, width_idx[key][e]: -1.0},
                    -np.inf,
                    0.0,
                )

        for local_key, band_var in lattice_idx.items():
            direction, segment_no, k, start = local_key
            t_vars = local_t_idx[local_key]
            loop_vars = local_loop_idx[local_key]
            local_segs = segs[start:start + k - 1]

            if direction == "up":
                start_name = f"up.{segment_no}.start"
                end_name = f"up.{segment_no}.end"
            else:
                start_name = f"down.{segment_no}.start"
                end_name = f"down.{segment_no}.end"

            for offset, seg in enumerate(local_segs):
                if direction == "up":
                    add_row(
                        {t_vars[offset + 1]: 1.0, t_vars[offset]: -1.0, loop_vars[offset]: -cycle},
                        seg.travel_time_up,
                        seg.travel_time_up,
                    )
                else:
                    add_row(
                        {t_vars[offset]: 1.0, t_vars[offset + 1]: -1.0, loop_vars[offset]: -cycle},
                        seg.travel_time_down,
                        seg.travel_time_down,
                    )

            for offset, abs_idx in enumerate(range(start, start + k)):
                start_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(start_name)]
                end_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(end_name)]
                add_row({t_vars[offset]: 1.0, start_var: -1.0}, 0.0, np.inf)
                add_row({t_vars[offset]: 1.0, band_var: 1.0, end_var: -1.0}, -np.inf, 0.0)

        for gidx, group in enumerate(self.config.balance_groups):
            gvar = balance_vars[gidx]
            for member in group.members:
                row = {gvar: 1.0}
                member_vars = band_member_vars_text(member)
                for var in member_vars:
                    row[var] = row.get(var, 0.0) - 1.0
                add_row(row, -np.inf, 0.0)

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
        if result.x is None:
            sol.status = "infeasible"
            return sol

        x = result.x
        sol.plan_choices = {int_names[i]: selected[i].name for i in range(n)}
        sol.segment_times = {
            int_names[i]: {
                term: float(x[idx_terms[i][j]])
                for j, term in enumerate(expr.term_names)
            }
            for i, expr in enumerate(exprs)
        }
        sol.window_choices = {}
        for i, expr in enumerate(exprs):
            entry: dict[str, int | str] = {"plan": selected[i].name}
            if expr.up_labels:
                entry["up_segment"] = expr.up_labels[0]
                entry["up_window"] = 0
            if expr.down_labels:
                entry["down_segment"] = expr.down_labels[0]
                entry["down_window"] = 0
            sol.window_choices[int_names[i]] = entry

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
            first = active_by_direction["up"][0]
            sol.band_start_up = {
                int_names[i]: float(x[t_idx[("up", first)][i]])
                for i in range(n)
            }
        if active_by_direction["down"]:
            first = active_by_direction["down"][0]
            sol.band_start_down = {
                int_names[i]: float(x[t_idx[("down", first)][i]])
                for i in range(n)
            }

        fill_solution_multi_window_bands(sol, arterial, max_loops=self.max_loops)
        fill_solution_window_band_ranges(sol, arterial)
        # 当前 solver 始终输出多段带，风格固定为 multi。
        sol.band_up_style = "multi"
        sol.band_down_style = "multi"

        band_loss = 0.0
        intersection_loss = 0.0
        for spec, terms, _slack, kind in resolved_constraints:
            if not spec.soft:
                continue
            expr_value = sum(coef * float(x[var]) for var, coef in terms)
            if spec.sense == ">=":
                violation = max(0.0, spec.rhs - expr_value)
            else:
                violation = max(0.0, expr_value - spec.rhs)
            weighted = spec.penalty * violation
            if kind == "band":
                band_loss += weighted
            else:
                intersection_loss += weighted

        band_objective = 0.0
        for group in self.config.sum_groups:
            for key, weight in group.terms.items():
                band_objective += weight * sum(float(x[var]) for var in band_member_vars_text(key))
        for gidx, group in enumerate(self.config.balance_groups):
            band_objective += group.weight * float(x[balance_vars[gidx]])
            if group.eps > 0:
                for member in group.members:
                    band_objective += group.weight * group.eps * sum(
                        float(x[var]) for var in band_member_vars_text(member)
                    )

        sol.band_objective = float(band_objective)
        sol.band_loss = float(band_loss)
        sol.band_score = float(band_objective - band_loss_weight * band_loss)
        sol.intersection_loss = float(intersection_loss)
        sol.objective = -float(result.fun)
        sol.status = "optimal" if result.success else result.message
        return sol
