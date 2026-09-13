"""FullFlexiblePhaseTuneSolver：段级窗口变量 + 多段带传播 + ObjectiveConfig。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...models import Arterial, SignalConstraint, SignalLoss, SignalPlan
from ...solution import Solution
from ..core.base import Solver, compute_effective_max_loops
from ..core.band_lattice import (fill_solution_local_band_records,
                                 fill_solution_missing_window_bands)
from ..core.margin import BandMarginConfig
from ..core.objective import ObjectiveConfig, parse_band_key
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
        max_bands: int | None = None,
        band_gap: float = 0.0,
        up_global_output: bool = True,
        down_global_output: bool = False,
        margin: BandMarginConfig | None = None,
    ) -> None:
        """函数名：__init__；参数：config、max_loops、最大 band 数、band 间隔、输出口径、边距；返回值：无；异常：无。"""
        self.config = config
        self.max_loops = max_loops
        self.max_bands = int(max_bands) if max_bands is not None else None
        self.band_gap = float(band_gap)
        self.up_global_output = up_global_output
        self.down_global_output = down_global_output
        self.margin = margin or BandMarginConfig()
        if self.max_bands is not None and self.max_bands <= 0:
            raise ValueError("max_bands 必须为正")
        if self.band_gap < 0:
            raise ValueError("band_gap 不能为负")

    def _band_instances_from_prior(self, prior: Solution | None) -> list[BandInstance]:
        """从 Stage 1 结果恢复 band 编号；缺失时回退到 max_bands 或默认 1。"""
        keys: list[BandInstance] = []
        for direction in ("up", "down"):
            if prior is not None and prior.multi_bandwidths.get(direction):
                numbers = sorted(int(v) for v in prior.multi_bandwidths[direction].keys())
            elif self.max_bands is not None:
                numbers = list(range(1, self.max_bands + 1))
            else:
                numbers = [1]
            for band_no in numbers:
                keys.append((direction, band_no))
        return keys

    @staticmethod
    def _local_objective_specs(config: ObjectiveConfig,
                               n: int) -> list[tuple[str, int, int]]:
        """函数名：_local_objective_specs；参数：config、n；返回值：局部窗口规格；异常：无。"""
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

    def _local_band_numbers_from_prior(self,
                                       prior: Solution | None,
                                       key: str,
                                       fallback_numbers: list[int]) -> list[int]:
        """从 Stage 1 结果恢复局部 band 编号，并限制局部最多 2 条。"""
        if prior is not None:
            choices = prior.local_band_window_choices.get(key)
            if choices:
                return sorted(int(v) for v in choices.keys())[:2]
        return list(fallback_numbers)[:2]

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

        # self.max_loops 只是手动下限；长路段会自动放大。
        effective_max_loops = compute_effective_max_loops(arterial, self.max_loops)

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

        band_instances = self._band_instances_from_prior(prior)
        if not band_instances:
            raise ValueError("当前方案下没有可用的绿波带 band")

        exprs = [window_exprs(plan, cycle) for plan in selected]
        tunable = [
            tunable_intersections is None or name in tunable_intersections
            for name in int_names
        ]
        active_by_direction = {
            "up": [band_no for direction, band_no in band_instances if direction == "up"],
            "down": [band_no for direction, band_no in band_instances if direction == "down"],
        }

        # ------------------------------------------------------------------
        # 从 Stage 1 恢复固定窗口选择和顺序选择。
        # ------------------------------------------------------------------
        def plan_windows(plan: SignalPlan, direction: str) -> list:
            return plan.up_segments if direction == "up" else plan.down_segments

        def global_window_no(direction: str, band_no: int, i: int) -> int:
            plan = selected[i]
            q = 1
            if prior is not None:
                choice = prior.band_window_choices.get(direction, {}).get(band_no, {}).get(int_names[i])
                if isinstance(choice, dict):
                    try:
                        q = int(choice.get("window", 1))
                    except (TypeError, ValueError):
                        q = 1
            windows = plan_windows(plan, direction)
            if not (1 <= q <= len(windows)):
                q = 1
            return q

        def local_window_no(direction: str, band_no: int, key: str, i: int) -> int:
            plan = selected[i]
            q = 1
            if prior is not None:
                choice = prior.local_band_window_choices.get(key, {}).get(band_no, {}).get(int_names[i])
                if isinstance(choice, dict):
                    try:
                        q = int(choice.get("window", 1))
                    except (TypeError, ValueError):
                        q = 1
            windows = plan_windows(plan, direction)
            if not (1 <= q <= len(windows)):
                q = 1
            return q

        def global_order_value(direction: str, r: int, s: int, i: int) -> int:
            """缺失时默认 r 在 s 前面，保证 Stage 2 仍有顺序约束。"""
            if prior is not None:
                try:
                    return int(prior.band_order_choices[direction][r][s][int_names[i]])
                except (KeyError, TypeError):
                    pass
            return 1

        def local_order_value(key: str, r: int, s: int, i: int) -> int:
            """缺失时默认 r 在 s 前面，保证 Stage 2 仍有顺序约束。"""
            if prior is not None:
                try:
                    return int(prior.local_band_order_choices[key][r][s][int_names[i]])
                except (KeyError, TypeError):
                    pass
            return 1

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
        local_bands_by_spec: dict[tuple[str, int, int], list[int]] = {}
        for direction, k, start in local_specs:
            key = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
            fallback = active_by_direction.get(direction, [1])[:2]
            band_numbers = self._local_band_numbers_from_prior(prior, key, fallback)
            local_bands_by_spec[(direction, k, start)] = band_numbers
            for band_no in band_numbers:
                local_key = (direction, band_no, k, start)
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
            specs: list[LinearSpec] = []
            for item in plan_losses:
                terms = {f"{intersection_name}.{term}": coef for term, coef in item.terms.items()}
                if item.lower_threshold is not None:
                    specs.append(LinearSpec(terms=terms, sense=">=", rhs=item.lower_threshold,
                                            soft=True, penalty=item.lower_slope,
                                            name=(f"{item.name}.lower" if item.name else "")))
                if item.upper_threshold is not None:
                    specs.append(LinearSpec(terms=terms, sense="<=", rhs=item.upper_threshold,
                                            soft=True,
                                            penalty=(item.upper_slope if item.upper_slope is not None else item.lower_slope),
                                            name=(f"{item.name}.upper" if item.name else "")))
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

        validation_ctx = TermValidationContext(
            intersection_names=int_names,
            segment_names=seg_names,
            active_directions={direction for direction, nums in active_by_direction.items() if nums},
            active_segment_numbers={direction: set(nums) for direction, nums in active_by_direction.items() if nums},
            balance_group_count=len(self.config.balance_groups),
            endpoint_terms_by_intersection={name: set(exprs[i].term_names) for i, name in enumerate(int_names)},
            plan_names_by_intersection={inter.name: {plan.name for plan in inter.plans} for inter in ints},
        )
        validation_ctx.validate_specs(
            [(spec, "band") for spec in band_specs]
            + [(spec, "intersection") for spec in intersection_specs]
        )

        def band_member_vars_text(key_text: str) -> list[int]:
            band = parse_band_key(key_text, n)
            vars_out: list[int] = []
            if key_text.endswith(".global"):
                for band_no in active_by_direction.get(band.direction, []):
                    vars_out.append(global_idx[(band.direction, band_no)])
                return vars_out
            for band_no in local_bands_by_spec.get((band.direction, band.k, band.start), []):
                key = (band.direction, band_no, band.k, band.start)
                if key in lattice_idx:
                    vars_out.append(lattice_idx[key])
            return vars_out

        def resolve_special_terms(name: str, coef: float) -> list[tuple[int, float]]:
            """解析特殊变量名。

            带宽相关的特殊变量保留：
            - b_up / b_down：对应方向所有全局 band 的 B 求和；
            - bU_* / bD_*：对应方向所有全局 band 在指定物理路段上的宽度求和；
            - B_bal：第一个 BalanceGroup 的组内最小值。

            时间相关的 tU_* / tD_* 已停用：它们过去只取第一条 band，
            在多 band 模型下容易误导，因此只保留注释并主动报错。
            """
            if name == "b_up":
                return [(global_idx[("up", band_no)], coef) for band_no in active_by_direction["up"]]
            if name == "b_down":
                return [(global_idx[("down", band_no)], coef) for band_no in active_by_direction["down"]]
            if name == "B_bal":
                gvar = balance_vars.get(0)
                return [] if gvar is None else [(gvar, coef)]

            # 时间相关特殊变量 tU_* / tD_* 已停用，解析逻辑保留为注释。
            # if name.startswith("tU_"):
            #     i = name_to_i.get(name[3:])
            #     if i is None or not active_by_direction["up"]:
            #         return []
            #     first_band = active_by_direction["up"][0]
            #     return [(t_idx[("up", first_band)][i], coef)]
            # if name.startswith("tD_"):
            #     i = name_to_i.get(name[3:])
            #     if i is None or not active_by_direction["down"]:
            #         return []
            #     first_band = active_by_direction["down"][0]
            #     return [(t_idx[("down", first_band)][i], coef)]

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
                return [
                    (width_idx[("down", band_no)][seg_idx], coef)
                    for band_no in active_by_direction["down"]
                ]
            if name.startswith("bU_"):
                seg_idx = seg_name_to_idx.get(name[3:])
                if seg_idx is None:
                    return []
                return [
                    (width_idx[("up", band_no)][seg_idx], coef)
                    for band_no in active_by_direction["up"]
                ]
            return []


        def resolve_term_terms(name: str, coef: float) -> list[tuple[int, float]]:
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

        hard_margin_up = self.margin.hard_margin_up * cycle
        hard_margin_down = self.margin.hard_margin_down * cycle
        soft_margin_up = self.margin.soft_margin_up * cycle
        soft_margin_down = self.margin.soft_margin_down * cycle
        penalty_up = self.margin.penalty_up
        penalty_down = self.margin.penalty_down

        margin_records: list[dict[str, object]] = []
        if self.margin.has_soft_margin:
            for direction, band_no in band_instances:
                key = (direction, band_no)
                soft_margin = soft_margin_up if direction == "up" else soft_margin_down
                penalty = penalty_up if direction == "up" else penalty_down
                if soft_margin <= 0.0 or penalty <= 0.0:
                    continue
                for i, expr in enumerate(exprs):
                    q = global_window_no(direction, band_no, i)
                    start_name = f"{direction}.{q}.start"
                    start_var = idx_terms[i][expr.term_names.index(start_name)]
                    slack = cur
                    cur += 1
                    margin_records.append({
                        "slack": slack,
                        "terms": {t_idx[key][i]: 1.0, start_var: -1.0},
                        "rhs": soft_margin,
                        "penalty": penalty,
                    })
                for e in range(m):
                    for i in (e, e + 1):
                        q = global_window_no(direction, band_no, i)
                        end_name = f"{direction}.{q}.end"
                        end_var = idx_terms[i][exprs[i].term_names.index(end_name)]
                        slack = cur
                        cur += 1
                        margin_records.append({
                            "slack": slack,
                            "terms": {
                                end_var: 1.0,
                                t_idx[key][i]: -1.0,
                                width_idx[key][e]: -1.0,
                            },
                            "rhs": soft_margin,
                            "penalty": penalty,
                        })

            for (direction, k, start), band_numbers in local_bands_by_spec.items():
                key = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
                soft_margin = soft_margin_up if direction == "up" else soft_margin_down
                penalty = penalty_up if direction == "up" else penalty_down
                if soft_margin <= 0.0 or penalty <= 0.0:
                    continue
                for band_no in band_numbers:
                    t_vars = local_t_idx[(direction, band_no, k, start)]
                    band_var = lattice_idx[(direction, band_no, k, start)]
                    for offset, abs_idx in enumerate(range(start, start + k)):
                        q = local_window_no(direction, band_no, key, abs_idx)
                        start_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(f"{direction}.{q}.start")]
                        end_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(f"{direction}.{q}.end")]
                        slack = cur
                        cur += 1
                        margin_records.append({
                            "slack": slack,
                            "terms": {t_vars[offset]: 1.0, start_var: -1.0},
                            "rhs": soft_margin,
                            "penalty": penalty,
                        })
                        slack = cur
                        cur += 1
                        margin_records.append({
                            "slack": slack,
                            "terms": {end_var: 1.0, t_vars[offset]: -1.0, band_var: -1.0},
                            "rhs": soft_margin,
                            "penalty": penalty,
                        })

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
                    if weight == 0.0:
                        continue
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
            for record in margin_records:
                c[record["slack"]] += band_loss_weight * record["penalty"]
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
            lb[loop_idx[key]] = -effective_max_loops
            ub[loop_idx[key]] = effective_max_loops
            ub[width_idx[key]] = cycle
            ub[global_idx[key]] = cycle
            integrality[loop_idx[key]] = 1
        for local_key, band_var in lattice_idx.items():
            ub[local_t_idx[local_key]] = cycle
            lb[local_loop_idx[local_key]] = -effective_max_loops
            ub[local_loop_idx[local_key]] = effective_max_loops
            ub[band_var] = cycle
            integrality[local_loop_idx[local_key]] = 1
        for gvar in balance_vars.values():
            ub[gvar] = cycle
        for spec, _, slack, _kind in resolved_constraints:
            if slack is not None:
                ub[slack] = cycle
        for record in margin_records:
            ub[record["slack"]] = cycle

        rows: list[np.ndarray] = []
        lo_list: list[float] = []
        hi_list: list[float] = []

        def add_row(coefs: dict[int, float], lower: float, upper: float) -> None:
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

        # ---------------- 全局 band：固定 Stage 1 窗口，只调端点 ----------------
        for direction, band_no in band_instances:
            key = (direction, band_no)
            margin_s = hard_margin_up if direction == "up" else hard_margin_down
            for e, seg in enumerate(segs):
                if direction == "up":
                    add_row(
                        {t_idx[key][e + 1]: 1.0, t_idx[key][e]: -1.0, loop_idx[key][e]: -cycle},
                        seg.travel_time_up,
                        seg.travel_time_up,
                    )
                else:
                    add_row(
                        {t_idx[key][e]: 1.0, t_idx[key][e + 1]: -1.0, loop_idx[key][e]: -cycle},
                        seg.travel_time_down,
                        seg.travel_time_down,
                    )

            for i in range(n):
                q = global_window_no(direction, band_no, i)
                start_var = idx_terms[i][exprs[i].term_names.index(f"{direction}.{q}.start")]
                add_row({t_idx[key][i]: 1.0, start_var: -1.0}, margin_s, np.inf)

            for e in range(m):
                for i in (e, e + 1):
                    q = global_window_no(direction, band_no, i)
                    end_var = idx_terms[i][exprs[i].term_names.index(f"{direction}.{q}.end")]
                    add_row(
                        {t_idx[key][i]: 1.0, width_idx[key][e]: 1.0, end_var: -1.0},
                        -np.inf,
                        -margin_s,
                    )
                add_row({global_idx[key]: 1.0, width_idx[key][e]: -1.0}, -np.inf, 0.0)

        # 固定 Stage 1 的顺序，不再引入 big-M 窗口选择。
        for direction in ("up", "down"):
            nums = active_by_direction[direction]
            for a in range(len(nums)):
                for b in range(a + 1, len(nums)):
                    r, s = nums[a], nums[b]
                    for i in range(n):
                        order = global_order_value(direction, r, s, i)
                        for e in incident_edge_indices(i):
                            if order == 1:
                                add_row(
                                    {t_idx[(direction, r)][i]: 1.0,
                                     width_idx[(direction, r)][e]: 1.0,
                                     t_idx[(direction, s)][i]: -1.0},
                                    -np.inf,
                                    -self.band_gap,
                                )
                            else:
                                add_row(
                                    {t_idx[(direction, s)][i]: 1.0,
                                     width_idx[(direction, s)][e]: 1.0,
                                     t_idx[(direction, r)][i]: -1.0},
                                    -np.inf,
                                    -self.band_gap,
                                )

        # ---------------- 局部/segment band：固定 Stage 1 窗口，只调端点 ----------------
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            key_text = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
            margin_s = hard_margin_up if direction == "up" else hard_margin_down
            local_segs = segs[start:start + k - 1]
            for band_no in band_numbers:
                local_key = (direction, band_no, k, start)
                t_vars = local_t_idx[local_key]
                loop_vars = local_loop_idx[local_key]
                band_var = lattice_idx[local_key]
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
                    q = local_window_no(direction, band_no, key_text, abs_idx)
                    start_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(f"{direction}.{q}.start")]
                    end_var = idx_terms[abs_idx][exprs[abs_idx].term_names.index(f"{direction}.{q}.end")]
                    add_row({t_vars[offset]: 1.0, start_var: -1.0}, margin_s, np.inf)
                    add_row({t_vars[offset]: 1.0, band_var: 1.0, end_var: -1.0}, -np.inf, -margin_s)

            for a in range(len(band_numbers)):
                for b in range(a + 1, len(band_numbers)):
                    r, s = band_numbers[a], band_numbers[b]
                    for offset, abs_idx in enumerate(range(start, start + k)):
                        order = local_order_value(key_text, r, s, abs_idx)
                        t_r = local_t_idx[(direction, r, k, start)][offset]
                        t_s = local_t_idx[(direction, s, k, start)][offset]
                        b_r = lattice_idx[(direction, r, k, start)]
                        b_s = lattice_idx[(direction, s, k, start)]
                        if order == 1:
                            add_row({t_r: 1.0, b_r: 1.0, t_s: -1.0}, -np.inf, -self.band_gap)
                        else:
                            add_row({t_s: 1.0, b_s: 1.0, t_r: -1.0}, -np.inf, -self.band_gap)

        for record in margin_records:
            add_row({**record["terms"], record["slack"]: 1.0}, record["rhs"], np.inf)

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

        # 输出时保留 Stage 1 的窗口/顺序选择；若 prior 缺失则按固定回退选择重建。
        sol.band_window_choices = {"up": {}, "down": {}}
        sol.band_order_choices = {"up": {}, "down": {}}
        for direction, band_no in band_instances:
            sol.band_window_choices[direction][band_no] = {}
            for i in range(n):
                q = global_window_no(direction, band_no, i)
                sol.band_window_choices[direction][band_no][int_names[i]] = {
                    "plan": selected[i].name,
                    "window": q,
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
                        order = global_order_value(direction, r, s, i)
                        sol.band_order_choices[direction][r][s][int_names[i]] = order

        sol.local_band_window_choices = {}
        sol.local_band_order_choices = {}
        local_records = []
        for (direction, k, start), band_numbers in local_bands_by_spec.items():
            key_text = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
            sol.local_band_window_choices.setdefault(key_text, {})
            sol.local_band_order_choices.setdefault(key_text, {})
            for band_no in band_numbers:
                choices: dict[str, dict[str, object]] = {}
                raw_times: list[float] = []
                for offset, abs_idx in enumerate(range(start, start + k)):
                    q = local_window_no(direction, band_no, key_text, abs_idx)
                    choices[int_names[abs_idx]] = {"plan": selected[abs_idx].name, "window": q}
                    raw_times.append(float(x[local_t_idx[(direction, band_no, k, start)][offset]]))
                sol.local_band_window_choices[key_text][band_no] = choices
                local_records.append({
                    "direction": direction,
                    "band_no": band_no,
                    "key": key_text,
                    "start": start,
                    "k": k,
                    "bandwidth": float(x[lattice_idx[(direction, band_no, k, start)]]),
                    "times": raw_times,
                    "window_choices": choices,
                })
            for a in range(len(band_numbers)):
                for b in range(a + 1, len(band_numbers)):
                    r, s = band_numbers[a], band_numbers[b]
                    sol.local_band_order_choices[key_text].setdefault(r, {})[s] = {}
                    for offset, abs_idx in enumerate(range(start, start + k)):
                        order = local_order_value(key_text, r, s, abs_idx)
                        sol.local_band_order_choices[key_text][r][s][int_names[abs_idx]] = order

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

        if active_by_direction["up"]:
            first_band = active_by_direction["up"][0]
            sol.band_start_up = {
                int_names[i]: float(x[t_idx[("up", first_band)][i]])
                for i in range(n)
            }
        if active_by_direction["down"]:
            first_band = active_by_direction["down"][0]
            sol.band_start_down = {
                int_names[i]: float(x[t_idx[("down", first_band)][i]])
                for i in range(n)
            }

        fill_solution_local_band_records(sol, arterial, local_records, clear=True)
        fill_solution_missing_window_bands(
            sol, arterial, max_window=5, max_loops=effective_max_loops, margin=self.margin
        )

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

        for record in margin_records:
            expr_value = sum(coef * float(x[var]) for var, coef in record["terms"].items())
            violation = max(0.0, float(record["rhs"]) - expr_value)
            band_loss += float(record["penalty"]) * violation

        band_objective = 0.0
        for group in self.config.sum_groups:
            for key, weight in group.terms.items():
                if weight == 0.0:
                    continue
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
