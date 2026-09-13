"""双向窗口带格 BandModel。

数学定义
--------
对每个方向 d ∈ {up, down}，每个路段 i：
    b[d, i] = 方向 d 第 i 段的基础带宽（秒）

对窗口大小 k（2 <= k <= n）和起点路口 j：
    B[d, k, j] = 方向 d、覆盖路口 j..j+k-1 的窗口带宽（秒）

核心约束
--------
    B[d, k, j] <= b[d, i]      for all i = j ... j+k-1

含义：窗口带的宽度不能超过窗口内任意一个路段的带宽。

特例
----
    k = 2  -> 分段带；
    k = n  -> 全局带；
    其他 k -> 小绿波带。

变量规模
--------
每个方向：
    Σ_{k=2}^{n} (n - k + 1) = O(n²)
两个方向合计 O(2 n²)。
10 个路口时，每方向 45 个窗口变量，全部为连续变量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .objective import BandKey, ObjectiveConfig


@dataclass
class BandModel:
    """带格变量索引表。

    b_idx[direction][segment_index]  -> 基础段变量下标
    B_idx[direction][k][start_index] -> 窗口带变量下标
    """

    n: int
    m: int
    directions: tuple[str, ...] = ("up", "down")

    def __post_init__(self) -> None:
        self.b_idx: dict[str, list[int]] = {d: [] for d in self.directions}
        self.B_idx: dict[str, dict[int, list[int]]] = {d: {} for d in self.directions}

        cur = 0
        for d in self.directions:
            self.b_idx[d] = list(range(cur, cur + self.m))
            cur += self.m

        for d in self.directions:
            self.B_idx[d] = {}
            for k in range(2, self.n + 1):
                self.B_idx[d][k] = list(range(cur, cur + (self.n - k + 1)))
                cur += self.n - k + 1

        self.nvar = cur

    def var_of(self, key: BandKey) -> int:
        """按 BandKey 返回变量下标。"""
        return self.B_idx[key.direction][key.k][key.start]

    def base_var(self, direction: str, segment_index: int) -> int:
        return self.b_idx[direction][segment_index]

    def add_lattice_constraints(self, add_row) -> None:
        """添加 B[d,k,j] <= b[d,i] 约束。"""
        for d in self.directions:
            for k in range(2, self.n + 1):
                for j in range(self.n - k + 1):
                    B_var = self.B_idx[d][k][j]
                    for i in range(j, j + k - 1):
                        add_row({B_var: 1.0, self.b_idx[d][i]: -1.0},
                                -float("inf"), 0.0)

    def referenced_keys(self, config: ObjectiveConfig) -> list[BandKey]:
        return sorted(config.all_band_keys(self.n), key=lambda b: (b.direction, b.k, b.start))

    def add_sum_group(self, c, group) -> None:
        """把 SumGroup 写入目标系数 c（max 问题；调用方需处理 min 负号）。"""
        for key, weight in group.terms.items():
            band = BandKey(
                direction=key.split(".", 1)[0],
                k=_parse_k_from_key(key, self.n),
                start=_parse_start_from_key(key, self.n),
            )
            var = self.var_of(band)
            c[var] += weight

    def add_balance_group(self, c, group, add_row, add_var) -> None:
        """创建组内 min 变量 B_g，写入 c 和约束。

        add_var(): 返回新变量下标；
        add_row(coefs, lo, hi): 添加一行约束。
        """
        var_g = add_var()
        c[var_g] += group.weight
        for member in group.members:
            band = _parse_band_key_from_str(member, self.n)
            member_var = self.var_of(band)
            add_row({var_g: 1.0, member_var: -1.0}, -float("inf"), 0.0)
        # ε 托底项
        if group.eps > 0:
            for member in group.members:
                band = _parse_band_key_from_str(member, self.n)
                c[self.var_of(band)] += group.weight * group.eps


def _parse_k_from_key(text: str, n: int) -> int:
    if text.endswith(".global"):
        return n
    if ".seg" in text:
        return 2
    if ".win" in text:
        return int(text.split(".win", 1)[1].split("@", 1)[0])
    raise ValueError(f"无法解析带标识: {text}")


def _parse_start_from_key(text: str, n: int) -> int:
    if text.endswith(".global"):
        return 0
    if ".seg" in text:
        seg_idx = int(text.split(".seg", 1)[1]) - 1
        return seg_idx
    if ".win" in text:
        rng = text.split("@", 1)[1]
        first = rng.split("-")[0]
        if first.startswith("I"):
            return int(first[1:]) - 1
        return int(first) - 1
    raise ValueError(f"无法解析带标识: {text}")


def _parse_band_key_from_str(text: str, n: int) -> BandKey:
    return BandKey(
        direction=text.split(".", 1)[0],
        k=_parse_k_from_key(text, n),
        start=_parse_start_from_key(text, n),
    )


def parse_window_key(key: str, names: list[str]) -> tuple[str | None, int | None, int | None]:
    """解析窗口带 key。

    新格式：
        "up.win3@I1-I3"   -> ("up", 3, 0)
        "down.win2@I1-I2" -> ("down", 2, 0)

    兼容旧格式：
        "win3@I1-I3"      -> ("down", 3, 0)
    """
    try:
        prefix, rng = key.split("@")
        if "." in prefix:
            direction, k_str = prefix.split(".", 1)
        else:
            direction, k_str = "down", prefix
        if direction not in ("up", "down"):
            return None, None, None
        k = int(k_str.replace("win", ""))
        first = rng.split("-")[0]
        return direction, k, names.index(first)
    except (ValueError, IndexError):
        return None, None, None


def unwrap_band_times(direction: str,
                      raw_times: list[float],
                      segs,
                      cycle: float) -> list[float]:
    """按传播方向把模周期时刻解包为绝对时刻序列。"""
    if not raw_times:
        return []

    series = [float(raw_times[0])]
    for i, seg in enumerate(segs):
        raw_next = float(raw_times[i + 1])
        if direction == "up":
            target = series[-1] + seg.travel_time_up
        elif direction == "down":
            target = series[-1] - seg.travel_time_down
        else:
            raise ValueError(f"unknown direction: {direction}")
        shift = round((target - raw_next) / cycle)
        series.append(raw_next + shift * cycle)
    return series


def _make_window_band_range_entry(key: str,
                                  names: list[str],
                                  t_series: list[float],
                                  bandwidth: float,
                                  segment_no: int | None = None) -> dict[str, object] | None:
    """构造单个窗口带实例的时间范围条目。"""
    direction, k, start = parse_window_key(key, names)
    if direction is None or k is None or start is None:
        return None
    if bandwidth <= 0:
        return None

    intersection_ranges: dict[str, dict[str, float]] = {}
    used_names = names[start:start + k]
    if len(used_names) != k:
        return None
    time_values: list[float] = []
    for idx, iname in enumerate(used_names, start=start):
        front = float(t_series[idx])
        tail = front + float(bandwidth)
        intersection_ranges[iname] = {
            "start": front,
            "end": tail,
        }
        time_values.extend([front, tail])

    return {
        "direction": direction,
        "segment_no": segment_no,
        "bandwidth": float(bandwidth),
        "intersections": used_names,
        "time_min": min(time_values),
        "time_max": max(time_values),
        "intersection_ranges": intersection_ranges,
    }


def _widest_window(windows):
    """返回最宽绿灯窗；没有则返回 None。"""
    if not windows:
        return None
    return max(windows, key=lambda w: w.width)


def _selected_green_windows(solution, arterial):
    """从 Solution 中还原每个路口最终采用的上/下行绿灯窗（秒）。

    优先级：
        1. segment_times：第二阶段段级优化后的窗口；
        2. phase_times + phases：旧相位路径回退；
        3. window_choices：第一阶段选中的方案窗口；
        4. 方案里最宽的窗口。

    Returns:
        dict[路口名, {"up": (start_s, end_s), "down": (start_s, end_s)}]
    """
    C = arterial.cycle
    out: dict[str, dict[str, tuple[float, float]]] = {}

    for inter in arterial.intersection_order:
        # 先锁定方案：优先用 Solution.plan_choices，否则退回第一个方案。
        plan = inter.plans[0]
        if solution is not None and solution.plan_choices:
            pname = solution.plan_choices.get(inter.name)
            if pname:
                try:
                    plan = inter.plan_by_name(pname)
                except KeyError:
                    plan = inter.plans[0]

        up_win = None
        down_win = None

        # 1) 段级优化后的窗口。
        st = solution.segment_times.get(inter.name) if solution and solution.segment_times else None
        if st:
            up_idx = None
            dn_idx = None
            wc = (solution.window_choices.get(inter.name)
                  if solution and solution.window_choices else None)
            try:
                up_idx = int(wc.get("up_window", -1)) + 1 if wc else None
                dn_idx = int(wc.get("down_window", -1)) + 1 if wc else None
            except (TypeError, ValueError):
                up_idx = None
                dn_idx = None
            if up_idx is not None:
                s_key = f"up.{up_idx}.start"
                e_key = f"up.{up_idx}.end"
                if s_key in st and e_key in st:
                    up_win = (float(st[s_key]), float(st[e_key]))
            if dn_idx is not None:
                s_key = f"down.{dn_idx}.start"
                e_key = f"down.{dn_idx}.end"
                if s_key in st and e_key in st:
                    down_win = (float(st[s_key]), float(st[e_key]))

        # 2) 相位优化后的窗口。
        # phase_start_times 会把 phase_lost_times 作为常数间隔计入。
        pt = solution.phase_times.get(inter.name) if solution and solution.phase_times else None
        if pt and plan.phases:
            from ..builders.signal_constraints import (direction_phase_name,
                                                       phase_start_times)
            starts = phase_start_times(plan, pt)
            wc = (solution.window_choices.get(inter.name)
                  if solution and solution.window_choices else None)
            if wc and wc.get("up_phase"):
                up_name = wc["up_phase"]
            else:
                try:
                    up_name = direction_phase_name(plan, "up")
                except (ValueError, NotImplementedError):
                    up_name = plan.up_phase
            if wc and wc.get("down_phase"):
                down_name = wc["down_phase"]
            else:
                try:
                    down_name = direction_phase_name(plan, "down")
                except (ValueError, NotImplementedError):
                    down_name = plan.down_phase
            if up_name in starts and up_name in pt:
                us = starts[up_name]
                ue = us + float(pt[up_name])
                up_win = (us, ue)
            if down_name in starts and down_name in pt:
                ds = starts[down_name]
                de = ds + float(pt[down_name])
                down_win = (ds, de)

        # 3) 方案/窗口选择信息。
        if (up_win is None or down_win is None) and solution and solution.window_choices:
            wc = solution.window_choices.get(inter.name)
            if wc:
                try:
                    up_idx = int(wc.get("up_window", -1))
                    dn_idx = int(wc.get("down_window", -1))
                    if up_win is None and 0 <= up_idx < len(plan.up_windows):
                        w = plan.up_windows[up_idx]
                        up_win = (w.start * C, w.end * C)
                    if down_win is None and 0 <= dn_idx < len(plan.down_windows):
                        w = plan.down_windows[dn_idx]
                        down_win = (w.start * C, w.end * C)
                except (TypeError, ValueError, KeyError):
                    pass

        # 4) 退回方案里最宽的窗口。
        if up_win is None:
            w = _widest_window(plan.up_windows)
            if w is not None:
                up_win = (w.start * C, w.end * C)
        if down_win is None:
            w = _widest_window(plan.down_windows)
            if w is not None:
                down_win = (w.start * C, w.end * C)

        out[inter.name] = {
            "up": up_win if up_win is not None else (0.0, 0.0),
            "down": down_win if down_win is not None else (0.0, 0.0),
        }

    return out


def _selected_segment_window_sets(solution, arterial):
    """还原每个路口、每个方向的全部段级绿灯窗（秒）。"""
    C = arterial.cycle
    out: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for inter in arterial.intersection_order:
        plan = inter.plans[0]
        if solution is not None and solution.plan_choices:
            pname = solution.plan_choices.get(inter.name)
            if pname:
                try:
                    plan = inter.plan_by_name(pname)
                except KeyError:
                    plan = inter.plans[0]

        up_windows: list[tuple[float, float]] = []
        down_windows: list[tuple[float, float]] = []

        st = solution.segment_times.get(inter.name) if solution and solution.segment_times else None
        if st:
            for seg_idx in range(1, len(plan.up_windows) + 1):
                s_key = f"up.{seg_idx}.start"
                e_key = f"up.{seg_idx}.end"
                if s_key in st and e_key in st:
                    up_windows.append((float(st[s_key]), float(st[e_key])))
            for seg_idx in range(1, len(plan.down_windows) + 1):
                s_key = f"down.{seg_idx}.start"
                e_key = f"down.{seg_idx}.end"
                if s_key in st and e_key in st:
                    down_windows.append((float(st[s_key]), float(st[e_key])))
        elif solution is not None and solution.phase_times and inter.name in solution.phase_times and plan.phases:
            pt = solution.phase_times[inter.name]
            from ..builders.signal_constraints import direction_phase_name, phase_start_times
            starts = phase_start_times(plan, pt)
            try:
                up_name = direction_phase_name(plan, "up")
                if up_name in starts and up_name in pt:
                    us = starts[up_name]
                    up_windows.append((us, us + float(pt[up_name])))
            except (ValueError, NotImplementedError):
                pass
            try:
                down_name = direction_phase_name(plan, "down")
                if down_name in starts and down_name in pt:
                    ds = starts[down_name]
                    down_windows.append((ds, ds + float(pt[down_name])))
            except (ValueError, NotImplementedError):
                pass
        else:
            up_windows = [(w.start * C, w.end * C) for w in plan.up_windows]
            down_windows = [(w.start * C, w.end * C) for w in plan.down_windows]

        out[inter.name] = {
            "up": up_windows,
            "down": down_windows,
        }

    return out


def _available_segment_numbers(window_sets,
                               int_names: list[str],
                               direction: str,
                               start: int,
                               k: int) -> list[int]:
    """返回局部窗口内共同可用的段号列表。"""
    counts = [
        len(window_sets.get(int_names[idx], {}).get(direction, []))
        for idx in range(start, start + k)
    ]
    max_count = min(counts, default=0)
    return list(range(1, max_count + 1))


def _solve_independent_window_band(direction: str,
                                   segment_no: int,
                                   start: int,
                                   k: int,
                                   int_names: list[str],
                                   window_sets,
                                   segs,
                                   cycle: float,
                                   max_loops: int) -> tuple[float, list[float]] | None:
    """独立求解单个局部窗口带的最大可行带宽。"""
    local_names = int_names[start:start + k]
    if len(local_names) != k:
        return None

    local_windows: list[tuple[float, float]] = []
    for name in local_names:
        windows = window_sets.get(name, {}).get(direction, [])
        if not (1 <= segment_no <= len(windows)):
            return None
        local_windows.append(tuple(float(v) for v in windows[segment_no - 1]))

    local_segs = segs[start:start + k - 1]
    if len(local_segs) != k - 1:
        return None

    idx_t = list(range(k))
    idx_m = list(range(k, k + k - 1))
    idx_b = k + k - 1
    nvar = idx_b + 1

    c = np.zeros(nvar)
    c[idx_b] = -1.0

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[idx_t] = cycle
    if idx_m:
        lb[idx_m] = -max_loops
        ub[idx_m] = max_loops
    ub[idx_b] = cycle

    integrality = np.zeros(nvar)
    if idx_m:
        integrality[idx_m] = 1

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

    for edge_idx, seg in enumerate(local_segs):
        if direction == "up":
            add_row(
                {
                    idx_t[edge_idx + 1]: 1.0,
                    idx_t[edge_idx]: -1.0,
                    idx_m[edge_idx]: -cycle,
                },
                seg.travel_time_up,
                seg.travel_time_up,
            )
        elif direction == "down":
            add_row(
                {
                    idx_t[edge_idx]: 1.0,
                    idx_t[edge_idx + 1]: -1.0,
                    idx_m[edge_idx]: -cycle,
                },
                seg.travel_time_down,
                seg.travel_time_down,
            )
        else:
            raise ValueError(f"unknown direction: {direction}")

    for local_idx, (start_s, end_s) in enumerate(local_windows):
        add_row({idx_t[local_idx]: 1.0}, start_s, np.inf)
        add_row({idx_t[local_idx]: 1.0, idx_b: 1.0}, -np.inf, end_s)

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
    if result.x is None or not result.success:
        return None

    raw_times = [float(result.x[index]) for index in idx_t]
    t_series = unwrap_band_times(direction, raw_times, local_segs, cycle)
    return float(result.x[idx_b]), t_series


def fill_solution_local_window_band_data(solution,
                                         arterial,
                                         max_window: int = 5,
                                         max_loops: int = 3,
                                         clear: bool = True) -> None:
    """按“独立局部带”语义回填所有窗口带宽和时间范围。"""
    int_names = [v.name for v in arterial.intersection_order]
    n = len(int_names)
    segs = arterial.segment_order
    cycle = arterial.cycle
    window_sets = _selected_segment_window_sets(solution, arterial)

    if clear:
        solution.multi_window_bands = {"up": {}, "down": {}}
        solution.window_bands.clear()
        solution.window_band_ranges.clear()

    aggregated: dict[str, float] = {}

    for direction in ("up", "down"):
        direction_out: dict[int, dict[str, float]] = {}
        for k in range(2, min(max_window, n) + 1):
            for start in range(0, n - k + 1):
                for segment_no in _available_segment_numbers(window_sets, int_names, direction, start, k):
                    solved = _solve_independent_window_band(
                        direction=direction,
                        segment_no=segment_no,
                        start=start,
                        k=k,
                        int_names=int_names,
                        window_sets=window_sets,
                        segs=segs,
                        cycle=cycle,
                        max_loops=max_loops,
                    )
                    if solved is None:
                        continue
                    bandwidth, t_series = solved
                    key = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
                    direction_out.setdefault(segment_no, {})[key] = float(bandwidth)
                    aggregated[key] = aggregated.get(key, 0.0) + float(bandwidth)
                    entry = {
                        "direction": direction,
                        "segment_no": segment_no,
                        "bandwidth": float(bandwidth),
                        "intersections": int_names[start:start + k],
                        "time_min": min(t_series) if t_series else 0.0,
                        "time_max": max((t + bandwidth) for t in t_series) if t_series else 0.0,
                        "intersection_ranges": {
                            name: {
                                "start": float(t_series[idx]),
                                "end": float(t_series[idx] + bandwidth),
                            }
                            for idx, name in enumerate(int_names[start:start + k])
                        },
                    }
                    if bandwidth > 0:
                        solution.window_band_ranges.setdefault(key, []).append(entry)

        solution.multi_window_bands[direction] = direction_out

    if aggregated:
        solution.window_bands.update(aggregated)


def fill_solution_window_bands(solution,
                               arterial,
                               max_window: int = 5,
                               clear: bool = True,
                               max_loops: int = 3) -> None:
    """按独立局部带语义回填窗口绿波带。"""
    fill_solution_local_window_band_data(
        solution,
        arterial,
        max_window=max_window,
        max_loops=max_loops,
        clear=clear,
    )


def fill_solution_window_band_ranges(solution,
                                     arterial,
                                     clear: bool = True) -> None:
    """根据当前解回填局部绿波带的时间范围。"""
    if solution.window_band_ranges:
        return

    int_names = [v.name for v in arterial.intersection_order]
    segs = arterial.segment_order
    cycle = arterial.cycle
    if clear:
        solution.window_band_ranges.clear()

    if solution.multi_window_bands and solution.multi_band_starts:
        for direction in ("up", "down"):
            direction_windows = solution.multi_window_bands.get(direction, {})
            direction_starts = solution.multi_band_starts.get(direction, {})
            for segment_no, key_to_bw in direction_windows.items():
                start_map = direction_starts.get(segment_no)
                if not start_map:
                    continue
                raw_times = [float(start_map[name]) for name in int_names]
                t_series = unwrap_band_times(direction, raw_times, segs, cycle)
                for key, bw in key_to_bw.items():
                    entry = _make_window_band_range_entry(
                        key=key,
                        names=int_names,
                        t_series=t_series,
                        bandwidth=float(bw),
                        segment_no=segment_no,
                    )
                    if entry is not None:
                        solution.window_band_ranges.setdefault(key, []).append(entry)
        return

    if not solution.window_bands:
        return

    if not solution.band_start_up and not solution.band_start_down:
        return

    t_up: list[float] = []
    t_down: list[float] = []
    if solution.band_start_up:
        t_up = [float(solution.band_start_up[int_names[0]])]
        for seg in segs:
            t_up.append(t_up[-1] + seg.travel_time_up)
    if solution.band_start_down:
        t_down = [0.0] * len(int_names)
        t_down[-1] = float(solution.band_start_down[int_names[-1]])
        for i in range(len(segs) - 1, -1, -1):
            t_down[i] = t_down[i + 1] + segs[i].travel_time_down

    for key, bw in solution.window_bands.items():
        direction, _k, _start = parse_window_key(key, int_names)
        t_series = t_up if direction == "up" else t_down
        if not t_series:
            continue
        entry = _make_window_band_range_entry(
            key=key,
            names=int_names,
            t_series=t_series,
            bandwidth=float(bw),
            segment_no=None,
        )
        if entry is not None:
            solution.window_band_ranges.setdefault(key, []).append(entry)


def fill_solution_multi_window_bands(solution,
                                     arterial,
                                     max_window: int = 5,
                                     clear: bool = True,
                                     max_loops: int = 3) -> None:
    """按独立局部带语义回填多段窗口带宽。"""
    fill_solution_local_window_band_data(
        solution,
        arterial,
        max_window=max_window,
        max_loops=max_loops,
        clear=clear,
    )


__all__ = [
    "BandModel",
    "fill_solution_local_window_band_data",
    "fill_solution_window_bands",
    "fill_solution_window_band_ranges",
    "fill_solution_multi_window_bands",
    "parse_window_key",
    "unwrap_band_times",
]
