"""窗口带后处理。

本模块只保留：

- `parse_window_key` / `unwrap_band_times`：窗口 key 解析与时刻解包；
- `fill_solution_local_window_band_data`：对每个子走廊独立求最大可行带宽；
- `fill_solution_window_band_ranges`：根据当前解回填窗口带时间范围；
- `fill_solution_multi_window_bands`：多段窗口带宽回填入口。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp



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


def _selected_segment_window_sets(solution, arterial, margin=None):
    """还原每个路口、每个方向的全部段级绿灯窗（秒）。

    margin 非空时，返回的是扣掉硬边距后的有效窗口。
    """
    C = arterial.cycle
    margin_up = float(margin.hard_margin_up) * C if margin is not None else 0.0
    margin_down = float(margin.hard_margin_down) * C if margin is not None else 0.0
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
                    start_s = float(st[s_key]) + margin_up
                    end_s = float(st[e_key]) - margin_up
                    if end_s > start_s:
                        up_windows.append((start_s, end_s))
            for seg_idx in range(1, len(plan.down_windows) + 1):
                s_key = f"down.{seg_idx}.start"
                e_key = f"down.{seg_idx}.end"
                if s_key in st and e_key in st:
                    start_s = float(st[s_key]) + margin_down
                    end_s = float(st[e_key]) - margin_down
                    if end_s > start_s:
                        down_windows.append((start_s, end_s))
        else:
            up_windows = [
                (w.start * C + margin_up, w.end * C - margin_up)
                for w in plan.up_windows
                if w.end * C - margin_up > w.start * C + margin_up
            ]
            down_windows = [
                (w.start * C + margin_down, w.end * C - margin_down)
                for w in plan.down_windows
                if w.end * C - margin_down > w.start * C + margin_down
            ]

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
                                         clear: bool = True,
                                         margin=None) -> None:
    """按“独立局部带”语义回填所有窗口带宽和时间范围。"""
    int_names = [v.name for v in arterial.intersection_order]
    n = len(int_names)
    segs = arterial.segment_order
    cycle = arterial.cycle
    window_sets = _selected_segment_window_sets(solution, arterial, margin=margin)

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
                                     max_loops: int = 3,
                                     margin=None) -> None:
    """按独立局部带语义回填多段窗口带宽。"""
    fill_solution_local_window_band_data(
        solution,
        arterial,
        max_window=max_window,
        max_loops=max_loops,
        clear=clear,
        margin=margin,
    )


__all__ = [
    "fill_solution_local_window_band_data",
    "fill_solution_window_band_ranges",
    "fill_solution_multi_window_bands",
    "parse_window_key",
    "unwrap_band_times",
]
