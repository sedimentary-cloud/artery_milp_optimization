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


def _solve_free_window_bands(direction: str,
                            start: int,
                            k: int,
                            int_names: list[str],
                            window_sets,
                            segs,
                            cycle: float,
                            max_loops: int,
                            n_bands: int = 3,
                            band_gap: float = 0.0) -> list[tuple[int, float, list[float], dict[str, int]]]:
    """独立求解同一个局部子走廊上的多条 band，允许每个路口自由选窗口。

    返回 ``[(band_no, bandwidth, t_series, window_choices), ...]``。
    该函数只用于后处理和画图，不参与主 MILP 目标。
    """
    if n_bands <= 0:
        return []

    local_names = int_names[start:start + k]
    if len(local_names) != k:
        return []

    options_by_offset: list[list[tuple[int, tuple[float, float]]]] = []
    for name in local_names:
        windows = [tuple(float(v) for v in w) for w in window_sets.get(name, {}).get(direction, [])]
        if not windows:
            return []
        options_by_offset.append([(q, w) for q, w in enumerate(windows, start=1)])

    local_segs = segs[start:start + k - 1]
    if len(local_segs) != k - 1:
        return []

    r_count = int(n_bands)
    n_t = r_count * k
    n_m = r_count * (k - 1)
    n_b = r_count
    cur = n_t + n_m + n_b

    y_idx: dict[tuple[int, int, int], int] = {}
    for r in range(r_count):
        for offset, opts in enumerate(options_by_offset):
            for q, _window in opts:
                y_idx[(r, offset, q)] = cur
                cur += 1

    pi_idx: dict[tuple[int, int, int], int] = {}
    for r in range(r_count):
        for s_local in range(r + 1, r_count):
            for offset in range(k):
                pi_idx[(r, s_local, offset)] = cur
                cur += 1

    nvar = cur
    c = np.zeros(nvar)
    for r in range(r_count):
        c[n_t + n_m + r] = -1.0

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)

    def t_var(r: int, offset: int) -> int:
        return r * k + offset

    def m_var(r: int, edge: int) -> int:
        return n_t + r * (k - 1) + edge

    def b_var(r: int) -> int:
        return n_t + n_m + r

    for r in range(r_count):
        for offset in range(k):
            ub[t_var(r, offset)] = cycle
        for edge in range(k - 1):
            lb[m_var(r, edge)] = -max_loops
            ub[m_var(r, edge)] = max_loops
        ub[b_var(r)] = cycle
    for j in y_idx.values():
        ub[j] = 1.0
    for j in pi_idx.values():
        ub[j] = 1.0

    integrality = np.zeros(nvar)
    for r in range(r_count):
        for edge in range(k - 1):
            integrality[m_var(r, edge)] = 1
    for j in y_idx.values():
        integrality[j] = 1
    for j in pi_idx.values():
        integrality[j] = 1

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

    for r in range(r_count):
        for edge_idx, seg in enumerate(local_segs):
            if direction == "up":
                add_row(
                    {
                        t_var(r, edge_idx + 1): 1.0,
                        t_var(r, edge_idx): -1.0,
                        m_var(r, edge_idx): -cycle,
                    },
                    seg.travel_time_up,
                    seg.travel_time_up,
                )
            elif direction == "down":
                add_row(
                    {
                        t_var(r, edge_idx): 1.0,
                        t_var(r, edge_idx + 1): -1.0,
                        m_var(r, edge_idx): -cycle,
                    },
                    seg.travel_time_down,
                    seg.travel_time_down,
                )
            else:
                raise ValueError(f"unknown direction: {direction}")

        for offset, opts in enumerate(options_by_offset):
            add_row(
                {y_idx[(r, offset, q)]: 1.0 for q, _window in opts},
                1.0,
                1.0,
            )
            start_terms = {
                y_idx[(r, offset, q)]: float(window[0])
                for q, window in opts
            }
            end_terms = {
                y_idx[(r, offset, q)]: -float(window[1])
                for q, window in opts
            }
            add_row({t_var(r, offset): -1.0, **start_terms}, -np.inf, 0.0)
            add_row(
                {t_var(r, offset): 1.0, b_var(r): 1.0, **end_terms},
                -np.inf,
                0.0,
            )

    order_m = 2.0 * cycle
    for r in range(r_count):
        for s_local in range(r + 1, r_count):
            for offset in range(k):
                pi = pi_idx[(r, s_local, offset)]
                add_row(
                    {
                        t_var(r, offset): 1.0,
                        b_var(r): 1.0,
                        t_var(s_local, offset): -1.0,
                        pi: order_m,
                    },
                    -np.inf,
                    order_m - band_gap,
                )
                add_row(
                    {
                        t_var(s_local, offset): 1.0,
                        b_var(s_local): 1.0,
                        t_var(r, offset): -1.0,
                        pi: -order_m,
                    },
                    -np.inf,
                    -band_gap,
                )

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
        return []

    out: list[tuple[int, float, list[float], dict[str, int]]] = []
    for r in range(r_count):
        raw_times = [float(result.x[t_var(r, offset)]) for offset in range(k)]
        t_series = unwrap_band_times(direction, raw_times, local_segs, cycle)
        choices: dict[str, int] = {}
        for offset, opts in enumerate(options_by_offset):
            q_best = max(
                opts,
                key=lambda item: float(result.x[y_idx[(r, offset, item[0])]]),
            )[0]
            choices[local_names[offset]] = int(q_best)
        out.append((r + 1, float(result.x[b_var(r)]), t_series, choices))
    return out


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


def fill_solution_local_band_records(solution,
                                     arterial,
                                     records: list[dict[str, object]],
                                     clear: bool = True) -> None:
    """根据主 MILP 的局部带变量直接回填窗口带结果。

    ``records`` 中每个元素包含：

    - direction: "up" / "down"
    - band_no: 局部 band 编号
    - key: 形如 "down.win3@I2-I4" 的窗口 key
    - start / k: 子走廊起点下标和路口数量
    - bandwidth: 该局部 band 的带宽（秒）
    - times: 子走廊各路口原始（mod cycle）到达时刻
    - window_choices: 路口名 -> {"plan": ..., "window": ...}
    """
    int_names = [v.name for v in arterial.intersection_order]
    segs = arterial.segment_order
    cycle = arterial.cycle

    if clear:
        solution.multi_window_bands = {"up": {}, "down": {}}
        solution.window_bands.clear()
        solution.window_band_ranges.clear()

    aggregated: dict[str, float] = {}
    for record in records:
        direction = str(record["direction"])
        band_no = int(record["band_no"])
        key = str(record["key"])
        start = int(record["start"])
        k = int(record["k"])
        bandwidth = float(record["bandwidth"])
        raw_times = [float(v) for v in record["times"]]  # type: ignore[arg-type]
        local_segs = segs[start:start + k - 1]
        t_series = unwrap_band_times(direction, raw_times, local_segs, cycle)
        used_names = int_names[start:start + k]

        solution.multi_window_bands.setdefault(direction, {}).setdefault(band_no, {})[key] = bandwidth
        aggregated[key] = aggregated.get(key, 0.0) + bandwidth

        if bandwidth <= 0 or len(used_names) != k:
            continue

        time_values: list[float] = []
        intersection_ranges: dict[str, dict[str, float]] = {}
        for idx, iname in enumerate(used_names):
            front = float(t_series[idx])
            tail = front + bandwidth
            intersection_ranges[iname] = {"start": front, "end": tail}
            time_values.extend([front, tail])

        entry = {
            "direction": direction,
            "band_no": band_no,
            "segment_no": None,
            "bandwidth": bandwidth,
            "intersections": used_names,
            "time_min": min(time_values),
            "time_max": max(time_values),
            "intersection_ranges": intersection_ranges,
            "window_choices": dict(record.get("window_choices", {})),
        }
        solution.window_band_ranges.setdefault(key, []).append(entry)

    if aggregated:
        solution.window_bands.update(aggregated)


def fill_solution_missing_window_bands(solution,
                                       arterial,
                                       max_window: int = 5,
                                       max_loops: int = 3,
                                       margin=None) -> None:
    """为所有方向、所有窗口长度补齐局部窗口带。

    主 MILP 只对 ObjectiveConfig 中显式出现的局部带建模；本函数使用
    当前选中的方案/端点，对尚未填充的 ``(direction, k, start)`` 组合做
    独立后处理，使 ``window_bands`` / ``window_band_ranges`` /
    ``multi_window_bands`` 像旧版一样包含完整的方向-长度集合。

    已有 key（通常是主 MILP 的自由窗口结果）不会被覆盖。
    """
    int_names = [v.name for v in arterial.intersection_order]
    n = len(int_names)
    segs = arterial.segment_order
    cycle = arterial.cycle
    window_sets = _selected_segment_window_sets(solution, arterial, margin=margin)

    existing_keys = set(solution.window_band_ranges.keys())
    added: dict[str, float] = {}

    # 从主解里推断每个方向建模了多少条 band；
    # 局部窗口最多补 2 条，避免对称性和变量数量过大。
    band_count = 1
    for direction_map in solution.multi_bandwidths.values():
        if direction_map:
            band_count = max(band_count, max(int(k) for k in direction_map.keys()))
    if band_count <= 0:
        band_count = 2
    local_band_count = min(2, band_count)

    for direction in ("up", "down"):
        for k in range(2, min(max_window, n) + 1):
            for start in range(0, n - k + 1):
                key = f"{direction}.win{k}@{int_names[start]}-{int_names[start + k - 1]}"
                if key in existing_keys:
                    continue

                # 后处理局部带也使用“自由窗口分配 + 多带顺序”：
                # 每个路口可以从当前方案的所有绿灯窗口里任选一个，
                # 并生成多条互不重叠的局部 band。
                solved_list = _solve_free_window_bands(
                    direction=direction,
                    start=start,
                    k=k,
                    int_names=int_names,
                    window_sets=window_sets,
                    segs=segs,
                    cycle=cycle,
                    max_loops=max_loops,
                    n_bands=local_band_count,
                )
                for band_no, bandwidth, t_series, window_choices in solved_list:
                    solution.multi_window_bands.setdefault(direction, {}).setdefault(
                        band_no, {}
                    )[key] = float(bandwidth)
                    added[key] = added.get(key, 0.0) + float(bandwidth)

                    if bandwidth <= 0:
                        continue

                    used_names = int_names[start:start + k]
                    entry = {
                        "direction": direction,
                        "band_no": band_no,
                        # 这里用 band_no 作为 segment_no 兼容键，
                        # 让 plot_time_space 在全局带宽为 0 时可以把这条
                        # 全走廊局部带回退绘制成 Global Band 样式。
                        "segment_no": band_no,
                        "bandwidth": float(bandwidth),
                        "intersections": used_names,
                        "time_min": min(t_series) if t_series else 0.0,
                        "time_max": max((t + bandwidth) for t in t_series) if t_series else 0.0,
                        "intersection_ranges": {
                            name: {
                                "start": float(t_series[idx]),
                                "end": float(t_series[idx] + bandwidth),
                            }
                            for idx, name in enumerate(used_names)
                        },
                        "window_choices": {
                            name: {
                                "plan": solution.plan_choices.get(name, ""),
                                "window": int(window_choices[name]),
                            }
                            for name in used_names
                        },
                    }
                    solution.window_band_ranges.setdefault(key, []).append(entry)

    for key, bandwidth in added.items():
        solution.window_bands[key] = solution.window_bands.get(key, 0.0) + float(bandwidth)


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
    "fill_solution_local_band_records",
    "fill_solution_missing_window_bands",
    "fill_solution_local_window_band_data",
    "fill_solution_window_band_ranges",
    "fill_solution_multi_window_bands",
    "parse_window_key",
    "unwrap_band_times",
]
