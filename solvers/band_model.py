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

from .objective_config import BandKey, ObjectiveConfig


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


def _widest_window(windows):
    """返回最宽绿灯窗；没有则返回 None。"""
    if not windows:
        return None
    return max(windows, key=lambda w: w.width)


def _selected_green_windows(solution, arterial):
    """从 Solution 中还原每个路口最终采用的上/下行绿灯窗（秒）。

    优先级：
        1. phase_times + phases：第二阶段相位优化后的窗口；
        2. window_choices：第一阶段选中的方案窗口；
        3. 方案里最宽的窗口。

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

        # 1) 相位优化后的窗口。
        # phase_start_times 会把 phase_lost_times 作为常数间隔计入。
        pt = solution.phase_times.get(inter.name) if solution and solution.phase_times else None
        if pt and plan.phases:
            from .phase import direction_phase_name, phase_start_times
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

        # 2) 方案/窗口选择信息。
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

        # 3) 退回方案里最宽的窗口。
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


def fill_solution_window_bands(solution,
                               arterial,
                               max_window: int = 5,
                               clear: bool = True) -> None:
    """从最终 Solution 重新计算并回填窗口绿波带。

    该函数只做后处理，不修改任何优化变量、目标值或带宽结果。

    与“直接取最终 bandwidth 的最小值”不同，这里利用最终解中的：
        - band_start_up / band_start_down
        - 最终采用的绿灯窗
    计算在给定带前沿 t 下，窗口内各路口还剩余多少绿灯时间：

        room_i = max(0, green_end_i - t_i)
        B_feasible[d,k,j] = min(room_i)  i = j ... j+k-1

    这样即使 objective="loss" 没有优化带宽，只要最终 t 落在绿灯窗内，
    仍然能回填出局部两两路口、三路口等“可行窗口绿波带”。

    Args:
        solution: 已求解的 Solution。
        arterial: 对应干线。
        max_window: 最大窗口大小，默认 5。
        clear: 是否先清空 solution.window_bands，默认 True。
    """
    int_names = [v.name for v in arterial.intersection_order]
    n = len(int_names)
    if clear:
        solution.window_bands.clear()

    windows = _selected_green_windows(solution, arterial)

    for direction, starts in (("up", solution.band_start_up),
                              ("down", solution.band_start_down)):
        if not starts:
            continue

        # k = 2..min(max_window, n)
        for k in range(2, min(max_window, n) + 1):
            for j in range(0, n - k + 1):
                rooms: list[float] = []
                for idx in range(j, j + k):
                    iname = int_names[idx]
                    t = starts.get(iname)
                    start_s, end_s = windows.get(iname, {}).get(
                        direction, (0.0, 0.0)
                    )
                    if t is None:
                        rooms = []
                        break
                    t = float(t)
                    # t 必须落在当前方向的绿灯窗内；若不在，则该窗口带不可行。
                    if t < start_s - 1e-7 or t > end_s + 1e-7:
                        room = 0.0
                    else:
                        room = max(0.0, end_s - t)
                    rooms.append(room)

                if not rooms:
                    continue
                bw = min(rooms)
                key = (f"{direction}.win{k}@"
                       f"{int_names[j]}-{int_names[j + k - 1]}")
                solution.window_bands[key] = float(bw)


__all__ = ["BandModel", "fill_solution_window_bands"]
