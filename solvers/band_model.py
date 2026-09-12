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


def fill_solution_window_bands(solution,
                               arterial,
                               max_window: int = 5,
                               clear: bool = True) -> None:
    """从最终 Solution 的逐路段带宽重新计算并回填窗口绿波带。

    该函数只做后处理，不修改任何优化变量、目标值或带宽结果。

    计算方式：
        对每个方向 d、窗口大小 k、起点 j：
            B[d,k,j] = min(
                sol.bandwidth_up/down[seg_j],
                ...,
                sol.bandwidth_up/down[seg_{j+k-2}],
            )

    Args:
        solution: 已求解的 Solution。
        arterial: 对应干线，用于拿路口名和路段名。
        max_window: 最大窗口大小，默认 5。
        clear: 是否先清空 solution.window_bands，默认 True。
    """
    int_names = [v.name for v in arterial.intersection_order]
    seg_names = [s.name for s in arterial.segment_order]
    n = len(int_names)

    if clear:
        solution.window_bands.clear()

    for direction, bandwidth in (("up", solution.bandwidth_up),
                                 ("down", solution.bandwidth_down)):
        if not bandwidth:
            continue

        # k = 2..min(max_window, n)
        for k in range(2, min(max_window, n) + 1):
            for j in range(0, n - k + 1):
                seg_slice = seg_names[j:j + k - 1]
                values = [float(bandwidth.get(name, 0.0))
                          for name in seg_slice]
                if not values:
                    continue
                bw = min(values)
                key = (f"{direction}.win{k}@"
                       f"{int_names[j]}-{int_names[j + k - 1]}")
                solution.window_bands[key] = float(bw)


__all__ = ["BandModel", "fill_solution_window_bands"]
