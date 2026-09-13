"""目标配置层：把“全局带 / 分段带 / 小绿波带”统一成可加权、可均衡的目标。

本文件只做“目标声明和校验”，不负责求解。求解器会读取配置并组装 MILP。

============================================================
一、带标识（BandKey）
============================================================
所有绿波带都用“方向 + 窗口大小 + 起点”来描述：

    BandKey(direction, k, start)

含义：
    direction: "up" 或 "down"；
    k:         窗口大小（连续多少个路口）；
               k = 2  -> 两两路口，也就是一个路段；
               k = n  -> 整条干线，也就是全局带；
               其他   -> 小绿波带。
    start:     窗口起点路口的 0-based 下标。

字符串写法：
    "up.global"        -> 上行全局带（k = n, start = 0）
    "down.seg2"        -> 下行第 2 段（k = 2, start = 1）
    "up.win3@I1-I3"    -> 上行从 I1 开始、连续 3 个路口（k = 3, start = 0）

当前解析规则：
    ".global" 表示 k = n；
    ".segX"   表示 k = 2，X 从 1 开始；
    ".winK@..." 表示 k = K，起点由 "I1" 这类路口名给出。

============================================================
二、目标的两类块
============================================================
一个 ObjectiveConfig 由若干 SumGroup 和若干 BalanceGroup 组成。

1. SumGroup：自由加权和
   ---------------------------------------------------------
   SumGroup(terms={"up.global": 1.0, "down.seg1": 0.5})

   数学含义：
       sum = 1.0 * B_up_global + 0.5 * B_down_seg1

   特点：
       - 任意方向、任意 k、任意窗口；
       - 每个 key 的权重可以自由设置；
       - 多个 SumGroup 也可以存在，但同一个带不要重复出现。

2. BalanceGroup：组内取 min
   ---------------------------------------------------------
   BalanceGroup(["up.global", "down.global"], weight=1.0, eps=0.01)

   数学含义：
       B_group <= "up.global"
       B_group <= "down.global"
       目标 += weight * B_group

   最大化 B_group 时，它自动等于两个成员里的较小值：
       B_group = min(B_up_global, B_down_global)

   ε 托底：
       目标 += weight * eps * (B_up_global + B_down_global)
   作用是：均衡达标后，继续把成员的“总量”往上榨，避免其他带摆烂。

   约束：
       - 组内成员的 k 必须相同；
         例如 up.global(k=n) 只能和 down.global(k=n) 比，
         不能拿 k=n 和 k=3 比，因为 k 大的天然更小。
       - 同一个带不能出现在多个 BalanceGroup；
       - 允许和 SumGroup 共存，用于 “sum + eps*min” 的复合目标。

============================================================
三、完整配置示例
============================================================
1) 双向加权和：
    ObjectiveConfig(sum_groups=[
        SumGroup({"up.global": 2.0, "down.global": 1.0})
    ])

2) 双向均衡：
    ObjectiveConfig(balance_groups=[
        BalanceGroup(["up.global", "down.global"], weight=1.0)
    ])

3) 复合目标：max b_up + b_down + eps * min(b_up, b_down)
    ObjectiveConfig(
        sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})],
        balance_groups=[
            BalanceGroup(["up.global", "down.global"], weight=0.1)
        ],
    )

4) 上行优先 + 下行分段/窗口：
    ObjectiveConfig(sum_groups=[
        SumGroup({
            "up.global": 1.0,
            "down.seg1": 1.0,
            "down.seg2": 1.0,
            "down.win3@I1-I3": 0.5,
        })
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BandKey:
    """窗口带格中的一个带标识。

    attributes:
        direction: "up" 或 "down"；
        k:         窗口大小（连续多少个路口）；2 表示分段带，n 表示全局带；
        start:     窗口起点路口的 0-based 下标。
    """
    direction: str       # "up" / "down"
    k: int               # 窗口大小：2 表示分段带，n 表示全局带
    start: int           # 起点路口下标，0-based

    def to_str(self) -> str:
        return f"{self.direction}.win{self.k}@{self.start}"


@dataclass
class SumGroup:
    """加权求和块：terms = {带标识: 权重}。

    示例：
        SumGroup({
            "up.global": 2.0,
            "down.global": 1.0,
            "down.seg1": 0.5,
        })

    数学含义：
        Σ weight * band

    特点：
        - 任意方向、任意 k、任意窗口；
        - 权重可以是任意正数（也可以用 0 表示不参与）。
    """
    terms: dict[str, float] = field(default_factory=dict)

    def parsed_terms(self):
        """返回 {BandKey: weight}，便于求解器直接使用。"""
        out = {}
        for key, w in self.terms.items():
            band = parse_band_key(key)
            out[band] = w
        return out


@dataclass
class BalanceGroup:
    """均衡块：组内取 min，组整体以 weight 进入目标。

    示例：
        BalanceGroup(["up.global", "down.global"], weight=1.0, eps=0.01)

    数学含义：
        B_group <= member    for each member
        目标 += weight * B_group
        目标 += weight * eps * Σ member

    attributes:
        members: 组内成员带标识列表；
        weight:  整个均衡组在目标里的权重；
        eps:     托底项权重，默认 0.01；设为 0 可关闭托底。

    约束：
        - 组内成员的 k 必须相同；
        - 同一个带不能出现在多个 BalanceGroup；
        - 允许与 SumGroup 共存。
    """
    members: list[str] = field(default_factory=list)
    weight: float = 1.0
    eps: float = 0.01  # 托底项：group_weight * eps * sum(members)


@dataclass
class ObjectiveConfig:
    """一个完整目标配置。

    attributes:
        sum_groups:     加权和块列表；
        balance_groups: 均衡块列表。

    数学形式：
        目标 = Σ SumGroup
             + Σ [ weight * B_group + weight * eps * Σ members ]
    """
    sum_groups: list[SumGroup] = field(default_factory=list)
    balance_groups: list[BalanceGroup] = field(default_factory=list)

    def validate(self, n_intersections: int) -> None:
        """检查配置是否合法。

        规则：
            1. SumGroup 里的同一个带不能重复；
            2. BalanceGroup 不能为空；
            3. BalanceGroup.weight 必须为正；
            4. BalanceGroup 成员的 k 必须相同；
            5. 同一个带不能被多个 BalanceGroup 引用；
            6. 允许带同时出现在 SumGroup 和 BalanceGroup，
               用于 “sum + eps*min” 复合目标，但语义由配置者负责。
        """
        n = n_intersections
        seen_in_sum: set[BandKey] = set()
        balance_member_to_group: dict[BandKey, BalanceGroup] = {}

        for g in self.sum_groups:
            for key, w in g.terms.items():
                band = parse_band_key(key, n)
                if band in seen_in_sum:
                    raise ValueError(f"带 {key} 在多个 SumGroup 中重复")
                seen_in_sum.add(band)

        for g in self.balance_groups:
            if not g.members:
                raise ValueError("BalanceGroup.members 不能为空")
            if g.weight <= 0:
                raise ValueError(f"BalanceGroup.weight 必须为正: {g.weight}")

            bands = [parse_band_key(m, n) for m in g.members]
            # 1) 同 k：min 只能比较同一类窗口
            ks = {b.k for b in bands}
            if len(ks) != 1:
                raise ValueError(
                    f"BalanceGroup 成员窗口大小必须相同，当前 k={sorted(ks)}"
                )
            # 2) 组内成员不可被多个 BalanceGroup 引用。
            #    允许与 SumGroup 同时出现，用于 balanced_composite 这类
            #    “sum + eps*min” 目标；语义由配置者负责。
            for b in bands:
                if b in balance_member_to_group:
                    raise ValueError(
                        f"带 {b.to_str()} 被多个 BalanceGroup 引用"
                    )
                balance_member_to_group[b] = g


def parse_band_key(text: str, n_intersections: int | None = None) -> BandKey:
    """解析带标识。

    支持：
      "up.global" / "down.global"     -> k=n 的全局带
      "up.seg1" / "down.seg2"         -> k=2 的分段带，seg 编号从 1 开始
      "up.win3@I1-I3"                 -> k=3，起点路口为 I1

    参数：
        text: 带标识字符串；
        n_intersections: 路口数量；解析 global 和 win 时需要。

    返回：
        BandKey(direction, k, start)
    """
    if text.endswith(".global"):
        direction = text.split(".", 1)[0]
        if direction not in ("up", "down"):
            raise ValueError(f"未知方向: {direction}")
        if n_intersections is None:
            raise ValueError("global 带需要 n_intersections")
        return BandKey(direction, n_intersections, 0)

    if ".seg" in text:
        direction, rest = text.split(".", 1)
        if direction not in ("up", "down"):
            raise ValueError(f"未知方向: {direction}")
        if not rest.startswith("seg"):
            raise ValueError(f"无法解析分段带标识: {text}")
        seg_idx = int(rest[3:]) - 1  # seg1 -> 0
        return BandKey(direction, 2, seg_idx)

    if ".win" in text:
        direction, rest = text.split(".", 1)
        if direction not in ("up", "down"):
            raise ValueError(f"未知方向: {direction}")
        if not rest.startswith("win"):
            raise ValueError(f"无法解析窗口带标识: {text}")
        k_part, rng = rest.split("@", 1)
        k = int(k_part[3:])
        first = rng.split("-")[0]
        # 路口名 -> 下标：默认解析为 "I" 开头，否则按文本序匹配
        if n_intersections is None:
            raise ValueError("win 带需要 n_intersections")
        if first.startswith("I"):
            start = int(first[1:]) - 1
        else:
            start = int(first) - 1
        return BandKey(direction, k, start)

    raise ValueError(f"无法解析带标识: {text}")


def composite_config(up_weight: float = 1.0,                 # 上行全局带权重
                     down_weight: float = 1.0,               # 下行全局带权重
                     objective_mode: str = "sum",            # 目标模式：sum / balanced / balanced_composite
                     balance_eps: float = 0.1,               # 复合目标里 min 项的权重
                     balance_terms: tuple[str, ...] = ("up", "down")) -> ObjectiveConfig:
    """CompositeBandSolver 的预设目标配置。

    生成目标：
        sum                 -> max up_weight*b_up + down_weight*b_down
        balanced            -> max min(b_up, b_down)
        balanced_composite  -> max up_weight*b_up + down_weight*b_down
                                    + balance_eps * min(b_up, b_down)
    """
    if objective_mode == "sum":                              # 情况 1：纯加权和
        return ObjectiveConfig(sum_groups=[SumGroup({        # 创建一个 SumGroup
            "up.global": up_weight,                          # 上行全局带权重
            "down.global": down_weight,                      # 下行全局带权重
        })])

    if objective_mode == "balanced":                         # 情况 2：纯均衡
        members = [f"{d}.global" for d in balance_terms]     # 例如 ["up.global", "down.global"]
        return ObjectiveConfig(balance_groups=[BalanceGroup(members, weight=1.0)])  # max min(members)

    if objective_mode == "balanced_composite":               # 情况 3：加权和 + 均衡托底
        members = [f"{d}.global" for d in balance_terms]     # 均衡组包含哪些全局带
        return ObjectiveConfig(
            sum_groups=[SumGroup({                           # 1) 先加常规加权和
                "up.global": up_weight,                      # 上行全局带权重
                "down.global": down_weight,                  # 下行全局带权重
            })],
            balance_groups=[BalanceGroup(members, weight=balance_eps)],  # 2) 再加 eps*min
        )

    raise ValueError(f"unknown objective_mode: {objective_mode}")  # 未知模式直接报错


def oneway_config(up_weight: float = 1.0,                        # 上行全局带权重
                  window_weights: dict[int, float] | None = None,  # 任意窗口权重：{k: w_k}
                  segment_down_weights: dict[str, float] | None = None,  # 下行逐段权重（可选）
                  n_intersections: int = 0,                      # 路口数量
                  normalize_window_weights: bool = True) -> ObjectiveConfig:
    """OneWayPrioritySolver 的预设目标配置（下行分段 + 任意窗口带）。

    参数：
        up_weight: 上行全局带 b_up_global 的权重；
        window_weights: 下行窗口权重字典：
            {2: w2}          -> 只奖励每个下行路段；
            {2: w2, 3: w3}   -> 再奖励每个下行三路口窗口；
            {2: w2, 3: w3, 4: w4, ...} -> 支持任意 k <= n；
            如果不写 2，默认 w2 = 1.0；
        segment_down_weights: 下行逐段权重，优先级高于 window_weights[2]；
            例如 {"seg1": 2.0, "seg3": 0.5}；
        n_intersections: 路口数量 n，用于展开所有 k 窗口。
        normalize_window_weights:
            是否按窗口数量归一化权重；默认 True。

            对窗口大小 k，内部每个窗口权重变为：
                w_k / (n - k + 1)

            这样“所有 k 窗口带的总权重”约为 w_k，
            不再随路口数量 n 线性放大。

    生成目标：
        max up_weight * b_up_global
          + Σ 下行每段权重 * b_down_seg
          + Σ_k w_k * 所有下行 k 窗口带

    若 normalize_window_weights=True，则最后一行内部实际为：
        Σ_k (w_k / 窗口数) * 所有 k 窗口带
    """
    ww = dict(window_weights or {})                              # 复制窗口权重，避免修改原字典
    if not ww:                                                   # 如果没有提供任何窗口权重
        ww = {2: 1.0}                                            # 默认只奖励下行每个路段

    seg_weights = dict(segment_down_weights or {})               # 复制逐段权重
    terms: dict[str, float] = {"up.global": up_weight}           # 先放上行全局带

    # k=2：每个下行路段。即使 ww 里没有 2，也用默认权重 1.0。
    # 局部段数量 = n-1，若开启归一化，则每个默认段权重除以 n-1。
    w2 = ww.get(2, 1.0)                                          # k=2 的默认权重
    n_seg = n_intersections - 1
    for i in range(n_seg):                                       # 遍历所有相邻路口段
        seg_name = f"seg{i+1}"                                   # seg1, seg2, ...
        if seg_name in seg_weights:
            # 显式逐段权重：用户自己指定，不再按数量缩放。
            terms[f"down.{seg_name}"] = seg_weights[seg_name]
        else:
            seg_weight = w2
            if normalize_window_weights and n_seg > 0:
                seg_weight = w2 / n_seg
            terms[f"down.{seg_name}"] = seg_weight

    # k>=3：任意窗口大小。
    for k, weight in ww.items():                                 # 遍历用户配置的每个 k
        if k == 2:                                               # k=2 已在上面处理
            continue
        if k < 2 or k > n_intersections:                         # 越界窗口直接报错
            raise ValueError(
                f"oneway_config: 非法窗口大小 k={k}, "
                f"要求 2 <= k <= {n_intersections}"
            )
        if weight <= 0:                                          # 非正权重不加入目标
            continue

        n_windows = n_intersections - k + 1                      # 该 k 的窗口数量
        effective_weight = weight
        if normalize_window_weights and n_windows > 0:
            effective_weight = weight / n_windows

        for start1 in range(1, n_intersections - k + 2):         # 起点路口编号从 I1 开始
            end1 = start1 + k - 1                                # 终点路口编号
            key = f"down.win{k}@I{start1}-I{end1}"               # 例如 down.win4@I1-I4
            terms[key] = effective_weight                        # 加入归一化后的窗口权重

    return ObjectiveConfig(sum_groups=[SumGroup(terms)])         # 所有项放进一个 SumGroup
