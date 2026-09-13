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
