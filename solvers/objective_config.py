"""目标配置层：把“全局带 / 分段带 / 小绿波带”统一为带标识。

目标由两类块组成：

SumGroup
    自由加权和：
        Σ w_key * band_key
    任意方向、任意窗口大小、任意权重。

BalanceGroup
    组内取 min：
        B_group <= band_member   for each member
        目标加入 weight * B_group
    默认带 ε 托底：
        + weight * eps * Σ band_member
    校验规则：
        - 组内成员窗口大小 k 必须相同；
        - 成员不可被多个 BalanceGroup 引用；
        - 允许与 SumGroup 共存，用于 sum + eps*min 的复合目标。

带标识示例：
    "up.global"            -> k = n
    "down.seg2"            -> k = 2
    "up.win3@I1-I3"        -> k = 3
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BandKey:
    """窗口带格中的一个带标识。"""
    direction: str       # "up" / "down"
    k: int               # 窗口大小：2 表示分段带，n 表示全局带
    start: int           # 起点路口下标，0-based

    def to_str(self) -> str:
        return f"{self.direction}.win{self.k}@{self.start}"


@dataclass
class SumGroup:
    """加权求和块：terms = {带标识: 权重}。"""
    terms: dict[str, float] = field(default_factory=dict)

    def parsed_terms(self):
        out = {}
        for key, w in self.terms.items():
            band = parse_band_key(key)
            out[band] = w
        return out


@dataclass
class BalanceGroup:
    """均衡块：组内取 min，组整体以 weight 进入目标。"""
    members: list[str] = field(default_factory=list)
    weight: float = 1.0
    eps: float = 0.01  # 托底项：group_weight * eps * sum(members)


@dataclass
class ObjectiveConfig:
    """一个完整目标配置。

    sum_groups: 任意方向的带自由加权求和；
    balance_groups: 组内取 min 后加权进入目标。
    """
    sum_groups: list[SumGroup] = field(default_factory=list)
    balance_groups: list[BalanceGroup] = field(default_factory=list)

    def validate(self, n_intersections: int) -> None:
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
            # 1) 同 k
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

    def all_band_keys(self, n_intersections: int) -> set[BandKey]:
        keys: set[BandKey] = set()
        for g in self.sum_groups:
            for key in g.terms:
                keys.add(parse_band_key(key, n_intersections))
        for g in self.balance_groups:
            for m in g.members:
                keys.add(parse_band_key(m, n_intersections))
        return keys


def parse_band_key(text: str, n_intersections: int | None = None) -> BandKey:
    """解析带标识。

    支持：
      "up.global" / "down.global"     -> k=n 的全局带
      "up.seg1" / "down.seg2"         -> k=2 的分段带，seg 编号从 1 开始
      "up.win3@I1-I3"                 -> k=3，起点路口为 I1
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
