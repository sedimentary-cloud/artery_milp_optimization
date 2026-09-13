"""统一的约束/损失 term 校验层。

背景
----
在引入本模块之前，``LinearSpec`` / ``SegmentLossSpec`` 里的 term 是在
各求解器内部边解析边建模的，失败模式不一致：

- Stage 1 引用某个候选方案不存在的段号，会直接抛 ``IndexError``；
- 端点后缀写错（如 ``up.1.foo``）在 Stage 1 会被静默当成 ``end``；
- 路口名拼错、特殊变量不存在、忘记写路口前缀等，会静默丢弃整条
  约束/损失，或把表达式截短后继续建模；
- 部分 term 解析失败时，约束含义会被静默改变。

本模块在 MILP 组装之前做一次统一的语法 + 可用性校验，把所有上述
情况变成一条清晰的 ``TermValidationError``，不再静默丢弃或截短。

term 分类
---------
1. 路口段端点：``{Int}.{dir}.{idx}.{start|end}``，例如
   ``I2.up.1.start`` / ``I3.down.2.end``。
2. 旧特殊带宽/传播变量：

   - ``b_up`` / ``b_down``、``B_bal``、``tU_{Int}`` / ``tD_{Int}``、
     ``bU_{Seg}`` / ``bD_{Seg}`` 等。

   这些特殊变量属于旧外部扩展接口，当前版本已不建议使用；
   Stage 1 / Stage 2 中的解析逻辑只保留注释，写入后会主动报错。

可用性规则
----------
- 路口段端点：
    - Stage 1：要求**所有候选方案**共同定义该端点（因为 Stage 1
      会对每个候选方案取端点常量）；
    - Stage 2：要求**选中方案**定义该端点。
- 特殊变量：要求对应方向、物理路段、均衡组在当前上下文中存在。

校验失败会一次性汇总所有错误，避免修一个报一个。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .signal_constraints import LinearSpec

_VALID_DIRECTIONS = ("up", "down")
_VALID_ENDPOINTS = ("start", "end")
_VALID_SENSES = ("<=", ">=", "=")
_EXACT_BAND_TERMS = {"b_up": "up", "b_down": "down"}


class TermValidationError(ValueError):
    """约束/损失 term 校验失败。"""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        message = "约束/损失 term 校验失败:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


@dataclass(frozen=True)
class EndpointRef:
    """路口段端点 term 的解析结果。"""

    intersection: str
    direction: str
    segment_no: int
    endpoint: str

    @property
    def local_name(self) -> str:
        """例如 ``up.1.start``。"""
        return f"{self.direction}.{self.segment_no}.{self.endpoint}"


@dataclass(frozen=True)
class SpecialRef:
    """特殊带宽/传播变量 term 的解析结果。"""

    kind: str                 # b_up / b_down / B_bal / tU / tD / bU / bD
    direction: str | None = None
    target: str | None = None  # 路口名或物理路段名


@dataclass
class TermValidationContext:
    """term 校验上下文。

    Attributes:
        intersection_names: 干线上的路口名列表（按顺序）。
        segment_names: 干线上的物理路段名列表（按顺序）。
        active_directions: 当前存在可用带宽变量的方向集合。
        active_segment_numbers: 方向 -> 当前存在全走廊带宽实例的段号集合。
        balance_group_count: ``ObjectiveConfig.balance_groups`` 的数量。
        endpoint_terms_by_intersection:
            路口名 -> 允许使用的局部端点 term 集合，例如
            ``{"I1": {"up.1.start", "up.1.end", "down.1.start", ...}}``。
        plan_names_by_intersection:
            路口名 -> 候选方案名集合，用于校验 ``plan_tags``。
    """

    intersection_names: list[str]
    segment_names: list[str]
    active_directions: set[str]
    active_segment_numbers: dict[str, set[int]]
    balance_group_count: int
    endpoint_terms_by_intersection: dict[str, set[str]]
    plan_names_by_intersection: dict[str, set[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def validate_specs(self, specs: Iterable[tuple[LinearSpec, str]]) -> None:
        """校验一批 ``(LinearSpec, kind)``；失败时抛 ``TermValidationError``。

        Args:
            specs: 待校验的约束/损失列表，``kind`` 仅用于报错信息。
        """
        errors: list[str] = []
        for idx, item in enumerate(specs):
            if not isinstance(item, tuple) or len(item) != 2:
                errors.append(f"第 {idx} 个 spec 格式非法，期望 (LinearSpec, kind)")
                continue
            spec, kind = item
            if not isinstance(spec, LinearSpec):
                errors.append(f"第 {idx} 个 spec 不是 LinearSpec")
                continue
            label = self._spec_label(spec, idx, kind)

            if spec.sense not in _VALID_SENSES:
                errors.append(f"[{label}] 非法 sense: {spec.sense!r}")
                continue
            if spec.soft and spec.sense == "=":
                errors.append(f"[{label}] 不支持 soft '=' 约束")
                continue
            if not spec.terms:
                errors.append(f"[{label}] terms 不能为空")
                continue

            if spec.plan_tags:
                self._validate_plan_tags(spec.plan_tags, label, errors)

            for term in spec.terms:
                try:
                    self.parse_term(term)
                except TermValidationError as exc:
                    for sub in exc.errors:
                        errors.append(f"[{label}] {sub}")
                except ValueError as exc:
                    errors.append(f"[{label}] term {term!r} 非法: {exc}")

        if errors:
            raise TermValidationError(errors)

    def parse_term(self, term: str) -> EndpointRef | SpecialRef:
        """解析并校验单个 term；成功返回解析结果，失败抛异常。"""
        if not isinstance(term, str) or not term:
            raise ValueError("term 必须是非空字符串")

        parts = term.split(".")
        # 优先级与求解器内部一致：先判断路口段端点。
        if len(parts) == 4 and parts[0] in self.intersection_names:
            return self._parse_endpoint(term, parts)

        special = self._parse_special(term)
        if special is not None:
            return special

        raise ValueError(
            "无法识别的 term；期望 {Int}.{up|down}.{段号}.{start|end}，"
            "或 b_up / b_down / B_bal / tU_{Int} / tD_{Int} / "
            "bU_{Seg} / bD_{Seg}"
        )

    # ------------------------------------------------------------------
    # 内部：端点
    # ------------------------------------------------------------------
    def _parse_endpoint(self, term: str, parts: list[str]) -> EndpointRef:
        intersection, direction, raw_idx, endpoint = parts

        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"方向 {direction!r} 非法，只能是 up / down（term={term!r}）"
            )
        if endpoint not in _VALID_ENDPOINTS:
            raise ValueError(
                f"端点 {endpoint!r} 非法，只能是 start / end（term={term!r}）"
            )
        try:
            segment_no = int(raw_idx)
        except ValueError as exc:
            raise ValueError(f"段号 {raw_idx!r} 不是整数（term={term!r}）") from exc
        if segment_no < 1:
            raise ValueError(f"段号必须从 1 开始，当前为 {segment_no}（term={term!r}）")

        local_name = f"{direction}.{segment_no}.{endpoint}"
        allowed = self.endpoint_terms_by_intersection.get(intersection, set())
        if local_name not in allowed:
            available = sorted(allowed) if allowed else []
            raise ValueError(
                f"路口 {intersection} 上不存在可用端点 {local_name!r}"
                f"（Stage 1 要求所有候选方案共同定义；"
                f"Stage 2 要求选中方案定义）。"
                f"当前可用端点示例: {available[:8]}"
            )
        return EndpointRef(intersection, direction, segment_no, endpoint)

    # ------------------------------------------------------------------
    # 内部：特殊变量
    # ------------------------------------------------------------------
    def _parse_special(self, term: str) -> SpecialRef | None:
        if term in _EXACT_BAND_TERMS:
            direction = _EXACT_BAND_TERMS[term]
            self._require_direction(term, direction)
            return SpecialRef(kind=term, direction=direction)

        if term == "B_bal":
            if self.balance_group_count <= 0:
                raise ValueError(
                    "B_bal 需要至少一个 BalanceGroup；"
                    "当前 ObjectiveConfig.balance_groups 为空"
                )
            return SpecialRef(kind="B_bal")

        for prefix, kind, direction in (
            ("tU_", "tU", "up"),
            ("tD_", "tD", "down"),
        ):
            if term.startswith(prefix):
                target = term[len(prefix):]
                if target not in self.intersection_names:
                    raise ValueError(
                        f"{kind} 变量引用了不存在的路口 {target!r}（term={term!r}）"
                    )
                self._require_direction(term, direction)
                return SpecialRef(kind=kind, direction=direction, target=target)

        for prefix, kind, direction in (
            ("bU_", "bU", "up"),
            ("bD_", "bD", "down"),
        ):
            if term.startswith(prefix):
                target = term[len(prefix):]
                self._require_segment(term, target, direction)
                return SpecialRef(kind=kind, direction=direction, target=target)

        return None

    def _require_direction(self, term: str, direction: str) -> None:
        if direction not in self.active_directions:
            raise ValueError(
                f"{direction} 方向当前没有可用的全走廊带宽变量"
                f"（term={term!r}）；请检查候选方案的绿波段配置"
            )

    def _require_segment(self, term: str, segment: str, direction: str) -> None:
        # bU_/bD_ 会作用在该方向所有活跃段号实例的同一物理路段宽度上；
        # 只要方向有带、且物理路段存在即可，不要求该路段本身是一个
        # 独立的“全走廊段号实例”。
        if segment not in self.segment_names:
            raise ValueError(
                f"{direction} 变量引用了不存在的物理路段 {segment!r}（term={term!r}）"
            )
        self._require_direction(term, direction)

    # ------------------------------------------------------------------
    # 内部：plan_tags
    # ------------------------------------------------------------------
    def _validate_plan_tags(self,
                            plan_tags: dict[str, str],
                            label: str,
                            errors: list[str]) -> None:
        for int_name, plan_name in plan_tags.items():
            if int_name not in self.intersection_names:
                errors.append(
                    f"[{label}] plan_tags 引用了不存在的路口 {int_name!r}"
                )
                continue
            known_plans = self.plan_names_by_intersection.get(int_name)
            if known_plans is not None and plan_name not in known_plans:
                errors.append(
                    f"[{label}] plan_tags 引用了路口 {int_name} 上"
                    f"不存在的方案 {plan_name!r}；"
                    f"候选方案: {sorted(known_plans)}"
                )

    @staticmethod
    def _spec_label(spec: LinearSpec, idx: int, kind: str) -> str:
        if spec.name:
            return f"{kind}:{spec.name}"
        terms = ", ".join(repr(t) for t in list(spec.terms)[:2])
        return f"{kind}:spec#{idx}({terms})"


# ----------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------
def common_endpoint_terms(plans: Sequence[object]) -> set[str]:
    """返回一组方案共同拥有的局部端点 term 集合。

    用于 Stage 1：只要某个候选方案缺少某端点，该端点就不可用于
    通用约束，否则求解器会在遍历候选方案时崩溃。
    """
    sets = [set(plan.all_segment_terms()) for plan in plans]  # type: ignore[attr-defined]
    if not sets:
        return set()
    return set.intersection(*sets)


def union_endpoint_terms(plans: Sequence[object]) -> set[str]:
    """返回一组方案出现过的所有局部端点 term 集合。"""
    out: set[str] = set()
    for plan in plans:
        out.update(plan.all_segment_terms())  # type: ignore[attr-defined]
    return out


__all__ = [
    "TermValidationError",
    "TermValidationContext",
    "EndpointRef",
    "SpecialRef",
    "common_endpoint_terms",
    "union_endpoint_terms",
]
