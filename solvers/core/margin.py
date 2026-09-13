"""绿波带硬/软边距配置。

语义
----
- 硬边距（hard margin）：绿波带必须离“有效绿灯窗口”的边界至少这么远。
  Stage 1 与 Stage 2 都使用；它通过缩小有效绿灯窗口实现。
- 软边距（soft margin）：在硬边距之外，希望绿波带离边界更远；
  如果实际边距小于 soft margin，就产生 band loss。
  只在 Stage 2 生效（方案已锁定，可以精确线性化）。

所有 margin 均使用“占周期比例”；求解器内部会乘以 `cycle` 转成秒。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BandMarginConfig:
    """硬/软边距配置。"""

    hard_margin_up: float = 0.0
    hard_margin_down: float = 0.0
    soft_margin_up: float = 0.0
    soft_margin_down: float = 0.0
    penalty_up: float = 1.0
    penalty_down: float = 1.0

    @property
    def has_soft_margin(self) -> bool:
        """是否存在需要建模的软边距。"""
        return (
            self.soft_margin_up > 0.0 and self.penalty_up > 0.0
        ) or (
            self.soft_margin_down > 0.0 and self.penalty_down > 0.0
        )

    def validate(self) -> None:
        """校验边距配置。"""
        for name, hard, soft in (
            ("up", self.hard_margin_up, self.soft_margin_up),
            ("down", self.hard_margin_down, self.soft_margin_down),
        ):
            if hard < 0.0:
                raise ValueError(f"{name} hard_margin 不能为负: {hard}")
            if soft < 0.0:
                raise ValueError(f"{name} soft_margin 不能为负: {soft}")
            if soft < hard:
                raise ValueError(
                    f"{name} 方向要求 soft_margin >= hard_margin，"
                    f"当前 soft={soft}, hard={hard}"
                )
        if self.penalty_up < 0.0:
            raise ValueError("penalty_up 不能为负")
        if self.penalty_down < 0.0:
            raise ValueError("penalty_down 不能为负")


__all__ = ["BandMarginConfig"]
