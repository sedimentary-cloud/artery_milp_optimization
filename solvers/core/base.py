"""求解器抽象基类。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ...models import Arterial
from ...solution import Solution


def compute_effective_max_loops(arterial: Arterial, user_max_loops: int) -> int:
    """计算实际使用的 loop 上界。

    ``self.max_loops`` 只作为用户手动下限；对于长路段，自动根据旅行时间放大：

        L_e = ceil(travel_time / cycle) + 1

    所有路段取最大值，保证同一个 max_loops 覆盖整条干线。
    """
    required = 0
    for segment in arterial.segment_order:
        for travel_time in (segment.travel_time_up, segment.travel_time_down):
            required = max(
                required,
                int(math.ceil(float(travel_time) / arterial.cycle)) + 1,
            )
    return max(int(user_max_loops), required)


class Solver(ABC):
    """所有求解器（MILP / GA / 混合）的统一接口。"""

    name: str = "base"

    @abstractmethod
    def solve(self, arterial: Arterial) -> Solution:
        """对给定干线求解，返回一套协调方案。"""
        ...

    def build(self, arterial: Arterial) -> None:
        """可选：预先构建模型（便于检查约束、导出模型文件）。"""
        return None
