"""求解器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Arterial
from ..solution import Solution


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
