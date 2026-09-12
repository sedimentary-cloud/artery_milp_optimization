"""FlexiblePhaseTuneSolver：BandModel + ObjectiveConfig + 相位变量。

这是新架构对外的第二阶段求解器：

    g_{i,p}          相位绿灯时长（秒）
    b[d,i]           基础段带宽
    B[d,k,j]         窗口带格
    tU_i / tD_i      带前沿时刻
    mU_i / mD_i      圈数

相位生成绿灯窗，绿灯窗约束基础段带宽，BandModel 负责窗口带格，
ObjectiveConfig 负责目标，损失系统负责 total_loss。

PhaseTuneSolver 是薄包装器：根据 mode 生成 ObjectiveConfig，
然后把所有参数原样转给 FlexiblePhaseTuneSolver。
"""

from __future__ import annotations

from ..models import Arterial
from ..solution import Solution
from .base import Solver
from .flexible_band_solver import composite_config, oneway_config
from .objective_config import ObjectiveConfig
from .full_flexible_phase_solver import FullFlexiblePhaseTuneSolver


class FlexiblePhaseTuneSolver(Solver):
    """完整新架构求解器：相位变量 + BandModel + ObjectiveConfig。"""

    name = "flexible-phase-tune"

    def __init__(self,
                 config: ObjectiveConfig,
                 mode: str = "global",
                 down_weight: float = 1.0,
                 up_weight: float = 1.0,
                 window_weights: dict[int, float] | None = None,
                 max_loops: int = 3) -> None:
        self.config = config
        self.mode = mode
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.window_weights = window_weights
        self.max_loops = max_loops

    def solve(self, arterial: Arterial,
              prior: Solution | None = None,
              loss_builder=None,
              constraint_builder=None,
              alignment_builder=None,
              max_loss: float | None = None,
              max_intersection_loss: float | None = None,
              band_loss_weight: float = 0.0,
              objective: str = "bandwidth",
              tunable_intersections: set[str] | None = None) -> Solution:
        down_global_output = (self.mode == "global")
        solver = FullFlexiblePhaseTuneSolver(
            config=self.config,
            max_loops=self.max_loops,
            mode=self.mode,
            up_global_output=True,
            down_global_output=down_global_output,
        )
        return solver.solve(
            arterial,
            prior=prior,
            loss_builder=loss_builder,
            constraint_builder=constraint_builder,
            alignment_builder=alignment_builder,
            max_loss=max_loss,
            max_intersection_loss=max_intersection_loss,
            band_loss_weight=band_loss_weight,
            objective=objective,
            tunable_intersections=tunable_intersections,
        )


class PhaseTuneSolver(Solver):
    """薄包装器：构造 ObjectiveConfig，然后交给 FlexiblePhaseTuneSolver。

    支持：
    - 默认带宽目标；
    - loss_builder / constraint_builder / alignment_builder；
    - max_loss / objective="loss"；
    - tunable_intersections；
    - global / oneway 两种模式。
    """

    name = "phase-tune"

    def __init__(self,
                 mode: str = "global",
                 down_weight: float = 1.0,
                 up_weight: float = 1.0,
                 window_weights: dict[int, float] | None = None,
                 max_loops: int = 3,
                 objective_mode: str = "sum",
                 balance_eps: float = 0.1,
                 balance_terms: tuple[str, ...] = ("up", "down"),
                 tunable_intersections: set[str] | None = None) -> None:
        self.mode = mode
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.window_weights = window_weights
        self.max_loops = max_loops
        self.objective_mode = objective_mode
        self.balance_eps = balance_eps
        self.balance_terms = tuple(balance_terms)
        self.tunable_intersections = tunable_intersections

    def _build_config(self, arterial: Arterial) -> ObjectiveConfig:
        if self.mode == "global":
            return composite_config(
                up_weight=self.up_weight,
                down_weight=self.down_weight,
                objective_mode=self.objective_mode,
                balance_eps=self.balance_eps,
                balance_terms=self.balance_terms,
            )
        if self.mode == "oneway":
            return oneway_config(
                up_weight=self.up_weight,
                window_weights=self.window_weights,
                n_intersections=len(arterial.intersection_order),
            )
        raise ValueError(f"unknown mode: {self.mode}")

    def solve(self, arterial: Arterial,
              prior: Solution | None = None,
              loss_builder=None,
              constraint_builder=None,
              alignment_builder=None,
              max_loss: float | None = None,
              max_intersection_loss: float | None = None,
              band_loss_weight: float = 0.0,
              objective: str = "bandwidth") -> Solution:
        config = self._build_config(arterial)
        solver = FlexiblePhaseTuneSolver(
            config=config,
            mode=self.mode,
            down_weight=self.down_weight,
            up_weight=self.up_weight,
            window_weights=self.window_weights,
            max_loops=self.max_loops,
        )
        return solver.solve(
            arterial,
            prior=prior,
            loss_builder=loss_builder,
            constraint_builder=constraint_builder,
            alignment_builder=alignment_builder,
            max_loss=max_loss,
            max_intersection_loss=max_intersection_loss,
            band_loss_weight=band_loss_weight,
            objective=objective,
            tunable_intersections=self.tunable_intersections,
        )
