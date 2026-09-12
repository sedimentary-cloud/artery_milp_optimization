"""FlexiblePhaseTuneSolver：相位微调 + BandModel + ObjectiveConfig。

当前作为桥接版本：
1. 先用现有 PhaseTuneSolver 做相位时长优化，得到 phase_times；
2. 用 phase_times 生成固定绿灯窗，锁住选中方案；
3. 再用 FlexibleBandSolver + ObjectiveConfig 做最终带宽组合优化。
"""

from __future__ import annotations

from ..models import Arterial, GreenWindow, Intersection, SignalPlan
from ..solution import Solution
from .base import Solver
from .flexible_band_solver import (FlexibleBandSolver, composite_config,
                                  oneway_config)
from .objective_config import ObjectiveConfig
from .staged import _LegacyPhaseTuneSolver


class FlexiblePhaseTuneSolver(Solver):
    """带相位微调的 FlexibleBandSolver 桥接版。"""

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
              max_loss=None,
              objective: str = "bandwidth") -> Solution:
        # 1) 相位时长优化（高级功能沿用 legacy 求解器）
        tuner = _LegacyPhaseTuneSolver(mode=self.mode,
                                       down_weight=self.down_weight,
                                       up_weight=self.up_weight,
                                       window_weights=self.window_weights,
                                       max_loops=self.max_loops)
        s_phase = tuner.solve(arterial, prior=prior,
                              loss_builder=loss_builder,
                              constraint_builder=constraint_builder,
                              alignment_builder=alignment_builder,
                              max_loss=max_loss,
                              objective=objective)
        if s_phase.status != "optimal":
            return s_phase

        selected = {}
        for inter in arterial.intersection_order:
            name = s_phase.plan_choices.get(inter.name) if s_phase.plan_choices else None
            plan = inter.plan_by_name(name) if name else inter.plans[0]
            selected[inter.name] = plan

        # 2) 用 phase_times 生成固定窗口，锁定方案
        new_intersections = {}
        for inter in arterial.intersection_order:
            plan = selected[inter.name]
            pt = s_phase.phase_times.get(inter.name, {})
            wu = plan.up_windows[0] if plan.up_windows else None
            wd = plan.down_windows[0] if plan.down_windows else None

            if plan.phases and pt:
                starts = {}
                acc = 0.0
                for ph in plan.phases:
                    starts[ph.name] = acc
                    acc += float(pt.get(ph.name, ph.green))
                if plan.up_phase in starts and plan.up_phase in pt:
                    us = starts[plan.up_phase]
                    wu = GreenWindow(us / arterial.cycle,
                                     (us + float(pt[plan.up_phase])) / arterial.cycle)
                if plan.down_phase in starts and plan.down_phase in pt:
                    ds = starts[plan.down_phase]
                    wd = GreenWindow(ds / arterial.cycle,
                                     (ds + float(pt[plan.down_phase])) / arterial.cycle)

            if wu is None or wd is None:
                raise ValueError(f"路口 {inter.name} 缺少可用绿灯窗")

            fixed_plan = SignalPlan(
                name=plan.name,
                up_windows=[wu],
                down_windows=[wd],
                phases=list(plan.phases),
                up_phase=plan.up_phase,
                down_phase=plan.down_phase,
                lost_time=plan.lost_time,
            )
            new_intersections[inter.name] = Intersection(inter.name, [fixed_plan])

        fixed_arterial = Arterial(
            cycle=arterial.cycle,
            intersections=new_intersections,
            segments=arterial.segments,
            order=arterial.order,
        )

        # 3) 用 BandModel + ObjectiveConfig 做最终带宽组合优化
        s_final = FlexibleBandSolver(
            self.config,
            max_loops=self.max_loops,
            name=self.name,
            up_style="global",
            down_style="global" if self.mode == "global" else "local",
            up_global_output=True,
            down_global_output=(self.mode == "global"),
        ).solve(fixed_arterial)

        # 4) 回填相位信息
        if s_final.status == "optimal":
            s_final.phase_times = s_phase.phase_times
            s_final.total_phase_loss = s_phase.total_phase_loss
            s_final.plan_choices = s_phase.plan_choices or s_final.plan_choices
        return s_final


class PhaseTuneSolver(Solver):
    """薄包装器：

    - 默认带宽目标：走 FlexiblePhaseTuneSolver（BandModel + ObjectiveConfig）；
    - 带 loss/constraint/alignment/max_loss/objective=loss 或 tunable_intersections：
      回退到 legacy PhaseTuneSolver，保持既有行为。
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

    def solve(self, arterial: Arterial,
              prior: Solution | None = None,
              loss_builder=None,
              constraint_builder=None,
              alignment_builder=None,
              max_loss=None,
              objective: str = "bandwidth") -> Solution:
        use_legacy = (
            loss_builder is not None
            or constraint_builder is not None
            or alignment_builder is not None
            or max_loss is not None
            or objective == "loss"
            or self.tunable_intersections is not None
        )
        if use_legacy:
            legacy = _LegacyPhaseTuneSolver(
                mode=self.mode,
                down_weight=self.down_weight,
                up_weight=self.up_weight,
                window_weights=self.window_weights,
                max_loops=self.max_loops,
                objective_mode=self.objective_mode,
                balance_eps=self.balance_eps,
                balance_terms=self.balance_terms,
            )
            return legacy.solve(arterial, prior=prior,
                                loss_builder=loss_builder,
                                constraint_builder=constraint_builder,
                                alignment_builder=alignment_builder,
                                max_loss=max_loss,
                                objective=objective)

        # 默认带宽目标：走新 BandModel + ObjectiveConfig 路径
        if self.mode == "global":
            cfg = composite_config(up_weight=self.up_weight,
                                   down_weight=self.down_weight,
                                   objective_mode=self.objective_mode,
                                   balance_eps=self.balance_eps,
                                   balance_terms=self.balance_terms)
        elif self.mode == "oneway":
            cfg = oneway_config(up_weight=self.up_weight,
                                window_weights=self.window_weights,
                                n_intersections=len(arterial.intersection_order))
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        solver = FlexiblePhaseTuneSolver(
            config=cfg,
            mode=self.mode,
            down_weight=self.down_weight,
            up_weight=self.up_weight,
            window_weights=self.window_weights,
            max_loops=self.max_loops,
        )
        return solver.solve(arterial, prior=prior)
