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
from .flexible_band_solver import FlexibleBandSolver
from .objective_config import ObjectiveConfig
from .staged import PhaseTuneSolver


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
              prior: Solution | None = None) -> Solution:
        # 1) 相位时长优化
        tuner = PhaseTuneSolver(mode=self.mode,
                                down_weight=self.down_weight,
                                up_weight=self.up_weight,
                                window_weights=self.window_weights,
                                max_loops=self.max_loops)
        s_phase = tuner.solve(arterial, prior=prior)
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
        s_final = FlexibleBandSolver(self.config,
                                     max_loops=self.max_loops,
                                     name=self.name,
                                     up_style="global",
                                     down_style="global" if self.mode == "global" else "local").solve(fixed_arterial)

        # 4) 回填相位信息
        if s_final.status == "optimal":
            s_final.phase_times = s_phase.phase_times
            s_final.total_phase_loss = s_phase.total_phase_loss
            s_final.plan_choices = s_phase.plan_choices or s_final.plan_choices
        return s_final
