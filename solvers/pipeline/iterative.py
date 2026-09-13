"""迭代式两阶段求解：不断把 Stage 2 调好的端点写回 Stage 1。

与 ``TwoStageSolver`` 的区别：
- ``TwoStageSolver`` 只跑一次 Stage 1 + Stage 2；
- ``IterativeTwoStageSolver`` 会把 Stage 2 微调后的绿灯端点写回
  ``Arterial``，再重新跑 Stage 1 的窗口分配，直到窗口分配稳定。

收敛判据只检查窗口分配：
- 如果本轮窗口分配与上一轮相同，视为收敛；
- 如果窗口分配进入循环，返回“重复状态出现前一轮”的解：
  A -> B -> A 返回 B，A -> B -> C -> B 返回 C；
- 每轮 Stage 2 都必须返回 optimal，否则抛出 ``RuntimeError``。
"""

from __future__ import annotations

from ...models import Arterial, GreenWindow, Intersection, SignalPlan
from ...solution import Solution
from ..core.base import Solver
from ..stage1.segmented_band import SegmentedBandSolver
from ..stage2.phase_tune import FullFlexiblePhaseTuneSolver
from .config import TwoStageConfig


class IterativeTwoStageSolver(Solver):
    """两阶段固定点迭代求解器。"""

    name = "iterative-two-stage"

    def __init__(self, config: TwoStageConfig, max_iterations: int = 10) -> None:
        """函数名：__init__；参数：TwoStageConfig、最大迭代次数；返回值：无；异常：ValueError。"""
        config.validate()
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations 必须为正")
        self.config = config
        self.max_iterations = int(max_iterations)
        self.history: list[Solution] = []

    def solve(self, arterial: Arterial) -> Solution:
        """函数名：solve；参数：arterial；返回值：Solution；异常：RuntimeError/ValueError。"""
        cfg = self.config
        mode = cfg.band.mode

        stage1 = SegmentedBandSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_global_output=True,
            down_global_output=(mode == "global"),
            margin=cfg.margin,
        )
        stage2 = FullFlexiblePhaseTuneSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_global_output=True,
            down_global_output=(mode == "global"),
            margin=cfg.margin,
        )

        current_arterial = arterial
        history: list[Solution] = []
        seen_states: dict[tuple, int] = {}
        previous_state: tuple | None = None

        for iteration in range(1, self.max_iterations + 1):
            s1 = stage1.solve(
                current_arterial,
                band_loss_weight=cfg.band.band_loss_weight,
            )
            if s1.status != "optimal":
                s1.status = f"{s1.status}|iterative_stage1_failed"
                s1.solver_msg = (
                    f"{s1.solver_msg}; iterative stopped at Stage 1 "
                    f"iteration {iteration}"
                )
                self.history = history
                return s1

            s2 = stage2.solve(
                current_arterial,
                prior=s1,
                loss_builder=cfg.intersection.loss_builder,
                constraint_builder=cfg.intersection.constraint_builder,
                band_loss_weight=cfg.band.band_loss_weight,
                objective="bandwidth",
                tunable_intersections=cfg.intersection.tunable_intersections,
            )
            if s2.status != "optimal":
                raise RuntimeError(
                    f"Stage 2 failed at iteration {iteration}: {s2.status}; "
                    f"{s2.solver_msg}"
                )

            history.append(s2)
            state = self._window_assignment_state(s2)

            # 连续两轮窗口分配相同：窗口分配已经稳定。
            if previous_state is not None and state == previous_state:
                s2.status = f"{s2.status}|iterative_converged"
                s2.solver_msg = (
                    f"{s2.solver_msg}; iterative converged at iteration {iteration}"
                )
                self.history = history
                return s2

            # 窗口分配进入循环：返回“重复状态出现前一轮”的解。
            # 例如 A -> B -> C -> B，选择 C；A -> B -> A，选择 B。
            if state in seen_states:
                result = history[-2]
                result.status = f"{result.status}|iterative_cycle"
                result.solver_msg = (
                    f"{result.solver_msg}; oscillation cycle detected at "
                    f"iteration {iteration}, returned predecessor of repeated state"
                )
                self.history = history
                return result

            seen_states[state] = len(history) - 1
            previous_state = state
            current_arterial = self._clone_arterial_with_tuned_windows(
                current_arterial, s2
            )

        last = history[-1]
        last.status = f"{last.status}|iterative_max_iter"
        last.solver_msg = (
            f"{last.solver_msg}; iterative reached max_iterations="
            f"{self.max_iterations}"
        )
        self.history = history
        return last

    @staticmethod
    def _window_assignment_state(solution: Solution) -> tuple:
        """函数名：_window_assignment_state；参数：solution；返回值：可哈希窗口分配状态。"""
        global_items: list[tuple] = []
        for direction in sorted(solution.band_window_choices.keys()):
            band_map = solution.band_window_choices.get(direction, {})
            for band_no in sorted(band_map.keys()):
                intersection_map = band_map[band_no]
                for intersection in sorted(intersection_map.keys()):
                    choice = intersection_map[intersection] or {}
                    global_items.append((
                        direction,
                        int(band_no),
                        intersection,
                        choice.get("plan"),
                        int(choice.get("window", 0)) if choice.get("window") is not None else None,
                    ))

        local_items: list[tuple] = []
        for key in sorted(solution.local_band_window_choices.keys()):
            band_map = solution.local_band_window_choices.get(key, {})
            for band_no in sorted(band_map.keys()):
                intersection_map = band_map[band_no]
                for intersection in sorted(intersection_map.keys()):
                    choice = intersection_map[intersection] or {}
                    local_items.append((
                        key,
                        int(band_no),
                        intersection,
                        choice.get("plan"),
                        int(choice.get("window", 0)) if choice.get("window") is not None else None,
                    ))

        return (tuple(global_items), tuple(local_items))

    @classmethod
    def _clone_arterial_with_tuned_windows(cls,
                                           arterial: Arterial,
                                           solution: Solution) -> Arterial:
        """函数名：_clone_arterial_with_tuned_windows；参数：arterial、solution；返回值：新 Arterial。"""
        new_intersections: dict[str, Intersection] = {}
        for inter in arterial.intersection_order:
            selected_name = solution.plan_choices.get(inter.name)
            new_plans: list[SignalPlan] = []
            for plan in inter.plans:
                if plan.name != selected_name:
                    new_plans.append(plan)
                    continue
                times = solution.segment_times.get(inter.name, {})
                new_plans.append(
                    cls._clone_plan_with_tuned_windows(plan, times, arterial.cycle)
                )
            new_intersections[inter.name] = Intersection(inter.name, new_plans)

        return Arterial(
            cycle=arterial.cycle,
            intersections=new_intersections,
            segments=dict(arterial.segments),
            order=list(arterial.order),
        )

    @staticmethod
    def _clone_plan_with_tuned_windows(plan: SignalPlan,
                                       times: dict[str, float],
                                       cycle: float) -> SignalPlan:
        """函数名：_clone_plan_with_tuned_windows；参数：plan、times、cycle；返回值：新 SignalPlan。"""
        if not times:
            return plan

        def build_segments(direction: str, original: list[GreenWindow]) -> list[GreenWindow]:
            result: list[GreenWindow] = []
            for idx, old in enumerate(original, start=1):
                start_key = f"{direction}.{idx}.start"
                end_key = f"{direction}.{idx}.end"
                if start_key not in times or end_key not in times:
                    result.append(old)
                    continue
                start = float(times[start_key]) / cycle
                end = float(times[end_key]) / cycle
                if not (0.0 <= start < end <= 1.0):
                    result.append(old)
                else:
                    result.append(GreenWindow(start, end))
            return result

        up_segments = build_segments("up", plan.up_segments)
        down_segments = build_segments("down", plan.down_segments)

        try:
            return SignalPlan(
                name=plan.name,
                up_segments=up_segments,
                down_segments=down_segments,
                signal_constraints=list(plan.signal_constraints),
                signal_losses=list(plan.signal_losses),
                metadata=dict(plan.metadata),
            )
        except ValueError:
            # 微调后的窗口不能通过 SignalPlan 校验时，保持原方案。
            return plan


__all__ = ["IterativeTwoStageSolver"]
